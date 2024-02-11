import copy
import math
import numpy as np
import sys
import logging
import pathlib
from typing import Any, Dict

import data
import model
import torch
from ruamel import yaml

import determined as det
from determined import pytorch

import utils
from coco_eval import CocoEvaluator
from coco_utils import get_coco_api_from_dataset


class CocoReducer(pytorch.MetricReducer):
    def __init__(self, dataset, iou_types):
    #def __init__(self, base_ds, iou_types, cat_ids=[]):
        self.dataset = dataset
        self.iou_types = iou_types
        self.reset()

    def reset(self):
        self.results = []

    def update(self, result):
        self.results.extend(result)

    def per_slot_reduce(self):
        return self.results

    def cross_slot_reduce(self, per_slot_metrics):
        metrics = {}
        coco = get_coco_api_from_dataset(self.dataset)
        coco_evaluator = CocoEvaluator(coco, self.iou_types)

        # Check if per_slot_metrics != ([],[]) or ([],)
        if any(slot for slot in per_slot_metrics):
            for results in per_slot_metrics:
                results_dict = {r[0]: r[1] for r in results}
                coco_evaluator.update(results_dict)

            for iou_type in coco_evaluator.iou_types:
                coco_eval = coco_evaluator.coco_eval[iou_type]
                a = coco_evaluator.eval_imgs[iou_type]
                # logging.warning(f"[{iou_type}] a.shape: {a[0].shape}")
                # logging.warning(f"a: {a}")
                coco_evaluator.eval_imgs[iou_type] = np.concatenate(
                    coco_evaluator.eval_imgs[iou_type], 2
                )
                coco_eval.evalImgs = list(coco_evaluator.eval_imgs[iou_type].flatten())
                coco_eval.params.imgIds = list(coco_evaluator.img_ids)
                # We need to perform a deepcopy here since this dictionary can be modified in a
                # custom accumulate call and we don't want that to change coco_eval.params.
                # See https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py#L315.
                coco_eval._paramsEval = copy.deepcopy(coco_eval.params)
            coco_evaluator.accumulate()
            coco_evaluator.summarize()
 
            for iou_type, coco_eval in coco_evaluator.coco_eval.items():
                coco_stats = coco_eval.stats.tolist()
                metrics[f"{iou_type}_mAP"] = coco_stats[0]
                # metrics[f"{iou_type}_mAP_50"] = coco_stats[1]
                # metrics[f"{iou_type}_mAP_75"] = coco_stats[2]
                # metrics[f"{iou_type}_mAP_small"] = coco_stats[3]
                # metrics[f"{iou_type}_mAP_medium"] = coco_stats[4]
                # metrics[f"{iou_type}_mAP_large"] = coco_stats[5]
        return metrics


class TorchVisionTrial(pytorch.PyTorchTrial):
    def __init__(self, context: pytorch.PyTorchTrialContext, hparams: Dict) -> None:
        self.context = context

        self.data_dir = pathlib.Path("data")
        self.data_url = (
            "https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip"
        )

        # Download the dataset
        utils.download_data(self.data_dir, self.data_url)

        self.dataset = data.get_dataset(self.data_dir) # No data preprocessing
        
        # Split it in train/val
        train_size = int(0.8 * len(self.dataset))
        test_size = len(self.dataset) - train_size
        self.train_dataset, self.test_dataset = torch.utils.data.random_split(self.dataset,[train_size, test_size])

        self.batch_size = 4
        self.per_slot_batch_size = self.batch_size // self.context.distributed.get_size()

        # Define model.
        # Initialize the model and wrap it using self.context.wrap_model().
        self.model = self.context.wrap_model(model.build_model(hparams=hparams))

        # Configure optimizer.
        # Initialize the optimizer and wrap it using self.context.wrap_optimizer().
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = self.context.wrap_optimizer(
            torch.optim.SGD(
            params,
            lr=hparams["learning_rate"],
            momentum=hparams["momentum"],
            weight_decay=hparams["weight_decay"]
            )
        )

        self.lr_scheduler = self.context.wrap_lr_scheduler(
            torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=3, gamma=0.1),
            step_mode=pytorch.LRScheduler.StepMode.STEP_EVERY_EPOCH,
        )

        self.reducer = self.context.wrap_reducer(CocoReducer(self.test_dataset, ['bbox', 'segm']))

    def loss_reduced(self, loss_dict) -> torch.Tensor:
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            logging.error(f"Loss is {loss_value}, stopping training")
            logging.info(loss_dict_reduced)
            sys.exit(1)
        
        return losses_reduced

    def build_training_data_loader(self) -> pytorch.DataLoader:
        return pytorch.DataLoader(
            self.train_dataset, 
            batch_size=self.per_slot_batch_size,
            collate_fn=utils.collate_fn)

    def build_validation_data_loader(self) -> pytorch.DataLoader:
        return pytorch.DataLoader(
            self.test_dataset, 
            batch_size=self.per_slot_batch_size,
            collate_fn=utils.collate_fn)

    def train_batch(
        self, batch: pytorch.TorchData, epoch_idx: int, batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        images, targets = batch

        images = list(image for image in images)
        targets = [{k: v for k, v in t.items()} for t in targets]
        
        # Define the training forward pass and calculate loss.
        loss_dict = self.model(images, targets)
        loss = self.loss_reduced(loss_dict)
        losses = sum(loss for loss in loss_dict.values())

        # Define the training backward pass and step the optimizer.
        self.context.backward(losses)
        self.context.step_optimizer(self.optimizer)

        return {"loss": loss}

    def evaluate_batch(self, batch: pytorch.TorchData, batch_idx: int) -> Dict[str, Any]:
        images, targets = batch

        images = list(image for image in images)
        targets = [{k: v for k, v in t.items()} for t in targets]

        metrics = {}

        outputs = self.model(images)
        outputs = [{k: v.to(torch.device("cpu")) for k, v in t.items()} for t in outputs]
        result = [(target["image_id"], output) for target, output in zip(targets, outputs)]

        self.reducer.update(result)

        return metrics


def run(local: bool = False):
    """Initializes the trial and runs the training loop.

    This method configures the appropriate training parameters for both local and on-cluster
    training modes. It is an example of a standalone training script that can run both locally and
    on-cluster without any code changes.

    To run the training code solely locally or on-cluster, remove the conditional parameter logic
    for the unneeded training mode.

    Arguments:
        local: Whether to run this script locally. Defaults to false (on-cluster training).
    """

    info = det.get_cluster_info()

    if local:
        # For convenience, use hparams from const.yaml for local mode.
        conf = yaml.safe_load(pathlib.Path("./const.yaml").read_text())
        hparams = conf["hyperparameters"]
        max_length = pytorch.Batch(5)  # Train for 100 batches.
        latest_checkpoint = None
    else:
        hparams = info.trial.hparams  # Get instance of hparam values from Determined cluster info.
        max_length = None  # On-cluster training trains for the searcher's configured length.
        latest_checkpoint = (
            info.latest_checkpoint
        )  # (Optional) Configure checkpoint for pause/resume functionality.

    with pytorch.init() as train_context:
        trial = TorchVisionTrial(train_context, hparams=hparams)
        trainer = pytorch.Trainer(trial, train_context)
        trainer.fit(max_length=max_length, latest_checkpoint=latest_checkpoint)


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format=det.LOG_FORMAT)

    local_training = det.get_cluster_info() is None
    run(local=local_training)
