import math
import copy
import sys
import logging
import pathlib
from typing import Any, Dict

import data
import model
import torch
from ruamel import yaml
from torch import nn
import torchvision

import determined as det
from determined import pytorch

import utils
import numpy as np
#from pycocotools.cocoeval import COCOeval
from coco_eval import CocoEvaluator


class TorchVisionTrial(pytorch.PyTorchTrial):
    def __init__(self, context: pytorch.PyTorchTrialContext, hparams: Dict) -> None:
        self.context = context

        # Trial-level constants.
        self.data_dir = pathlib.Path("data")
        self.data_url = (
            "https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip"
        )

        utils.download_data(self.data_dir, self.data_url)

        self.dataset = data.get_dataset(self.data_dir) # No data preprocessing
        
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

        self.warmup_factor = 1.0 / 1000
        self.warmup_iters = min(1000, len(self.dataset) - 1)

        self.lr_scheduler = self.context.wrap_lr_scheduler(
            torch.optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=self.warmup_factor, total_iters=self.warmup_iters
            ),
            step_mode=pytorch.LRScheduler.StepMode.STEP_EVERY_EPOCH,
        )

        # self.lr_scheduler = self.context.wrap_lr_scheduler(
        #     torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=3, gamma=0.1),
        #     step_mode=pytorch.LRScheduler.StepMode.STEP_EVERY_EPOCH,
        # )
        
        self.evaluator = CocoEvaluator(self.test_dataset, self.model)

    def loss_reduced(self, loss_dict):
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
        self.val_data_loader = pytorch.DataLoader(
            self.test_dataset, 
            batch_size=self.per_slot_batch_size,
            collate_fn=utils.collate_fn)
        return self.val_data_loader

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

        outputs = self.model(images, targets)
        outputs = [{k: v.to(torch.device("cpu")) for k, v in t.items()} for t in outputs]
        res = {target["image_id"]: output for target, output in zip(targets, outputs)}

        self.evaluator.update(res)

        # accumulate predictions from all images
        self.evaluator.accumulate()
        self.evaluator.summarize()
        metrics = {}
        for iou_type, coco_eval in self.evaluator.coco_eval.items():
            logging.info(f"IoU type: {iou_type}")
            coco_eval.summarize()
            metrics[f"iou_{iou_type}"] = coco_eval.stats[0]

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
