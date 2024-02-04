from dataset import PennFudanDataset

import pathlib
from typing import Any

import filelock

import torch
from torchvision.transforms import v2 as T


def get_transform():
    transforms = []
    # if train:
    #     transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)

def get_dataset(data_dir: pathlib.Path) -> Any:
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Use a file lock so that only one worker on each node downloads.
    ### What happens with random indices if used by multiple workers?
    with filelock.FileLock(str(data_dir / "lock")):
        return PennFudanDataset(
                root=str(data_dir / "PennFudanPed"),
                transforms=get_transform())
