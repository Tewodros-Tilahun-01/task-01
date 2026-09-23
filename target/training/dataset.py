"""
dataset.py — CIFAR-10 DataLoaders.

Used exclusively by train.py; the inference server never imports this.
"""

import pathlib

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# CIFAR-10 per-channel statistics
MEAN = (0.4914, 0.4822, 0.4465)
STD  = (0.2470, 0.2435, 0.2616)

_DATA_DIR = pathlib.Path(__file__).parent.parent / "data"


def get_loaders(batch_size: int = 128, num_workers: int = 2):
    """
    Download CIFAR-10 (if needed) and return (train_loader, test_loader).

    Training transform applies random horizontal flip and random crop for
    data augmentation; test transform only normalises.
    """
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    train_set = datasets.CIFAR10(
        root=_DATA_DIR, train=True, download=True, transform=train_transform
    )
    test_set = datasets.CIFAR10(
        root=_DATA_DIR, train=False, download=True, transform=test_transform
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, test_loader
