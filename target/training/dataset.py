"""
dataset.py — CIFAR-10 DataLoaders.

Used exclusively by train.py; the inference server never imports this.
"""

import pathlib

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# CIFAR-10 per-channel statistics
MEAN = (0.4914, 0.4822, 0.4465)
STD  = (0.2470, 0.2435, 0.2616)

_DATA_DIR = pathlib.Path(__file__).parent.parent / "data"


def get_loaders(batch_size: int = 128, num_workers: int = 2, val_size: int = 5000, seed: int = 42):
    """
    Download CIFAR-10 (if needed) and return (train_loader, val_loader, test_loader).

    Training transform applies random horizontal flip and random crop for
    data augmentation; validation and test transforms only normalise.

    A fixed 5,000 images are held out from the 50,000 training images for validation.
    """
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    # Two views of the same training data: augmented and clean
    train_aug   = datasets.CIFAR10(_DATA_DIR, train=True, download=True, transform=train_transform)
    train_clean = datasets.CIFAR10(_DATA_DIR, train=True, download=True, transform=eval_transform)
    test_set    = datasets.CIFAR10(_DATA_DIR, train=False, download=True, transform=eval_transform)

    # Reproducible split
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(train_aug), generator=g).tolist()
    val_idx, train_idx = perm[:val_size], perm[val_size:]

    train_set = Subset(train_aug, train_idx)    # augmented
    val_set   = Subset(train_clean, val_idx)    # NOT augmented

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
