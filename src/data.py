"""CIFAR-10 transforms and reproducible train/validation splitting."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

IMAGE_SIZE = 224
MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)


def split_indices(total: int, val_size: int, seed: int) -> tuple[list[int], list[int]]:
    """Split one training dataset without sharing samples between partitions."""
    if total < 2 or not 0 < val_size < total:
        raise ValueError("val_size must leave at least one sample in each partition")
    indices = torch.randperm(total, generator=torch.Generator().manual_seed(seed)).tolist()
    return indices[val_size:], indices[:val_size]


def image_transform(*, training: bool) -> transforms.Compose:
    """Use light augmentation for training and deterministic evaluation."""
    operations = []
    if training:
        operations.extend(
            [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
        )
    operations.extend(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    return transforms.Compose(operations)


def make_train_val_loaders(
    data_dir: str | Path, *, batch_size: int = 32, workers: int = 2, seed: int = 42
) -> tuple[DataLoader, DataLoader]:
    """Return 45,000 training and 5,000 validation images from CIFAR-10."""
    data_dir = Path(data_dir)
    train_dataset = datasets.CIFAR10(
        data_dir, train=True, download=True, transform=image_transform(training=True)
    )
    val_dataset = datasets.CIFAR10(
        data_dir, train=True, download=True, transform=image_transform(training=False)
    )
    train_indices, val_indices = split_indices(len(train_dataset), 5_000, seed)
    loader_options = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(
        Subset(train_dataset, train_indices), shuffle=True, **loader_options
    )
    val_loader = DataLoader(
        Subset(val_dataset, val_indices), shuffle=False, **loader_options
    )
    return train_loader, val_loader


def make_test_loader(
    data_dir: str | Path, *, batch_size: int = 32, workers: int = 2
) -> DataLoader:
    """Load the untouched CIFAR-10 test partition for final evaluation."""
    test_dataset = datasets.CIFAR10(
        Path(data_dir), train=False, download=True, transform=image_transform(training=False)
    )
    return DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )
