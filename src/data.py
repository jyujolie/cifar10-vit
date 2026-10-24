"""CIFAR-10 transforms and reproducible train/validation splitting."""

from collections.abc import Sequence
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode, RandAugment

IMAGE_SIZE = 224
MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)


def stratified_split_indices(
    labels: Sequence[int], val_per_class: int, seed: int
) -> tuple[list[int], list[int]]:
    """Split indices so that every class contributes exactly `val_per_class` samples.

    A purely random split lets class proportions drift, which makes the validation
    accuracy noisy and the "best checkpoint" choice unreliable. Stratifying fixes
    the composition: the metric then only reflects model quality.
    """
    labels = list(labels)
    if not labels:
        raise ValueError("labels are empty")
    if val_per_class < 1:
        raise ValueError("val_per_class must be positive")

    generator = torch.Generator().manual_seed(seed)
    val_indices: list[int] = []
    for label in sorted(set(labels)):
        pool = [index for index, value in enumerate(labels) if value == label]
        if len(pool) <= val_per_class:
            raise ValueError(
                f"class {label} has {len(pool)} samples, too few for "
                f"val_per_class={val_per_class}"
            )
        permutation = torch.randperm(len(pool), generator=generator).tolist()
        val_indices.extend(pool[position] for position in permutation[:val_per_class])

    validation = set(val_indices)
    train_indices = [index for index in range(len(labels)) if index not in validation]
    return train_indices, val_indices


def image_transform(
    *,
    training: bool,
    rand_augment_ops: int = 0,
    rand_augment_magnitude: int = 0,
) -> transforms.Compose:
    """Use light augmentation for training and deterministic evaluation.

    `rand_augment_ops` and `rand_augment_magnitude` add RandAugment on top of the
    basic crop and flip. Both must be positive to enable it, or both zero to keep
    the plain pipeline; mixing the two is rejected so a half-configured run cannot
    silently happen. RandAugment is only ever applied to the training transform.
    """
    if (rand_augment_ops == 0) != (rand_augment_magnitude == 0):
        raise ValueError("rand_augment_ops and rand_augment_magnitude must be given together")
    if rand_augment_ops < 0 or rand_augment_magnitude < 0:
        raise ValueError("rand_augment_ops and rand_augment_magnitude cannot be negative")

    operations = []
    if training:
        operations.extend(
            [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
        )
        if rand_augment_ops > 0:
            operations.append(
                RandAugment(num_ops=rand_augment_ops, magnitude=rand_augment_magnitude)
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
    data_dir: str | Path, *, batch_size: int = 32, workers: int = 2, seed: int = 42,
    val_per_class: int = 500, rand_augment_ops: int = 0, rand_augment_magnitude: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Return 45,000 training and 5,000 validation images from CIFAR-10.

    The validation split is stratified: exactly `val_per_class` images per class,
    which keeps the 10 classes in their natural 10% proportion and makes the
    validation accuracy comparable across runs. RandAugment is applied to the
    training transform only.
    """
    data_dir = Path(data_dir)
    train_dataset = datasets.CIFAR10(
        data_dir, train=True, download=True,
        transform=image_transform(
            training=True,
            rand_augment_ops=rand_augment_ops,
            rand_augment_magnitude=rand_augment_magnitude,
        ),
    )
    val_dataset = datasets.CIFAR10(
        data_dir, train=True, download=True, transform=image_transform(training=False)
    )
    train_indices, val_indices = stratified_split_indices(
        train_dataset.targets, val_per_class=val_per_class, seed=seed
    )
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
