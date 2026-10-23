"""Save one original CIFAR-10 training image from each class."""

from pathlib import Path

import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR10


def main() -> None:
    dataset = CIFAR10(Path("data"), train=True, download=True)
    print(f"Training images supplied by CIFAR-10: {len(dataset):,}")
    print(f"Classes: {', '.join(dataset.classes)}")

    first_by_class = {}
    for index, label in enumerate(dataset.targets):
        first_by_class.setdefault(label, index)
        if len(first_by_class) == 10:
            break

    figure, axes = plt.subplots(2, 5, figsize=(11, 5))
    for label, axis in enumerate(axes.flat):
        image, _ = dataset[first_by_class[label]]
        axis.imshow(image)
        axis.set_title(dataset.classes[label])
        axis.axis("off")
    figure.suptitle("CIFAR-10: one 32×32 image from each class")
    figure.tight_layout()
    output = Path("runs/data_preview.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)
    print(f"Preview saved to: {output}")


if __name__ == "__main__":
    main()
