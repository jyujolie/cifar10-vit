"""Confirm that the project's Python can run a small CUDA calculation."""

import sys

import torch
import torchvision


def main() -> None:
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"torchvision: {torchvision.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        raise SystemExit("GPU unavailable in this Python environment. Check the PyTorch CUDA installation.")

    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    a = torch.tensor([1.0, 2.0, 3.0], device=device)
    print(f"GPU calculation: {a.sum().item():.1f} (expected 6.0)")


if __name__ == "__main__":
    main()
