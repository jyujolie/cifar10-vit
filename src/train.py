"""Fine-tune a pretrained ViT Tiny on CIFAR-10."""

import argparse
import json
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.data import make_train_val_loaders
from src.model import MODEL_NAME, build_model


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    max_batches: int | None = None,
    show_progress: bool = False,
) -> dict[str, float | int]:
    """Run one training or validation epoch and report sample-weighted metrics."""
    training = optimizer is not None
    model.train(training)
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    samples = 0

    iterator = tqdm(loader, total=max_batches or len(loader), disable=not show_progress)
    for batch_index, (images, labels) in enumerate(iterator):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, labels)
            if training:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        count = labels.size(0)
        samples += count
        total_loss += loss.detach().item() * count
        correct += (logits.detach().argmax(dim=1) == labels).sum().item()

    if samples == 0:
        raise ValueError("The data loader supplied no images")
    return {"loss": total_loss / samples, "accuracy": correct / samples, "samples": samples}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--run-dir", type=Path, default=Path("runs/vit-tiny"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        raise SystemExit("epochs and batch-size must be positive; workers cannot be negative")
    if args.max_train_batches is not None and args.max_train_batches < 1:
        raise SystemExit("max-train-batches must be positive")
    if args.max_val_batches is not None and args.max_val_batches < 1:
        raise SystemExit("max-val-batches must be positive")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    if device.type == "cpu":
        print("CUDA is unavailable; training will be much slower on CPU.", flush=True)

    train_loader, val_loader = make_train_val_loaders(
        args.data_dir, batch_size=args.batch_size, workers=args.workers, seed=args.seed
    )
    model = build_model(pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    smoke_mode = args.max_train_batches is not None or args.max_val_batches is not None
    history = []
    best_accuracy = -1.0

    try:
        for epoch in range(1, args.epochs + 1):
            train_metrics = run_epoch(
                model, train_loader, device, optimizer=optimizer, scaler=scaler,
                max_batches=args.max_train_batches, show_progress=True,
            )
            val_metrics = run_epoch(
                model, val_loader, device, max_batches=args.max_val_batches,
                show_progress=True,
            )
            record = {"epoch": epoch, "train": train_metrics, "validation": val_metrics}
            history.append(record)
            (args.run_dir / "history.json").write_text(
                json.dumps({"model": MODEL_NAME, "seed": args.seed, "smoke_mode": smoke_mode,
                            "history": history}, indent=2), encoding="utf-8"
            )
            print(
                f"Epoch {epoch}/{args.epochs}: train loss={train_metrics['loss']:.4f}, "
                f"train acc={train_metrics['accuracy']:.2%}, "
                f"val loss={val_metrics['loss']:.4f}, val acc={val_metrics['accuracy']:.2%}",
                flush=True,
            )
            if val_metrics["accuracy"] > best_accuracy:
                best_accuracy = val_metrics["accuracy"]
                torch.save(
                    {"model_state": model.state_dict(), "model_name": MODEL_NAME,
                     "seed": args.seed, "epoch": epoch, "val_accuracy": best_accuracy,
                     "smoke_mode": smoke_mode},
                    args.run_dir / "best.pt",
                )
    except torch.cuda.OutOfMemoryError as error:
        raise SystemExit("GPU memory exhausted. Retry with --batch-size 16 or 8.") from error

    print(f"Best validation accuracy: {best_accuracy:.2%}", flush=True)
    print(f"Checkpoint: {args.run_dir / 'best.pt'}", flush=True)


if __name__ == "__main__":
    main()
