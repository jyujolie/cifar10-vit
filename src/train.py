"""Fine-tune a pretrained ViT Tiny on CIFAR-10."""

import argparse
import json
import math
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.data import make_train_val_loaders
from src.model import MODEL_NAME, build_model


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    epochs: int,
    warmup_epochs: int,
    base_lr: float,
    min_lr: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup followed by cosine decay down to `min_lr`.

    ViT fine-tuning is sensitive to the first steps: a full-size learning rate on
    freshly initialised attention tends to destroy pretrained features. The warmup
    ramps the rate up gently, and the cosine tail keeps improving the model instead
    of bouncing around a constant rate near the end of training.
    """

    def lr_at(epoch: int) -> float:
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        # Divide by (epochs - warmup - 1) so the *last trained epoch* lands exactly
        # on min_lr; epoch indices run from 0 to epochs-1.
        span = max(1, epochs - warmup_epochs - 1)
        progress = (epoch - warmup_epochs) / span
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        floor = min_lr / base_lr
        return floor + (1.0 - floor) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_at)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    max_batches: int | None = None,
    show_progress: bool = False,
    clip_grad_norm: float | None = None,
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
                    # AMP requires unscale_() before clipping, otherwise the norm is
                    # computed on the scaled (huge) gradients.
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    if clip_grad_norm is not None:
                        nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if clip_grad_norm is not None:
                        nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
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
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--clip-grad-norm", type=float, default=1.0)
    parser.add_argument("--val-per-class", type=int, default=500)
    parser.add_argument("--rand-augment-ops", type=int, default=0)
    parser.add_argument("--rand-augment-magnitude", type=int, default=0)
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
    if not 0 <= args.warmup_epochs < args.epochs:
        raise SystemExit("warmup-epochs must be in [0, epochs)")
    if not 0 <= args.min_lr < args.lr:
        raise SystemExit("min-lr must be non-negative and below --lr")
    if args.clip_grad_norm is not None and args.clip_grad_norm <= 0:
        raise SystemExit("clip-grad-norm must be positive when given")
    if args.val_per_class < 1:
        raise SystemExit("val-per-class must be positive")
    if (args.rand_augment_ops == 0) != (args.rand_augment_magnitude == 0):
        raise SystemExit("rand-augment-ops and rand-augment-magnitude must be given together")
    if args.rand_augment_ops < 0 or not 0 <= args.rand_augment_magnitude <= 30:
        raise SystemExit("rand-augment-ops must be non-negative and magnitude within 0..30")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    if device.type == "cpu":
        print("CUDA is unavailable; training will be much slower on CPU.", flush=True)

    train_loader, val_loader = make_train_val_loaders(
        args.data_dir, batch_size=args.batch_size, workers=args.workers,
        seed=args.seed, val_per_class=args.val_per_class,
        rand_augment_ops=args.rand_augment_ops,
        rand_augment_magnitude=args.rand_augment_magnitude,
    )
    model = build_model(pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    scheduler = build_scheduler(
        optimizer, epochs=args.epochs, warmup_epochs=args.warmup_epochs,
        base_lr=args.lr, min_lr=args.min_lr,
    )
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
                clip_grad_norm=args.clip_grad_norm,
            )
            val_metrics = run_epoch(
                model, val_loader, device, max_batches=args.max_val_batches,
                show_progress=True,
            )
            record = {
                "epoch": epoch,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "train": train_metrics,
                "validation": val_metrics,
            }
            history.append(record)
            (args.run_dir / "history.json").write_text(
                json.dumps({"model": MODEL_NAME, "seed": args.seed, "smoke_mode": smoke_mode,
                            "optimizer": "adamw", "weight_decay": 0.05,
                            "batch_size": args.batch_size, "epochs_planned": args.epochs,
                            "warmup_epochs": args.warmup_epochs, "min_lr": args.min_lr,
                            "clip_grad_norm": args.clip_grad_norm,
                            "val_per_class": args.val_per_class,
                            "rand_augment_ops": args.rand_augment_ops,
                            "rand_augment_magnitude": args.rand_augment_magnitude,
                            "history": history}, indent=2), encoding="utf-8"
            )
            print(
                f"Epoch {epoch}/{args.epochs}: lr={record['learning_rate']:.2e}, "
                f"train loss={train_metrics['loss']:.4f}, "
                f"train acc={train_metrics['accuracy']:.2%}, "
                f"val loss={val_metrics['loss']:.4f}, val acc={val_metrics['accuracy']:.2%}",
                flush=True,
            )
            scheduler.step()
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
