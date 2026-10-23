"""Evaluate a saved checkpoint once on CIFAR-10's held-out test set."""

import argparse
import json
import random
import shutil
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from tqdm.auto import tqdm

from src.data import make_test_loader
from src.model import MODEL_NAME, build_model


def summarize_predictions(labels, predictions, *, num_classes=10):
    labels, predictions = np.asarray(labels), np.asarray(predictions)
    if labels.shape != predictions.shape or labels.size == 0:
        raise ValueError('Expected nonempty, equally sized labels and predictions')
    return {
        'accuracy': float((labels == predictions).mean()),
        'correct': int((labels == predictions).sum()),
        'samples': int(labels.size),
        'confusion_matrix': confusion_matrix(
            labels, predictions, labels=list(range(num_classes))
        ).tolist(),
    }


def select_examples(labels, predictions, *, count=8):
    rng = random.Random(42)
    correct = [i for i, (a, b) in enumerate(zip(labels, predictions)) if a == b]
    wrong = [i for i, (a, b) in enumerate(zip(labels, predictions)) if a != b]
    return (rng.sample(correct, min(count, len(correct))),
            rng.sample(wrong, min(count, len(wrong))))


def plot_examples(dataset, indices, labels, predictions, confidences, title, path):
    if not indices:
        return
    rows = (len(indices) + 3) // 4
    figure, axes = plt.subplots(rows, 4, figsize=(12, rows * 3), squeeze=False)
    for axis in axes.flat:
        axis.axis('off')
    for axis, index in zip(axes.flat, indices):
        axis.imshow(dataset.data[index])
        axis.set_title(
            f'True: {dataset.classes[labels[index]]}\n'
            f'Pred: {dataset.classes[predictions[index]]} ({confidences[index]:.1%})',
            fontsize=10,
        )
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def plot_history(history_path, output):
    if not history_path.exists():
        return
    records = json.loads(history_path.read_text(encoding='utf-8'))['history']
    epochs = [row['epoch'] for row in records]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for split, label in [('train', 'Training'), ('validation', 'Validation')]:
        axes[0].plot(epochs, [r[split]['loss'] for r in records], marker='o', label=label)
        axes[1].plot(epochs, [100*r[split]['accuracy'] for r in records], marker='o', label=label)
    for axis in axes:
        axis.set_xlabel('Epoch')
        axis.set_xticks(epochs)
        axis.legend()
        axis.grid(alpha=0.25)
    axes[0].set_ylabel('Cross-entropy loss')
    axes[1].set_ylabel('Accuracy (%)')
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=Path('runs/vit-tiny/best.pt'))
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--output-dir', type=Path, default=Path('reports'))
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=0)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if checkpoint.get('smoke_mode'):
        raise SystemExit('Choose a full-training checkpoint; this checkpoint is from a smoke run.')
    if checkpoint['model_name'] != MODEL_NAME:
        raise SystemExit('The checkpoint model does not match this project.')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(pretrained=False)
    model.load_state_dict(checkpoint['model_state'])
    model.to(device).eval()
    loader = make_test_loader(args.data_dir, batch_size=args.batch_size, workers=args.workers)
    labels, predictions, confidences = [], [], []
    print(f'Evaluating checkpoint epoch {checkpoint["epoch"]} on {device}', flush=True)
    with torch.inference_mode():
        for images, targets in tqdm(loader):
            with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
                logits = model(images.to(device))
            confidence, prediction = logits.float().softmax(dim=1).max(dim=1)
            labels.extend(targets.tolist())
            predictions.extend(prediction.cpu().tolist())
            confidences.extend(confidence.cpu().tolist())

    metrics = summarize_predictions(labels, predictions)
    metrics.update(model=MODEL_NAME, checkpoint_epoch=checkpoint['epoch'],
                   validation_accuracy=checkpoint['val_accuracy'], seed=checkpoint['seed'])
    metrics['per_class'] = classification_report(
        labels, predictions, labels=list(range(10)), target_names=loader.dataset.classes,
        output_dict=True, zero_division=0,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'test_metrics.json').write_text(
        json.dumps(metrics, indent=2), encoding='utf-8'
    )
    np.savez(args.output_dir / 'test_predictions.npz', labels=labels,
             predictions=predictions, confidences=confidences)
    figure, axis = plt.subplots(figsize=(9, 8))
    matrix = np.asarray(metrics['confusion_matrix'])
    image = axis.imshow(matrix, cmap='Blues')
    axis.set_xticks(range(10), loader.dataset.classes, rotation=45, ha='right')
    axis.set_yticks(range(10), loader.dataset.classes)
    axis.set_xlabel('Predicted class')
    axis.set_ylabel('True class')
    axis.set_title(f'CIFAR-10 test accuracy: {metrics["accuracy"]:.2%}')
    for row in range(10):
        for col in range(10):
            axis.text(col, row, str(matrix[row, col]), ha='center', va='center',
                      color='white' if matrix[row, col] > matrix.max()/2 else 'black', fontsize=8)
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(args.output_dir / 'confusion_matrix.png', dpi=160)
    plt.close(figure)
    correct, wrong = select_examples(labels, predictions)
    for indices, title, filename in [
        (correct, 'Correct predictions (seeded random sample)', 'correct_predictions.png'),
        (wrong, 'Incorrect predictions (seeded random sample)', 'incorrect_predictions.png'),
    ]:
        plot_examples(loader.dataset, indices, labels, predictions, confidences,
                      title, args.output_dir / filename)
    plot_history(args.checkpoint.parent / 'history.json', args.output_dir / 'training_curves.png')
    if (args.checkpoint.parent / 'history.json').exists():
        shutil.copyfile(args.checkpoint.parent / 'history.json', args.output_dir / 'training_history.json')
    print(f'Test accuracy: {metrics["accuracy"]:.2%} ({metrics["correct"]}/{metrics["samples"]})')
    print(f'Reports saved to: {args.output_dir.resolve()}')


if __name__ == '__main__':
    main()
