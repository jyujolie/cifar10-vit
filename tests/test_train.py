import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.train import run_epoch


def test_validation_accuracy_counts_predictions():
    images = torch.zeros(4, 1, 2, 2)
    labels = torch.tensor([0, 1, 0, 1])
    loader = DataLoader(TensorDataset(images, labels), batch_size=2)
    model = nn.Sequential(nn.Flatten(), nn.Linear(4, 2))
    with torch.no_grad():
        model[1].weight.zero_()
        model[1].bias.copy_(torch.tensor([-1.0, 1.0]))

    metrics = run_epoch(model, loader, torch.device("cpu"))

    assert metrics["samples"] == 4
    assert metrics["accuracy"] == 0.5
    assert metrics["loss"] > 0


def test_training_updates_parameters():
    images = torch.ones(2, 1, 2, 2)
    labels = torch.tensor([0, 0])
    loader = DataLoader(TensorDataset(images, labels), batch_size=2)
    model = nn.Sequential(nn.Flatten(), nn.Linear(4, 2))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    before = model[1].weight.detach().clone()

    run_epoch(model, loader, torch.device("cpu"), optimizer=optimizer)

    assert not torch.equal(before, model[1].weight)
