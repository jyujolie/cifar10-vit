import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.train import build_scheduler, run_epoch


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


def test_gradient_clipping_keeps_updates_within_bound():
    images = torch.ones(4, 1, 2, 2) * 50.0
    labels = torch.tensor([0, 0, 1, 1])
    loader = DataLoader(TensorDataset(images, labels), batch_size=4)
    model = nn.Sequential(nn.Flatten(), nn.Linear(4, 2))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    before = model[1].weight.detach().clone()

    run_epoch(model, loader, torch.device("cpu"), optimizer=optimizer,
              clip_grad_norm=0.01)

    step = (model[1].weight.detach() - before).abs().max().item()
    # With lr=0.1 and a clipped norm of 0.01 the weight move cannot exceed 0.001.
    assert step <= 0.001 + 1e-6


def test_scheduler_warms_up_then_decays_cosine():
    model = nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    scheduler = build_scheduler(optimizer, epochs=10, warmup_epochs=2,
                                base_lr=1e-3, min_lr=1e-4)

    rates = []
    for _ in range(10):
        rates.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()

    assert rates[0] == 5e-4          # half of base lr during warmup
    assert rates[1] == 1e-3          # warmup completes
    assert rates[2] == 1e-3          # cosine starts at its peak
    assert rates[3] < rates[2]       # monotone decay after warmup
    assert rates[9] == 1e-4          # ends exactly at min_lr

