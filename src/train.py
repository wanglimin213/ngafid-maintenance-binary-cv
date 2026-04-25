from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


@dataclass
class EpochMetrics:
    loss: float
    accuracy: float


def binary_accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = (torch.sigmoid(logits) >= 0.5).float()
    return (preds == targets).float().mean().item()


def run_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    max_steps: Optional[int] = None,
    desc: str = "train",
) -> EpochMetrics:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    iterator = tqdm(loader, desc=desc, leave=False)
    for step, (x, y) in enumerate(iterator, start=1):
        if max_steps is not None and step > max_steps:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.set_grad_enabled(is_train):
            logits = model(x)
            loss = criterion(logits, y)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        batch_size = y.shape[0]
        total_loss += loss.item() * batch_size
        preds = (torch.sigmoid(logits) >= 0.5).float()
        total_correct += (preds == y).sum().item()
        total_count += batch_size
        iterator.set_postfix(loss=total_loss / max(total_count, 1), acc=total_correct / max(total_count, 1))

    return EpochMetrics(loss=total_loss / total_count, accuracy=total_correct / total_count)


def evaluate(model, loader, criterion, device, desc="val") -> EpochMetrics:
    with torch.no_grad():
        return run_one_epoch(model, loader, criterion, device, optimizer=None, max_steps=None, desc=desc)
