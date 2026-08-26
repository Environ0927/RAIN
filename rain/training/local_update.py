"""Deterministic client-local SGD used by all plaintext convergence methods."""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class LocalUpdate:
    update: np.ndarray
    buffers: dict[str, object]
    examples_seen: int
    mean_loss: float


class LocalSGDTrainer:
    """Reuse one model replica while resetting model and optimizer per client."""

    def __init__(
        self,
        model,
        *,
        learning_rate: float,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
        amp: bool = False,
    ) -> None:
        import torch

        if learning_rate <= 0:
            raise ValueError("client learning rate must be positive")
        if not 0 <= momentum < 1:
            raise ValueError("client momentum must lie in [0, 1)")
        if weight_decay < 0:
            raise ValueError("client weight decay must be non-negative")
        if nesterov and momentum <= 0:
            raise ValueError("Nesterov momentum requires positive momentum")
        self.model = copy.deepcopy(model)
        self.learning_rate = float(learning_rate)
        self.momentum = float(momentum)
        self.weight_decay = float(weight_decay)
        self.nesterov = bool(nesterov)
        self.amp = bool(amp)
        self.device = next(model.parameters()).device
        self._torch = torch

    def run(self, global_model, batches, loss_fn) -> LocalUpdate:
        torch = self._torch
        self.model.load_state_dict(global_model.state_dict())
        self.model.train()
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
            nesterov=self.nesterov,
        )
        examples_seen = 0
        loss_sum = 0.0
        for values, targets in batches:
            values = values.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=self.device.type,
                enabled=self.amp and self.device.type == "cuda",
            ):
                loss = loss_fn(self.model(values), targets)
            loss.backward()
            optimizer.step()
            count = int(targets.numel())
            examples_seen += count
            loss_sum += float(loss.detach()) * count
        if examples_seen == 0:
            raise ValueError("local SGD requires at least one example")
        global_parameters = dict(global_model.named_parameters())
        update = torch.cat([
            (global_parameters[name].detach() - parameter.detach()).reshape(-1)
            for name, parameter in self.model.named_parameters()
        ]).to(torch.float32).cpu().numpy()
        buffers = {
            name: value.detach().cpu().clone()
            for name, value in self.model.named_buffers()
        }
        return LocalUpdate(
            update=np.ascontiguousarray(update),
            buffers=buffers,
            examples_seen=examples_seen,
            mean_loss=loss_sum / examples_seen,
        )


def apply_weighted_buffers(model, local_updates: list[LocalUpdate], weights: np.ndarray) -> None:
    """Apply the same client weights to BatchNorm and other model buffers."""

    if not local_updates:
        return
    values = np.asarray(weights, dtype=np.float64)
    if values.shape != (len(local_updates),) or np.any(values < 0) or values.sum() <= 0:
        raise ValueError("buffer weights must be non-negative and non-empty")
    values /= values.sum()
    torch = __import__("torch")
    with torch.no_grad():
        for name, target in model.named_buffers():
            rows = [item.buffers[name] for item in local_updates]
            if target.is_floating_point() or target.is_complex():
                averaged = torch.zeros_like(rows[0], dtype=torch.float64)
                for row, weight in zip(rows, values):
                    averaged.add_(row.to(torch.float64), alpha=float(weight))
                target.copy_(averaged.to(device=target.device, dtype=target.dtype))
            else:
                stacked = torch.stack([row.to(torch.int64) for row in rows])
                target.copy_(stacked.max(dim=0).values.to(device=target.device, dtype=target.dtype))
