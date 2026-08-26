"""Wrapper around the repository's existing server_data/server_label root set."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np


class ReferenceProvider:
    """Compute a fresh dense reference-gradient sign from the current model."""

    def __init__(
        self,
        server_data,
        server_label,
        *,
        loss_fn,
        seed: int,
        indices: Iterable[int] | None = None,
    ) -> None:
        if len(server_data) == 0 or len(server_label) == 0:
            raise ValueError("RAIN requires a non-empty root dataset")
        if len(server_data) != len(server_label):
            raise ValueError("root data and labels must have equal length")
        self.server_data = server_data
        self.server_label = server_label
        self.loss_fn = loss_fn
        self.seed = int(seed)
        self.indices = list(range(len(server_data))) if indices is None else [int(x) for x in indices]
        if not self.indices or min(self.indices) < 0 or max(self.indices) >= len(server_data):
            raise ValueError("root indices are empty or out of range")
        self.calls = 0

    def gradient(self, model) -> np.ndarray:
        import torch

        model.zero_grad(set_to_none=True)
        was_training = model.training
        model.eval()
        index = torch.as_tensor(self.indices, dtype=torch.long, device=self.server_label.device)
        output = model(self.server_data.index_select(0, index))
        loss = self.loss_fn(output, self.server_label.index_select(0, index))
        loss.backward()
        flattened = torch.cat([parameter.grad.detach().reshape(-1) for parameter in model.parameters()])
        gradient = flattened.to(torch.float32).cpu().numpy()
        model.zero_grad(set_to_none=True)
        model.train(was_training)
        self.calls += 1
        return np.ascontiguousarray(gradient)

    def direction(self, model) -> np.ndarray:
        return np.ascontiguousarray((self.gradient(model) >= 0).astype(np.uint8))

    def input_batch(self, *, limit: int = 32):
        """Return a deterministic reference-input batch for representation defenses."""
        import torch

        if limit < 1:
            raise ValueError("reference input limit must be positive")
        selected = self.indices[:limit]
        index = torch.as_tensor(selected, dtype=torch.long, device=self.server_data.device)
        return self.server_data.index_select(0, index)

    def save_manifest(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({
            "schema_version": 1,
            "seed": self.seed,
            "indices": self.indices,
            "sample_count": len(self.indices),
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class LoaderReferenceProvider:
    """Memory-bounded reference provider for index-backed image datasets."""

    def __init__(
        self,
        dataset,
        indices: Iterable[int],
        *,
        loss_fn,
        device,
        batch_size: int = 32,
        workers: int = 0,
        seed: int = 0,
    ) -> None:
        self.dataset = dataset
        self.indices = [int(value) for value in indices]
        if not self.indices or min(self.indices) < 0 or max(self.indices) >= len(dataset):
            raise ValueError("reference indices are empty or out of range")
        if batch_size <= 0 or workers < 0:
            raise ValueError("batch_size must be positive and workers non-negative")
        self.loss_fn = loss_fn
        self.device = device
        self.batch_size = int(batch_size)
        self.workers = int(workers)
        self.seed = int(seed)
        self.calls = 0

    def gradient(self, model) -> np.ndarray:
        import torch
        from torch.utils.data import DataLoader, Subset

        generator = torch.Generator().manual_seed(self.seed)
        loader = DataLoader(
            Subset(self.dataset, self.indices), batch_size=self.batch_size,
            shuffle=False, num_workers=self.workers, generator=generator,
            pin_memory=self.device.type == "cuda",
        )
        was_training = model.training
        model.eval()
        model.zero_grad(set_to_none=True)
        for values, targets in loader:
            values = values.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            # Accumulate a sample-sum gradient; the positive scalar does not
            # affect the final dense reference sign.
            self.loss_fn(model(values), targets).mul(values.shape[0]).backward()
        flattened = torch.cat([parameter.grad.detach().reshape(-1) for parameter in model.parameters()])
        gradient = flattened.to(torch.float32).cpu().numpy()
        model.zero_grad(set_to_none=True)
        model.train(was_training)
        self.calls += 1
        return np.ascontiguousarray(gradient)

    def direction(self, model) -> np.ndarray:
        return np.ascontiguousarray((self.gradient(model) >= 0).astype(np.uint8))

    def input_batch(self, *, limit: int = 32):
        """Return a deterministic reference-input batch for representation defenses."""
        import torch
        from torch.utils.data import DataLoader, Subset

        if limit < 1:
            raise ValueError("reference input limit must be positive")
        loader = DataLoader(
            Subset(self.dataset, self.indices[:limit]), batch_size=min(limit, len(self.indices)),
            shuffle=False, num_workers=self.workers, pin_memory=self.device.type == "cuda",
        )
        values, _targets = next(iter(loader))
        return values.to(self.device, non_blocking=True)

    def save_manifest(self, path: str | Path) -> None:
        target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({
            "schema_version": 1, "seed": self.seed, "indices": self.indices,
            "sample_count": len(self.indices), "batch_size": self.batch_size,
            "workers": self.workers,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
