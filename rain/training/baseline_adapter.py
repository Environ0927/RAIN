"""Stateful adapter for robustness baselines with model-update interfaces.

The comparison rules in :mod:`aggregation_rules` update a PyTorch model in
place. This adapter reconstructs their expected per-parameter client
gradients, executes the rule on a model copy, and returns the resulting flat
update to the unified trainer. Stateful methods expose serializable state so a
checkpoint resume is equivalent to an uninterrupted run.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Any

import numpy as np


ADAPTER_AGGREGATIONS = frozenset({
    "shieldfl", "signguard", "foolsgold", "divide-and-conquer",
    "contra", "romoa", "flare",
})


def normalize_aggregation_name(name: str) -> str:
    value = str(name).lower().replace("_", "-")
    if value in {"dnc", "divide-conquer", "divide-and-conquer"}:
        return "divide-and-conquer"
    return value


def _identity_attack(values, *_args, **_kwargs):
    return values


def _cpu_state(value: Any) -> Any:
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, list):
        return [_cpu_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_state(item) for item in value)
    if isinstance(value, dict):
        return {key: _cpu_state(item) for key, item in value.items()}
    return value


def _device_state(value: Any, device) -> Any:
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None and isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, list):
        return [_device_state(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_device_state(item, device) for item in value)
    if isinstance(value, dict):
        return {key: _device_state(item, device) for key, item in value.items()}
    return value


@dataclass(frozen=True, slots=True)
class AggregationResult:
    update: np.ndarray
    aggregation_seconds: float


class BaselineAggregationAdapter:
    """Run one comparison baseline through the unified flat-update interface."""

    def __init__(
        self,
        method: str,
        *,
        model,
        reference_provider,
        protocol: dict[str, object],
        device,
        client_count: int,
        seed: int,
    ) -> None:
        normalized = normalize_aggregation_name(method)
        if normalized not in ADAPTER_AGGREGATIONS:
            raise ValueError(f"unsupported adapter aggregation: {method}")
        if client_count < 2:
            raise ValueError("adapter-based robustness baselines require at least two clients")
        self.method = normalized
        self.model = model
        self.reference_provider = reference_provider
        self.protocol = dict(protocol)
        self.device = device
        self.client_count = int(client_count)
        self.seed = int(seed)
        self._state: dict[str, Any] = {}

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "method": self.method,
            "client_count": self.client_count,
            "state": _cpu_state(self._state),
        }

    def load_state_dict(self, value: dict[str, Any]) -> None:
        if value.get("schema_version") != 1 or value.get("method") != self.method:
            raise ValueError("aggregation checkpoint is incompatible")
        if int(value.get("client_count", -1)) != self.client_count:
            raise ValueError("aggregation client count changed across resume")
        self._state = _device_state(dict(value.get("state", {})), self.device)

    def _gradients(self, matrix: np.ndarray):
        import torch

        templates = list(self.model.parameters())
        expected = sum(parameter.numel() for parameter in templates)
        if matrix.shape != (self.client_count, expected):
            raise ValueError("update matrix shape does not match model and clients")
        gradients = []
        for row in matrix:
            offset = 0
            client = []
            for parameter in templates:
                count = parameter.numel()
                value = torch.from_numpy(np.ascontiguousarray(row[offset:offset + count]))
                client.append(value.to(self.device).reshape(parameter.shape))
                offset += count
            gradients.append(client)
        return gradients

    def _initialize_state(self, dimension: int) -> None:
        import torch

        if self._state:
            return
        shape = (dimension, 1)
        if self.method == "shieldfl":
            self._state = {"previous_global": 0, "previous_gradients": []}
        elif self.method == "foolsgold":
            self._state = {
                "gradient_history": [torch.zeros(shape, device=self.device) for _ in range(self.client_count)]
            }
        elif self.method == "contra":
            self._state = {
                "gradient_history": [torch.zeros(shape, device=self.device) for _ in range(self.client_count)],
                "reputation": torch.ones(self.client_count, device=self.device),
                "cos_dist": torch.zeros(
                    (self.client_count, self.client_count), dtype=torch.double, device=self.device
                ),
            }
        elif self.method == "romoa":
            flattened = torch.cat([value.detach().flatten() for value in self.model.parameters()])[:, None]
            generator = torch.Generator(device=self.device).manual_seed(self.seed)
            jitter = torch.normal(
                mean=0.0, std=1e-7, size=shape, generator=generator, device=self.device
            )
            self._state = {
                "previous_global": flattened + jitter,
                "sanitization_factor": torch.full(
                    (self.client_count, dimension), 1.0 / self.client_count, device=self.device
                ),
            }

    def aggregate(
        self,
        updates: np.ndarray,
        *,
        malicious_clients: int,
        round_id: int,
    ) -> AggregationResult:
        import torch
        import aggregation_rules

        matrix = np.asarray(updates, dtype=np.float32)
        if matrix.ndim != 2 or not np.isfinite(matrix).all():
            raise ValueError("updates must be a finite 2-D matrix")
        if matrix.shape[0] != self.client_count:
            raise ValueError("adapter methods require a fixed client count")
        if not 0 <= malicious_clients < self.client_count:
            raise ValueError("malicious_clients must lie in [0, client_count)")
        self._initialize_state(matrix.shape[1])
        gradients = self._gradients(matrix)
        clone = copy.deepcopy(self.model)
        # These rules update parameters directly instead of using an
        # optimizer. The adapter model is disposable and never participates
        # in autograd, so disabling gradients makes those updates explicit and
        # compatible with current PyTorch leaf-tensor checks.
        clone.requires_grad_(False)
        before = torch.cat([parameter.detach().flatten() for parameter in clone.parameters()]).clone()
        started = time.perf_counter()

        if self.method == "shieldfl":
            previous_global, previous_gradients = aggregation_rules.shieldfl(
                gradients, clone, 1.0, malicious_clients, _identity_attack, self.device,
                self._state["previous_global"], round_id, self._state["previous_gradients"],
            )
            self._state.update(
                previous_global=previous_global, previous_gradients=previous_gradients
            )
        elif self.method == "divide-and-conquer":
            dimension = matrix.shape[1]
            bound = min(dimension, int(self.protocol.get("dnc_b", min(10_000, dimension))))
            if bound < 2:
                raise ValueError("divide-and-conquer requires at least two update coordinates")
            aggregation_rules.divide_and_conquer(
                gradients, clone, 1.0, malicious_clients, _identity_attack, self.device,
                niters=int(self.protocol.get("dnc_niters", 1)),
                c=float(self.protocol.get("dnc_c", 1.0)), b=bound,
            )
        elif self.method == "foolsgold":
            self._state["gradient_history"] = aggregation_rules.foolsgold(
                gradients, clone, 1.0, malicious_clients, _identity_attack, self.device,
                gradient_history=self._state["gradient_history"],
            )
        elif self.method == "contra":
            history, reputation, cos_dist = aggregation_rules.contra(
                gradients, clone, 1.0, malicious_clients, _identity_attack, self.device,
                gradient_history=self._state["gradient_history"],
                reputation=self._state["reputation"], cos_dist=self._state["cos_dist"],
                C=float(self.protocol.get("selection_fraction", 1.0)),
            )
            self._state.update(
                gradient_history=history, reputation=reputation, cos_dist=cos_dist
            )
        elif self.method == "signguard":
            aggregation_rules.signguard(
                gradients, clone, 1.0, malicious_clients, _identity_attack,
                self.device, seed=self.seed + round_id,
            )
        elif self.method == "flare":
            if self.reference_provider is None:
                raise ValueError("FLARE requires a non-empty public reference dataset")
            server_inputs = self.reference_provider.input_batch(
                limit=int(self.protocol.get("flare_reference_batch", 32))
            )
            aggregation_rules.flare(
                gradients, clone, 1.0, malicious_clients, _identity_attack,
                self.device, server_inputs,
            )
        elif self.method == "romoa":
            factor, previous_global = aggregation_rules.romoa(
                gradients, clone, 1.0, malicious_clients, _identity_attack, self.device,
                F=self._state["sanitization_factor"],
                prev_global_update=self._state["previous_global"], seed=self.seed + round_id,
            )
            self._state.update(
                sanitization_factor=factor, previous_global=previous_global
            )
        else:  # pragma: no cover - constructor validation makes this unreachable.
            raise AssertionError(self.method)

        after = torch.cat([parameter.detach().flatten() for parameter in clone.parameters()])
        update = (before - after).to(torch.float32).cpu().numpy()
        return AggregationResult(
            update=np.ascontiguousarray(update),
            aggregation_seconds=time.perf_counter() - started,
        )
