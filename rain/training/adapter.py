"""Thin PyTorch adapter; no cryptographic logic lives in the legacy trainer."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .coordinator import RainRoundCoordinator, RoundResult


@dataclass(slots=True)
class TorchRainAdapter:
    coordinator: RainRoundCoordinator
    reference_provider: object
    learning_rate: float

    @staticmethod
    def flatten_gradients(grad_list) -> np.ndarray:
        import torch

        rows = []
        for gradients in grad_list:
            if torch.is_tensor(gradients):
                rows.append(gradients.detach().reshape(-1).cpu().numpy())
            else:
                if any(value is None for value in gradients):
                    raise ValueError("client gradient list contains None")
                rows.append(torch.cat([value.detach().reshape(-1) for value in gradients]).cpu().numpy())
        if not rows:
            raise ValueError("grad_list must not be empty")
        return np.ascontiguousarray(np.stack(rows), dtype=np.float32)

    def step(self, *, grad_list, model, round_id: int) -> RoundResult:
        import torch

        reference = self.reference_provider.direction(model)
        result = self.coordinator.run_round(
            self.flatten_gradients(grad_list), reference, round_id=round_id
        )
        direction = torch.from_numpy(result.direction_bits.astype(np.float32) * 2.0 - 1.0)
        offset = 0
        with torch.no_grad():
            for parameter in model.parameters():
                count = parameter.numel()
                update = direction[offset : offset + count].reshape(parameter.shape)
                parameter.sub_(self.learning_rate * update.to(parameter.device, dtype=parameter.dtype))
                offset += count
        if offset != direction.numel():
            raise RuntimeError("aggregated direction does not match model parameter count")
        return result

    def state_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "learning_rate": self.learning_rate,
            "coordinator": self.coordinator.state_dict(),
            "reference_calls": getattr(self.reference_provider, "calls", None),
        }
