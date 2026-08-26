import numpy as np
import pytest


torch = pytest.importorskip("torch")


class _Reference:
    def input_batch(self, *, limit=32):
        return torch.arange(16, dtype=torch.float32).reshape(4, 4)[:limit]


@pytest.mark.parametrize("method", [
    "shieldfl", "signguard", "foolsgold", "divide-and-conquer",
    "contra", "romoa", "flare",
])
def test_legacy_baseline_runs_through_flat_update_adapter(method):
    from rain.training.legacy_adapter import LegacyAggregationAdapter

    torch.manual_seed(3)
    model = torch.nn.Linear(4, 3)
    dimension = sum(value.numel() for value in model.parameters())
    updates = np.random.default_rng(7).normal(size=(4, dimension)).astype(np.float32)
    adapter = LegacyAggregationAdapter(
        method, model=model, reference_provider=_Reference(),
        protocol={"dnc_b": dimension, "dnc_niters": 1, "dnc_c": 1.0},
        device=torch.device("cpu"), client_count=4, seed=11,
    )
    result = adapter.aggregate(updates, malicious_clients=0, round_id=0)
    assert result.update.shape == (dimension,)
    assert np.isfinite(result.update).all()
    assert result.aggregation_seconds >= 0


def test_stateful_baseline_checkpoint_roundtrip():
    from rain.training.legacy_adapter import LegacyAggregationAdapter

    model = torch.nn.Linear(4, 3)
    dimension = sum(value.numel() for value in model.parameters())
    updates = np.random.default_rng(17).normal(size=(4, dimension)).astype(np.float32)
    first = LegacyAggregationAdapter(
        "foolsgold", model=model, reference_provider=_Reference(), protocol={},
        device=torch.device("cpu"), client_count=4, seed=5,
    )
    first.aggregate(updates, malicious_clients=0, round_id=0)
    state = first.state_dict()
    second = LegacyAggregationAdapter(
        "foolsgold", model=model, reference_provider=_Reference(), protocol={},
        device=torch.device("cpu"), client_count=4, seed=5,
    )
    second.load_state_dict(state)
    expected = first.aggregate(updates, malicious_clients=0, round_id=1).update
    actual = second.aggregate(updates, malicious_clients=0, round_id=1).update
    assert np.allclose(actual, expected)
