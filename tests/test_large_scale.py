import numpy as np
import pytest

from rain.protocol.aggregation import secure_rain_aggregate
from rain.protocol.boolean_sharing import reconstruct_bits, share_bits
from rain.protocol.oracle import rain_aggregate_oracle
from rain.training.large_data import (
    FEMNISTDataset, partition_femnist_dataset, partition_large_dataset,
    prepare_femnist_cache,
)
from rain.transport import LogicalTransport


def test_large_partition_is_deterministic_disjoint_and_complete(tmp_path):
    targets = np.repeat(np.arange(100), 20)
    kwargs = dict(clients=12, root_size=100, calibration_size=100, root_bias=.1,
                  dirichlet_alpha=.5, seed=17, minimum_client_size=2)
    left = partition_large_dataset(targets, **kwargs)
    right = partition_large_dataset(targets, **kwargs)
    assert left == right
    groups = [set(left.root_indices), set(left.calibration_indices)] + [set(values) for values in left.client_indices]
    assert all(groups[i].isdisjoint(groups[j]) for i in range(len(groups)) for j in range(i + 1, len(groups)))
    assert set.union(*groups) == set(range(targets.size))
    assert min(map(len, left.client_indices)) >= 2
    left.save(tmp_path / "partition.json")
    assert (tmp_path / "partition.json").exists()


def test_bound_aware_protocol_handles_large_dense_direction():
    rng = np.random.default_rng(9)
    bits = rng.integers(0, 2, size=(2, 50_000), dtype=np.uint8)
    reference = rng.integers(0, 2, size=50_000, dtype=np.uint8)
    shares = share_bits(bits, rng=np.random.default_rng(10))
    result = secure_rain_aggregate(
        shuffled_share0=shares[0], shuffled_share1=shares[1],
        reference_bits=reference, tau=.4, round_id=1, batch_id=0,
        seed=11, transport=LogicalTransport(), chunk_size=16_384,
    )
    actual = reconstruct_bits(result.server0_direction_share, result.server1_direction_share)
    assert np.array_equal(actual, rain_aggregate_oracle(bits, reference, tau=.4))
    # Bound-aware A2B needs far fewer than 64 bits for this configuration.
    assert result.primitives.boolean_and_gates < 64 * (bits.shape[0] + bits.shape[1])


def test_femnist_leaf_cache_preserves_writer_boundaries(tmp_path):
    base = tmp_path / "femnist" / "data"
    for split in ("train", "test"):
        directory = base / split; directory.mkdir(parents=True)
        payload = {
            "users": ["writer-a", "writer-b"], "num_samples": [2, 1],
            "user_data": {
                "writer-a": {"x": [[0.0] * 784, [1.0] * 784], "y": [0, 61]},
                "writer-b": {"x": [[0.5] * 784], "y": [7]},
            },
        }
        (directory / "all.json").write_text(__import__("json").dumps(payload), encoding="utf-8")
    cache = prepare_femnist_cache(tmp_path, "train")
    dataset = FEMNISTDataset(cache)
    assert len(dataset) == 3
    assert dataset.writer_ids == ("writer-a", "writer-b")
    assert dataset.writer_indices == ((0, 1), (2,))
    assert dataset.targets.tolist() == [0, 61, 7]
    # An unchanged source signature reuses the same processed cache.
    assert prepare_femnist_cache(tmp_path, "train") == cache


def test_femnist_partition_is_natural_and_writer_disjoint():
    writer_indices = tuple(tuple(range(index * 20, (index + 1) * 20)) for index in range(12))
    writer_ids = tuple(f"writer-{index}" for index in range(12))
    targets = np.arange(240, dtype=np.int64) % 62
    kwargs = dict(clients=4, root_size=30, calibration_size=20,
                  root_bias=1 / 62, seed=19, minimum_client_size=2)
    left = partition_femnist_dataset(targets, writer_indices, writer_ids, **kwargs)
    right = partition_femnist_dataset(targets, writer_indices, writer_ids, **kwargs)
    assert left == right
    assert left.partition_kind == "natural-writer"
    assert len(left.client_ids) == 4
    expected = {writer_ids.index(name): set(values)
                for name, values in zip(left.client_ids, left.client_indices)}
    assert all(values == set(writer_indices[index]) for index, values in expected.items())
    public = set(left.root_indices) | set(left.calibration_indices)
    assert all(public.isdisjoint(values) for values in map(set, left.client_indices))
    assert set(left.root_indices).isdisjoint(left.calibration_indices)


def test_large_models_shapes_and_parameter_scale():
    torch = pytest.importorskip("torch")
    from models.resnet_large import resnet34_small, resnet50_small
    resnet34 = resnet34_small(100)
    resnet50 = resnet50_small(200)
    assert resnet34(torch.randn(1, 3, 32, 32)).shape == (1, 100)
    assert resnet50(torch.randn(1, 3, 64, 64)).shape == (1, 200)
    assert 21_000_000 < sum(p.numel() for p in resnet34.parameters()) < 22_000_000
    assert 23_000_000 < sum(p.numel() for p in resnet50.parameters()) < 25_000_000
    from models.simple_cnn import SimpleCNN1C
    femnist = SimpleCNN1C(62)
    assert femnist(torch.randn(2, 1, 28, 28)).shape == (2, 62)
    assert sum(p.numel() for p in femnist.parameters()) == 3_246_270
