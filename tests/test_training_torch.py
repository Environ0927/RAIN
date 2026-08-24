import numpy as np
import pytest


torch = pytest.importorskip("torch")


def test_tiny_multiround_training_and_reference_refresh(tmp_path):
    from rain.training.adapter import TorchRainAdapter
    from rain.training.coordinator import RainRoundCoordinator
    from rain.training.reference import ReferenceProvider

    torch.manual_seed(2)
    model = torch.nn.Linear(3, 2)
    root_x = torch.tensor([[1., 0., 1.], [0., 1., -1.], [1., 1., 0.]])
    root_y = torch.tensor([0, 1, 0])
    provider = ReferenceProvider(root_x, root_y, loss_fn=torch.nn.CrossEntropyLoss(), seed=4)
    adapter = TorchRainAdapter(RainRoundCoordinator(
        clip_norm=100, noise_multiplier=0, tau=.4, chunk_size=3, seed=5
    ), provider, .01)
    for round_id in range(2):
        gradients = []
        for shift in range(3):
            model.zero_grad(set_to_none=True)
            loss = model(root_x.roll(shift, 0)).sum()
            loss.backward()
            gradients.append([p.grad.detach().clone() for p in model.parameters()])
        result = adapter.step(grad_list=gradients, model=model, round_id=round_id)
        assert result.direction_bits.size == sum(p.numel() for p in model.parameters())
    assert provider.calls == 2
    provider.save_manifest(tmp_path / "reference.json")
    assert (tmp_path / "reference.json").exists()


def test_checkpoint_material_roundtrip(tmp_path):
    model = torch.nn.Linear(2, 1)
    path = tmp_path / "checkpoint.pt"
    torch.save({"schema_version": 1, "model": model.state_dict(), "next_round": 3}, path)
    loaded = torch.load(path, map_location="cpu")
    clone = torch.nn.Linear(2, 1); clone.load_state_dict(loaded["model"])
    assert loaded["next_round"] == 3
    assert all(torch.equal(a, b) for a, b in zip(model.parameters(), clone.parameters()))
