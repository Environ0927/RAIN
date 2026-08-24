import numpy as np
import pytest


torch = pytest.importorskip("torch")

from rain.training.local_update import LocalSGDTrainer, apply_weighted_buffers


def test_local_sgd_returns_global_minus_local_parameter_delta():
    model = torch.nn.Linear(1, 2, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0], [-1.0]]))
    values = torch.tensor([[1.0]])
    targets = torch.tensor([1])
    loss_fn = torch.nn.CrossEntropyLoss()
    expected = model.weight.detach().clone()
    loss_fn(model(values), targets).backward()
    expected_update = 0.1 * model.weight.grad.detach().reshape(-1)
    model.zero_grad(set_to_none=True)
    trainer = LocalSGDTrainer(model, learning_rate=0.1)
    result = trainer.run(model, [(values, targets)], loss_fn)
    assert result.examples_seen == 1
    assert np.allclose(result.update, expected_update.numpy(), atol=1e-6)
    assert torch.equal(model.weight, expected)


def test_buffer_averaging_uses_client_weights():
    model = torch.nn.Sequential(torch.nn.BatchNorm1d(2), torch.nn.Linear(2, 2))
    trainer = LocalSGDTrainer(model, learning_rate=0.01)
    loss_fn = torch.nn.CrossEntropyLoss()
    left = trainer.run(model, [(torch.tensor([[1.0, 3.0], [2.0, 4.0]]), torch.tensor([0, 1]))], loss_fn)
    right = trainer.run(model, [(torch.tensor([[5.0, 7.0], [6.0, 8.0]]), torch.tensor([0, 1]))], loss_fn)
    apply_weighted_buffers(model, [left, right], np.asarray([1.0, 3.0]))
    expected = left.buffers["0.running_mean"] * 0.25 + right.buffers["0.running_mean"] * 0.75
    assert torch.allclose(model[0].running_mean.cpu(), expected)
