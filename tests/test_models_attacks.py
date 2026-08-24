import numpy as np
import pytest

from rain.training.attacks import ATTACKS, AttackContext, apply_attack


def test_all_six_attacks_exist_and_preserve_shape():
    updates = np.random.default_rng(1).normal(size=(8, 20))
    assert set(ATTACKS) == {"krum", "min-max", "scaling", "attack-dpfl", "raa", "rsca"}
    for name in ATTACKS:
        context = AttackContext(2, 4, target_direction=np.ones(20, dtype=np.int8), hamming_budget=3)
        result = apply_attack(name, updates, context)
        assert result.shape == updates.shape
        assert np.array_equal(result[2:], updates[2:])


@pytest.mark.parametrize("name", ["raa", "rsca"])
def test_sign_attacks_respect_exact_hamming_budget(name):
    updates = np.ones((5, 31))
    target = -np.ones(31, dtype=np.int8)
    result = apply_attack(name, updates, AttackContext(2, 7, target, 6))
    signs = np.where(result[:2] >= 0, 1, -1)
    assert np.all(np.sum(signs != target, axis=1) == 6)


def test_models_forward_backward():
    torch = pytest.importorskip("torch")
    from models.fnet import FashionNet
    from models.resnet_cifar import resnet18_cifar
    fashion = FashionNet()
    fashion(torch.randn(2, 1, 28, 28)).sum().backward()
    cifar = resnet18_cifar()
    output = cifar(torch.randn(2, 3, 32, 32))
    assert output.shape == (2, 10)
    output.sum().backward()
    assert 10_000_000 < sum(p.numel() for p in cifar.parameters()) < 12_000_000


def test_scaling_backdoor_image_pipeline():
    torch = pytest.importorskip("torch")
    from attacks import add_backdoor, scaling_attack_insert_backdoor
    images = [torch.zeros(4, 1, 28, 28), torch.zeros(4, 1, 28, 28)]
    labels = [torch.ones(4, dtype=torch.long), torch.ones(4, dtype=torch.long)]
    poisoned_data, poisoned_labels = scaling_attack_insert_backdoor(
        images, labels, "MNIST", 1, torch.device("cpu")
    )
    assert poisoned_data[0].shape[0] == 6
    assert poisoned_labels[0][-2:].eq(0).all()
    triggered, targets = add_backdoor(torch.zeros(3, 3, 32, 32), torch.ones(3, dtype=torch.long), "CIFAR10")
    assert targets.eq(0).all()
    assert triggered[:, :, -3:, -3:].eq(triggered.amax(dim=(1, 2, 3), keepdim=True)).all()
