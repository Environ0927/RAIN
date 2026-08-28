import numpy as np
import pytest

from rain.training.attacks import ATTACKS, AttackContext, apply_attack


def test_current_paper_attacks_and_compatibility_rsca_exist_and_preserve_shape():
    updates = np.random.default_rng(1).normal(size=(8, 20))
    assert set(ATTACKS) == {
        "krum", "min-max", "scaling", "attack-dpfl", "raa", "woaa", "rsca",
    }
    for name in ATTACKS:
        context = AttackContext(
            2, 4, target_direction=np.ones(20, dtype=np.int8),
            hamming_budget=3, scale=0.4 if name == "woaa" else 10.0,
        )
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


def test_rsca_coordinates_the_same_high_impact_flip_set():
    updates = np.vstack([
        np.ones((3, 12)),
        np.tile(np.arange(1, 13, dtype=np.float64), (4, 1)),
    ])
    reference = np.ones(12, dtype=np.int8)
    adversarial = -np.arange(1, 13, dtype=np.float64)
    result = apply_attack(
        "rsca",
        updates,
        AttackContext(
            malicious_clients=3,
            seed=8,
            target_direction=reference,
            hamming_budget=4,
            adversarial_direction=adversarial,
        ),
    )
    signs = np.where(result[:3] >= 0, 1, -1)
    assert np.array_equal(signs[0], signs[1])
    assert np.array_equal(signs[1], signs[2])
    assert np.array_equal(np.flatnonzero(signs[0] != reference), np.arange(8, 12))


def test_woaa_selects_the_best_radius_and_coordinates_clients():
    updates = np.vstack([np.ones((2, 5)), np.ones((3, 5))])
    reference = np.ones(5, dtype=np.int8)
    # Flipping the two negative-product coordinates improves the directional
    # term enough to offset the Hamming penalty at tau=0.6.
    adversarial = np.array([-4.0, -3.0, 1.0, 1.0, 1.0])
    result = apply_attack(
        "woaa", updates,
        AttackContext(
            malicious_clients=2, seed=2, target_direction=reference,
            scale=0.6, adversarial_direction=adversarial,
        ),
    )
    signs = np.where(result[:2] >= 0, 1, -1)
    assert np.array_equal(signs[0], signs[1])
    assert np.array_equal(np.flatnonzero(signs[0] != reference), np.array([0, 1]))


def test_attack_dpfl_matches_noisy_target_norm_without_changing_benign_rows():
    updates = np.array([
        [3.0, 4.0],
        [2.0, 0.0],
        [2.0, 0.0],
        [2.0, 0.0],
    ])
    result = apply_attack(
        "attack-dpfl", updates,
        AttackContext(malicious_clients=1, seed=3, noise_std=0.0),
    )
    benign_mean_norm = np.linalg.norm(-updates[1:].mean(axis=0))
    assert np.linalg.norm(result[0]) == pytest.approx(benign_mean_norm)
    assert np.array_equal(result[1:], updates[1:])


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
    tiny = resnet18_cifar(num_classes=200)
    tiny_output = tiny(torch.randn(2, 3, 64, 64))
    assert tiny_output.shape == (2, 200)
    assert sum(p.numel() for p in tiny.parameters()) == 11_271_432


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
