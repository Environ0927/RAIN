import pytest

from rain.privacy import noise_multiplier_from_epsilon0


def test_paper_epsilon0_mapping_uses_replacement_sensitivity():
    assert noise_multiplier_from_epsilon0(10) == pytest.approx(0.2)
    assert noise_multiplier_from_epsilon0(5, kappa=0.5) == pytest.approx(0.2)


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan")])
def test_paper_epsilon0_mapping_rejects_invalid_values(value):
    with pytest.raises(ValueError):
        noise_multiplier_from_epsilon0(value)
