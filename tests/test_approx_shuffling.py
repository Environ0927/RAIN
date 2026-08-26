import math

import pytest

from rain.privacy.approx_shuffling import (
    approximate_shuffling_epsilon,
    theorem_domain,
)


def test_theorem_formula_and_metadata():
    report = approximate_shuffling_epsilon(
        clients=1_000_000, local_epsilon=2.0, delta=1e-5
    )
    expected = math.sqrt(64 * math.exp(2.0) * math.log(4e5) / 1_000_000)
    assert report.central_epsilon == pytest.approx(expected)
    assert report.adjacency == "removal"
    assert "III.1" in report.theorem


def test_bound_decreases_with_more_clients():
    small = approximate_shuffling_epsilon(
        clients=1_000_000, local_epsilon=2.0, delta=1e-5
    )
    large = approximate_shuffling_epsilon(
        clients=4_000_000, local_epsilon=2.0, delta=1e-5
    )
    assert large.central_epsilon == pytest.approx(small.central_epsilon / 2)


@pytest.mark.parametrize(
    "clients,local_epsilon,delta",
    [(1, 2.0, 1e-5), (1000, 0.9, 1e-5), (1000, 2.0, 0.0), (1000, 20.0, 1e-5)],
)
def test_invalid_or_out_of_domain_parameters_are_rejected(clients, local_epsilon, delta):
    valid, _ = theorem_domain(clients, local_epsilon, delta)
    assert not valid
    with pytest.raises(ValueError, match="outside Theorem"):
        approximate_shuffling_epsilon(
            clients=clients, local_epsilon=local_epsilon, delta=delta
        )
