import pytest

from rain.cli.train import _effective_learning_rate


def test_cosine_restarts_hits_high_and_low_endpoints():
    training = {
        "rounds": 1000,
        "learning_rate": 0.01,
        "minimum_learning_rate": 0.001,
        "lr_schedule": "cosine_restarts",
        "lr_restart_period": 5,
    }

    assert _effective_learning_rate(training, 0) == pytest.approx(0.01)
    assert _effective_learning_rate(training, 4) == pytest.approx(0.001)
    assert _effective_learning_rate(training, 5) == pytest.approx(0.01)


def test_unknown_schedule_is_rejected():
    with pytest.raises(ValueError, match="unknown lr_schedule"):
        _effective_learning_rate({"learning_rate": 0.1, "lr_schedule": "invented"}, 0)


def test_randomized_restarts_are_seeded_and_bounded():
    training = {
        "rounds": 1000,
        "learning_rate": 0.01,
        "minimum_learning_rate": 0.001,
        "lr_schedule": "randomized_cosine_restarts",
        "lr_schedule_seed": 7,
        "lr_restart_period_min": 5,
        "lr_restart_period_max": 9,
        "lr_restart_scale_min": 0.5,
        "lr_restart_scale_max": 1.2,
    }
    first = [_effective_learning_rate(training, round_id) for round_id in range(30)]
    second = [_effective_learning_rate(training, round_id) for round_id in range(30)]

    assert first == second
    assert min(first) >= 0.001
    assert max(first) <= 0.012
    assert len({round(value, 8) for value in first}) > 10


def test_randomized_pulses_are_short_seeded_and_bounded():
    training = {
        "rounds": 1000,
        "learning_rate": 0.01,
        "minimum_learning_rate": 0.001,
        "lr_schedule": "randomized_lr_pulses",
        "lr_schedule_seed": 11,
        "lr_initial_decay_rounds": 10,
        "lr_pulse_interval_min": 8,
        "lr_pulse_interval_max": 12,
        "lr_pulse_duration_min": 2,
        "lr_pulse_duration_max": 4,
        "lr_pulse_scale_min": 0.4,
        "lr_pulse_scale_max": 0.7,
    }
    first = [_effective_learning_rate(training, round_id) for round_id in range(50)]
    second = [_effective_learning_rate(training, round_id) for round_id in range(50)]

    assert first == second
    assert first[0] == pytest.approx(0.01)
    assert first[9] == pytest.approx(0.001)
    assert min(first) >= 0.001
    assert max(first) == pytest.approx(0.01)
    assert first[10] > 0.001
    assert any(
        first[index] == pytest.approx(0.001)
        and first[index + 1] == pytest.approx(0.001)
        and first[index + 2] == pytest.approx(0.001)
        for index in range(10, len(first) - 2)
    )


def test_randomized_pulse_rejects_overlapping_duration_range():
    training = {
        "rounds": 100,
        "learning_rate": 0.01,
        "minimum_learning_rate": 0.001,
        "lr_schedule": "randomized_lr_pulses",
        "lr_initial_decay_rounds": 10,
        "lr_pulse_interval_min": 4,
        "lr_pulse_interval_max": 8,
        "lr_pulse_duration_min": 2,
        "lr_pulse_duration_max": 4,
        "lr_pulse_scale_min": 0.4,
        "lr_pulse_scale_max": 0.7,
    }

    with pytest.raises(ValueError, match="duration must be shorter"):
        _effective_learning_rate(training, 10)
