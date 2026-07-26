import pytest

from uavcl.engine import _validate_state_health


def test_state_health_accepts_stable_norm():
    _validate_state_health(
        current=54.0,
        previous=53.0,
        train_cfg={"max_state_norm": 10000.0, "max_state_norm_growth": 10.0},
        task_id=2,
    )


def test_state_health_rejects_absolute_explosion():
    with pytest.raises(FloatingPointError, match="vượt ngưỡng"):
        _validate_state_health(
            current=10000.0,
            previous=54.0,
            train_cfg={"max_state_norm": 10000.0},
            task_id=3,
        )


def test_state_health_rejects_relative_explosion():
    with pytest.raises(FloatingPointError, match="tăng 20.00x"):
        _validate_state_health(
            current=1080.0,
            previous=54.0,
            train_cfg={"max_state_norm_growth": 10.0},
            task_id=4,
        )
