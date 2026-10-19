from lis2.early_stop import BaseEarlyStopping


def test_loss_early_stopping_requires_patience_and_resets_on_improvement():
    early_stopper = BaseEarlyStopping.from_config(
        {"type": "LossEarlyStopping", "patience": 2, "min_delta": 0.1}
    )

    assert early_stopper.step(1.0) is False
    assert early_stopper.step(0.95) is False
    assert early_stopper.counter == 1
    assert early_stopper.step(0.8) is False
    assert early_stopper.best_loss == 0.8
    assert early_stopper.counter == 0
    assert early_stopper.step(0.7) is False


def test_loss_early_stopping_triggers_after_consecutive_non_improvements():
    early_stopper = BaseEarlyStopping.from_config(
        {"type": "LossEarlyStopping", "patience": 2}
    )

    assert early_stopper.step(1.0) is False
    assert early_stopper.step(1.0) is False
    assert early_stopper.step(1.1) is True


def test_loss_early_stopping_reset_clears_state():
    early_stopper = BaseEarlyStopping.from_config(
        {"type": "LossEarlyStopping", "patience": 1}
    )
    early_stopper.step(1.0)
    assert early_stopper.step(1.0) is True

    early_stopper.reset()

    assert early_stopper.best_loss is None
    assert early_stopper.counter == 0
    assert early_stopper.step(0.5) is False
