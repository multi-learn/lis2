import numpy as np
import pytest

from lis2.datasets.transforms import (
    apply_extended_transform,
    apply_noise_transform,
    transform,
)


@pytest.mark.parametrize(
    ("num_tf", "expected"),
    [
        (0, lambda data: data),
        (1, np.fliplr),
        (2, np.flipud),
        (3, np.rot90),
        (4, lambda data: np.rot90(data, 2)),
        (5, lambda data: np.rot90(data, 3)),
        (6, lambda data: np.fliplr(np.flipud(data))),
        (7, lambda data: np.fliplr(np.rot90(data))),
        (8, lambda data: np.fliplr(np.rot90(data, 2))),
        (9, lambda data: np.fliplr(np.rot90(data, 3))),
        (10, lambda data: np.flipud(np.rot90(data))),
        (11, lambda data: np.flipud(np.rot90(data, 2))),
        (12, lambda data: np.flipud(np.rot90(data, 3))),
        (13, lambda data: np.fliplr(np.flipud(np.rot90(data)))),
        (14, lambda data: np.fliplr(np.flipud(np.rot90(data, 2)))),
        (15, lambda data: np.fliplr(np.flipud(np.rot90(data, 3)))),
    ],
)
def test_transform_preserves_shape_and_applies_expected_orientation(num_tf, expected):
    data = np.arange(6).reshape(1, 2, 3)

    result = transform(data, num_tf)

    np.testing.assert_array_equal(result, expected(data[0]).reshape(data.shape))
    assert result.shape == data.shape
    assert not np.shares_memory(result, data)


def test_apply_noise_transform_perturbs_first_pair_only(monkeypatch):
    monkeypatch.setattr(
        np.random, "standard_normal", lambda shape: np.ones(shape, dtype=np.float64)
    )
    first = np.zeros((2, 2), dtype=np.float32)
    second = np.ones((2, 2), dtype=np.float32)
    third = np.full((2, 2), 3)
    data = [first, second, third]

    result = apply_noise_transform(data, input_noise_var=0.25, output_noise_var=0.5)

    np.testing.assert_array_equal(result[0], np.full((2, 2), 0.25))
    np.testing.assert_array_equal(result[1], np.full((2, 2), 1.5))
    assert result[0].dtype == np.float32
    assert result[1].dtype == np.float32
    assert result[2] is third
    np.testing.assert_array_equal(first, np.zeros((2, 2)))


def test_apply_extended_transform_uses_one_transform_for_all_arrays(monkeypatch):
    class FixedTransform:
        def randint(self, low, high):
            assert (low, high) == (0, 15)
            return 3

    monkeypatch.setattr(
        np.random, "standard_normal", lambda shape: np.ones(shape, dtype=np.float64)
    )
    first = np.arange(6).reshape(2, 3)
    second = np.arange(6, 12).reshape(2, 3)

    result = apply_extended_transform(
        [first, second], FixedTransform(), [0.5, 1.0]
    )

    expected_first = np.rot90(first + 0.5).copy().reshape(first.shape)
    expected_second = np.rot90(second + 1.0).copy().reshape(second.shape)
    np.testing.assert_array_equal(result[0], expected_first)
    np.testing.assert_array_equal(result[1], expected_second)
    np.testing.assert_array_equal(first, np.arange(6).reshape(2, 3))
