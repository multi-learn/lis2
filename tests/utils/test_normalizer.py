import numpy as np

import lis2.utils.normalizer as normalizer


def test_normalize_direct():
    x = np.array([1.0, 2.0, 4.0])
    x_norm = normalizer.normalize_direct(x)
    assert x_norm.shape == (3,)
    assert x_norm.min() == 0.0
    assert x_norm.max() == 1.0
    assert x_norm[0] == 0.0
    assert x_norm[2] == 1.0


def test_normalize_histo():
    x = np.array([1.0, 2.0, 4.0, 4.0, 4.0])
    x_norm = normalizer.normalize_histo(x)
    assert x_norm.shape == (5,)
    assert x_norm.min() == 0.2
    assert x_norm[0] == 0.2
    assert x_norm[4] == 1.0


def test_normalize_adapt_histo():
    x = np.array([[1.0, 2.0, 4.0, 4.0, 4.0], [1.0, 2.0, 4.0, 4.0, 4.0]])
    x_norm = normalizer.normalize_adapt_histo(x)
    print(x_norm)
    assert x_norm.shape == (2, 5)
    assert x_norm.min() == 0.0
    assert x_norm[0, 0] == 0.0
    assert x_norm[1, 4] == 1.0

def test_normalize_direct_with_explicit_bounds():
    x = np.array([2.0, 4.0, 8.0])

    result = normalizer.normalize_direct(x, xmin=1.0, zmax=7.0)

    np.testing.assert_allclose(result, [1 / 7, 3 / 7, 1])
    np.testing.assert_array_equal(x, [2.0, 4.0, 8.0])


def test_normalize_direct_constant_input():
    x = np.array([3.0, 3.0, 3.0])

    result = normalizer.normalize_direct(x)

    np.testing.assert_array_equal(result, [0.0, 0.0, 0.0])
