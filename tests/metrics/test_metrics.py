import numpy as np
import pytest
from skimage.metrics import structural_similarity
from sklearn.metrics import average_precision_score, roc_auc_score

from core.metrics import Metrics, Metric


@pytest.fixture
def setup_data():
    # Données en matrices 2D (ex. segmentation binaire)
    pred = np.array([
        [0.8, 0.2, 0.6],
        [0.1, 0.7, 0.9],
        [0.5, 0.3, 0.8]
    ])
    target = np.array([
        [1, 0, 1],
        [0, 1, 1],
        [1, 0, 1]
    ])
    idx = np.ones_like(target)  # Masque pour tous les éléments
    return pred, target, idx


def test_average_precision(setup_data):
    pred, target, idx = setup_data
    metric = Metric.from_config({'type': 'AveragePrecision'})
    metric.update(pred.flatten(), target.flatten(), idx.flatten())
    computed_value = metric.compute()

    # Calcul de la précision moyenne attendue
    pred_bin = np.round(pred).astype(int)
    target_bin = np.round(target).astype(int)
    expected_value = average_precision_score(target_bin.flatten(), pred_bin.flatten())

    assert pytest.approx(computed_value, 0.00001) == expected_value
    metric.reset()
    assert metric.result == 0
    assert metric.averaging_coef == 0


def test_dice(setup_data):
    pred, target, idx = setup_data
    metric = Metric.from_config({'type': 'Dice', 'threshold': 0.5})
    metric.update(pred, target, idx)
    computed_value = metric.compute()

    # Calcul attendu pour le Dice
    pred_segmented = (pred >= 0.5).astype(int)
    TP = np.sum((pred_segmented + target) == 2)
    FP = np.sum((2 * pred_segmented + target) == 2)
    FN = np.sum((pred_segmented + 2 * target) == 2)
    expected_value = 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 1.0

    assert pytest.approx(computed_value, 0.00001) == expected_value
    metric.reset()
    assert metric.result == 0
    assert metric.averaging_coef == 0


def test_roc_auc(setup_data):
    pred, target, idx = setup_data
    metric = Metric.from_config({'type': 'ROCAUCScore'})
    metric.update(pred, target, idx)
    computed_value = metric.compute()

    expected_value = roc_auc_score(target.flatten(), np.round(pred.flatten()))
    assert pytest.approx(computed_value, 0.00001) == expected_value
    metric.reset()
    assert metric.result == 0
    assert metric.averaging_coef == 0


def test_mssim(setup_data):
    pred, target, idx = setup_data
    win_size = 3
    metric = Metric.from_config({'type': 'MSSIM', 'threshold': 0.5, 'win_size': win_size})
    metric.update(pred, target, idx)
    computed_value = metric.compute()

    pred_segmented = (pred >= 0.5).astype(np.int32)
    mssim_values = [
        structural_similarity(pred_segmented[i], target[i], data_range=1, win_size=win_size)
        for i in range(pred_segmented.shape[0])
    ]
    expected_value = np.mean(mssim_values)

    assert pytest.approx(computed_value, 0.00001) == expected_value
    metric.reset()
    assert metric.result == 0
    assert metric.averaging_coef == 0


def test_metrics_container(setup_data):
    pred, target, idx = setup_data
    metrics_configs = [
        {'type': 'AveragePrecision'},
        {'type': 'Dice', 'threshold': 0.5},
        {'type': 'ROCAUCScore'},
        {'type': 'MSSIM', 'threshold': 0.5, 'win_size': 3}
    ]
    container = Metrics(metrics_configs)
    container.update(pred, target, idx)

    results = container.compute()
    assert 'average_precision' in results
    assert 'dice' in results
    assert 'roc_auc' in results
    assert 'mean_ssim' in results

    container.reset()
    for metric in container.metrics:
        assert metric.result == 0
        assert metric.averaging_coef == 0


def test_compute_without_update():
    """Test if compute raises an exception when called before update."""
    metric = Metric.from_config({'type': 'AveragePrecision'})
    with pytest.raises(ValueError, match="No data to compute metric"):
        metric.compute()


def test_mssim_win_size_exceeds_image(setup_data):
    """Test MSSIM raises an exception if win_size is larger than image dimensions."""
    pred, target, idx = setup_data
    metric = Metric.from_config({'type': 'MSSIM', 'threshold': 0.5, 'win_size': 5})

    with pytest.raises(ValueError, match="win_size exceeds image extent"):
        metric.update(pred[:2, :2], target[:2, :2], idx[:2, :2])  # Image of size 2x2, win_size of 5


def test_dice_non_binary_values(setup_data):
    """Test Dice metric raises an exception if pred or target contains non-binary values."""
    pred, target, idx = setup_data
    metric = Metric.from_config({'type': 'Dice', 'threshold': 0.5})

    # Introduce non-binary values in target
    target_with_non_binary = np.array([
        [1, 0.5, 1],
        [0, 1.5, 1],
        [1, 0, 1]
    ])

    with pytest.raises(ValueError, match="The groundtruth_images tensor should be a 0-1 map"):
        metric.update(pred, target_with_non_binary, idx)


def test_metrics_container_with_invalid_metric():
    """Test Metrics container raises an exception if an invalid metric type is given."""
    metrics_configs = [
        {'type': 'AveragePrecision'},
        {'type': 'InvalidMetricType'}  # Invalid metric type
    ]

    with pytest.raises(ValueError, match="Type 'InvalidMetricType' not found"):
        Metrics(metrics_configs)
