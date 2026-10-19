import logging

import numpy as np
import pytest
import torch

from lis2.datasets import BaseDataset
from lis2.models.base_model import BaseModel
from lis2.segmenter import Segmenter


class EchoModel:
    def __init__(self):
        self.device = None
        self.evaluating = False
        self.input_device = None

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.evaluating = True
        return self

    def __call__(self, **batch):
        self.input_device = batch["predictions"].device
        return batch["predictions"]


@pytest.fixture(params=["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
def execution_device(request):
    return torch.device(request.param)


def make_segmenter(
    *,
    source_data,
    batch,
    monkeypatch,
    device=None,
    missing=False,
    no_segmenter=False,
    output_path=None,
):
    segmenter = object.__new__(Segmenter)
    segmenter.logger = logging.getLogger("test.segmenter")
    segmenter.model = EchoModel()
    segmenter.dataset = []
    segmenter.source_data = source_data
    segmenter.source_header = object()
    segmenter.device = (
        device
        if device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    segmenter.batch_size = 4
    segmenter.missing = missing
    segmenter.no_segmenter = no_segmenter
    segmenter.output_path = output_path
    monkeypatch.setattr(
        "lis2.segmenter.DataLoader",
        lambda dataset, **kwargs: [batch],
    )
    return segmenter


def test_segmenter_initialization_loads_model_dataset_and_fits_data(monkeypatch):
    model = EchoModel()
    dataset = [object()]
    source_data = np.ones((2, 3))
    source_header = object()
    monkeypatch.setattr(
        BaseModel,
        "from_snapshot",
        classmethod(lambda cls, snapshot: model),
    )
    monkeypatch.setattr(
        BaseDataset,
        "from_config",
        classmethod(lambda cls, config: dataset),
    )
    monkeypatch.setattr("lis2.segmenter.fits.getdata", lambda source: source_data)
    monkeypatch.setattr("lis2.segmenter.fits.getheader", lambda source: source_header)

    segmenter = Segmenter.from_config(
        {
            "model_snapshot": "model.pt",
            "source": "source.fits",
            "dataset": {"type": "ExampleDataset"},
        }
    )

    assert segmenter.model is model
    assert segmenter.dataset is dataset
    assert segmenter.source_data is source_data
    assert segmenter.source_header is source_header
    expected_device = "cuda" if torch.cuda.is_available() else "cpu"
    assert segmenter.device.type == expected_device


def test_segment_averages_patches_and_skips_invalid_or_wrong_sized_patches(
    monkeypatch, execution_device
):
    batch = {
        "predictions": torch.full((3, 1, 2, 2), 0.75),
        "positions": torch.tensor(
            [
                [[0, 2], [0, 2]],
                [[1, 1], [2, 4]],
                [[0, 1], [3, 4]],
            ]
        ),
    }
    segmenter = make_segmenter(
        source_data=np.zeros((2, 4), dtype=np.float32),
        batch=batch,
        monkeypatch=monkeypatch,
        device=execution_device,
        output_path=None,
    )
    monkeypatch.setattr(
        "lis2.segmenter.skm.footprint_rectangle",
        lambda shape: np.ones(shape, dtype=bool),
        raising=False,
    )
    monkeypatch.setattr("lis2.segmenter.skm.erosion", lambda image, footprint: image)
    monkeypatch.setattr(
        "lis2.segmenter.skm.reconstruction",
        lambda seed, image: image,
    )

    result = segmenter.segment(save_output=False)

    np.testing.assert_array_equal(
        result,
        np.array([[1, 1, 0, 0], [1, 1, 0, 0]], dtype=np.float32),
    )
    assert segmenter.model.device == segmenter.device
    assert segmenter.model.evaluating
    assert segmenter.model.input_device.type == execution_device.type


def test_segment_applies_missing_map_to_patch(monkeypatch, execution_device):
    batch = {
        "predictions": torch.full((1, 1, 2, 2), 0.9),
        "positions": torch.tensor([[[0, 2], [0, 2]]]),
        "missmap": torch.tensor([[[1, 0], [1, 0]]]),
    }
    segmenter = make_segmenter(
        source_data=np.zeros((2, 2), dtype=np.float32),
        batch=batch,
        monkeypatch=monkeypatch,
        device=execution_device,
        missing=True,
    )
    original_squeeze = torch.squeeze

    def squeeze_tensor_or_array(value):
        if isinstance(value, np.ndarray):
            value = torch.from_numpy(value)
        return original_squeeze(value)

    monkeypatch.setattr("lis2.segmenter.torch.squeeze", squeeze_tensor_or_array)
    monkeypatch.setattr(
        "lis2.segmenter.skm.footprint_rectangle",
        lambda shape: np.ones(shape, dtype=bool),
        raising=False,
    )
    monkeypatch.setattr("lis2.segmenter.skm.erosion", lambda image, footprint: image)
    monkeypatch.setattr(
        "lis2.segmenter.skm.reconstruction",
        lambda seed, image: image,
    )

    result = segmenter.segment(save_output=False)

    np.testing.assert_array_equal(result, np.array([[1, 0], [1, 0]]))
    assert segmenter.model.input_device.type == execution_device.type


def test_segment_saves_output_when_requested(monkeypatch, execution_device):
    batch = {
        "predictions": torch.full((1, 1, 2, 2), 0.9),
        "positions": torch.tensor([[[0, 2], [0, 2]]]),
    }
    segmenter = make_segmenter(
        source_data=np.zeros((2, 2), dtype=np.float32),
        batch=batch,
        monkeypatch=monkeypatch,
        device=execution_device,
        output_path="result.fits",
    )
    monkeypatch.setattr(
        "lis2.segmenter.skm.footprint_rectangle",
        lambda shape: np.ones(shape, dtype=bool),
        raising=False,
    )
    monkeypatch.setattr("lis2.segmenter.skm.erosion", lambda image, footprint: image)
    monkeypatch.setattr(
        "lis2.segmenter.skm.reconstruction",
        lambda seed, image: image,
    )
    saved = []
    monkeypatch.setattr(segmenter, "_save_output", saved.append)

    result = segmenter.segment()

    assert saved == [result]


def test_process_patch_returns_batch_when_segmentation_disabled(monkeypatch):
    batch = {"predictions": torch.ones((1, 1, 2, 2))}
    segmenter = make_segmenter(
        source_data=np.zeros((2, 2)),
        batch=batch,
        monkeypatch=monkeypatch,
        no_segmenter=True,
    )

    assert segmenter._process_patch(batch) is batch


def test_post_process_passthrough_normalizes_without_morphology():
    segmenter = object.__new__(Segmenter)
    segmenter.logger = logging.getLogger("test.segmenter")
    segmenter.no_segmenter = True
    segmentation = np.array([[1.0, 0.0], [0.5, 0.0]])
    counts = np.array([[2, 0], [1, 0]])

    result = segmenter._post_process(segmentation, counts)

    np.testing.assert_array_equal(result, np.array([[0.5, 0.0], [0.5, 0.0]]))


def test_save_output_writes_fits_with_header_and_overwrite(monkeypatch):
    segmenter = object.__new__(Segmenter)
    segmenter.logger = logging.getLogger("test.segmenter")
    segmenter.output_path = "result.fits"
    segmenter.source_header = object()
    calls = []
    monkeypatch.setattr(
        "lis2.segmenter.fits.writeto",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    data = np.ones((2, 2))
    segmenter._save_output(data)

    assert calls == [
        (
            ("result.fits",),
            {"data": data, "header": segmenter.source_header, "overwrite": True},
        )
    ]


def test_save_output_reports_write_errors(monkeypatch, capsys):
    segmenter = object.__new__(Segmenter)
    segmenter.logger = logging.getLogger("test.segmenter")
    segmenter.output_path = "result.fits"
    segmenter.source_header = object()

    def fail_write(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("lis2.segmenter.fits.writeto", fail_write)

    segmenter._save_output(np.ones((2, 2)))

    assert "Error saving output file: disk full" in capsys.readouterr().out
