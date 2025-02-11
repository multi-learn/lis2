from tempfile import tempdir
from unittest.mock import patch

import pytest
import torch
from configurable import GlobalConfig

from core.trainer import Trainer
from tests.trainer.mocks import (
    MockDataset,
    MockModel,
    MockOptimizer,
    MockScheduler,
    MockEarlyStopping,
    MockMetrics,
)


def set_seed(seed):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

@pytest.fixture
def trainer_config():
    c = {
        'output_dir': tempdir,
        'run_name': 'test_run',
        'epochs': 2,  # Reduced epochs for faster testing
        'model': {
            'type': 'MockModel',
            'name': 'mock_model'
        },
        'optimizer': {
            'type': 'MockOptimizer',
        },
        'scheduler': {
            'type': 'MockScheduler',
        },
        'train_dataset': {
            'type': 'MockDataset',
        },
        'val_dataset': {
            'type': 'MockDataset',
        },
        'early_stopper': {
            'type': 'MockEarlyStopping'
        },
        'split_ratio': 0.8,
        'batch_size': 2,
        'num_workers': 1,  # Use 0 to avoid multiprocessing in tests
        'save_interval': 1,
        'metrics': [
            {'type': 'MockMetrics', 'name': 'mock_metric_1'},
            {'type': 'MockMetrics', 'name': 'mock_metric_2'}
        ]
    }
    GlobalConfig(config=c)
    return c


@pytest.fixture(params=['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu'])
def device(request):
    """Fixture to run tests on both CPU and GPU if available."""
    if request.param == 'cuda':
        return torch.device('cuda:0')
    else:
        return torch.device('cpu')


@patch('datasets.BaseDataset', MockDataset)
@patch('models.custom_model.BaseModel', MockModel)
@patch('core.optim.BaseOptimizer', MockOptimizer)
@patch('core.scheduler.BaseScheduler', MockScheduler)
@patch('core.early_stop.EarlyStopping', MockEarlyStopping)
@patch('core.metrics.Metrics', MockMetrics)
def test_trainer_initialization(trainer_config, device):
    set_seed(42)
    trainer = Trainer.from_config(trainer_config)
    trainer.model.to(device)  # Move model to specified device
    assert trainer.model is not None, "Model should be initialized"
    assert trainer.optimizer is not None, "Optimizer should be initialized"
    assert trainer.scheduler is not None, "Scheduler should be initialized"
    # Use string comparison to handle 'cuda' vs 'cuda:0'
    assert str(next(trainer.model.parameters()).device) == str(device), "Model is not on the expected device"


@patch('datasets.BaseDataset', MockDataset)
@patch('models.custom_model.BaseModel', MockModel)
def test_run_batch(trainer_config, device):
    set_seed(42)
    trainer = Trainer.from_config(trainer_config)
    trainer.model.to(device)
    batch = {
        'inputs': torch.randn(2, 3, 3, device=device, dtype=torch.float32),  # [batch_size=2, 3,3]
        'target': torch.rand(2, 3, 3, device=device, dtype=torch.float32),  # [batch_size=2, 3,3]
        'labelled': torch.ones(2, 3, 3, device=device, dtype=torch.uint8)  # [batch_size=2, 3,3]
    }
    loss, idx_sum = trainer._run_batch(batch)
    assert loss.item() >= 0, "Loss should be non-negative"
    assert idx_sum == torch.numel(batch['labelled']), "Sum of indices should match number of labelled elements"


@patch('datasets.BaseDataset', MockDataset)
def test_save_snapshot(tmp_path, trainer_config, device):
    set_seed(42)
    trainer = Trainer.from_config(trainer_config)
    trainer.model.to(device)
    snapshot_path = tmp_path / "snapshot.pt"
    trainer._save_snapshot(epoch=0, path=str(snapshot_path), loss=0.5)
    assert snapshot_path.exists(), "Snapshot file should exist after saving"


@patch('datasets.BaseDataset', MockDataset)
def test_create_dataloader(trainer_config, device):
    set_seed(42)
    trainer = Trainer.from_config(trainer_config)
    dataloader = trainer._create_dataloader(MockDataset(), is_train=True)
    batch = next(iter(dataloader))
    batch = {k: v.to(device) for k, v in batch.items()}
    assert 'inputs' in batch, "Batch should contain 'inputs'"
    assert 'target' in batch, "Batch should contain 'target'"
    assert 'labelled' in batch, "Batch should contain 'labelled'"
    # Check shapes
    assert batch['inputs'].shape == (2, 30, 30), f"Expected 'inputs' shape (2,30,30), got {batch['inputs'].shape}"
    assert batch['target'].shape == (2, 30, 30), f"Expected 'target' shape (2,30,30), got {batch['target'].shape}"
    assert batch['labelled'].shape == (2, 30, 30), f"Expected 'labelled' shape (2,30,30), got {batch['labelled'].shape}"


@patch('datasets.BaseDataset', MockDataset)
@patch('models.custom_model.BaseModel', MockModel)
@patch('core.optim.BaseOptimizer', MockOptimizer)
def test_train_method(trainer_config, device):
    """Test the train method of the Trainer class."""
    set_seed(42)
    trainer = Trainer.from_config(trainer_config, force_device=device)
    trainer.epochs = 10
    initial_weight = trainer.model.conv1.weight.clone()
    trainer.train()
    updated_weight = trainer.model.conv1.weight
    assert torch.equal(initial_weight,
                       updated_weight), "Les paramètres du modèle ne se sont pas mis à jour durant l'entraînement"


@patch('datasets.BaseDataset', MockDataset)
def test_run_loop_validation(trainer_config, device):
    """Test the validation loop."""
    set_seed(42)
    trainer = Trainer.from_config(trainer_config, force_device=device)
    dataloader = trainer._create_dataloader(MockDataset(), is_train=False)
    avg_loss = trainer._run_loop_validation(epoch=0, custom_dataloader=dataloader)
    assert avg_loss.item() >= 0, "Validation loss should be non-negative"


@patch('datasets.BaseDataset', MockDataset)
def test_from_snapshot(tmp_path, trainer_config, device):
    """Test loading trainer from a saved snapshot."""
    set_seed(42)
    trainer = Trainer.from_config(trainer_config, force_device=device)
    snapshot_path = tmp_path / "snapshot.pt"
    trainer._save_snapshot(epoch=10, path=str(snapshot_path), loss=0.5)
    # Load from snapshot
    trainer_loaded = Trainer.from_snapshot(str(snapshot_path))
    assert trainer_loaded.epochs_run == 10, "Loaded trainer should have epochs_run set to 1"
    assert trainer_loaded.best_loss == 0.5, "Loaded trainer should have best_loss set to 0.5"
    # Ensure loaded model is on the correct device
    trainer_loaded.model.to(device)
    assert str(next(trainer_loaded.model.parameters()).device) == str(
        device), "Loaded model is not on the expected device"
