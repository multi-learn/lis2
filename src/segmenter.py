from pathlib import Path
from typing import Union, Dict, Any

import astropy.io.fits as fits
import numpy as np
import skimage.morphology as skm
import torch
from configurable import Configurable, Schema, Config
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.datasets import BaseDataset
from src.models.base_model import BaseModel


class Segmenter(Configurable):
    """
    Segmenter class for performing segmentation tasks with configurable models and datasets.

    The Segmenter class is designed to handle segmentation tasks using a configurable model and dataset.
    It supports loading models from snapshots, processing data in batches, and saving the segmentation
    output to a FITS file. The class also includes options to handle missing data and disable segmentation.

    Configuration:
        - **name** (str): The name of the segmentation run.
        - **model_snapshot** (Union[Path, str]): Path to the model snapshot file.
        - **source** (Union[Path, str]): Path to the source data file.
        - **dataset** (Config): Configuration for the dataset.
        - **batch_size** (int): Batch size for processing data. Default is 32.
        - **missing** (bool): Flag to handle missing data. Default is False.
        - **no_segmenter** (bool): Flag to disable segmentation. Default is False.
        - **output_path** (str): Path to save the output file. Default is "output.fits".

    Example Configuration (YAML):
        .. code-block:: yaml

            name: "example_segmentation"
            model_snapshot: "path/to/model_snapshot.pt"
            source: "path/to/source_data.fits"
            dataset:
                type: "ExampleDataset"
                dataset_path: "path/to/dataset.h5"
                learning_mode: "onevsall"
                toEncode: ["positions"]
            batch_size: 32
            missing: False
            no_segmenter: False
            output_path: "segmentation_output.fits"

    Aliases:
        model
        source
        dataset
    """

    config_schema = {
        "model_snapshot": Schema(type=Union[Path, str], aliases=["model"]),
        "source": Schema(type=Union[Path, str], aliases=["source"]),
        "dataset": Schema(type=Config, aliases=["dataset"]),
        "batch_size": Schema(int, default=32),
        "missing": Schema(bool, default=False),
        "no_segmenter": Schema(bool, default=False),
        "output_path": Schema(str, default="output.fits"),
    }

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Initializes the Segmenter class from configuration.
        Uses the TypedConfigurable mechanism to automatically apply configurations.
        """
        super().__init__(*args, **kwargs)

        self.logger.debug("Initializing Segmenter")

        self.logger.debug(f"Loading model from snapshot: {self.model_snapshot}")
        self.model = BaseModel.from_snapshot(self.model_snapshot)
        self.logger.debug(f"Model loaded: {self.model}")

        self.logger.debug(f"Loading dataset from config: {self.dataset}")
        self.dataset = BaseDataset.from_config(self.dataset)
        self.logger.debug(f"Dataset loaded: {self.dataset}")

        self.logger.debug(f"Loading source data from: {self.source}")
        self.source_data = fits.getdata(self.source)
        self.source_header = fits.getheader(self.source)
        self.logger.debug(f"Source data shape: {self.source_data.shape}")

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.debug(f"Using device: {self.device}")

    def segment(self, save_output: bool = True) -> np.ndarray:
        """
        Perform the segmentation on the dataset and save the output.

        Args:
            save_output (bool): Whether to save the output segmentation map.

        Returns:
            np.ndarray: The segmentation map.
        """
        self.logger.debug("Starting segmentation process")
        self.model.to(self.device)
        self.model.eval()

        dataloader = DataLoader(
            self.dataset, num_workers=1, batch_size=self.batch_size, shuffle=False
        )
        self.logger.debug(f"Dataloader created with batch size: {self.batch_size}")

        segmentation_map = np.zeros_like(self.source_data)
        count_map = np.zeros_like(segmentation_map)

        with torch.no_grad():
            for batch_idx, samples in enumerate(
                tqdm(dataloader, desc="Processing Patches", unit="batch")
            ):
                self.logger.debug(f"Processing batch {batch_idx}")
                samples = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in samples.items()
                }
                patch_segmented = self._process_patch(samples)
                positions = samples.get("positions", None)

                for i in range(patch_segmented.shape[0]):
                    x_start, x_end, y_start, y_end = (
                        int(positions[i][0][0]),
                        int(positions[i][0][1]),
                        int(positions[i][1][0]),
                        int(positions[i][1][1]),
                    )
                    self.logger.debug(
                        f"Patch {i} position: x({x_start}:{x_end}), y({y_start}:{y_end})"
                    )

                    if x_start >= x_end or y_start >= y_end:
                        self.logger.warning(
                            f"Invalid patch position for patch {i}: x_start >= x_end or y_start >= y_end"
                        )
                        continue

                    segmentation_patch = torch.squeeze(
                        patch_segmented[i].cpu().detach()
                    ).numpy()
                    expected_shape = (x_end - x_start, y_end - y_start)
                    if segmentation_patch.shape == expected_shape:
                        segmentation_map[
                            x_start:x_end, y_start:y_end
                        ] += segmentation_patch

                    if self.missing:
                        missmap = samples["missmap"][i].cpu().numpy()
                        segmentation_map[x_start:x_end, y_start:y_end] *= torch.squeeze(
                            missmap
                        ).numpy()
                    count_map[x_start:x_end, y_start:y_end] += 1

            segmentation_map = self._post_process(segmentation_map, count_map)
            if save_output and self.output_path:
                self._save_output(segmentation_map)
            return segmentation_map

    def _process_patch(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Pass the patch through the model, handling various model types.

        Args:
            batch (Dict[str, torch.Tensor]): A batch of samples to process.

        Returns:
            torch.Tensor: The segmented patches.
        """
        self.logger.debug("Processing a patch through the model")
        if self.no_segmenter:
            self.logger.debug("No segmenter mode, returning batch as is")
            return batch
        return self.model(**batch)

    def _post_process(
        self, segmentation_map: np.ndarray, count_map: np.ndarray
    ) -> np.ndarray:
        """
        Post-process the segmentation output, including normalizing and saving the output.

        Args:
            segmentation_map (np.ndarray): The segmentation map to post-process.
            count_map (np.ndarray): The count map used for normalization.

        Returns:
            np.ndarray: The post-processed segmentation map.
        """
        self.logger.debug("Post-processing the segmentation map...")
        idx = count_map > 0
        self.logger.debug(f"Number of pixels to normalize: {np.sum(idx)}")
        segmentation_map[idx] /= count_map[idx]

        if not self.no_segmenter:
            self.logger.debug("Applying threshold and morphological operations")
            segmentation_map[segmentation_map > 0.5] = 1
            segmentation_map[segmentation_map <= 0.5] = 0
            erode = skm.erosion(segmentation_map, skm.square(4))
            segmentation_map = skm.reconstruction(erode, segmentation_map)
        self.logger.debug("Done post-processing the segmentation map")
        return segmentation_map

    def _save_output(self, segmentation_map: np.ndarray) -> None:
        """
        Save the segmentation map to a FITS file.

        Args:
            segmentation_map (np.ndarray): The segmentation map to save.
        """
        self.logger.debug(f"Saving segmentation map to '{self.output_path}'")
        try:
            fits.writeto(
                self.output_path,
                data=segmentation_map,
                header=self.source_header,
                overwrite=True,
            )
            self.logger.debug(f"Segmentation map saved to '{self.output_path}'")
        except Exception as e:
            self.logger.error(f"Error saving output file: {e}", exc_info=True)
            print(f"Error saving output file: {e}")


if __name__ == "__main__":
    config = {
        "model_snapshot": "sample_merged/run_fold_0/best.pt",
        "source": "/mnt/data/WORK/BigSF/data/spine_merged.fits",
        "dataset": {
            "type": "FilamentsDataset",
            "dataset_path": "/mnt/data/WORK/BigSF/data/minidatav1/fold_0_test.h5",
            "learning_mode": "onevsall",
            "toEncode": ["positions"],
        },
        "batch_size": 16,
        "missing": False,
        "output_path": "segmentation_output.fits",
    }

    segmenter = Segmenter.from_config(config)
    segmenter.segment()
