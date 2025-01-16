import torch
import numpy as np
import skimage.morphology as skm
import astropy.io.fits as fits
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import logging

from PNRIA.configs.config import Customizable, Schema, Config
from PNRIA.torch_c.dataset import BaseDataset
from PNRIA.torch_c.models.custom_model import BaseModel


class Segmenter(Customizable):
    """
    Segmenter class for performing segmentation tasks with customizable models and datasets.
    This class utilizes the configurations from TypedCustomizable for easy customization.
    """

    config_schema = {
        'model_snapshot': Schema(type=str, aliases=["model"]),
        'source': Schema(type=str, aliases=["source"]),
        'dataset': Schema(type=Config, aliases=["dataset"]),
        'normalization_mode': Schema(str, optional=True, default="none"),
        'batch_size': Schema(int, optional=True, default=100),
        'missing': Schema(bool, optional=True, default=False),
        'no_segmenter': Schema(bool, optional=True, default=False),
        'output_path': Schema(str, optional=True, default="output.fits"),
    }

    def __init__(self, *args, **kwargs):
        """
        Initializes the Segmenter class from configuration.
        Uses the TypedCustomizable mechanism to automatically apply configurations.
        """
        super().__init__(*args, **kwargs)  # Initialize with Customizable logic

        # Initialize logger
        self.logger = logging.getLogger(__name__)
        self.logger.debug("Initializing Segmenter")

        # Model and dataset instantiation using the configuration
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

    def segment(self) -> None:
        """
        Perform the segmentation on the dataset and save the output.
        """
        self.logger.debug("Starting segmentation process")
        self.model.to(self.device)
        self.model.eval()

        dataloader = DataLoader(self.dataset, num_workers=1, batch_size=self.batch_size, shuffle=False)
        self.logger.debug(f"Dataloader created with batch size: {self.batch_size}")

        segmentation_map = np.zeros_like(self.source_data)
        count_map = np.zeros_like(segmentation_map)

        with torch.no_grad():
            # Adding tqdm to track the progress
            for batch_idx, samples in enumerate(tqdm(dataloader, desc="Processing Patches", unit="batch")):
                self.logger.debug(f"Processing batch {batch_idx}")
                samples = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in samples.items()}
                patch_segmented = self._process_patch(samples)
                positions = samples.get('positions', None)

                for i in range(patch_segmented.shape[0]):
                    x_start, x_end, y_start, y_end = (
                        int(positions[i][0][0]),
                        int(positions[i][0][1]),
                        int(positions[i][1][0]),
                        int(positions[i][1][1]),
                    )
                    self.logger.debug(f"Patch {i} position: x({x_start}:{x_end}), y({y_start}:{y_end})")

                    if x_start >= x_end or y_start >= y_end:
                        self.logger.warning(f"Invalid patch position for patch {i}: x_start >= x_end or y_start >= y_end")
                        continue

                    segmentation_patch = torch.squeeze(patch_segmented[i].cpu().detach()).numpy()
                    expected_shape = (x_end - x_start, y_end - y_start)
                    if segmentation_patch.shape == expected_shape:
                        segmentation_map[x_start:x_end, y_start:y_end] += segmentation_patch

                    if self.missing:
                        missmap = samples["missmap"][i].cpu().numpy()
                        segmentation_map[x_start:x_end, y_start:y_end] *= torch.squeeze(missmap).numpy()
                    count_map[x_start:x_end, y_start:y_end] += 1

            self._post_process(segmentation_map, count_map)

    def _process_patch(self, batch):
        """
        Pass the patch through the model, handling various model types.
        """
        self.logger.debug("Processing a patch through the model")
        if self.no_segmenter:
            self.logger.debug("No segmenter mode, returning batch as is")
            return batch
        return self.model(**batch)

    def _post_process(self, segmentation_map: np.ndarray, count_map: np.ndarray) -> None:
        """
        Post-process the segmentation output, including normalizing and saving the output.
        """
        self.logger.info("Post-processing the segmentation map...")
        idx = count_map > 0
        self.logger.debug(f"Number of pixels to normalize: {np.sum(idx)}")
        segmentation_map[idx] /= count_map[idx]

        if not self.no_segmenter:
            self.logger.debug("Applying threshold and morphological operations")
            segmentation_map[segmentation_map > 0.5] = 1
            segmentation_map[segmentation_map <= 0.5] = 0
            erode = skm.erosion(segmentation_map, skm.square(4))
            segmentation_map = skm.reconstruction(erode, segmentation_map)
        self.logger.info("Done post-processing the segmentation map")
        self._save_output(segmentation_map)

    def _save_output(self, segmentation_map: np.ndarray) -> None:
        """
        Save the segmentation map to a FITS file.
        """
        self.logger.debug(f"Saving segmentation map to '{self.output_path}'")
        try:
            fits.writeto(self.output_path, data=segmentation_map, header=self.source_header, overwrite=True)
            self.logger.info(f"Segmentation map saved to '{self.output_path}'")
        except Exception as e:
            self.logger.error(f"Error saving output file: {e}", exc_info=True)
            print(f"Error saving output file: {e}")


# Example usage:
if __name__ == "__main__":

    config = {
        'model_snapshot': "./run/best.pt",
        'source': '/mnt/data/WORK/BigSF/data/spine_merged.fits',
        'dataset': {
            'type': 'FilamentsDataset',
            'dataset_path': '/mnt/data/WORK/BigSF/data/minidatav1/fold_0_test.h5',
            'learning_mode': 'onevsall',
            'normalization_mode': 'log10',
            'toEncode': ['positions']
        },
        'batch_size': 16,
        'missing': False,
        'output_path': 'segmentation_output.fits',
    }

    segmenter = Segmenter.from_config(config)
    segmenter.segment()
