import abc
from pathlib import Path
from typing import Union, Tuple

import astropy.io.fits as fits
import h5py
import numpy as np
from configurable import TypedConfigurable, Schema
from tqdm import tqdm

import src.utils.normalizer as norma


#
# class BaseMosaicBuilding(abc.ABC, TypedConfigurable):
#
#     @abc.abstractmethod
#     def mosaic_building(self):self):
#         pass
#
#
# class FilamentMosaicBuilding(BaseMosaicBuilding):
#
#     config_schema = {
#         "files_dir": Schema(str),
#         "output_dir": Schema(str),
#         "one_file": Schema(bool, default=False, optional=True),
#         "hdu_number": Schema(int, default=0),
#         "avoid_missing": Schema(str, default=False, optional=True),
#         "missing_value": Schema(float, default=0),
#         "binarize": Schema(bool, default=False, optional=True),
#         "conservative": Schema(bool, default=False, optional=True),
#     }
#
#     def build_mosaic(self, files_dir):
#         """
#         For each file get the data from the other using merging
#
#         Parameters
#         ----------
#         files_dir: str
#             The directory with the files for the composition
#
#         Returns
#         -------
#         The blended results for each input file with the corresponding filenames
#         """
#         files = utils.get_sorted_file_list(files_dir)
#         hdus = [fits.open(Path(files_dir) / f) for f in files]
#
#         new_hdus = []
#         for hdu in hdus:
#             header = hdu[0].header.copy()
#
#             a, f = reproject.mosaicking.reproject_and_coadd(
#                 hdus, header, reproject_function=reproject.reproject_interp
#             )
#             idx = a > 0
#             a[idx] = 1
#
#             new_hdu = fits.ImageHDU(data=a, header=header)
#             new_hdus.append(new_hdu)
#
#         return new_hdus, files
#
#     def build_unified_mosaic(self,
#         files_dir,
#         naxis1,
#         naxis2,
#         hdu_number=0,
#         avoid_missing=False,
#         missing_value=1.0,
#         binarize=False,
#         conservative=False,
#     ):
#         """
#         Build a mosaic using all the files inside a directory
#
#         Parameters
#         ----------
#         files_dir: str
#             The directory with the files for the composition
#         naxis1: int
#             The width of the new image
#         naxis2: int
#             The height of the new image
#         hdu_number: int, optional
#             The number of the HDU inside the fits
#         avoid_missing: bool, optional
#             True if we want to avoid missing values (below 1) problems
#         missing_value: float, optional
#             The threshold for detecting missing values
#         binarize: bool, optional
#             True for a binarized result
#         conservative: bool, optional
#             If True, apply a conservative binarization
#
#         Returns
#         -------
#         The a full mosaic file
#         """
#         files = utils.get_sorted_file_list(files_dir)
#         hdus = [(fits.open(Path(files_dir) / f))[hdu_number] for f in files]
#         new_header = hdus[0].header.copy()
#         new_header["NAXIS1"] = naxis1
#         new_header["NAXIS2"] = naxis2
#         new_header["CRPIX1"] = naxis1 // 2
#         new_header["CRPIX2"] = naxis2 // 2
#         new_header["CRVAL1"] = 180.0
#         new_header["CRVAL2"] = 0.0
#
#         # Convert all missing values into NaN (avoid stupid means)
#         if avoid_missing:
#             for hdu in hdus:
#                 idx = hdu.data < missing_value
#                 hdu.data[idx] = np.nan
#
#         a, f = reproject.mosaicking.reproject_and_coadd(
#             hdus, new_header, reproject_function=reproject.reproject_interp
#         )
#
#         # Put everything to 0 if not filament
#         if binarize:
#             a[np.isnan(a)] = 0.0
#             if conservative:
#                 a[a < 0.6] = 0.0
#                 a[a > 0] = 1.0
#             else:
#                 a[a > 0.2] = 1.0
#                 a[a < 0.5] = 0.0
#
#         return a, new_header
#
#     def mosaic_building(self):
#
#         if not self.one_file:
#             reshdus, files_list = self.build_mosaic(self.files_dir)
#
#             for fhdu, file in zip(reshdus, files_list):
#                 fhdu.writeto(Path(self.output_dir) / file)
#         else:
#             data, header = self.build_unified_mosaic(
#                 self.files_dir,
#                 114000,
#                 1800,
#                 self.hdu_number,
#                 self.avoid_missing,
#                 self.missing_value,
#                 self.binarize,
#                 self.conservative,
#             )
#             fits.writeto(
#                 Path(self.output_dir) / "merge_result.fits",
#                 data=data,
#                 header=header,
#                 overwrite=True,
#             )


class BasePatchExtraction(abc.ABC, TypedConfigurable):
    """
    Abstract base class for patch extraction.

    Defines the interface for classes that perform patch extraction from images or datasets.

    Methods:
        extract_patches():
            Abstract method to be implemented in subclasses for extracting patches.
    """

    @abc.abstractmethod
    def extract_patches(self):
        pass


class PatchExtraction(BasePatchExtraction):
    """
    A class for extracting patches from images or datasets and saving them into an HDF5 file.

    Attributes:
        config_schema (dict): Defines the configuration schema with the following keys:
            - "image" (str): Path to the input image file.
            - "target" (str): Path to the target image file.
            - "missing" (str): Path to the missing data mask file.
            - "background" (str): Path to the background image file.
            - "output" (str, default="dataset"): Name of the output HDF5 file.
            - "patch_size" (int, default=64): Size of the patches to extract.

    Methods:
        __init__():
            Initializes the class by loading image, target, missing, and background data.

        extract_patches():
            Extracts patches from the input image and saves them into an HDF5 file.

        _create_hdf(output, patch_size):
            Creates an HDF5 file with datasets for patches, positions, targets, and labels.

        _hdf_incrementation(hdf_data, y, x, patch_size, hdf_current_size, image, missing, background, target):
            Updates HDF5 data with new patches and corresponding metadata.

        _flush_into_hdf5(hdf, hdf_data, current_index, current_size, patch_size):
            Writes buffered data to the HDF5 file and flushes it to disk.
    """

    config_schema = {
        "image": Schema(Union[str, Path]),
        "target": Schema(Union[str, Path]),
        "missing": Schema(Union[str, Path]),
        "background": Schema(Union[str, Path]),
        "output": Schema(Union[str, Path], default="dataset"),
        "patch_size": Schema(int, default=64),
    }

    def __init__(self):

        self.patch_size = tuple((self.patch_size, self.patch_size))

        self.image = fits.getdata(self.image)
        self.target = fits.getdata(self.target)
        self.missing = fits.getdata(self.missing)
        self.background = fits.getdata(self.background)

    def preconditions(self):
        assert str(self.image).endswith(".fits")
        assert str(self.target).endswith(".fits")
        assert str(self.missing).endswith(".fits")
        assert str(self.background).endswith(".fits")

    def extract_patches(self) -> h5py.File:
        """
        Extracts patches from the input image and saves them into an HDF5 file.

        This method iterates through the input image, extracts patches of a defined size, and stores them in an HDF5 file.
        If the HDF5 file already exists, the extraction is skipped. The method utilizes an incremental approach to
        optimize memory usage by buffering patches before writing them to disk.

        Returns:
            h5py.File: A reference to the HDF5 file containing the extracted patches.

        Raises:
            FileNotFoundError: If the input image is not loaded properly.
        """

        path_h5 = Path(self.output) / "patches.h5"
        if path_h5.exists():
            self.logger.info(
                f"HDF5 file {path_h5} already exists. Skipping patches extraction. If you want to run again, delete patches.h5 or change the output folder"
            )
        else:
            hdf_cache = 1000  # The number of patches before a flush

            hdf_data = [
                np.zeros(
                    (hdf_cache, self.patch_size[0], self.patch_size[1])
                ),  # Patches
                np.zeros((hdf_cache, 2, 2)),  # Positions
                np.zeros((hdf_cache, self.patch_size[0], self.patch_size[1])),  # Target
                np.zeros(
                    (hdf_cache, self.patch_size[0], self.patch_size[1])
                ),  # Labelled
            ]
            if type(self.output) == str:
                self.output = Path(self.output)

            hdf_files = self._create_hdf(f"{self.output / 'patches'}", self.patch_size)
            current_size = 0
            current_index = 0
            total_patches = (self.image.shape[0] - self.patch_size[0] + 1) * (
                self.image.shape[1] - self.patch_size[1] + 1
            )

            with tqdm(total=total_patches, desc="Processing patches") as pbar:
                for y in range(0, self.image.shape[0] - self.patch_size[0] + 1):
                    for x in range(0, self.image.shape[1] - self.patch_size[1] + 1):
                        pbar.update(1)
                        current_size, hdf_data = self._hdf_incrementation(
                            hdf_data,
                            y,
                            x,
                            self.patch_size,
                            current_size,
                            self.image,
                            self.missing,
                            self.background,
                            self.target,
                        )
                        if current_size == hdf_cache:
                            self._flush_into_hdf5(
                                hdf_files,
                                hdf_data,
                                current_index,
                                hdf_cache,
                                self.patch_size,
                            )
                            current_index += hdf_cache
                            current_size = 0

            # Final flush for remaining data

            if current_size > 0:
                self._flush_into_hdf5(
                    hdf_files,
                    hdf_data,
                    current_index,
                    current_size,
                    self.patch_size,
                )
            hdf_files.close()

    def _create_hdf(self, output: str, patch_size: Tuple[int, int]) -> h5py.File:
        """
        Creates an HDF5 file to store extracted patches and related metadata.

        This method initializes an HDF5 file with datasets for patches, positions, spines, and labeled data.
        Each dataset is created with gzip compression and configured to allow dynamic resizing.

        Args:
            output (str): The base name of the output HDF5 file (without the extension).
            patch_size (Tuple[int, int]): The dimensions (height, width) of the patches to be stored.

        Returns:
            h5py.File: A reference to the created HDF5 file, ready for data insertion.
        """

        hdf = h5py.File(output + ".h5", "w")

        hdf.create_dataset(
            "patches",
            (1, patch_size[0], patch_size[1], 1),
            maxshape=(None, patch_size[0], patch_size[1], 1),
            compression="gzip",
            compression_opts=7,
        )
        hdf.create_dataset(
            "positions",
            (1, 2, 2, 1),
            maxshape=(None, 2, 2, 1),
            compression="gzip",
            compression_opts=7,
        )
        hdf.create_dataset(
            "spines",
            (1, patch_size[0], patch_size[1], 1),
            maxshape=(None, patch_size[0], patch_size[1], 1),
            compression="gzip",
            compression_opts=7,
        )
        hdf.create_dataset(
            "labelled",
            (1, patch_size[0], patch_size[1], 1),
            maxshape=(None, patch_size[0], patch_size[1], 1),
            compression="gzip",
            compression_opts=7,
        )
        return hdf

    def _hdf_incrementation(
        self,
        hdf_data: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        y: int,
        x: int,
        patch_size: Tuple[int, int],
        hdf_current_size: int,
        image: np.ndarray,
        missing: np.ndarray,
        background: np.ndarray,
        target: np.ndarray,
    ) -> Tuple[int, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Extracts a patch and updates the HDF5 data buffer with new patches and metadata.

        This method extracts a patch from the input image and computes its corresponding
        missing, background, and target masks. The patch is normalized if it contains
        valid pixel values. The updated patch and metadata are stored in the buffer
        before being written to the HDF5 file.

        Args:
            hdf_data (Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]): The buffer
                storing patches, positions, targets, and labeled masks before writing to HDF5.
            y (int): The y-coordinate of the patch's top-left corner.
            x (int): The x-coordinate of the patch's top-left corner.
            patch_size (Tuple[int, int]): The dimensions (height, width) of the patch.
            hdf_current_size (int): The current index in the buffer before adding the new patch.
            image (np.ndarray): The input image from which patches are extracted.
            missing (np.ndarray): The missing data mask.
            background (np.ndarray): The background image.
            target (np.ndarray): The target image.

        Returns:
            Tuple[int, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
            The updated buffer index and the modified HDF5 data buffer.
        """
        p = image[y : y + patch_size[0], x : x + patch_size[1]].copy()
        idx = np.isnan(p)
        p[idx] = 0
        b = background[y : y + patch_size[0], x : x + patch_size[1]]
        t = target[y : y + patch_size[0], x : x + patch_size[1]]
        m = missing[y : y + patch_size[0], x : x + patch_size[1]]
        l = (b + t) * m
        l[l > 0] = 1
        midx = p > 0
        if midx.any() and np.sum(l) > 0:
            p[midx] = norma.normalize_direct(p[midx])
            position = [[y, y + patch_size[0]], [x, x + patch_size[1]]]
            hdf_data[2][hdf_current_size, :, :] = t
            hdf_data[0][hdf_current_size, :, :] = p
            hdf_data[3][hdf_current_size, :, :] = l
            hdf_data[1][hdf_current_size, :, :] = position
            hdf_current_size += 1
        return hdf_current_size, hdf_data

    def _flush_into_hdf5(
        self,
        hdf: h5py.File,
        hdf_data: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        current_index: int,
        current_size: int,
        patch_size: Tuple[int, int],
    ) -> None:
        """
        Writes buffered patch data into the HDF5 file and flushes it to disk.

        This method resizes the existing datasets in the HDF5 file and appends new patches,
        positions, spines, and labeled data. It ensures efficient storage management by
        incrementally updating the file instead of rewriting it entirely.

        Args:
            hdf (h5py.File): The HDF5 file object where patches and metadata are stored.
            hdf_data (Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]): A tuple containing
                buffered patches, positions, spines, and labeled data.
            current_index (int): The index in the dataset where new patches should be inserted.
            current_size (int): The number of patches to write in this flush operation.
            patch_size (Tuple[int, int]): The dimensions (height, width) of each patch.

        Returns:
            None
        """
        hdf["patches"].resize(
            (current_index + current_size, patch_size[0], patch_size[1], 1)
        )
        hdf["patches"][current_index : current_index + current_size, :, :, :] = (
            hdf_data[0][:current_size, :, :, np.newaxis]
        )
        hdf["positions"].resize((current_index + current_size, 2, 2, 1))
        hdf["positions"][current_index : current_index + current_size, :, :, :] = (
            hdf_data[1][:current_size, :, :, np.newaxis]
        )
        hdf["spines"].resize(
            (current_index + current_size, patch_size[0], patch_size[1], 1)
        )
        hdf["spines"][current_index : current_index + current_size, :, :, :] = hdf_data[
            2
        ][:current_size, :, :, np.newaxis]
        hdf["labelled"].resize(
            (current_index + current_size, patch_size[0], patch_size[1], 1)
        )
        hdf["labelled"][current_index : current_index + current_size, :, :, :] = (
            hdf_data[3][:current_size, :, :, np.newaxis]
        )
        hdf.flush()
