import abc
from pathlib import Path
from typing import Union, Tuple, List

import astropy.io.fits as fits
import h5py
import numpy as np
import reproject
import reproject.mosaicking
from configurable import TypedConfigurable, Schema, Configurable
from tqdm import tqdm

from src import utils as norma
from .utils import get_sorted_file_list


class FilamentMosaicBuilding(Configurable):
    """
    FilamentMosaicBuilding for building mosaics from FITS files.

    This class provides methods to build mosaics from FITS files located in a specified directory. It supports both individual file processing and unified mosaic creation.

    Configuration:
        - **files_dir** (str): The directory containing the FITS files.
        - **output_dir** (str): The directory to save the output files.
        - **fits_file_names** (List): A list of FITS file names to process.
        - **one_file** (bool): Whether to create a single unified mosaic. Default is False.
        - **hdu_number** (int): The HDU number to use from the FITS files. Default is 0.
        - **avoid_missing** (bool): Whether to avoid missing values. Default is False.
        - **missing_value** (float): The threshold for detecting missing values. Default is 1.0.
        - **binarize** (bool): Whether to binarize the result. Default is False.
        - **conservative** (bool): Whether to apply conservative binarization. Default is False.

    Example Configuration (YAML):
        .. code-block:: yaml

            files_dir: "/path/to/files"
            output_dir: "/path/to/output"
            fits_file_names: ["file1", "file2"]
            one_file: False
            hdu_number: 0
            avoid_missing: False
            missing_value: 1.0
            binarize: False
            conservative: False
    """

    config_schema = {
        "files_dir": Schema(str),
        "output_dir": Schema(str),
        "fits_file_names": Schema(List),
        "map_resolution": Schema(Tuple[int, int]),
        "one_file": Schema(bool, default=False, optional=True),
        "hdu_number": Schema(int, default=0),
        "avoid_missing": Schema(str, default=False, optional=True),
        "missing_value": Schema(float, default=1.0),
        "binarize": Schema(bool, default=False, optional=True),
        "conservative": Schema(bool, default=False, optional=True),
    }

    def build_mosaic(self, current_folder):
        """Build a mosaic for each file in the specified directory.

        For each file, get the data from the other files using merging.

        Parameters
        ----------
        current_folder : str
            The directory containing the files for the composition.

        Returns
        -------
        tuple
            A tuple containing the blended results for each input file and the corresponding filenames.
        """
        files = get_sorted_file_list(current_folder)
        hdus = [fits.open(Path(current_folder) / f) for f in files]
        new_hdus = []
        for hdu in hdus:
            header = hdu[0].header.copy()

            a, f = reproject.mosaicking.reproject_and_coadd(
                hdus, header, reproject_function=reproject.reproject_interp
            )
            idx = a > 0
            a[idx] = 1

            new_hdu = fits.ImageHDU(data=a, header=header)
            new_hdus.append(new_hdu)

        return new_hdus, files

    def build_unified_mosaic(
        self,
        current_folder,
        naxis1,
        naxis2,
    ):
        """Build a unified mosaic using all the files inside a directory.

        Parameters
        ----------
        current_folder : str
            The directory containing the files for the composition.
        naxis1 : int
            The width of the new image.
        naxis2 : int
            The height of the new image.

        Returns
        -------
        tuple
            A tuple containing the full mosaic data and the new header.
        """
        files = get_sorted_file_list(Path(current_folder))
        hdus = [(fits.open(Path(current_folder) / f))[self.hdu_number] for f in files]

        new_header = hdus[0].header.copy()
        new_header["NAXIS1"] = naxis1
        new_header["NAXIS2"] = naxis2
        new_header["CRPIX1"] = naxis1 // 2
        new_header["CRPIX2"] = naxis2 // 2
        new_header["CRVAL1"] = 180.0
        new_header["CRVAL2"] = 0.0

        # Convert all missing values into NaN (avoid stupid means)
        if self.avoid_missing:
            for hdu in hdus:
                idx = hdu.data < self.missing_value
                hdu.data[idx] = np.nan

        a, f = reproject.mosaicking.reproject_and_coadd(
            hdus, new_header, reproject_function=reproject.reproject_interp
        )

        # Put everything to 0 if not filament
        if self.binarize:
            a[np.isnan(a)] = 0.0
            if self.conservative:
                a[a < 0.6] = 0.0
                a[a > 0] = 1.0
            else:
                a[a > 0.2] = 1.0
                a[a < 0.5] = 0.0

        return a, new_header

    def mosaic_building(self):
        """Build mosaics for all specified FITS files.

        This method processes each FITS file in the specified directory and builds mosaics based on the configuration.
        Chose whether to build only one file (build_unified_mosaic) or multiple files (build_mosaic).
        """
        for file_name in self.fits_file_names:
            current_folder = Path(self.files_dir) / file_name

            if not self.one_file:
                reshdus, files_list = self.build_mosaic(current_folder)

                for fhdu, file in zip(reshdus, files_list):
                    output_path = Path(self.output_dir) / f"{file_name}_{file}"
                    print(f"output to {output_path}")
                    fhdu.writeto(output_path, overwrite=True)
            else:
                data, header = self.build_unified_mosaic(
                    file_name,
                    current_folder,
                    self.map_resolution[0],
                    self.map_resolution[1],
                )
                output_path = Path(self.output_dir) / f"{file_name}_merged.fits"

                print(f"Writing unified mosaic to {output_path}")
                fits.writeto(output_path, data=data, header=header, overwrite=True)


class BasePatchExtraction(abc.ABC, TypedConfigurable):
    """
    Abstract base class for patch extraction.

    Defines the interface for classes that perform patch extraction.
    """

    @abc.abstractmethod
    def extract_patches(self):
        pass


class PatchExtraction(BasePatchExtraction):
    """
    PatchExtraction for extracting patches from images and storing them in an HDF5 file.

    This class extracts patches from input images and stores them efficiently in an HDF5 file,
    ensuring optimized memory usage through incremental storage.

    Configuration:
    **image** (str | Path): Path to the input image file.
    **target** (str | Path): Path to the target image file.
    **missing** (str | Path): Path to the missing data mask file.
    **background** (str | Path): Path to the background image file.
    **output** (str | Path): Name of the output HDF5 file. Default is "dataset".
    **patch_size** (int): Size of the patches to extract. Default is 64.

    Example Configuration (Python Dict):
        .. code-block:: python

            {
                "image": "path/to/image.fits",
                "target": "path/to/target.fits",
                "missing": "path/to/missing.fits",
                "background": "path/to/background.fits",
                "output": "dataset",
                "patch_size": 64
            }
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
        """
        Validates that all required input files are FITS files.

        Raises:
            AssertionError: If any input file does not have a .fits extension.
        """
        assert str(self.image).endswith(".fits")
        assert str(self.target).endswith(".fits")
        assert str(self.missing).endswith(".fits")
        assert str(self.background).endswith(".fits")

    def extract_patches(self) -> h5py.File:
        """
        Extracts patches from the input image and saves them into an HDF5 file.

        This method iterates through the input image, extracts patches of a defined size,
        and stores them in an HDF5 file. If the file already exists, the extraction is skipped.

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

            hdf_files = self._create_hdf()
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
                            current_size,
                        )
                        if current_size == hdf_cache:
                            self._flush_into_hdf5(
                                hdf_files,
                                hdf_data,
                                current_index,
                                hdf_cache,
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
                )
            hdf_files.close()

    def _create_hdf(self) -> h5py.File:
        """
        Creates an HDF5 file to store extracted patches and related metadata.

        Args:
            output (str): The base name of the output HDF5 file (without the extension).
            patch_size (Tuple[int, int]): The dimensions (height, width) of the patches to be stored.

        Returns:
            h5py.File: A reference to the created HDF5 file.
        """
        hdf = h5py.File(self.output / "patches.h5", "w")

        hdf.create_dataset(
            "patches",
            (1, self.patch_size[0], self.patch_size[1], 1),
            maxshape=(None, self.patch_size[0], self.patch_size[1], 1),
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
            (1, self.patch_size[0], self.patch_size[1], 1),
            maxshape=(None, self.patch_size[0], self.patch_size[1], 1),
            compression="gzip",
            compression_opts=7,
        )
        hdf.create_dataset(
            "labelled",
            (1, self.patch_size[0], self.patch_size[1], 1),
            maxshape=(None, self.patch_size[0], self.patch_size[1], 1),
            compression="gzip",
            compression_opts=7,
        )
        return hdf

    def _hdf_incrementation(
        self,
        hdf_data: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        y: int,
        x: int,
        hdf_current_size: int,
    ) -> Tuple[int, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Extracts a patch and updates the HDF5 data buffer with new patches and metadata.

        Args:
            hdf_data (Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]): The buffer storing patches.
            y (int): The y-coordinate of the patch's top-left corner.
            x (int): The x-coordinate of the patch's top-left corner.
            patch_size (Tuple[int, int]): The dimensions (height, width) of the patch.
            hdf_current_size (int): The current index in the buffer before adding the new patch.
            image (np.ndarray): The input image.
            missing (np.ndarray): The missing data mask.
            background (np.ndarray): The background image.
            target (np.ndarray): The target image.

        Returns:
            Tuple[int, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
            The updated buffer index and the modified HDF5 data buffer.
        """
        p = self.image[y : y + self.patch_size[0], x : x + self.patch_size[1]].copy()
        idx = np.isnan(p)
        p[idx] = 0
        b = self.background[y : y + self.patch_size[0], x : x + self.patch_size[1]]
        t = self.target[y : y + self.patch_size[0], x : x + self.patch_size[1]]
        m = self.missing[y : y + self.patch_size[0], x : x + self.patch_size[1]]
        l = (b + t) * m
        l[l > 0] = 1
        midx = p > 0
        if midx.any() and np.sum(l) > 0:
            p[midx] = norma.normalize_direct(p[midx])
            position = [[y, y + self.patch_size[0]], [x, x + self.patch_size[1]]]
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
    ) -> None:
        """
        Writes buffered patch data into the HDF5 file and flushes it to disk.

        Args:
            hdf (h5py.File): The HDF5 file object where patches are stored.
            hdf_data (Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]): Buffered patch data.
            current_index (int): The index in the dataset where new patches should be inserted.
            current_size (int): The number of patches to write.
            patch_size (Tuple[int, int]): The dimensions (height, width) of each patch.
        """
        hdf["patches"].resize(
            (current_index + current_size, self.patch_size[0], self.patch_size[1], 1)
        )
        hdf["patches"][current_index : current_index + current_size, :, :, :] = (
            hdf_data[0][:current_size, :, :, np.newaxis]
        )
        hdf["positions"].resize((current_index + current_size, 2, 2, 1))
        hdf["positions"][current_index : current_index + current_size, :, :, :] = (
            hdf_data[1][:current_size, :, :, np.newaxis]
        )
        hdf["spines"].resize(
            (current_index + current_size, self.patch_size[0], self.patch_size[1], 1)
        )
        hdf["spines"][current_index : current_index + current_size, :, :, :] = hdf_data[
            2
        ][:current_size, :, :, np.newaxis]
        hdf["labelled"].resize(
            (current_index + current_size, self.patch_size[0], self.patch_size[1], 1)
        )
        hdf["labelled"][current_index : current_index + current_size, :, :, :] = (
            hdf_data[3][:current_size, :, :, np.newaxis]
        )
        hdf.flush()
