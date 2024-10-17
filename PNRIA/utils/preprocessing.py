import abc
import h5py 

import astropy.io.fits as fits
import numpy as np
import reproject.mosaicking

from PNRIA.configs.config import TypedCustomizable, Schema
import deep_filaments.io.utils as utils
import deep_filaments.utils.normalizer as norma

class BasePreprocessing(abc.ABC, TypedCustomizable):
    pass

class BasePatchExtraction(BasePreprocessing):

    @abc.abstractmethod
    def create_folds():
        """ 
        Create k folds, in train/test or train/valid/test
        using random shuffling or naïve pattern
        Stride should be modifiable, also depend on the fold ? (i.e. different stride for train/test)
        """
        pass
    
    @abc.abstractmethod    
    def create_masks():
        """
        Create masks according to the choice of folds ?
        """
        pass
    
    @abc.abstractmethod
    def extract_patches():
        """
        Should extract the patches according to the parameters chosen for the k-folds
        """
        pass
    
class BaseMosaicBuilding(BasePreprocessing):
    
    @abc.abstractmethod
    def mosaic_building():
        pass

class FilamentPatchExtraction(BasePatchExtraction):
    
    config_schema = {
        'image': Schema(str),
        'target': Schema(str),
        'missing': Schema(str),
        'background': Schema(str),
        "output": Schema(str, default='dataset'),
        'test_area_size': Schema(int, default=0),
        'patch_size': Schema(int, default=64),
        'stride': Schema(int),
        'k': Schema(int, default=10),
        'use_validation': Schema(bool, default=True)
    }   
    
    def __init__(self):

        self.patch_size = tuple((self.patch_size, self.patch_size))

        if self.image.endswith(".fits"):
            self.image = fits.getdata(self.image)

        if self.target.endswith(".fits"):
            self.target = fits.getdata(self.target)

        if self.missing.endswith(".fits"):
            self.missing = fits.getdata(self.missing)

        if self.background.endswith(".fits"):
            self.background = fits.getdata(self.background)
            
        self.image = self.image[:, :1140]
        self.target = self.target[:, :1140]
        self.missing = self.missing [:, :1140]
        self.background = self.background[:, :1140]

        
    def create_folds(self):
        fold_mask = self.create_masks()
        self.extract_patches(
            fold_mask=fold_mask,
            use_validation=self.use_validation
        )


    def create_masks(self):
        rng = np.random.default_rng()
        area_size = self.test_area_size
        fold_mask = np.full((self.k, self.image.shape[0], self.image.shape[1]), False)
        x, y = 0, 0
        while x + area_size <= self.image.shape[1]:
            fold = rng.integers(0, self.k)
            y = 0
            while y + area_size <= self.image.shape[0]:
                fold_mask[fold, y : y + area_size, x : x + area_size] = True
                fold = (fold + 1) % self.k
                y += area_size
            x += area_size
        return fold_mask


    def extract_patches(
        self,
        fold_mask,
        use_validation,
    ):
        """
        Extract patches from a source, with a target and a missing data map
        All the data are assumed to be of the same size

        Parameters
        ----------
        source: np.ndarray
            The source data (aka n(h2) map)
        target: np.ndarray
            The target (aka RoI or spines) data
        missing: np.ndarray
            The missing data map (with 1 where data are missing, 0 otherwise)
        background: np.ndarray
            The background data (with 1 for background, 0 otherwise)
        output: str
            The name of the output file (HDF5 file)
        patch_size: tuple[int, int]
            The size of one patch
        overlap: int, optional
            The overlap between 2 patches (0 by default)
        normalizer: callable, optional
            The normalizing function (avoid missing data)
        conservative: bool, optional
            Update the missing data with unlabelled pixels
        use_validation: bool, optional
            Whether to include the validation set (default: True)

        Returns
        -------
        A HDF5 reference to the set of patches
        """
        # Configuration variables
        hdf_cache = 1000  # The number of patches before a flush
        
        # Precompute patch area for efficient comparison
        patch_area = self.patch_size[0] * self.patch_size[1]
        
        # Initialize dictionaries
        hdf_data = {'train': {}, 'test': {}}
        hdf_files = {'train': {}, 'test': {}}
        current_size = {'train': {}, 'test': {}}
        current_index = {'train': {}, 'test': {}}
        masks = {'train': {}, 'test': {}}
        modes = ['train', 'test']

        if use_validation:
            hdf_data['validation'] = {}
            hdf_files['validation'] = {}
            current_size['validation'] = {}
            current_index['validation'] = {}
            masks['validation'] = {}
            modes = ['train', 'validation', 'test']
        

        # Setup HDF5 datasets and masks for each fold
        for fold in range(self.k):
            for mode in modes:
                hdf_data[mode][fold] = [
                    np.zeros((hdf_cache, self.patch_size[0], self.patch_size[1])),  # Patches
                    np.zeros((hdf_cache, 2, 2)),  # Positions
                    np.zeros((hdf_cache, self.patch_size[0], self.patch_size[1])),  # Target
                    np.zeros((hdf_cache, self.patch_size[0], self.patch_size[1])),  # Labelled
                ]
                hdf_files[mode][fold] = self.create_hdf(f"{self.output}fold_{fold}_{mode}", self.patch_size)
                current_size[mode][fold] = 0
                current_index[mode][fold] = 0
            
            # Create masks
            test_mask = fold_mask[fold] + fold_mask[(fold + 1) % self.k]
            train_mask = np.full(self.image.shape, True) & ~test_mask

            if use_validation:
                validation_mask = fold_mask[(fold + 2) % self.k] + fold_mask[(fold + 3) % self.k]
                train_mask &= ~validation_mask
                masks['validation'][fold] = validation_mask

            masks['test'][fold] = test_mask
            masks['train'][fold] = train_mask
            
            
        # Get patches with corresponding masks and missing data map
        for y in range(0, self.image.shape[0] - self.patch_size[0] + 1, self.stride):
            for x in range(0, self.image.shape[1] - self.patch_size[1] + 1, self.stride):
                for fold in range(self.k):
                    for mode in modes:
                        mask_sum = masks[mode][fold][y:y + self.patch_size[0], x:x + self.patch_size[1]].sum()
                        if mask_sum == patch_area:
                            current_size[mode][fold], hdf_data[mode][fold] = self.hdf_incrementation(
                                hdf_data[mode][fold],
                                y, x, self.patch_size,
                                current_size[mode][fold],
                                self.image, self.missing, self.background, self.target
                            )
                            # Flush when needed
                            if current_size[mode][fold] == hdf_cache:
                                self.flush_into_hdf5(
                                    hdf_files[mode][fold],
                                    hdf_data[mode][fold],
                                    current_index[mode][fold],
                                    hdf_cache,
                                    self.patch_size,
                                )
                                current_index[mode][fold] += hdf_cache
                                current_size[mode][fold] = 0

        # Final flush for remaining data
        for fold in range(self.k):
            for mode in modes:
                if current_size[mode][fold] > 0:
                    self.flush_into_hdf5(
                        hdf_files[mode][fold],
                        hdf_data[mode][fold],
                        current_index[mode][fold],
                        current_size[mode][fold],
                        self.patch_size,
                    )
                hdf_files[mode][fold].close()


    def hdf_incrementation(self, hdf, y, x, patch_size, hdf_current_size, image, missing, background, target):
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
            hdf[2][hdf_current_size, :, :] = t
            hdf[0][hdf_current_size, :, :] = p
            hdf[3][hdf_current_size, :, :] = l
            hdf[1][hdf_current_size, :, :] = position
            hdf_current_size += 1
        return hdf_current_size, hdf

    def create_hdf(self, output, patch_size):
        # Creation of the HDF5 file
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

    
    def flush_into_hdf5(self, hdf, data, current_index, size, patch_size):
        """
        Flush the current data into the hdf5 file

        Parameters
        ----------
        hdf: h5py.File
            The hdf5 file object
        data: tuple
            The different data (patches, maskss/targets, missmap/missing, backgrounds)
        current_index: int
            The current index inside the dataset
        size: int
            The size of the data (current number of patches)
        patch_size:
            The size of the patches
        """
        hdf["patches"].resize((current_index + size, patch_size[0], patch_size[1], 1))
        hdf["patches"][current_index : current_index + size, :, :, :] = data[0][
            :size, :, :, np.newaxis
        ]
        hdf["positions"].resize((current_index + size, 2, 2, 1))
        hdf["positions"][current_index : current_index + size, :, :, :] = data[1][
            :size, :, :, np.newaxis
        ]
        hdf["spines"].resize((current_index + size, patch_size[0], patch_size[1], 1))
        hdf["spines"][current_index : current_index + size, :, :, :] = data[2][
            :size, :, :, np.newaxis
        ]
        hdf["labelled"].resize((current_index + size, patch_size[0], patch_size[1], 1))
        hdf["labelled"][current_index : current_index + size, :, :, :] = data[3][
            :size, :, :, np.newaxis
        ]
        hdf.flush()

        
class FilamentMosaicBuilding(BaseMosaicBuilding):
    
    config_schema = {
            'files_dir': Schema(str),
            'output_dir': Schema(str),
            'one_file': Schema(bool, default=False, optional=True),
            'hdu_number': Schema(int, default=0),
            'avoid_missing': Schema(str, default=False, optional=True),
            "missing_value": Schema(float, default=0),
            "binarize": Schema(bool, default=False, optional=True),
            'conservative': Schema(bool, default=False, optional=True),
        }   
    
    def build_mosaic(files_dir):
            """
            For each file get the data from the other using merging

            Parameters
            ----------
            files_dir: str
                The directory with the files for the composition

            Returns
            -------
            The blended results for each input file with the corresponding filenames
            """
            files = utils.get_sorted_file_list(files_dir)
            hdus = [fits.open(files_dir + "/" + f) for f in files]

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
        files_dir,
        naxis1,
        naxis2,
        hdu_number=0,
        avoid_missing=False,
        missing_value=1.0,
        binarize=False,
        conservative=False,
    ):
        """
        Build a mosaic using all the files inside a directory

        Parameters
        ----------
        files_dir: str
            The directory with the files for the composition
        naxis1: int
            The width of the new image
        naxis2: int
            The height of the new image
        hdu_number: int, optional
            The number of the HDU inside the fits
        avoid_missing: bool, optional
            True if we want to avoid missing values (below 1) problems
        missing_value: float, optional
            The threshold for detecting missing values
        binarize: bool, optional
            True for a binarized result
        conservative: bool, optional
            If True, apply a conservative binarization

        Returns
        -------
        The a full mosaic file
        """
        files = utils.get_sorted_file_list(files_dir)
        hdus = [(fits.open(files_dir + "/" + f))[hdu_number] for f in files]
        new_header = hdus[0].header.copy()
        new_header["NAXIS1"] = naxis1
        new_header["NAXIS2"] = naxis2
        new_header["CRPIX1"] = naxis1 // 2
        new_header["CRPIX2"] = naxis2 // 2
        new_header["CRVAL1"] = 180.0
        new_header["CRVAL2"] = 0.0

        # Convert all missing values into NaN (avoid stupid means)
        if avoid_missing:
            for hdu in hdus:
                idx = hdu.data < missing_value
                hdu.data[idx] = np.nan

        a, f = reproject.mosaicking.reproject_and_coadd(
            hdus, new_header, reproject_function=reproject.reproject_interp
        )

        # Put everything to 0 if not filament
        if binarize:
            a[np.isnan(a)] = 0.0
            if conservative:
                a[a < 0.6] = 0.0
                a[a > 0] = 1.0
            else:
                a[a > 0.2] = 1.0
                a[a < 0.5] = 0.0

        return a, new_header
    
    def mosaic_building(self):
        
        if not self.one_file:
            reshdus, files_list = self.build_mosaic(self.files_dir)

            for fhdu, file in zip(reshdus, files_list):
                fhdu.writeto(self.output_dir + "/" + file)
        else:
            data, header = self.build_unified_mosaic(
                self.files_dir,
                114000,
                1800,
                self.hdu_number,
                self.avoid_missing,
                self.missing_value,
                self.binarize,
                self.conservative,
            )
            fits.writeto(self.output_dir + "/merge_result.fits", data=data, header=header, overwrite=True)
