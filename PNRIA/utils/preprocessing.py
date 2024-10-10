import abc
import h5py 

import astropy.io.fits as fits
import numpy as np

from PNRIA.configs.config import TypedCustomizable, Schema

class BasePreprocessing(abc.ABC, TypedCustomizable):

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
    
    @abc.abstractmethod
    def mosaic_building():
        pass
    
class FilamentPreprocessing(BasePreprocessing):
    
    config_schema = {
        'image': Schema(str),
        'normed_image': Schema(str),
        'roi': Schema(str),
        'missing': Schema(str),
        'background': Schema(str),
        "output": Schema(str, default='dataset'),
        "train_overlap": Schema(int, default=0, optional=True),
        'kfold_overlap': Schema(int, default=0, optional=True),
        'test_overlap': Schema(int, default=0, optional=True),
        'test_area_size': Schema(int, default=0),
        'patch_size': Schema(int, default=64),
        'stride': Schema(int),
        'k': Schema(int, default=10),
        'k_fold_mode': Schema(str, default='naive')
    }   
    
    def __init__(self):
        assert self.k_fold_mode in {"naive", "random"}, "Learning_mode must be one of {naive, random}"
            # Manage default patch size
        self.patch_size = tuple((self.patch_size, self.patch_size))

        if self.image.endswith(".fits"):
            self.image = fits.getdata(self.image)

        if self.normed_image.endswith(".fits"):
            self.normed_image = fits.getdata(self.normed_image)

        if self.roi.endswith(".fits"):
            self.roi = fits.getdata(self.roi)

        if self.missing.endswith(".fits"):
            self.missing = fits.getdata(self.missing)

        if self.background.endswith(".fits"):
            self.background = fits.getdata(self.background)
            
        self.image = self.image[:, :1140]
        self.roi = self.roi[:, :1140]
        self.missing = self.missing [:, :1140]
        self.background = self.background[:, :1140]
    
    def create_folds(self):
    
        test_fold_mask = self.create_masks()
        
        hdfs_fold = [
            self.extract_patches(
                output=self.output + f"fold_{i}",
                mask=test_fold_mask[i]
        )
        for i in range(len(test_fold_mask))
        ]
        for hdfs in hdfs_fold:
            [hdf.close() for hdf in hdfs]
    
    def create_masks(self):
        test_area_size = self.test_area_size
        test_fold_mask = np.array([np.zeros_like(self.image) for _ in range(self.k)])
        if self.k_fold_mode == "random":
            x, y = 0, 0
            while y + test_area_size <= self.shape[1]:
                fold = np.random.randint(0, self.k)
                x = 0
                while x + test_area_size <= self.image.shape[0]:
                    test_fold_mask[fold][x : x + test_area_size, y : y + test_area_size] = 1
                    fold = (fold + 1) % self.k
                    x += test_area_size - self.kfold_overlap
                    if self.shape[0] - test_area_size < x < self.shape[0] - self.kfold_overlap:
                        x = self.image.shape[0] - test_area_size
                y += test_area_size - self.kfold_overlap
                if self.image.shape[1] - test_area_size < y < self.image.shape[1] - self.kfold_overlap:
                    y = self.image.shape[1] - test_area_size
        else:
            for i in range(len(test_fold_mask)):
                test_fold_mask[i][:, int(i * self.image.shape[1] / self.k) : int((i + 1) * self.image.shape[1] / self.k)] = 1
        return test_fold_mask

    def extract_patches(
        self,
        output,
        mask,
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

        Returns
        -------
        A HDF5 reference to the set of patches
        """
        
        # tmp = np.log10(source[np.nonzero(source)])
        # data_min = np.min(tmp)
        # data_max = np.max(tmp)
        data_min = -1000
        data_max = 1000
        if mask is not None:
            assert mask.shape == self.image.shape, "The source and the mask have to have the same size"
            train = True
            # Configuration variables
            hdf2_cache = 1000  # The number of patches before a flush
            hdf2_current_size = 0
            hdf2_current_index = 0
            # Initialize the results
            hdf2_data = [
                np.zeros((hdf2_cache, self.patch_size[0], self.patch_size[1])),   # Patches
                np.zeros((hdf2_cache, 2, 2)),    # Positions
                np.zeros((hdf2_cache, self.patch_size[0], self.patch_size[1])),  # Target
                np.zeros((hdf2_cache, self.patch_size[0], self.patch_size[1])),  # Normed
                np.zeros((hdf2_cache, self.patch_size[0], self.patch_size[1])),  # Missing
                np.zeros((hdf2_cache, self.patch_size[0], self.patch_size[1])),  # Background
                ]
            hdf2_hdf = self.create_hdf(output + "_test", self.patch_size, data_min, data_max, True, True, True, True)

        if train:
            assert self.roi is not None, "For creating a training dataset, a target map is required"
            assert self.missing is not None, "For creating a training dataset, a missing map is required"
            assert self.background is not None, "For creating a training dataset, a background map is required"

        # Configuration variables
        hdf1_cache = 1000  # The number of patches before a flush
        hdf1_current_size = 0
        hdf1_current_index = 0

        # Initialize the results
        hdf1_data = [
            np.zeros((hdf1_cache, self.patch_size[0], self.patch_size[1])),   # Patches
            np.zeros((hdf1_cache, 2, 2))    # Positions
            ]

        if train:
            hdf1_data.append(np.zeros((hdf1_cache, self.patch_size[0], self.patch_size[1])))  # Target
            hdf1_data.append(np.zeros((hdf1_cache, self.patch_size[0], self.patch_size[1])))  # Normed
            hdf1_data.append(np.zeros((hdf1_cache, self.patch_size[0], self.patch_size[1])))  # Missing
            hdf1_data.append(np.zeros((hdf1_cache, self.patch_size[0], self.patch_size[1])))  # Background
            hdf1_hdf = self.create_hdf(output + "_train", self.patch_size, data_min, data_max, True, True, True, True)
        else:
            hdf1_hdf = self.create_hdf(output + "_test", self.patch_size, data_min, data_max)

        # Get patches with corresponding masks and missing data map
        x = 0
        while x <= self.image.shape[0] - self.patch_size[0]:
            x_train_bool = x % (self.patch_size[0] - self.train_overlap) == 0
            x_test_bool = x % (self.patch_size[0] - self.test_overlap) == 0
            if x_train_bool or x_test_bool:
                y = 0
                while y <= self.image.shape[1] - self.patch_size[1]:
                    y_train_bool = y % (self.patch_size[1] - self.train_overlap) == 0
                    y_test_bool = y % (self.patch_size[1] - self.test_overlap) == 0
                    if y_train_bool or y_test_bool:
                        if mask is not None:
                            submask = mask[x : x + self.patch_size[0], y : y + self.patch_size[1]]
                            if submask.sum() == self.patch_size[0] * self.patch_size[1] and y_test_bool and x_test_bool:
                                hdf2_data, hdf2_current_size = self.hdf_icrementation(hdf2_data, x, y, self.patch_size, hdf2_current_size, self.image, train=True, test=True, normed_image=self.normed_image, missing=self.missing, background=self.background, target=self.roi)
                            if submask.sum() == 0 and y_train_bool and x_train_bool:
                                hdf1_data, hdf1_current_size = self.hdf_icrementation(hdf1_data, x, y, self.patch_size, hdf1_current_size, self.image, train=True, test=False, normed_image=self.normed_image, missing=self.missing, background=self.background, target=self.roi)
                        else:
                            if train and y_train_bool and x_train_bool:
                                hdf1_data, hdf1_current_size = self.hdf_icrementation(hdf1_data, x, y, self.patch_size, hdf1_current_size, self.image, train=True, test=False, normed_image=self.normed_image, missing=self.missing, background=self.background, target=self.roi)
                            elif not train and y_test_bool and x_test_bool:
                                hdf1_data, hdf1_current_size = self.hdf_icrementation(hdf1_data, x, y, self.patch_size, hdf1_current_size, self.image, train=False, test=True, normed_image=self.normed_image, missing=None, background=None, target=None)

                        # Flush when needed
                        if hdf1_current_size == hdf1_cache:
                            self.flush_into_hdf5(
                                hdf1_hdf,
                                hdf1_data,
                                hdf1_current_index,
                                hdf1_cache,
                                self.patch_size,
                            )
                            hdf1_current_index += hdf1_cache
                            hdf1_current_size = 0
                        if mask is not None:
                            # Flush when needed
                            if hdf2_current_size == hdf2_cache:
                                self.flush_into_hdf5(
                                    hdf2_hdf,
                                    hdf2_data,
                                    hdf2_current_index,
                                    hdf2_cache,
                                    self.patch_size,
                                )
                                hdf2_current_index += hdf2_cache
                                hdf2_current_size = 0
                    y += 1
            x += 1

        # Final flush
        if hdf1_current_size > 0:
            self.flush_into_hdf5(
                hdf1_hdf,
                hdf1_data,
                hdf1_current_index,
                hdf1_current_size,
                self.patch_size,
            )
        # Final flush
        if mask is not None:
            if hdf2_current_size > 0:
                self.flush_into_hdf5(
                    hdf2_hdf,
                    hdf2_data,
                    hdf2_current_index,
                    hdf2_current_size,
                    self.patch_size,
                )
        if mask is not None:
            return [hdf1_hdf, hdf2_hdf]
        else:
            return [hdf1_hdf]
            
    def hdf_icrementation(self, hdf, x, y, patch_size, hdf_current_size, source, train=False, test=False, normed_image=None, missing=None, background=None, target=None):
        p = source[x : x + patch_size[0], y : y + patch_size[1]]
        position = [[x, x + patch_size[0]], [y, y + patch_size[1]]]
        idx = np.isnan(p)
        p[idx] = 0

        if train:   
            m = missing[x : x + patch_size[0], y : y + patch_size[1]]
            n = normed_image[x : x + patch_size[0], y : y + patch_size[1]]
            b = background[x : x + patch_size[0], y : y + patch_size[1]]
            t = target[x : x + patch_size[0], y : y + patch_size[1]]
            m[idx] = 0

            if test or (not test and np.sum(m) > 1.0):
                hdf[2][hdf_current_size, :, :] = t
                hdf[3][hdf_current_size, :, :] = m
                hdf[5][hdf_current_size, :, :] = n
                hdf[4][hdf_current_size, :, :] = b
                hdf[0][hdf_current_size, :, :] = p
                hdf[1][hdf_current_size, :, :] = position
                hdf_current_size += 1
        else:
            hdf[0][hdf_current_size, :, :] = p
            hdf[1][hdf_current_size, :, :] = position
            hdf_current_size += 1

        return hdf, hdf_current_size

    def create_hdf(self, output, patch_size, data_min, data_max, missing=False, target=False, background=False, normed=False):
        # Creation of the HDF5 file
        hdf = h5py.File(output + ".h5", "w")

        hdf.create_dataset(
            "patches",
            (1, patch_size[0], patch_size[1], 1),
            maxshape=(None, patch_size[0], patch_size[1], 1),
            compression="gzip",
            compression_opts=7,
        )
        hdf.attrs["min"] = data_min
        hdf.attrs["max"] = data_max
        hdf.create_dataset(
            "positions",
            (1, 2, 2, 1),
            maxshape=(None, 2, 2, 1),
            compression="gzip",
            compression_opts=7,
        )
        if missing:
            hdf.create_dataset(
                "missing",
                (1, patch_size[0], patch_size[1], 1),
                maxshape=(None, patch_size[0], patch_size[1], 1),
                compression="gzip",
                compression_opts=7,
            )
        if target:
            hdf.create_dataset(
                "spines",
                (1, patch_size[0], patch_size[1], 1),
                maxshape=(None, patch_size[0], patch_size[1], 1),
                compression="gzip",
                compression_opts=7,
            )
        if background:
            hdf.create_dataset(
                "background",
                (1, patch_size[0], patch_size[1], 1),
                maxshape=(None, patch_size[0], patch_size[1], 1),
                compression="gzip",
                compression_opts=7,
            )
        if normed:
            hdf.create_dataset(
                "normed",
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
            The different data (patches, masks/targets, missmap/missing, backgrounds)
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

        if "spines" in hdf:
            hdf["spines"].resize((current_index + size, patch_size[0], patch_size[1], 1))
            hdf["spines"][current_index : current_index + size, :, :, :] = data[2][
                :size, :, :, np.newaxis
            ]

        if "missing" in hdf:
            hdf["missing"].resize((current_index + size, patch_size[0], patch_size[1], 1))
            hdf["missing"][current_index : current_index + size, :, :, :] = data[3][
                :size, :, :, np.newaxis
            ]

        if "background" in hdf:    
            hdf["background"].resize((current_index + size, patch_size[0], patch_size[1], 1))
            hdf["background"][current_index : current_index + size, :, :, :] = data[4][
                :size, :, :, np.newaxis
            ]
        
        if "normed" in hdf:
            hdf["normed"].resize((current_index + size, patch_size[0], patch_size[1], 1))
            hdf["normed"][current_index : current_index + size, :, :, :] = data[2][
                :size, :, :, np.newaxis
            ]

        hdf.flush()
        
    def mosaic_building():
        pass