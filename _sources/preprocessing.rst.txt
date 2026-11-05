Preprocessing Module
====================

.. currentmodule:: preprocessing

This module provides different ``preprocessing`` classes, which are responsible respectively for the building of the mosaics and extracting the patches from these mosaics.


Mosaic Building
---------------

.. autoclass:: lis2.preprocessing.FilamentMosaicBuilding
   :members:
   :undoc-members:
   :show-inheritance:


The FilamentMosaicBuilding class constructs mosaics from multiple FITS files. Below is a breakdown of its internal process:


1. Mosaic Processing (mosaic_building)

Iterates through the list of FITS files in fits_file_names.
Chooses between (using ``one_file``):

- Multiple Mosaic Mode (build_mosaic): Processes each FITS file separately and saves individual mosaics.
- Unified Mosaic Mode (build_unified_mosaic): Combines all FITS files into a single large-scale mosaic. One file is created for each file in ``fits_file_names``

Writes the resulting mosaics to output_dir, ensuring proper file handling.


2. Multiple files creation (build_mosaic)

Retrieves FITS files from ``files_dir``.
Opens and processes each FITS file.
Uses reproject.reproject_and_coadd to align and merge data into a consistent format.
Applies a binary mask (idx > 0 → 1) to highlight filament regions.
Returns a list of processed HDUs and corresponding filenames.

3. Single file creation (build_unified_mosaic)

Loads all FITS files from ``files_dir`` and extracts the specified HDU.
Creates a new FITS header with updated spatial metadata:

- NAXIS1, NAXIS2: Defines the dimensions of the final mosaic.
- CRPIX1, CRPIX2: Sets the central reference pixel.
- CRVAL1, CRVAL2: Defines the reference celestial coordinates.

Converts missing values to NaN if avoid_missing is enabled.
Uses reproject.reproject_and_coadd to merge all files into a single mosaic.
Applies binarization based on the binarize and conservative settings.

- If conservative=True: Keeps only high-confidence pixels (a > 0.6).
- Otherwise: Uses a broader threshold (a > 0.2).

Returns the final mosaic and its updated header.

Patch Extraction
----------------

.. autoclass:: lis2.preprocessing.PatchExtraction
   :members:
   :undoc-members:
   :show-inheritance:

The PatchExtraction class extracts image patches and stores them efficiently in an HDF5 file for later use. Below is a structured breakdown of its internal process:

1. Initialization

    Takes as inputs different fits files created previously (i.e. image, target, missing and background). 

2. Preconditions

    The preconditions method ensures that all input files have the correct .fits format.
    Raises an error if any file is missing or has an invalid format.

3. Patch Extraction (extract_patches)

    Iterates over the inputs, extracting patches of a specified size (patch_size).
    Stores extracted patches in an HDF5 file for efficient storage and retrieval.
    Uses a caching mechanism (hdf_cache) to optimize memory usage by buffering patches before writing them to disk.
    If the output HDF5 file already exists, the extraction process is skipped unless the file is deleted or the output folder is changed.


Subtleties

    Efficient Storage: Uses incremental writing to store patches efficiently without excessive memory overhead.
    Data Integrity Checks: Ensures that only valid patches are stored, avoiding empty or uninformative samples.
    Handles missing data by replacing NaN values with zeros.
    Computes a labelled mask (labelled) based on missing, background, and target data.

Custom Extraction
-----------------

Writing general data pipeline is most of the time inadequate, so we suggest you write your own pipeline. To do so, here are detailled
steps to follow.

First, create a class that heritates from ``BasePatchExtraction`` and write a ``config_schema``:


.. code-block:: python

    class CustomPatchExtraction(BasePatchExtraction):
        config_schema = {
            "name": Schema(str, optional=True, default="patches"),
            "dataset_path": Schema(Union[str, Path]),
            "keys": Schema(List[str]),
            "dimensions": Schema(List(Tuple[int, int])),
            "output": Schema([Path]),
            "patch_size": Schema(int, default=64),
        }
        
        self.data = {}
        self.dim = {}
        for idx, key in enumerate(keys):
            self.data[key] = fits.getdata(dataset_path / f"{key}.fits")
            self.dim[key] = self.dimensions[idx]

In this example, we suppose that ``dimensions`` and ``keys`` have the same number of elements, are given in the same order, and that the keys to store in the ``hdf`` files are the same as 
the input keys.

This class must implement ``extract_patches``, but in practice we suggest to implement ``_create_hdf``, ``_hdf_incrementation`` and ``_flush_into_hdf`` as in ``PatchExtraction``. More on that latter.


.. code-block:: python

    def extract_patches(self) --> h5py.File:
        path_h5 = Path(self.output) / f"{self.name}.h5"
        if path_h5.exists():
            self.logger.info(
                f"HDF5 file {path_h5} already exists. Skipping patches extraction. If you want to run again, delete patches.h5 or change the output folder"

First thing is to check whether a ``h5`` file already exist to avoid unnacessary computation. Then, we create numpy arrays to store our data, using a cache system to optimize the memory usage :

.. code-block:: python

    else:
        for key in self.keys:
            hdf_data[key] = [
                np.zeros((hdf_cache, self.dim[key][0], self.dim[key][1]))
            ]

Again, we suppose that we have the same output keys as the input keys. Now, we need to create our ``hdf``. To do so, we write a ``_create_hdf`` function:

.. code-block:: python

    def _create_hdf(self) -> h5py.File:
        hdf = h5py.File(self.output / self.name / ".h5", "w")
        for key in self.keys:
            hdf.create_dataset(
                key,
                (1, self.dim[key][0], self.dim[key][1], 1),
                maxshape=(None, self.dim[key][0], self.dim[key][1], 1),
                compression="gzip",
                compression_opts=7,
            )
        return hdf

Then, back to ``extract_patches``, we create the ``hdf``, and initialize some indexes to keep track of our location in the image. We consider that the first given key is the reference for the dimensions : 

.. code-block:: python

    hdf_files = self._create_hdf()
    current_size = 0
    current_index = 0
    total_patches = (self.dim[self.keys[0]][0] - self.patch_size[0] + 1) * (
        self.dim[self.keys[0]][1] - self.patch_size[1] + 1
    )

Now we can loop on the data to make some computation on it. Again, we use the first key as reference.

.. code-block:: python

    with tqdm(total=total_patches, desc="Processing patches") as pbar:
        for y in range(0, self.dim[self.keys[0]][0] - self.patch_size[0] + 1) :
            for x in range(0, self.dim[self.keys[0]][0] - self.patch_size[0] + 1):
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

All we have left to do is to write ``_flush_into_hdf5`` and ``_hdf_incrementation``. Let's start with the first:

.. code-block:: python

    def _flush_into_hdf5(
        self,
        hdf: h5py.File,
        hdf_data: Tuple,
        current_index: int,
        current_size: int,
    ) -> None:

        for key in self.keys:
            hdf[key].resize(
                (current_index + current_size, self.patch_size[0], self.patch_size[1], 1)
            )
            hdf[key][current_index : current_index + current_size, :, :, :] = (
                hdf_data[0][:current_size, :, :, np.newaxis]
            )
        hdf.flush()

Then,  ``_hdf_incrementation``. It is probably in this function that the most modifications will have to happen. Some problem may arise if:
- Output keys are not the same as input keys,
- Some specific computation needs to be done on certain keys

Here is an example in a simple case:

.. code-block:: python

    def _hdf_incrementation(
        self,
        hdf_data: Tuple,
        y: int,
        x: int,
        hdf_current_size: int,
    ) -> Tuple[int, Tuple]:

        for key in self.keys:
            hdf_data[key][hdf_current_size, :, :] = self.data[key][y : y + self.patch_size[0], x : x + self.patch_size[1]]
            hdf_current_size += 1

        return hdf_current_size, hdf_data

For a more sophisticated example, please refer to ``PatchExtraction``

Full code : 

.. code-block:: python

    class CustomPatchExtraction(BasePatchExtraction):
        config_schema = {
            "name": Schema(str, optional=True, default="patches"),
            "dataset_path": Schema(Union[str, Path]),
            "keys": Schema(List[str]),
            "dimensions": Schema(List(Tuple[int, int])),
            "output": Schema([Path]),
            "patch_size": Schema(int, default=64),
        }
        
        self.data = {}
        self.dim = {}
        for idx, key in enumerate(keys):
            self.data[key] = fits.getdata(dataset_path / f"{key}.fits")
            self.dim[key] = self.dimensions[idx]

        def extract_patches(self) --> h5py.File:
            path_h5 = Path(self.output) / f"{self.name}.h5"
            if path_h5.exists():
                self.logger.info(
                    f"HDF5 file {path_h5} already exists. Skipping patches extraction. If you want to run again, delete patches.h5 or change the output folder"
            else:
                for key in self.keys:
                    hdf_data[key] = [
                        np.zeros((hdf_cache, self.dim[key][0], self.dim[key][1]))
                    ]

                hdf_files = self._create_hdf()
                current_size = 0
                current_index = 0
                total_patches = (self.dim[self.keys[0]][0] - self.patch_size[0] + 1) * (
                    self.dim[self.keys[0]][1] - self.patch_size[1] + 1
                )

                with tqdm(total=total_patches, desc="Processing patches") as pbar:
                    for y in range(0, self.dim[self.keys[0]][0] - self.patch_size[0] + 1) :
                        for x in range(0, self.dim[self.keys[0]][0] - self.patch_size[0] + 1):
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
            hdf = h5py.File(self.output / self.name / ".h5", "w")
            for key in self.keys:
                hdf.create_dataset(
                    key,
                    (1, self.dim[key][0], self.dim[key][1], 1),
                    maxshape=(None, self.dim[key][0], self.dim[key][1], 1),
                    compression="gzip",
                    compression_opts=7,
                )
            return hdf
        
        def _flush_into_hdf5(
            self,
            hdf: h5py.File,
            hdf_data: Tuple,
            current_index: int,
            current_size: int,
        ) -> None:

            for key in self.keys:
                hdf[key].resize(
                    (current_index + current_size, self.patch_size[0], self.patch_size[1], 1)
                )
                hdf[key][current_index : current_index + current_size, :, :, :] = (
                    hdf_data[0][:current_size, :, :, np.newaxis]
                )
            hdf.flush()
        
        def _hdf_incrementation(
            self,
            hdf_data: Tuple,
            y: int,
            x: int,
            hdf_current_size: int,
        ) -> Tuple[int, Tuple]:

            for key in self.keys:
                hdf_data[key][hdf_current_size, :, :] = self.data[key][y : y + self.patch_size[0], x : x + self.patch_size[1]]
                hdf_current_size += 1

            return hdf_current_size, hdf_data