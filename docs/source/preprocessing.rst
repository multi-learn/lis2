Preprocessing Module
====================

.. currentmodule:: preprocessing

This module provides different ``preprocessing`` classes, which are responsible respectively for the building of the mosaics and extracting the patches from these mosaics.


Mosaic Building
---------------

.. autoclass:: src.preprocessing.FilamentMosaicBuilding
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

.. autoclass:: src.preprocessing.PatchExtraction
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
