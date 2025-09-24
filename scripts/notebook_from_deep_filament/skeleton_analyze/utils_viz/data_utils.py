import pandas as pd
from reproject.adaptive.high_level import reproject_adaptive

from astropy.io import fits
from astropy.wcs import WCS

import numpy as np
import matplotlib.pyplot as plt


def open_fits_data(path):

    # Open the FITS file
    hdul = fits.open(path)

    # Access the primary data (typically image data)
    primary_hdu = hdul[0]  # Primary HDU (Header/Data Unit)

    header = primary_hdu.header
    data = primary_hdu.data

    # Close the file when done
    hdul.close()

    return header, data


def show_fits_data(image_data, figsize=(10, 10)):
    if isinstance(image_data, str):
        # Open the FITS file
        hdul = fits.open(image_data)

        # Get the image data
        image_data = hdul[0].data

        hdul.close()

    # Set the figure size
    plt.figure(figsize=figsize)

    # Display the image
    plt.imshow(image_data, cmap='gray', origin='lower')
    plt.colorbar()
    plt.show()
    

def reformat_threed(threed_data, twoddata, name_newfile='reprojected_3d_fits_file.fits'):
    hdul = fits.open(threed_data)
    hdu_3d = hdul[0]
    data_3d = hdu_3d.data  # Shape: (788, 302, 602)

    wcs_3d_1 = WCS(hdu_3d.header.copy())
    wcs_3d_1 = wcs_3d_1.celestial

    target_shape_2d = (313, 157)
    reprojected_data_3d = np.empty((data_3d.shape[0], target_shape_2d[0], target_shape_2d[1]))

    # Step 5: Reproject each 2D slice individually
    for i in range(data_3d.shape[0]):        

        slice = data_3d[i,:,:]

        # Reproject the current 2D slice onto the target 2D WCS
        reprojected_slice, footprint = reproject_adaptive((slice, wcs_3d_1), twoddata)

        # Store the reprojected slice in the 3D array
        reprojected_data_3d[i, :, :] = reprojected_slice

    # Step 1: Identify NaN values in the reprojected 3D data
    image_nan_locs = np.isnan(reprojected_data_3d)

    # Step 2: Calculate the minimum of the non-NaN values
    min_value = np.nanmin(reprojected_data_3d)  # np.nanmin ignores NaNs and finds the minimum of the remaining data

    # Step 3: Replace NaNs with the minimum value
    reprojected_data_3d[image_nan_locs] = min_value

    # Step 4: Save the reprojected 3D data to a new FITS file
    hdu_2d = fits.PrimaryHDU(data=reprojected_data_3d, header=hdu_3d.header)
    hdu_2d.writeto(name_newfile, overwrite=True)

    # Step 5: Close the original 3D FITS file
    hdul.close()


def convert_to_kms(CRPIX3, CDELT3, CRVAL3, NAXIS3, z=None):

    if z is None :
        z_max = NAXIS3
        z = np.linspace(0,z_max,z_max)
    
    return (z-CRPIX3) * CDELT3 + CRVAL3