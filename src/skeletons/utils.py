import numpy as np

import skimage.measure as measure

from astropy.io import fits 
from astropy.wcs import WCS

from reproject import reproject_interp, reproject_adaptive


def reproject_2Dspines23Ddata(path_3Ddata, path_3Ddata_reprojected, path_spine):
    """
    Tool to reproject 3D data.

    Reproject 3D data into 2d mask header and registers it into fits file at path_3Ddata_reprojected. 
    The 2d data may cover a larger area than the 3D data, we will select only the mutual area.

    Args:
        path_3Ddata (str) : path to existing fits 3D data.
        path_3Ddata_reprojected (str) : fits path to register reprojected 3Ddata data.
        path_spine (str) : path to existing fits 2D mask.

    Returns:
        Array with reprojected 3D data, corresponding header, recentered mask on corresponding zone, corresponding header of mask.
    """
    data_3D = fits.open(path_3Ddata)
    data_spine = fits.open(path_spine)

    data_s2_wcs = WCS(data_spine[0].header)

    wcs_3d_1 = WCS(data_3D[0].header.copy())
    wcs_3d_1 = wcs_3d_1.celestial

    # Get corners coordinates of 3Ddata data
    corners_world = wcs_3d_1.calc_footprint() 
    min_ra, min_dec = corners_world.min(axis=0)
    max_ra, max_dec = corners_world.max(axis=0)

    # Find these corners inside 2D mask data
    min_pix_x, min_pix_y = data_s2_wcs.world_to_pixel_values(min_ra, min_dec)
    max_pix_x, max_pix_y = data_s2_wcs.world_to_pixel_values(max_ra, max_dec)
    min_pix_x, max_pix_x = int(min_pix_x), int(max_pix_x)
    min_pix_y, max_pix_y = int(min_pix_y), int(max_pix_y)

    data_2d_cropped = (data_spine[0].data[min_pix_y:max_pix_y, max_pix_x:min_pix_x]>0).astype(np.int8)

    wcs_target_slice = data_s2_wcs.deepcopy()
    wcs_target_slice.wcs.crpix = [data_s2_wcs.wcs.crpix[0] - max_pix_x, data_s2_wcs.wcs.crpix[1] - min_pix_y]

    # Reproject 3d data into 2D mask resolution
    target_shape_2d = data_2d_cropped.shape
    reprojected_data_3d = np.empty((data_3D[0].data.shape[0], target_shape_2d[0], target_shape_2d[1]))

    for i in range(data_3D[0].data.shape[0]):        
        slice = data_3D[0].data[i,:,:]
        reprojected_slice, _ = reproject_interp((slice, wcs_3d_1), wcs_target_slice, shape_out=target_shape_2d)
        reprojected_data_3d[i, :, :] = reprojected_slice
    header_reprojected = wcs_target_slice.to_header()  # Convert WCS to header
    header_reprojected["NAXIS"] = data_3D[0].header["NAXIS"]
    header_reprojected["NAXIS3"] = data_3D[0].header["NAXIS3"]
    header_reprojected["CTYPE3"] = data_3D[0].header["CTYPE3"] 
    header_reprojected["CDELT3"] = data_3D[0].header["CDELT3"]
    header_reprojected["CRPIX3"] = data_3D[0].header["CRPIX3"]
    header_reprojected["CRVAL3"] = data_3D[0].header["CRVAL3"]

    fits.writeto(path_3Ddata_reprojected, reprojected_data_3d, header_reprojected, overwrite=True)

    return reprojected_data_3d, header_reprojected, data_2d_cropped, wcs_target_slice.to_header()


def get_skeleton_instances(skeletons):
    """
    Get individual skeletons from skeletonized mask.

    This function labels connected components in the given skeletonized mask, 
    assigning a unique integer label to each distinct skeleton. This allows for 
    separate analysis of individual skeleton structures.

    It uses skimage.measure.label with a connectivity of 2 so pixels are connected if they share a corner or an edge.

    Args:
        skeletons (np.array) : output of skeletonization tool. A 2D array containing all skeletons.

    Returns:
        List of binary masks for each skeleton.
    """
    labeled_skeleton, num_labels = measure.label(skeletons, return_num=True, connectivity=2)
    individual_skeletons = []
    for label in range(1, num_labels + 1):
        individual_skeleton = (labeled_skeleton == label).astype(np.uint8)
        individual_skeletons.append(individual_skeleton)
    return individual_skeletons


def convert_to_kms(CRPIX3, CDELT3, CRVAL3, NAXIS3, z=None):
    """
    Convert z coordinate to velocity with header attributes.

    Args:
        CRPIX3 (float) : Index of reference pixel on axis.
        CDELT3 (float) : Pixel size on axis.
        CRVAL3 (float) : Value at reference pixel on axis.
        NAXIS3 (float) : Size of axis.
        z (np.array, None) : either an array or a single z coordinate.

    Returns:
        Value or axis converted to velocity.
    """
    if z is None :
        z_max = NAXIS3
        z = np.linspace(0,z_max,z_max)
    
    return (z-CRPIX3) * CDELT3 + CRVAL3