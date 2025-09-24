import lis2.skeletons.utils as utils
import os
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import matplotlib.pyplot as plt

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

if __name__ == '__main__':
     print(os.getcwd())
     path_3Ddata = "../../../BIGSF_DATA/Clustering_COHRS/COHRS_10p50_0p00_CUBE_3T2_R2.fit"
     path_3Ddata_reprojected = "../../../BIGSF_DATA/Clustering_COHRS/COHRS_reprojected_10p50_0p00_CUBE_3T2_R2.fit"
     # path_spines = "../../../BIGSF_DATA/Clustering_COHRS/COHRS_labels_clustered.fits"
     path_spines = "../../../BIGSF_DATA/Clustering_COHRS/new_full_mask_002011.fits"
     path_skeleton = "../../../BIGSF_DATA/Clustering_COHRS/data_test/all_skeletons_all_clusters_data_3d_skeletons.fits"
     data_3D = fits.open(path_3Ddata)
     data_spine = fits.open(path_spines)
     skeleton = fits.open(path_skeleton)
     data_s2 = data_spine
     show_fits_data(data_s2[0].data )
     print(skeleton[0].header)
     # reprojected_data_3d, header_reprojected, data_2d_cropped, wcs_target_slice_header = utils.reproject_2Dspines23Ddata(path_3Ddata, path_3Ddata_reprojected, path_spines)
     # print(data_3D[0].header)
     # data_s2_wcs = WCS(data_spine[0].header)
     # wcs_3d_1 = WCS(data_3D[0].header.copy())
     # wcs_3d_1_celestial = wcs_3d_1.celestial
     # corners_world = wcs_3d_1_celestial.calc_footprint()
     # min_ra, min_dec = corners_world.min(axis=0)
     # max_ra, max_dec = corners_world.max(axis=0)
     # min_pix_x, min_pix_y = data_s2_wcs.world_to_pixel_values(min_ra, min_dec)
     # max_pix_x, max_pix_y = data_s2_wcs.world_to_pixel_values(max_ra, max_dec)
     # min_pix_x, max_pix_x = int(min_pix_x), int(max_pix_x)
     # min_pix_y, max_pix_y = int(min_pix_y), int(max_pix_y)
     # data_2d_cropped = (data_spine[0].data[min_pix_y:max_pix_y, max_pix_x:min_pix_x]>0).astype(np.int8)
     # wcs_target_slice = data_s2_wcs.deepcopy()
     # wcs_target_slice.wcs.crpix = [data_s2_wcs.wcs.crpix[0] - max_pix_x, data_s2_wcs.wcs.crpix[1] - min_pix_y]
     # print(data_spine[0].header)
