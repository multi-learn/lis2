import os
import numpy as np
from tqdm import tqdm

from astropy.io import fits 

from configurable import Configurable, Schema, Config

from .clustering import BaseClustering
from .denoising import BaseDenoising
from .distance import BaseDistance
from .skeletonize import BaseSkeletonize
from .subclustering import BaseSubClustering
from .utils import convert_to_kms, get_skeleton_instances, reproject_2Dspines23Ddata

class ClusterCube(Configurable) :
    """
    ClusterCube for cutting mask into sub filaments.

    This method allows users to use COHS data to cut filaments masks obtained from HI-GAL.
    
    Configuration:
        - **threshold** (float): Density threshold to consider a point in x,y spectrum
        - **speed_threshold** (float):  Threshold of speed difference so two points of a spectrum don't belong to the same velocity plane.
        - **min_len_souspic** (int): Minimum number of points in a pic to be registered.
        - **data3D_path** (str): Path to 3D data, data must be existing.
        - **data3D_reprojected_path** (str): Path to register reprojecter 3D data.
        - **clustered_data_folder** (str): Path to register clustered skeletons.
        - **skeleton_tool** (Config): Configuration for the skeletonization tool (BaseSkeletonize).
        - **distance** (Config): Configuration for the distance metric (BaseDistance).
        - **clustering_method** (Config): Configuration for the clustering method (BaseClustering).
        - **subclustering_method** (Config): Configuration for the clustering method (BaseSubClustering).
        - **denoising_method** (Config): Configuration for the denoising method (BaseDenoising). 
        - **modes** (list) : List of fits to register into clustered_data_folder, available modes are [all_3d_skeletons, 3d_skeletons, 2d_skeletons]
    """
    config_schema = {
        'threshold': Schema(float, default=0.5),
        'speed_threshold': Schema(float, default=10),
        'min_len_souspic': Schema(int, default=3),
        'data3D_path': Schema(str),
        'data3D_reprojected_path': Schema(str),
        'mask_toreproject_data_path': Schema(str),
        'clustered_data_folder': Schema(str),
        'skeleton_tool': Schema(Config, default={"type":"NoSkeleton"}),
        'distance': Schema(Config, default={"type":"DistanceEuclieanExp"}),
        'clustering_method': Schema(Config, default={"type":"ClusteringDBSCAN"}),
        'subclustering_method': Schema(Config),
        'denoising_method': Schema(Config, default={"type":"NoDenoising"}),
        'modes': Schema(list, default=[]),
        }

    def __init__(self):
        """
        Initialzes skeleton, distance, clustering and denoising tools and reproject 3D data into mask format.
        """
        self.skeleton_tool = BaseSkeletonize.from_config(self.skeleton_tool)
        self.distance = BaseDistance.from_config(self.distance)
        self.clustering_method = BaseClustering.from_config(self.clustering_method)
        self.subclustering_method = BaseSubClustering.from_config(self.subclustering_method) if self.subclustering_method["type"] is not None else None
        self.denoising_method = BaseDenoising.from_config(self.denoising_method)

        self.data3D_reprojected, self.data3D_reprojected_header, self.mask_reprojected_data, self.mask_reprojected_header = reproject_2Dspines23Ddata(self.data3D_path, self.data3D_reprojected_path, self.mask_toreproject_data_path)
        
        
    def get_clustered_cube(self):
        """
        Obtain clustered skeletons for all 3D data file.

        The algorithms skeletonizes each mask and then proceed to cluster extracted points from denoised signal 
        of the skeleton according to 3D data and distance metric.

        It registers the clustered skeletons inside different fits files.
        """        
        skeletons = self.skeleton_tool.get_skeletons(self.mask_reprojected_data)
        individual_skeletons = get_skeleton_instances(skeletons)

        all_skeletons_all_clusters_data = np.zeros_like(self.data3D_reprojected)
        for idx_sk, sk in enumerate(tqdm(individual_skeletons, "interate skeletons")) : 
            # Get region of this skeleton 
            y_coords, x_coords = np.where(sk == 1)
            points = list(zip(x_coords, y_coords))

            # Init list of points for this skeleton
            multiplied_points_y_x_z_v_s = []

            # Go through the signal in x,y to retrieve all the pics above threshold if pic is longer than min_len_souspic
            for i, (x,y) in enumerate(points):
                smoothed_signal = self.denoising_method.get_denoised_signal(self.data3D_reprojected[:, y, x])

                sub_pic = []
                for idx, s in enumerate(smoothed_signal):
                    if len(sub_pic)>0 and abs(convert_to_kms(self.data3D_reprojected_header["CRPIX3"], self.data3D_reprojected_header["CDELT3"], self.data3D_reprojected_header["CRVAL3"], self.data3D_reprojected_header["NAXIS3"], min(sub_pic)) 
                                              - convert_to_kms(self.data3D_reprojected_header["CRPIX3"], self.data3D_reprojected_header["CDELT3"], self.data3D_reprojected_header["CRVAL3"], self.data3D_reprojected_header["NAXIS3"], idx)) > self.speed_threshold:
                        if len(sub_pic) >= self.min_len_souspic :
                            multiplied_points_y_x_z_v_s.extend([(y, x, z, convert_to_kms(self.data3D_reprojected_header["CRPIX3"], self.data3D_reprojected_header["CDELT3"], self.data3D_reprojected_header["CRVAL3"], self.data3D_reprojected_header["NAXIS3"], z), s) for z in sub_pic])
                        sub_pic = []
                    elif s < self.threshold and len(sub_pic)>0:
                        if len(sub_pic) >= self.min_len_souspic :
                            multiplied_points_y_x_z_v_s.extend([(y, x, z, convert_to_kms(self.data3D_reprojected_header["CRPIX3"], self.data3D_reprojected_header["CDELT3"], self.data3D_reprojected_header["CRVAL3"], self.data3D_reprojected_header["NAXIS3"], z), s) for z in sub_pic])
                        sub_pic = []
                    elif s > self.threshold :
                        sub_pic.append(idx)

            # Get coords of pics in full 3D data
            y = [item[0] for item in multiplied_points_y_x_z_v_s]
            x = [item[1] for item in multiplied_points_y_x_z_v_s]
            z = [item[2] for item in multiplied_points_y_x_z_v_s]
            s = [item[4] for item in multiplied_points_y_x_z_v_s]

            # Compute distance between all the points x,y,z coordinates along velocity and density at these coordinates.
            n_points = len(multiplied_points_y_x_z_v_s)
            distance_matrix = np.zeros((n_points, n_points))
            for i in range(n_points):
                p1 = multiplied_points_y_x_z_v_s[i]
                for j in range(i + 1, n_points):  
                    p2 = multiplied_points_y_x_z_v_s[j]
                    d = self.distance.get_distance(p1, p2)
                    distance_matrix[i,j] = d
                    distance_matrix[j,i] = d

            if n_points > 0 :
                # Clustering
                labels = self.clustering_method.predict(distance_matrix)

                if self.subclustering_method is not None :
                    labels = self.subclustering_method.predict(x, y, z, self.data3D_reprojected.shape, labels)
                                                                            
                for mode in self.modes : 
                    if mode == "3d_skeletons" :
                        cluster_data = np.zeros_like(self.data3D_reprojected)
                        for i in range (len(labels)) :
                            l = labels[i] + 1
                            xi = x[i] 
                            yi = y[i] 
                            zi = z[i]
                            cluster_data[zi, yi, xi] = l
                        filename = f"{self.clustered_data_folder}/skeleton_{idx_sk}/3d_skeletons.fits"
                        os.makedirs(os.path.dirname(filename), exist_ok=True)
                        fits.writeto(filename, cluster_data, self.data3D_reprojected_header, overwrite=True)  

                    if mode == "all_3d_skeletons" :
                        for i in range (len(labels)) :
                            l = labels[i] + 1
                            xi = x[i] 
                            yi = y[i] 
                            zi = z[i]
                            all_skeletons_all_clusters_data[zi, yi, xi] = l

                    if mode == "2d_skeletons" :
                        for cluster_idx in np.unique(labels) :
                            if cluster_idx != -1 :
                                cluster_data = np.zeros((self.data3D_reprojected.shape[1:]))
                                idxs = np.argwhere(labels==cluster_idx).flatten()
                                xi = [x[px] for px in idxs]
                                yi = [y[py] for py in idxs]
                                for (py, px) in zip(yi,xi):
                                    cluster_data[py, px] = 1

                                filename = f"{self.clustered_data_folder}/skeleton_{idx_sk}/cluster_{cluster_idx}/2d_skeleton.fits"
                                os.makedirs(os.path.dirname(filename), exist_ok=True)
                                fits.writeto(filename, cluster_data, self.mask_reprojected_header, overwrite=True)
                    
            else :
                print("not enough sample in skeleton to cluster")
                print(idx_sk)

        filename = f"{self.clustered_data_folder}/all_skeletons_all_clusters_data_3d_skeletons.fits"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        fits.writeto(filename, all_skeletons_all_clusters_data, self.data3D_reprojected_header, overwrite=True) 