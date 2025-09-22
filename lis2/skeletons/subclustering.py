import abc

import numpy as np

import skimage.measure as measure

from configurable import TypedConfigurable, Schema


class BaseSubClustering(abc.ABC, TypedConfigurable):
    """
    BaseSubClustering for all sub clustering algorithms.

    This class serves as a template for creating nested clustering inside already existing clusters.
    Subclasses must implement the abstract methods to define specific behavior for these steps.

    Configuration:
        - **name** (str): The name of the subclustering algorithm, BaseSubClustering, used.
    """

    @abc.abstractmethod
    def predict(self, x, y, z, shape3d, labels):
        """
        Obtain sub clustering method. Fit and predict.

        Args:
            x (np.array) : x coordinates of points inside skeleton.
            y (np.array) : y coordinates of points inside skeleton.
            z (np.array) : z coordinates of points inside skeleton.
            shape3d : shape of 3D data.
            labels (np.array) : list of assigned labels per a previous clustering algorithm.

        Returns:
            New labels. Integers between -1 (outliers which are not registered in .fits files) and infinity.
        """
        pass


class SubClusteringSkimageLabel(BaseSubClustering):
    """
    Identifies and labels sub-clusters within clusters using the skimage.measure.label function.

    This function assigns a unique integer label to each connected component within a cluster,
    allowing for the separate analysis of individual sub-clusters.

    It utilizes `skimage.measure.label`, with connectivity configurable between 1 and 3.
    Clusters with a size smaller than **min_samples** are not registered.

    Configuration:
        - **min_samples** (int): The number of samples inside a cluster for it not to be erased.
        - **connectivity** (int): Maximum number of orthogonal hops to consider a pixel/voxel as a neighbor. Accepted values are ranging from 1 to 3.
    """

    config_schema = {
        "min_samples": Schema(int, default=7),
        "connectivity": Schema(int, default=2),
    }

    def preconditions(self):
        """
        Validate the preconditions for the object.

        Raises:
            AssertionError: If the connectivity value is not within the valid range (1 to 3).
        """
        assert (
            1 <= self.connectivity <= 3
        ), "Connectivity must be an integer between 1 and 3."

    def predict(self, x, y, z, shape3d, labels):
        label_start = 0
        labels_sub_clustering = np.zeros(shape3d)

        for cluster_idx in np.unique(labels):
            idxs = [i for i, label in enumerate(labels) if label == cluster_idx]
            xi = [x[pxx] for pxx in idxs]
            yi = [y[py] for py in idxs]
            zi = [z[pz] for pz in idxs]
            cluster_data = np.zeros(shape3d)
            for pz, py, pxx in zip(zi, yi, xi):
                cluster_data[pz, py, pxx] = 1

            # Apply label function for skimage to cut clusters
            labeled_skeleton, num_labels = measure.label(
                cluster_data, return_num=True, connectivity=self.connectivity
            )

            # Filter out small clusters
            for label in range(1, num_labels + 1):
                if np.sum(labeled_skeleton == label) >= self.min_samples:
                    labels_sub_clustering[labeled_skeleton == label] = (
                        label + label_start
                    )

            label_start = label_start + num_labels

        labels = [
            (
                labels_sub_clustering[pz, py, pxx] - 1
                if labels_sub_clustering[pz, py, pxx] > 0
                else -1
            )
            for (pz, py, pxx) in zip(z, y, x)
        ]

        return np.array(labels)
