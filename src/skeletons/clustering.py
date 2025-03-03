import abc

import networkx as nx
from networkx.algorithms.clique import find_cliques

import numpy as np

from sklearn.cluster import DBSCAN, AgglomerativeClustering

from configurable import TypedConfigurable, Schema

class BaseClustering(TypedConfigurable):
    """
    BaseClustering for all clustering algorithms.

    Configuration :
        - **name** (str) : The name of clustering algorithm, BaseClustering, used.
    """
    @abc.abstractmethod
    def predict(self, data):
        """
        Obtain clustering method. Fit and predict.

        Args:
            data (np.array): can be a distance matrix or a binary array representing adjacency.

        Returns:
            Labels
        """
        pass    


class ClusteringDBSCAN(BaseClustering):
    """
    Performs density-based clustering using the DBSCAN algorithm.

    DBSCAN (Density-Based Spatial Clustering of Applications with Noise) identifies clusters 
    as dense regions of data points separated by areas of lower density. It does not require 
    the number of clusters to be predefined and can detect noise points that do not belong to 
    any cluster.
    
    Configuration:
        - **eps** (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other. 
        - **min_samples** (int): The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
    """
    config_schema = {
        'eps': Schema(float, default=2.0),
        'min_samples': Schema(int, default=3),
    }

    def predict(self, data):
        clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric="precomputed")
        labels = clustering.fit_predict(data)
        return labels

class ClusteringAgglomerative(BaseClustering):
    """
    Performs agglomerative hierarchical clustering using pairwise linkage distances.

    This method applies Agglomerative Clustering from scikit-learn, recursively merging clusters 
    based on a linkage criterion until the linkage distance exceeds `distance_threshold`.
    
    Configuration:
        - **distance_threshold** (float): The linkage distance threshold at or above which clusters will not be merged. 
    """
    config_schema = {
        'distance_threshold': Schema(float, default=15.0),
    }

    def predict(self, data):
        clustering = AgglomerativeClustering( distance_threshold=self.distance_threshold, metric='precomputed', n_clusters=None, linkage="average")
        labels = clustering.fit_predict(data)
        return labels

class ClusteringSpeed(BaseClustering):
    """
    Identifies clusters based on speed similarity using a graph-based approach.

    This clustering method constructs a graph where nodes represent data points, and edges 
    indicate that the speed difference between two points is within `speed_threshold`. 
    Cliques (fully connected subgraphs) are then identified and merged to form maximal fully connected subgraphs.

    Configuration:
        - **speed_threshold** (float) : The maximum allowable speed difference 
          for two points to be considered connected in the graph.
    """
    config_schema = {
        'speed_threshold': Schema(float, default=10.0),
    }

    def predict(self, data):
        G = nx.from_numpy_array((data<=self.speed_threshold).astype(np.int8))
        cliques = list(find_cliques(G))

        merged_cliques = []
        while cliques:
            clique = cliques.pop(0)
            to_merge = []
            for other_clique in cliques:
                if set(clique) & set(other_clique):
                    to_merge.append(other_clique)
            clique += [node for other_clique in to_merge for node in other_clique]
            cliques = [other_clique for other_clique in cliques if other_clique not in to_merge]
            merged_cliques.append(clique)

        node_to_clique = {}
        for clique_id, clique in enumerate(merged_cliques):
            for node in clique:
                node_to_clique[node] = clique_id
        labels = [node_to_clique[node] for node in range(len(data))]


        return labels
    