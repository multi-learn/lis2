import numpy as np
import pandas as pd
from collections import deque

from scipy.stats import wasserstein_distance

from sklearn.cluster import DBSCAN, AgglomerativeClustering

import plotly.graph_objects as go

from utils_viz.utils import get_histogram


# VARIABLES :
limit_wd = 0.005
limit_ecart_kms = 10


def are_adjacent(p1, p2):
    return (abs(p1[0] - p2[0]) <= 1) and (abs(p1[1] - p2[1]) <= 1)

def get_adjacency_list(points):
    adjacency_list = {point: [] for point in points}

    for i, p1 in enumerate(adjacency_list.keys()):
        for j, p2 in enumerate(adjacency_list.keys()):
            if p1!=p2 and p2 not in adjacency_list[p1] and are_adjacent(p1, p2):
                adjacency_list[p1].append(p2)

    return adjacency_list





def filter_pics(pics, signals, adjacency_matrix):
    #TODO verifier tous les voisins ou plus loin (théorie ça suffit mais demander à Annie)
    # Fonction pour vérifier l'intersection de deux plages de pics
    #TODO qqch comme wasserstein avec un seuil 
    def intersect(range1, range2):
        return not (range1[1] < range2[0] or range2[1] < range1[0])
    def wasserstein(signal_i, signal_j):
        wd  = wasserstein_distance(signal_i, signal_j)
        return wd<=limit_wd

    points = list(pics.keys())
    filtered_peaks = []
    filtered_points = []
    filtered_signals = []
    for point_i in points:        
        for peak_i_idx, peak in enumerate(pics[point_i]):
            # Vérifier si ce pic partage une plage commune avec un pic d'un point adjacent
            has_common_peak = False
            for point_j in adjacency_matrix[point_i]:
                for peak_j_idx, adj_peak in enumerate(pics[point_j]):
                    #if intersect(peak, adj_peak):
                        #print("intersect")
                    if wasserstein(signals[point_i][peak_i_idx], signals[point_j][peak_j_idx]):
                        has_common_peak = True
                        break
                if has_common_peak:
                    break
            
            # Si le pic a une plage commune avec un adjacent, on le garde
            if has_common_peak:
                filtered_peaks.append(peak)
                filtered_points.append(point_i)
                filtered_signals.append(signals[point_i][peak_i_idx])
    
    return filtered_peaks, filtered_signals, filtered_points





def bfs_distance(start_point, target_point, points, adjacency_list):
    visited = {point: False for point in points}
    distance = {point: float("-1") for point in points}  
    distance[start_point] = 0
    
    queue = deque([start_point])
    visited[start_point] = True
    
    while queue:
        current = queue.popleft()
        
        for neighbor in adjacency_list[current]:
            if not visited[neighbor]:
                visited[neighbor] = True
                # Calculer la distance
                distance[neighbor] = distance[current] + 1
                queue.append(neighbor)
                
                if neighbor == target_point:
                    return distance[target_point]  
    
    return distance[target_point]  

def compare_two_pics(pic1,pic2):
    dist = max(abs(pic1[0]-pic2[1]), abs(pic1[1]-pic2[0]))
    return 1 if dist>=limit_ecart_kms else 0




def get_distance_total(filtered_points, filtered_pics, adjacency_list, alpha, beta):
    n_signals = len(filtered_points)

    distance_matrix_path1 = np.zeros((n_signals, n_signals))
    distance_matrix_path2 = np.zeros((n_signals, n_signals))

    n_filtered_signals = len(filtered_pics)
    for i in range(n_filtered_signals):
        for j in range(i + 1, n_filtered_signals):  
            dist_kms = compare_two_pics(filtered_pics[i],filtered_pics[j])
            distance_matrix_path1[i, j] = dist_kms
            distance_matrix_path1[j, i] = dist_kms
            
            if filtered_points[i] == filtered_points[j]:  # Identical points condition
                distance_matrix_path2[i, j] = 0
                distance_matrix_path2[j,i] = 0

            else:
                dist = bfs_distance(filtered_points[i], filtered_points[j], filtered_points, adjacency_list)
                distance_matrix_path2[i, j] = dist
                distance_matrix_path2[j, i] = dist  

    max_distance = np.max(distance_matrix_path2)  
    distance_matrix_path2[distance_matrix_path2 == -1] = max_distance
    distance_matrix_path2 = distance_matrix_path2 / max_distance  

    distance_matrix_path = alpha * distance_matrix_path1 + beta * distance_matrix_path2
    max_distance = np.max(distance_matrix_path)  
    distance_matrix_path = distance_matrix_path / max_distance  

    return distance_matrix_path

def get_distance_wasser(n_signals, common_bin_edges, smoothed_signals):

    distance_matrix_wasser = np.zeros((n_signals, n_signals))
    for i in range(n_signals):
        for j in range(i + 1, n_signals):
            data_hist_X, bin_centers_X = get_histogram(smoothed_signals[i], common_bin_edges)
            data_hist_Y, bin_centers_Y = get_histogram(smoothed_signals[j], common_bin_edges)

            dist = wasserstein_distance(
                bin_centers_X, bin_centers_Y, 
                u_weights=data_hist_X, v_weights=data_hist_Y)
            
            distance_matrix_wasser[i, j] = dist
            distance_matrix_wasser[j, i] = dist  

    max_distance = np.max(distance_matrix_wasser)  
    if max_distance > 0:  
        distance_matrix_wasser = distance_matrix_wasser / max_distance  

    return distance_matrix_wasser





def clustering(clustering_method, data, names, distance_matrix, eps_dbscan, min_samples, distance_threshold):
    if clustering_method == "DBSCAN" :
        clustering_model = DBSCAN(eps=eps_dbscan, min_samples=min_samples, metric='precomputed').fit(distance_matrix)

    elif clustering_method == "AgglomerativeClustering" :
        clustering_model = AgglomerativeClustering(n_clusters=None, linkage="average",
                                                   distance_threshold=distance_threshold, metric='precomputed').fit(distance_matrix)

    labels = clustering_model.labels_

    unique_labels = set(labels)
    print(unique_labels)
    cluster_data = {}
    cluster_names = {}
    cluster_profiles = []

    for label in unique_labels:
        if label != -1:  # Ignore outliers points
            cluster_indices = np.where(labels == label)[0]
            sub_data = np.array([data[ci] for ci in cluster_indices])
            sub_names = np.array([names[ci] for ci in cluster_indices])
            cluster_profile = np.mean(sub_data, axis=0)
            cluster_profiles.append(cluster_profile)

            cluster_data[label] = sub_data
            cluster_names[label] = sub_names

    cluster_profiles_df = pd.DataFrame(cluster_profiles, columns=[f'{i+1}' for i in range(data[0].shape[0])])
    cluster_profiles_df['Cluster'] = range(len(cluster_profiles_df))

    return cluster_data, cluster_names, labels, cluster_profiles_df