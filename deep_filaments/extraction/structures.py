"""
Structures extraction code
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas
import networkx as nx


def create_graph_from_structure(structure):
    """
    Create a graph from a given structure (one node by pixel)

    Parameters
    ----------
    structure: np.ndarray
        The image with the structure

    Returns
    -------
    The graph representing the structure
    """
    g = nx.Graph()
    count_id = 0
    nodes_id = np.zeros(structure.shape, dtype="i") - 1

    for i in range(structure.shape[0]):
        for j in range(structure.shape[1]):
            if structure[i, j] == 0:
                continue

            # New node
            g.add_node(count_id, position=(i, j))
            nodes_id[i, j] = count_id

            # Link with previous nodes
            if i - 1 >= 0 and nodes_id[i - 1, j] >= 0:
                g.add_edge(nodes_id[i - 1, j], count_id)
            if j - 1 >= 0 and nodes_id[i, j - 1] >= 0:
                g.add_edge(nodes_id[i, j - 1], count_id)
            if i - 1 >= 0 and j - 1 >= 0 and nodes_id[i - 1, j - 1] >= 0:
                g.add_edge(nodes_id[i - 1, j - 1], count_id)
            if (
                i - 1 >= 0
                and j + 1 < structure.shape[1]
                and nodes_id[i - 1, j + 1] >= 0
            ):
                g.add_edge(nodes_id[i - 1, j + 1], count_id)

            # Update ID
            count_id += 1

    return g


def get_centerline(graph):
    """
    Compute the centerline of a given structure

    Parameters
    ----------
    graph: nx.Graph
        The graph representing the structure

    Returns
    -------
    The nodes forming the centerline and its length
    """
    end_nodes = []
    for n in graph.nodes:
        if graph.degree[n] == 1:
            end_nodes.append(n)

    path_length = 0
    path = []
    for source_node_i in range(len(end_nodes)):
        for target_node_i in range(source_node_i + 1, len(end_nodes)):
            sp = nx.shortest_path(
                graph, source=end_nodes[source_node_i], target=end_nodes[target_node_i]
            )
            if len(sp) > path_length:
                path = sp
                path_length = len(sp)

    return path, path_length


def extract_information(structure, roi, density):
    """
    Extract information from a given structure

    Parameters
    ----------
    structure: np.ndarray
        The image with the structures
    roi: np.ndarray
        The region of interest for each structure (using the same label)
    density: np.ndarray
        The density map

    Returns
    -------
    A dictionary with the information
    """
    nb_points = np.sum(structure)

    str_a0 = np.sum(structure, axis=0)
    pos = np.argwhere(str_a0 > 0)
    bb_up_y = np.squeeze(pos[0])
    bb_down_y = np.squeeze(pos[-1])

    str_a1 = np.sum(structure, axis=1)
    pos = np.argwhere(str_a1 > 0)
    bb_up_x = np.squeeze(pos[0])
    bb_down_x = np.squeeze(pos[-1])

    # Pad the bounding box with one pixel
    res = {
        "bb_up_x": int(bb_up_x) - 1,
        "bb_up_y": int(bb_up_y) - 1,
        "bb_down_x": int(bb_down_x) + 2,
        "bb_down_y": int(bb_down_y) + 2,
        "nb_points": int(nb_points),
        "roi_points": int(np.sum(roi)),
    }

    # Reduce the structure using the ROI
    structure_roi = structure[bb_up_x - 1 : bb_down_x + 2, bb_up_y - 1 : bb_down_y + 2]
    graph = create_graph_from_structure(structure_roi)
    centerline, c_length = get_centerline(graph)
    res["length_centerline"] = c_length

    # Density stats
    density_roi = density[bb_up_x - 1 : bb_down_x + 2, bb_up_y - 1 : bb_down_y + 2]
    res["mean_density"] = np.sum(density_roi) / np.sum(structure_roi)

    return res


def structures_extraction(image, roi_image, density):
    """
    Extract structures from a given labelled image

    Parameters
    ----------
    image: np.ndarray
        The input labelled image
    roi_image: np.ndarray
        The region of interest for each structure (using the same label)
    density: np.ndarray
        The density map

    Returns
    -------
    A pandas dataframe with structural information
    """
    struct_list = list()
    nb_structures = np.max(image)

    for i in range(1, nb_structures + 1):
        structure = image.copy()
        structure[image != i] = 0
        structure[structure > 0] = 1

        roi = roi_image.copy()
        roi[roi_image != i] = 0
        roi[roi > 0] = 1

        infos = extract_information(structure, roi, density)
        infos["ID"] = i

        struct_list.append(infos)

    return pandas.DataFrame(struct_list)
