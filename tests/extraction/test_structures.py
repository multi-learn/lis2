from unittest import TestCase

import numpy as np
import networkx as nx

import deep_filaments.extraction.structures as stc


def create_structure():
    img = np.zeros((10, 10), dtype="i")
    img[3, 3] = 1
    img[3, 2] = 1
    img[3, 1] = 1
    img[2, 3] = 1
    img[1, 3] = 1
    img[3, 4] = 1
    img[3, 5] = 1
    img[3, 6] = 1
    return img


class TestStructures(TestCase):
    def test_create_graph_from_structure(self):
        img = create_structure()

        res = [
            [0, 1, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 1, 1, 1, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 1, 1, 0, 1, 0, 0, 0],
            [0, 1, 0, 1, 0, 1, 0, 0],
            [0, 1, 0, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 1, 0, 1],
            [0, 0, 0, 0, 0, 0, 1, 0],
        ]

        g = stc.create_graph_from_structure(img)
        g_a = nx.adjacency_matrix(g).todense()
        self.assertTrue(np.linalg.norm(res - g_a) < 1e-9)

    def test_get_centerline(self):
        img = create_structure()
        g = stc.create_graph_from_structure(img)
        centerline, length = stc.get_centerline(g)
        truth = [2, 3, 4, 5, 6, 7]

        self.assertEqual(length, 6)
        self.assertEqual(centerline, truth)

    def test_extract_information(self):
        img = create_structure()
        res = stc.extract_information(img, img, img)
        roi = img[res["bb_up_x"] : res["bb_down_x"], res["bb_up_y"] : res["bb_down_y"]]

        truth = [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ]

        self.assertTrue(np.linalg.norm(roi - truth) < 1e-9)

    def test_structures_extraction(self):
        img = create_structure()
        res = stc.structures_extraction(img, img, img)
        self.assertEqual(res.iloc[0]["roi_points"], 8)
        self.assertEqual(res.iloc[0]["nb_points"], 8)
        self.assertEqual(res.iloc[0]["ID"], 1)
