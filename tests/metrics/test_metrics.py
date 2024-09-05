from unittest import TestCase

from numpy.testing import assert_almost_equal
import torch

from deep_filaments.metrics.metrics import dice_index, pixel_acc, positive_part_accuracy


class TestMetrics(TestCase):
    def test_dice(self):
        tensor_shape = (3, 4, 8)
        segmentation1 = torch.ones(tensor_shape)
        ground_truth1 = torch.ones(tensor_shape)
        assert_almost_equal(dice_index(segmentation1, ground_truth1), 1)

        segmentation2 = torch.zeros(tensor_shape)
        assert_almost_equal(dice_index(segmentation2, ground_truth1), 0)

        segmentation3 = torch.tensor([0, 1, 1])
        ground_truth3 = torch.tensor([1, 1, 1])
        assert_almost_equal(dice_index(segmentation3, ground_truth3), 0.8)

        tensor_shape4 = (8, 8)
        segmentation4 = torch.ones(tensor_shape4)
        segmentation4[:4] = 0
        ground_truth4 = torch.ones(tensor_shape4)
        expected_dice = 2 / 3
        assert_almost_equal(dice_index(segmentation4, ground_truth4), expected_dice)

        segmentation5 = torch.tensor([0, 1, 3])
        ground_truth5 = torch.tensor([1, 1, 1])
        with self.assertRaises(ValueError):
            dice_index(segmentation5, ground_truth5)

        segmentation6 = torch.tensor([0, 1, 3])
        ground_truth6 = torch.tensor([2, 1, 3])
        with self.assertRaises(ValueError):
            dice_index(segmentation6, ground_truth6)

    def test_pixel_acc(self):
        tensor_shape = (3, 4, 8)
        segmentation1 = torch.ones(tensor_shape)
        ground_truth1 = torch.ones(tensor_shape)
        assert_almost_equal(pixel_acc(segmentation1, ground_truth1), 1)

        segmentation2 = torch.zeros(tensor_shape)
        assert_almost_equal(pixel_acc(segmentation2, ground_truth1), 0)

        tensor_shape3 = (3, 3)
        segmentation3 = torch.ones(tensor_shape3)
        segmentation3[:2] = 0
        ground_truth3 = torch.ones(tensor_shape3)
        expected_dice = 1 / 3
        assert_almost_equal(pixel_acc(segmentation3, ground_truth3), expected_dice)

    def test_positive_part_accuracy(self):
        tensor_shape = (4, 4)
        segmentation = torch.ones(tensor_shape)
        ground_truth = torch.ones(tensor_shape)
        assert_almost_equal(positive_part_accuracy(segmentation, ground_truth), 1)
