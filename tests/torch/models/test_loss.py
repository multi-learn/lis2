from unittest import TestCase

from numpy.testing import assert_almost_equal
import torch

from deep_filaments.torch.models import DiceLoss, BinaryCrossEntropyDiceSum


class TestLoss(TestCase):
    def test_DiceLoss(self):
        loss = DiceLoss()
        tensor_shape = (3, 4, 8)
        segmentation1 = torch.ones(tensor_shape)
        ground_truth1 = torch.ones(tensor_shape)
        assert_almost_equal(loss(segmentation1, ground_truth1).item(), 0)

        segmentation2 = torch.zeros(tensor_shape)
        assert_almost_equal(loss(segmentation2, ground_truth1).item(), 0.96)

    def test_BinaryCrossEntropyDiceSum(self):
        loss = BinaryCrossEntropyDiceSum()
        tensor_shape = (3, 4, 8)
        segmentation1 = torch.ones(tensor_shape)
        ground_truth1 = torch.ones(tensor_shape)
        assert_almost_equal(loss(segmentation1, ground_truth1).item(), 0)

        segmentation2 = torch.zeros(tensor_shape)
        assert_almost_equal(loss(segmentation2, ground_truth1).item(), 50.4799995)
