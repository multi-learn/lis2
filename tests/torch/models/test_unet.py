from unittest import TestCase

from deep_filaments.torch.models import UNet


class TestUNet(TestCase):
    def test_pytorch_unet(self):
        model = UNet()
        self.assertIsNotNone(model)
