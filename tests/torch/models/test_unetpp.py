from unittest import TestCase

from deep_filaments.torch.models import UNetPP


class TestUNetPP(TestCase):
    def test_pytorch_unet(self):
        model = UNetPP()
        self.assertIsNotNone(model)
