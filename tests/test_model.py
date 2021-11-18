import unittest

import torch

from pggan.models import Discriminator
from pggan.models.modules import MinibatchStdDev


class TestDiscriminator(unittest.TestCase):
    def test_minibatch_stddev_forward(self):
        minibatch_stddev = MinibatchStdDev()

        B, C, H, W = 4, 10, 2, 2
        x = torch.rand(B, C, H, W)
        result = minibatch_stddev(x)
        self.assertEqual(result.size(0), B)
        self.assertEqual(result.size(1), C+1)
        self.assertEqual(result.size(2), H)
        self.assertEqual(result.size(3), W)

    def test_discriminator(self):
        discriminator = Discriminator()
        input_image_sizes = [4, 8, 16, 32, 64, 128, 256, 512, 1024]

        for input_image_size in input_image_sizes:

            B, C, H, W = 2, 3, input_image_size, input_image_size
            x = torch.rand(B, C, H, W)

            discriminator.set_input_image_size(input_image_size)
            result = discriminator(x)
            self.assertEqual(result.dim(), 2)
            self.assertEqual(result.size(0), B)
            self.assertEqual(result.size(1), 1)
