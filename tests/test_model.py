import unittest

import torch

from pggan.models import Discriminator
from pggan.models.modules import MinibatchStdDev


class TestDiscriminator(unittest.TestCase):
    def test_discriminator(self):
        discriminator = Discriminator()
        B, C, H, W = 1, 3, 4, 4
        x = torch.rand([B, C, H, W])
        result = discriminator(x)
        self.assertEqual(result.size(0), B)
        self.assertEqual(result.size(1), 1)

    def test_minibatch_stddev_forward(self):
        minibatch_stddev = MinibatchStdDev()

        B, C, H, W = 4, 10, 2, 2
        x = torch.rand(B, C, H, W)
        result = minibatch_stddev(x)
        self.assertEqual(result.size(0), B)
        self.assertEqual(result.size(1), C+1)
        self.assertEqual(result.size(2), H)
        self.assertEqual(result.size(3), W)
