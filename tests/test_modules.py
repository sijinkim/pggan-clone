import unittest

import numpy as np
import torch

from pggan.models import Generator
from pggan.models import Discriminator
from pggan.models.modules import MinibatchStdDev
from pggan.models.modules import PixelwiseNorm


class TestGenerator(unittest.TestCase):
    def test_get_out_channels(self):
        resolutions = [1, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        ref_channels = [512, 512, 512, 512, 512, 256, 128, 64, 32, 16]
        generator = Generator()

        for res, ref in zip(resolutions, ref_channels):
            self.assertEqual(generator._get_out_channels(res), ref)

    def test_forward_without_fade_in(self):
        with torch.no_grad():
            generator = Generator()
            resolutions = [4, 8, 16, 32, 64, 128, 256, 512, 1024]

            B, C, H, W = 1, 512, 1, 1
            x = torch.randn(B, C, H, W)

            for res in resolutions:
                generator.set_output_image_size(res)
                result = generator(x)
                self.assertEqual(
                    result.shape, (B, 3, res, res))

    def test_forward_with_fade_in(self):
        with torch.no_grad():
            generator = Generator()
            generator.fade_in_weight = 0.5
            resolutions = [4, 8, 16, 32, 64, 128, 256, 512, 1024]

            B, C, H, W = 3, 512, 1, 1
            x = torch.randn(B, C, H, W)

            for res in resolutions:
                generator.set_output_image_size(res)
                result = generator(x)
                self.assertEqual(
                    result.shape, (B, 3, res, res))

    def test_pixel_wise_norm_method(self):
        with torch.no_grad():
            generator = Generator()
            generator.fade_in_weight = 0.2
            generator.pixel_wise_norm_method = 'LC'
            resolutions = [4, 8, 16, 32, 64]

            B, C, H, W = 5, 512, 1, 1
            x = torch.randn(B, C, H, W)

            for res in resolutions:
                generator.set_output_image_size(res)
                result = generator(x)
                self.assertEqual(
                    result.shape, (B, 3, res, res))


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

    def test_forward_without_fade_in(self):
        with torch.no_grad():
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

            channels = [16, 32, 64, 128, 256, 512]
            self.assertSetEqual(
                set([int(fromRGB_key.split('to_')[-1]) for fromRGB_key in discriminator.fromRGBs.keys()]),
                set(channels)
            )

    def test_forward_with_fade_in(self):
        with torch.no_grad():
            discriminator = Discriminator()
            discriminator.fade_in_weight = 0.5
            input_image_sizes = [8, 16, 32, 64, 128, 256]

            for input_image_size in input_image_sizes:

                B, C, H, W = 2, 3, input_image_size, input_image_size
                x = torch.rand(B, C, H, W)
                discriminator.set_input_image_size(input_image_size)
                result = discriminator(x)
                self.assertEqual(result.dim(), 2)
                self.assertEqual(result.size(0), B)
                self.assertEqual(result.size(1), 1)
