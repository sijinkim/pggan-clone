import unittest

from pggan.helper import get_channel_size


class TestHelper(unittest.TestCase):

    def test_get_channel_size_of_conv_blocks(self):
        channel_512_to_16 = [(512, 512),
                             (512, 512),
                             (512, 512),
                             (512, 512),
                             (512, 256),
                             (256, 128),
                             (128, 64),
                             (64, 32),
                             (32, 16),
                             (16, 3)]

        channels_256_to_32 = [(256, 256),
                              (256, 256),
                              (256, 256),
                              (256, 256),
                              (256, 128),
                              (128, 64),
                              (64, 32),
                              (32, 3)]

        self.assertEqual(get_channel_size(512, 16), channel_512_to_16)
        self.assertEqual(get_channel_size(256, 32), channels_256_to_32)
