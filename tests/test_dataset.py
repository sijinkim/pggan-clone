import unittest
from pathlib import Path

import torch

from pggan.dataset import CelebAHQ


class TestDataset(unittest.TestCase):
    def setUp(self):
        data_root = Path('./tests/data_root').absolute()
        self.dataset = CelebAHQ(data_root, 'train')

    def test_is_tensor(self):
        sample_data = self.dataset[0]
        self.assertIsInstance(sample_data, torch.Tensor)

    def test_data_shape(self):
        sample_data = self.dataset[0]
        self.assertEqual(sample_data.shape, torch.Size([3, 1024, 1024]))

    def test_data_scale(self):
        sample_data = self.dataset[0]
        self.assertGreaterEqual(sample_data.min(), 0.0)
        self.assertLessEqual(sample_data.max(), 1.0)
