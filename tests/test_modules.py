import unittest
from pathlib import Path

import numpy as np
import torch

from pggan.models import Generator


class TestGenerator(unittest.TestCase):
    def setUp(self):
        self.generator = Generator()
        self.sample_input = torch.from_numpy(
            np.random.randn(512, 1, 1)).float()

    def test_output_shape(self):
        self.assertEqual(torch.Size([1, 3, 1024, 1024]),
                         self.generator(torch.unsqueeze(self.sample_input, 0)).shape)
