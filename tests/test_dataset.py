
import unittest
from pathlib import Path


class TestDataset(unittest.TestCase):
    def setUp(self):
        data_root = Path('./tests/data_root').absolute()
        print(data_root)
