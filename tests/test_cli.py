import unittest
from pggan.__main__ import train, get_parser


class TestCLI(unittest.TestCase):
    def setUp(self):
        parser = get_parser()
        self.args = parser.parse_args([
            'train',
            '--epoch', '3',
            '--config_file', './tests/configs/config_test.yaml',
            '--data_root', './tests/data_root',
            '--output_root', './output_root',
            '--checkpoint_period', '1',
        ])

    def test_cli_train(self):
        train(self.args)

        self.assertEqual(self.args.output_root.joinpath('0001.pt').exists(), True)
        self.assertEqual(self.args.output_root.joinpath('0002.pt').exists(), True)
        self.assertEqual(self.args.output_root.joinpath('0003.pt').exists(), True)
        self.assertEqual(self.args.output_root.joinpath('best.pt').exists(), True)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.args.output_root)
