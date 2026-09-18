import unittest

from harness.runner.transforms import TRANSFORMS


class V053Tests(unittest.TestCase):

    def test_four_arms(self):
        self.assertEqual(
            set(TRANSFORMS),
            {
                'noop',
                'selection_only',
                'representation_only',
                'selection_representation'
            }
        )


if __name__ == '__main__':
    unittest.main()