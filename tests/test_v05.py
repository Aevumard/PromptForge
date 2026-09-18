import unittest
from harness.runner.transforms import TRANSFORMS


class V05Tests(unittest.TestCase):

    def test_all_arms_exist(self):
        self.assertEqual(
            set(TRANSFORMS),
            {
                'noop',
                'selection_only',
                'representation_only',
                'selection_representation'
            }
        )

    def test_selection_excludes_noise(self):
        ctx = {
            'required': ['a'],
            'data': {
                'a': 'required',
                'noise': 'irrelevant'
            }
        }

        result = TRANSFORMS['selection_only'](ctx)

        self.assertEqual(result['value'], {'a': 'required'})
        self.assertIn('noise', result['excluded'])


if __name__ == '__main__':
    unittest.main()