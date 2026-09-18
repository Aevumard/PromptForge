import unittest
from pathlib import Path
import json

from harness.runner.transforms import TRANSFORMS
from harness.runner.executor import execute


class V04Tests(unittest.TestCase):

    def setUp(self):
        root = Path(__file__).resolve().parents[1]
        self.tasks = json.loads(
            (root / 'tasks' / 'suite.json').read_text(
                encoding='utf-8'
            )
        )['tasks']

    def test_metric_is_not_token_claim(self):
        context = {
            'required': ['a'],
            'data': {'a': 'hello world', 'noise': 'more text'}
        }
        task = {
            'expected': {'a': 'hello world'}
        }
        compiled = TRANSFORMS['selection_only'](context)
        result = execute(task, compiled)
        self.assertGreater(result['input_chars'], 0)

    def test_selection_reduces_serialized_size(self):
        for task in self.tasks:
            context = {
                'required': task['required'],
                'data': task['data']
            }
            selected = TRANSFORMS['selection_only'](context)
            noop = TRANSFORMS['noop'](context)

            selected_size = execute(
                {'expected': {k: task['data'][k] for k in task['required']}},
                selected
            )['input_chars']

            noop_size = execute(
                {'expected': {k: task['data'][k] for k in task['required']}},
                noop
            )['input_chars']

            self.assertLessEqual(selected_size, noop_size)

if __name__ == '__main__':
    unittest.main()