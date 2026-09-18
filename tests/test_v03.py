import json
import unittest
from pathlib import Path
from harness.runner.transforms import TRANSFORMS
from harness.evaluators.deterministic import evaluate

class V03Tests(unittest.TestCase):
    def setUp(self):
        root=Path(__file__).resolve().parents[1]
        self.tasks=json.loads((root/'tasks'/'suite.json').read_text(encoding='utf-8'))['tasks']

    def test_four_tasks(self):
        self.assertEqual(len(self.tasks),4)

    def test_four_arms(self):
        self.assertEqual(set(TRANSFORMS),{'noop','selection_only','representation_only','selection_representation'})

    def test_selection_preserves_required(self):
        for t in self.tasks:
            ctx={'required':t['required'],'data':t['data']}
            c=TRANSFORMS['selection_only'](ctx)
            expected={k:t['data'][k] for k in t['required']}
            ok,_=evaluate(expected,c['value'])
            self.assertTrue(ok)

    def test_selection_records_exclusions(self):
        for t in self.tasks:
            ctx={'required':t['required'],'data':t['data']}
            c=TRANSFORMS['selection_only'](ctx)
            self.assertGreaterEqual(len(c['excluded']),1)

if __name__=='__main__':
    unittest.main()