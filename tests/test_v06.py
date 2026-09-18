import json
import unittest
from pathlib import Path

from harness.runner.aggregation import paired
from harness.runner.transforms import (
    V06_ARM_SEQUENCES,
    apply_sequence,
    compile_v06_arm,
    decode_representation_a,
    decode_representation_b,
)


class V06Tests(unittest.TestCase):
    def test_required_arm_sequences(self):
        self.assertEqual(
            V06_ARM_SEQUENCES['selection_only'],
            ['selection']
        )
        self.assertEqual(
            V06_ARM_SEQUENCES['selection_representation'],
            ['selection', 'representation']
        )

    def test_representation_a_is_lossless_and_structurally_distinct(self):
        ctx = {
            'required': ['a', 'b'],
            'data': {'a': 'one', 'b': 2, 'noise': 'x'}
        }
        selected = apply_sequence(ctx, ['selection'])
        represented = apply_sequence(ctx, ['selection', 'representation_A'])

        self.assertEqual(
            decode_representation_a(represented['value']),
            selected['value']
        )
        self.assertNotEqual(represented['value'], selected['value'])

    def test_representation_b_is_lossless_and_structurally_distinct(self):
        ctx = {
            'required': ['a', 'b'],
            'data': {'a': 'one', 'b': 2, 'noise': 'x'}
        }
        selected = apply_sequence(ctx, ['selection'])
        represented = apply_sequence(ctx, ['selection', 'representation_B'])

        self.assertEqual(
            decode_representation_b(represented['value']),
            selected['value']
        )
        self.assertNotEqual(represented['value'], selected['value'])

    def test_selection_representation_uses_two_stages(self):
        ctx = {
            'required': ['a'],
            'data': {'a': 'required', 'noise': 'irrelevant'}
        }
        compiled = compile_v06_arm(ctx, 'selection_representation')

        self.assertEqual(
            compiled['transform_sequence'],
            ['selection', 'representation']
        )
        self.assertEqual(compiled['excluded'], ['noise'])

    def test_schema_contains_v06_fields(self):
        root = Path(__file__).resolve().parents[1]
        schema = json.loads(
            (root / 'schemas' / 'run.schema.json').read_text(
                encoding='utf-8'
            )
        )
        self.assertIn('quality_pass', schema['required'])
        self.assertIn('transform_sequence', schema['required'])
        self.assertIn('quality_pass', schema['properties'])
        self.assertEqual(
            schema['properties']['quality_pass']['type'],
            'boolean'
        )
        self.assertEqual(
            schema['properties']['transform_sequence']['type'],
            'array'
        )

    def _run(self, repetition, input_tokens, output_tokens, passed=True):
        return {
            'repetition': repetition,
            'status': 'PASS',
            'metrics': {
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'reasoning_tokens': min(output_tokens, 10),
                'total_tokens': input_tokens + output_tokens,
                'latency_ms': 1
            },
            'verification': {'passed': passed},
            'execution': {'provider_status': 'MODEL_OK'}
        }

    def test_compensation_ratio_exact_formula(self):
        control = [self._run(1, 100, 20)]
        treatment = [self._run(1, 90, 25)]

        result = paired(control, treatment)
        row = result['rows'][0]

        self.assertEqual(row['input_tokens_delta'], -10)
        self.assertEqual(row['output_tokens_delta'], 5)
        self.assertEqual(row['downstream_compensation'], 5)
        self.assertEqual(row['input_removed'], 10)
        self.assertEqual(row['compensation_ratio'], 0.5)
        self.assertEqual(result['compensation_ratio']['mean'], 0.5)

    def test_compensation_ratio_not_applicable_when_input_not_reduced(self):
        control = [self._run(1, 100, 20)]
        treatment = [self._run(1, 105, 25)]

        result = paired(control, treatment)
        row = result['rows'][0]

        self.assertIsNone(row['compensation_ratio'])
        self.assertEqual(result['compensation_ratio']['n'], 0)


if __name__ == '__main__':
    unittest.main()
