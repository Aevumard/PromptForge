import unittest

from benchmarks.v27_adversarial import run_benchmark


class V271AdversarialBenchmarkTests(unittest.TestCase):
    def test_all_adversarial_cases_pass(self) -> None:
        report = run_benchmark()
        self.assertEqual(report.schema_version, "promptforge-v27.1-adversarial.v1")
        self.assertEqual(report.total_cases, 8)
        self.assertEqual(report.failed_cases, 0)
        self.assertEqual(report.pass_rate, 1.0)

    def test_report_is_fully_auditable(self) -> None:
        report = run_benchmark()
        for result in report.results:
            self.assertTrue(result.case_id)
            self.assertTrue(result.category)
            self.assertTrue(result.checks)
            self.assertTrue(result.details)


if __name__ == "__main__":
    unittest.main()
