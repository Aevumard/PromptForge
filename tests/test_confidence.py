import unittest

from promptforge import (
    ConfidenceCalibrator,
    ConfidenceCalibrationPolicy,
    ConfidenceObservation,
)


class ConfidenceCalibrationTests(unittest.TestCase):
    def _observations(self):
        rows = []
        for i in range(10):
            rows.append(
                ConfidenceObservation(
                    f"low-{i}",
                    0.2,
                    outcome=(i < 2),
                    timestamp=10 + i,
                    family="a",
                )
            )
        for i in range(10):
            rows.append(
                ConfidenceObservation(
                    f"high-{i}",
                    0.8,
                    outcome=(i < 8),
                    timestamp=30 + i,
                    family="b",
                )
            )
        return rows

    def test_supported_bins_are_calibrated_and_metrics_are_descriptive(self):
        model = ConfidenceCalibrator().fit(
            self._observations(),
            policy=ConfidenceCalibrationPolicy(
                bins=5,
                min_bin_observations=5,
                min_total_observations=10,
            ),
        )

        low = model.assess(0.2)
        high = model.assess(0.8)

        self.assertEqual(low.status, "calibrated")
        self.assertEqual(high.status, "calibrated")
        self.assertLess(low.calibrated_confidence, high.calibrated_confidence)
        self.assertAlmostEqual(
            model.metrics.empirical_success_rate,
            0.5,
        )
        self.assertGreaterEqual(model.metrics.brier_score, 0.0)
        self.assertGreaterEqual(model.metrics.expected_calibration_error, 0.0)

    def test_isotonic_order_is_deterministic_even_when_raw_rates_cross(self):
        observations = [
            ConfidenceObservation(f"a-{i}", 0.2, outcome=(i == 0))
            for i in range(5)
        ] + [
            ConfidenceObservation(f"b-{i}", 0.8, outcome=(i < 5))
            for i in range(5)
        ]

        model_a = ConfidenceCalibrator().fit(
            observations,
            policy=ConfidenceCalibrationPolicy(
                bins=5,
                min_bin_observations=5,
                min_total_observations=5,
            ),
        )
        model_b = ConfidenceCalibrator().fit(
            list(reversed(observations)),
            policy=ConfidenceCalibrationPolicy(
                bins=5,
                min_bin_observations=5,
                min_total_observations=5,
            ),
        )

        self.assertEqual(model_a.to_dict(), model_b.to_dict())
        rates = [
            value
            for value in model_a.calibrated_bin_rates
            if value is not None
        ]
        self.assertEqual(rates, sorted(rates))

    def test_insufficient_total_data_fails_closed_to_raw_confidence(self):
        model = ConfidenceCalibrator().fit(
            self._observations()[:6],
            policy=ConfidenceCalibrationPolicy(
                bins=5,
                min_bin_observations=2,
                min_total_observations=10,
            ),
        )

        result = model.assess(0.8)

        self.assertEqual(result.status, "insufficient_total_data")
        self.assertEqual(result.calibrated_confidence, 0.8)
        self.assertEqual(result.adjustment, 0.0)

    def test_sparse_bin_uses_global_fallback_with_bounded_adjustment(self):
        model = ConfidenceCalibrator().fit(
            self._observations(),
            policy=ConfidenceCalibrationPolicy(
                bins=10,
                min_bin_observations=5,
                min_total_observations=10,
                max_adjustment=0.10,
            ),
        )

        result = model.assess(0.95)

        self.assertEqual(result.status, "global_fallback")
        self.assertLessEqual(abs(result.adjustment), 0.10)

    def test_temporal_boundary_excludes_future_and_required_future_fails_closed(self):
        observations = self._observations() + [
            ConfidenceObservation(
                "future",
                0.99,
                outcome=True,
                timestamp=100,
            )
        ]

        model = ConfidenceCalibrator().fit(
            observations,
            policy=ConfidenceCalibrationPolicy(
                bins=5,
                min_bin_observations=2,
                min_total_observations=10,
                cutoff=50,
            ),
        )

        self.assertIn("future", model.future_excluded_ids)
        self.assertNotIn("future", model.included_observation_ids)

        with self.assertRaises(ValueError):
            ConfidenceCalibrator().fit(
                observations,
                policy=ConfidenceCalibrationPolicy(
                    cutoff=50,
                    required_observation_ids=("future",),
                ),
            )

    def test_unknown_time_is_excluded_by_default_under_cutoff(self):
        model = ConfidenceCalibrator().fit(
            [
                ConfidenceObservation("known", 0.5, True, timestamp=10),
                ConfidenceObservation("unknown", 0.5, False, timestamp=None),
            ],
            policy=ConfidenceCalibrationPolicy(
                cutoff=20,
                min_total_observations=1,
            ),
        )

        self.assertEqual(model.unknown_time_excluded_ids, ("unknown",))
        self.assertEqual(model.included_observation_ids, ("known",))


if __name__ == "__main__":
    unittest.main()
