from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np


APP_DIRECTORY = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP_DIRECTORY))

from model import (  # noqa: E402
    compute_spatial_spectrum,
    continuous_measurement_noise_intensity,
    discrete_measurement_noise,
    minimum_sensor_count,
    steady_state_clarity_lower_bound,
)


class PaperRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.spectrum = compute_spatial_spectrum(5.0, 0.5, 1.0, 2.0)

    def clarity(self, sensor_count: int) -> float:
        return steady_state_clarity_lower_bound(
            sensor_count,
            0.2,
            60.0,
            2.0,
            self.spectrum.eigenvalues,
        )

    def test_default_matches_paper_bound(self) -> None:
        self.assertAlmostEqual(self.clarity(7), 0.711807281080, places=10)

    def test_table_3_minimum_sensor_counts(self) -> None:
        expected = {0.5: 1, 0.6: 3, 0.7: 7, 0.8: 21, 0.9: 112}
        for target, expected_count in expected.items():
            with self.subTest(target=target):
                result = minimum_sensor_count(
                    target,
                    500,
                    0.2,
                    60.0,
                    2.0,
                    self.spectrum.eigenvalues,
                )
                self.assertIsNotNone(result)
                self.assertEqual(result[0], expected_count)

    def test_measurement_variance_input_sets_continuous_intensity(self) -> None:
        sigma_c_squared_a = continuous_measurement_noise_intensity(10.0, 0.02)
        sigma_c_squared_b = continuous_measurement_noise_intensity(10.0, 0.05)
        self.assertAlmostEqual(sigma_c_squared_a, 0.2)
        self.assertAlmostEqual(sigma_c_squared_b, 0.5)

        variance, standard_deviation = discrete_measurement_noise(
            sigma_c_squared_a, 0.02
        )
        self.assertAlmostEqual(variance, 10.0)
        self.assertAlmostEqual(standard_deviation, math.sqrt(10.0))

        clarity_a = steady_state_clarity_lower_bound(
            7, sigma_c_squared_a, 60.0, 2.0, self.spectrum.eigenvalues
        )
        clarity_b = steady_state_clarity_lower_bound(
            7, sigma_c_squared_b, 60.0, 2.0, self.spectrum.eigenvalues
        )
        self.assertGreater(clarity_a, clarity_b)

    def test_spectrum_has_expected_shape_and_is_finite(self) -> None:
        self.assertEqual(self.spectrum.grid_points, 121)
        self.assertEqual(self.spectrum.eigenvalues.shape, (121,))
        self.assertTrue(np.all(np.isfinite(self.spectrum.eigenvalues)))
        self.assertTrue(np.all(self.spectrum.eigenvalues >= 0.0))

    def test_invalid_inputs_fail_explicitly(self) -> None:
        with self.assertRaisesRegex(ValueError, "divide"):
            compute_spatial_spectrum(5.0, 0.3, 1.0, 2.0)
        with self.assertRaisesRegex(ValueError, "greater than zero"):
            discrete_measurement_noise(0.2, 0.0)
        with self.assertRaisesRegex(ValueError, "greater than zero"):
            continuous_measurement_noise_intensity(0.0, 0.02)
        with self.assertRaisesRegex(ValueError, "positive integer"):
            steady_state_clarity_lower_bound(
                7.5, 0.2, 60.0, 2.0, self.spectrum.eigenvalues
            )
        with self.assertRaisesRegex(ValueError, "strictly between"):
            minimum_sensor_count(1.0, 500, 0.2, 60.0, 2.0, self.spectrum.eigenvalues)


if __name__ == "__main__":
    unittest.main()
