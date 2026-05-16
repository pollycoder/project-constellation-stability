import unittest

from project_constellation_stability.stability_analysis import (
    analyze_constellation_stability,
)


class TestStabilityAnalysis(unittest.TestCase):
    def test_analyze_stable_case(self) -> None:
        result = analyze_constellation_stability(
            nominal_radius=7000.0,
            observed_radii=[7000.1, 6999.9, 7000.2],
            tolerance=0.25,
        )

        self.assertTrue(result.is_stable)
        self.assertAlmostEqual(result.max_deviation, 0.2, places=6)
        self.assertAlmostEqual(result.mean_deviation, 0.1333333333, places=6)

    def test_analyze_unstable_case(self) -> None:
        result = analyze_constellation_stability(
            nominal_radius=7000.0,
            observed_radii=[7000.0, 7000.8],
            tolerance=0.5,
        )

        self.assertFalse(result.is_stable)
        self.assertAlmostEqual(result.max_deviation, 0.8, places=6)

    def test_empty_observed_data_raises(self) -> None:
        with self.assertRaises(ValueError):
            analyze_constellation_stability(
                nominal_radius=7000.0,
                observed_radii=[],
                tolerance=0.2,
            )


if __name__ == "__main__":
    unittest.main()
