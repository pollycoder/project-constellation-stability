import unittest

from project_constellation_stability.plotting import build_radius_timeseries


class TestPlotting(unittest.TestCase):
    def test_build_radius_timeseries(self) -> None:
        points = build_radius_timeseries(
            observed_radii=[7000.0, 7000.2, 6999.9],
            start_time=10.0,
            step=0.5,
        )
        self.assertEqual(points, [(10.0, 7000.0), (10.5, 7000.2), (11.0, 6999.9)])

    def test_empty_series_raises(self) -> None:
        with self.assertRaises(ValueError):
            build_radius_timeseries([])

    def test_non_positive_step_raises(self) -> None:
        with self.assertRaises(ValueError):
            build_radius_timeseries([7000.0], step=0)


if __name__ == "__main__":
    unittest.main()
