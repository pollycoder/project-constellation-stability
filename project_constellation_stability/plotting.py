"""Plotting helpers for constellation stability coursework."""

from typing import Iterable, List, Tuple


def build_radius_timeseries(
    observed_radii: Iterable[float],
    start_time: float = 0.0,
    step: float = 1.0,
) -> List[Tuple[float, float]]:
    """Build (time, radius) pairs for plotting radius changes."""
    if step <= 0:
        raise ValueError("step must be positive")

    series: List[Tuple[float, float]] = []
    for index, radius in enumerate(observed_radii):
        series.append((start_time + index * step, float(radius)))

    if not series:
        raise ValueError("observed_radii must not be empty")

    return series
