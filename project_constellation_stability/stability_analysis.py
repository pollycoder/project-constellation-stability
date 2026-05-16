"""Stability analysis utilities for constellation coursework."""

from dataclasses import dataclass
from typing import Iterable, List


@dataclass(frozen=True)
class StabilityResult:
    """Summary of stability analysis."""

    baseline_radius: float
    max_deviation: float
    mean_deviation: float
    is_stable: bool


def analyze_constellation_stability(
    nominal_radius: float,
    observed_radii: Iterable[float],
    tolerance: float,
) -> StabilityResult:
    """Analyze constellation stability by deviation from nominal orbit radius."""
    if nominal_radius <= 0:
        raise ValueError("nominal_radius must be positive")
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")

    radii: List[float] = [float(value) for value in observed_radii]
    if not radii:
        raise ValueError("observed_radii must not be empty")

    deviations = [abs(radius - nominal_radius) for radius in radii]
    max_deviation = max(deviations)
    mean_deviation = sum(deviations) / len(deviations)

    return StabilityResult(
        baseline_radius=nominal_radius,
        max_deviation=max_deviation,
        mean_deviation=mean_deviation,
        is_stable=max_deviation <= tolerance,
    )
