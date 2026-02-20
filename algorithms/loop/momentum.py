"""
Glucose momentum calculation using linear regression.

Based on LoopKit's GlucoseMath.swift linearMomentumEffect() function (lines 84-128).
Predicts short-term glucose trend based on recent CGM velocity.

All timestamps passed to this module should be in MINUTES (consistent with the
rest of the Python Loop codebase). The caller is responsible for converting
unix timestamps to minutes.
"""

from typing import List, Tuple
import numpy as np


class MomentumCalculator:
    """
    Calculate glucose momentum effect from recent CGM trend.

    Momentum captures the short-term velocity of glucose changes and projects
    it forward. This helps Loop anticipate continued rises or falls.

    All time parameters are in MINUTES to match the rest of Python Loop.
    """

    # Constants matching GlucoseMath.swift (converted to minutes)
    MOMENTUM_DATA_INTERVAL = 15.0   # 15 minutes (lookback window)
    MOMENTUM_DURATION = 15.0        # 15 minutes (projection duration)
    DEFAULT_DELTA = 5.0             # 5 minutes (time step)
    VELOCITY_MAX = 4.0              # 4 mg/dL per minute

    def __init__(self,
                 momentum_duration_minutes: float = None,
                 velocity_max_mg_dL_per_min: float = None,
                 min_samples: int = 3,
                 max_data_interval_minutes: float = None,
                 momentum_data_interval_minutes: float = None):
        self.momentum_duration = momentum_duration_minutes or self.MOMENTUM_DURATION
        self.velocity_max = velocity_max_mg_dL_per_min or self.VELOCITY_MAX
        self.min_samples = min_samples
        self.max_interval = max_data_interval_minutes or 5.5  # 5.5 min gap tolerance
        self.data_interval = momentum_data_interval_minutes or self.MOMENTUM_DATA_INTERVAL
        self.delta = self.DEFAULT_DELTA

    def is_continuous(self, samples: List[Tuple[float, float]]) -> bool:
        """Check if glucose samples are continuous (no large gaps)."""
        if len(samples) < 2:
            return True
        for i in range(1, len(samples)):
            time_gap = samples[i][0] - samples[i-1][0]
            if time_gap > self.max_interval:
                return False
        return True

    def linear_regression(self, points: List[Tuple[float, float]]) -> Tuple[float, float]:
        """
        Calculate linear regression slope and intercept.

        Returns:
            Tuple of (slope, intercept) where slope is in mg/dL per MINUTE
        """
        if len(points) < 2:
            return 0.0, 0.0

        times = np.array([t for t, _ in points])
        glucoses = np.array([g for _, g in points])

        coeffs = np.polyfit(times, glucoses, deg=1)
        slope = coeffs[0]   # mg/dL per minute
        intercept = coeffs[1]

        return slope, intercept

    def calculate_momentum_effect(self,
                                  glucose_samples: List[Tuple[float, float]],
                                  current_time: float,
                                  time_step: float = None) -> List[Tuple[float, float]]:
        """
        Calculate linear momentum effect from recent glucose trend.

        Follows LoopKit's linearMomentumEffect() exactly:
        1. Filter to samples within momentumDataInterval (15 min)
        2. Validate: min 3 samples, continuous (no gaps > ~5.5 min)
        3. Linear regression -> slope (mg/dL per minute)
        4. Cap at velocity maximum (4 mg/dL/min)
        5. Project forward: effect(t) = max(0, t - lastSample.time) * slope

        Args:
            glucose_samples: [(time_minutes, glucose_mg_dL), ...]
            current_time: Current time in minutes
            time_step: Minutes between projection points (default 5)

        Returns:
            List of (time_minutes, cumulative_effect_mg_dL) tuples.
            Empty list if insufficient/invalid data.
        """
        if time_step is None:
            time_step = self.delta

        # Validate sample count
        if len(glucose_samples) < self.min_samples:
            return []

        # Filter to recent samples within data interval (last 15 min)
        recent_samples = [
            (t, g) for t, g in glucose_samples
            if current_time - self.data_interval <= t <= current_time
        ]

        if len(recent_samples) < self.min_samples:
            return []

        recent_samples = sorted(recent_samples, key=lambda x: x[0])

        # Check continuity
        if not self.is_continuous(recent_samples):
            return []

        # Linear regression: slope in mg/dL per minute
        slope, intercept = self.linear_regression(recent_samples)

        # Cap velocity at maximum (4 mg/dL/min)
        limited_slope = slope
        if abs(limited_slope) > self.velocity_max:
            limited_slope = np.sign(limited_slope) * self.velocity_max

        # Project forward from last sample time
        # LoopKit formula: effect = max(0, date.timeIntervalSince(lastSample.startDate)) * limitedSlope
        last_sample_time = recent_samples[-1][0]

        # Generate effects at regular intervals from last sample time
        start_date = last_sample_time
        end_date = last_sample_time + self.momentum_duration + time_step

        momentum_effects = []
        date = start_date
        while date <= end_date:
            elapsed = max(0, date - last_sample_time)
            effect = elapsed * limited_slope
            momentum_effects.append((date, effect))
            date += time_step

        return momentum_effects

    def calculate(self,
                 glucose_samples: List[Tuple[float, float]],
                 current_time: float) -> List[Tuple[float, float]]:
        """Convenience method matching the interface expected by loop_algorithm.py."""
        return self.calculate_momentum_effect(glucose_samples, current_time)


def create_momentum_calculator(**kwargs) -> MomentumCalculator:
    """Factory function to create a MomentumCalculator with default settings."""
    return MomentumCalculator(**kwargs)
