"""
Insulin action curve models.

Based on LoopKit's InsulinMath.swift and insulin model implementations.
"""

import numpy as np
from typing import Optional
from enum import Enum


class InsulinModelType(Enum):
    """Types of insulin action models."""
    RAPID_ACTING_ADULT = "rapid_acting_adult"  # Humalog/Novolog/Apidra (adult curve)
    RAPID_ACTING_CHILD = "rapid_acting_child"  # Child curve (faster)
    FIASP = "fiasp"  # Ultra-rapid Fiasp
    LYUMJEV = "lyumjev"  # Ultra-rapid Lyumjev
    EXPONENTIAL = "exponential"  # Custom exponential with DIA


class InsulinModel:
    """Base class for insulin action models."""

    def __init__(self, action_duration: float):
        """
        Initialize insulin model.

        Args:
            action_duration: Duration of insulin action in hours
        """
        self.action_duration = action_duration

    def percent_activity_at_time(self, time_hours: float) -> float:
        """
        Calculate insulin activity as a percentage at a given time.

        Args:
            time_hours: Time since dose in hours

        Returns:
            Activity percentage (0-1)
        """
        raise NotImplementedError

    def percent_absorbed_at_time(self, time_hours: float) -> float:
        """
        Calculate cumulative insulin absorption percentage at a given time.

        Args:
            time_hours: Time since dose in hours

        Returns:
            Absorption percentage (0-1)
        """
        raise NotImplementedError


class ExponentialInsulinModel(InsulinModel):
    """
    Exponential insulin action curve.

    This is the model used by Loop. It's based on an exponential decay
    with a peak time that's a function of the duration of insulin action.
    """

    def __init__(self, action_duration: float):
        """
        Initialize exponential insulin model.

        Args:
            action_duration: Duration of insulin action (DIA) in hours
        """
        super().__init__(action_duration)
        # Peak time is approximately 75 minutes for typical rapid-acting insulin
        # For DIA of 6 hours, this gives peak around 1.25 hours
        self.peak_time = action_duration / 4.8

    def percent_activity_at_time(self, time_hours: float) -> float:
        """
        Calculate insulin activity using exponential curve.

        The activity curve is:
        - 0 before time 0
        - Rises to peak at peak_time
        - Decays exponentially to 0 at action_duration

        Args:
            time_hours: Time since dose in hours

        Returns:
            Activity percentage (0-1)
        """
        if time_hours < 0:
            return 0.0
        if time_hours >= self.action_duration:
            return 0.0

        tau = self.peak_time * (1 - self.peak_time / self.action_duration) / (1 - 2 * self.peak_time / self.action_duration)
        a = 2 * tau / self.action_duration
        S = 1 / (1 - a + (1 + a) * np.exp(-self.action_duration / tau))

        activity = S * (1 - time_hours / self.action_duration) * np.exp(-time_hours / tau)
        return max(0.0, activity)

    def percent_absorbed_at_time(self, time_hours: float) -> float:
        """
        Calculate cumulative insulin absorption (IOB depletion).

        This is the integral of the activity curve from 0 to time_hours.

        Args:
            time_hours: Time since dose in hours

        Returns:
            Percentage absorbed (0-1)
        """
        if time_hours <= 0:
            return 0.0
        if time_hours >= self.action_duration:
            return 1.0

        tau = self.peak_time * (1 - self.peak_time / self.action_duration) / (1 - 2 * self.peak_time / self.action_duration)
        a = 2 * tau / self.action_duration
        S = 1 / (1 - a + (1 + a) * np.exp(-self.action_duration / tau))

        # Integral of activity curve
        absorbed = 1 - S * (1 - time_hours / self.action_duration) * np.exp(-time_hours / tau) * \
                   (1 + time_hours / tau) / (1 + self.action_duration / tau)

        return max(0.0, min(1.0, absorbed))


class WalshInsulinModel(InsulinModel):
    """
    Walsh insulin curve (piecewise linear approximation).

    Based on "Using Insulin" by John Walsh.
    This is a simpler model used in some older pump algorithms.
    """

    def __init__(self, action_duration: float):
        """
        Initialize Walsh insulin model.

        Args:
            action_duration: Duration of insulin action in hours
        """
        super().__init__(action_duration)

    def percent_activity_at_time(self, time_hours: float) -> float:
        """Calculate activity using Walsh piecewise linear curve."""
        if time_hours < 0 or time_hours >= self.action_duration:
            return 0.0

        # Walsh curve peaks at 75 minutes (1.25 hours) for most rapid-acting insulin
        peak_time = 1.25
        peak_activity = 1.0

        if time_hours <= peak_time:
            # Rising phase
            return peak_activity * (time_hours / peak_time)
        else:
            # Falling phase
            return peak_activity * (self.action_duration - time_hours) / (self.action_duration - peak_time)

    def percent_absorbed_at_time(self, time_hours: float) -> float:
        """Calculate absorption using Walsh curve."""
        if time_hours <= 0:
            return 0.0
        if time_hours >= self.action_duration:
            return 1.0

        peak_time = 1.25

        if time_hours <= peak_time:
            # Area under rising triangle
            return 0.5 * (time_hours / peak_time) ** 2
        else:
            # Area under rising triangle + area under falling triangle
            rising_area = 0.5
            time_in_fall = time_hours - peak_time
            fall_duration = self.action_duration - peak_time
            fall_progress = time_in_fall / fall_duration
            falling_area = 0.5 * (1 - (1 - fall_progress) ** 2)
            return rising_area + falling_area


def get_insulin_model(model_type: InsulinModelType, action_duration: float = 6.0) -> InsulinModel:
    """
    Factory function to get an insulin model.

    Args:
        model_type: Type of insulin model
        action_duration: Duration of insulin action in hours

    Returns:
        InsulinModel instance
    """
    if model_type in [InsulinModelType.RAPID_ACTING_ADULT, InsulinModelType.RAPID_ACTING_CHILD,
                      InsulinModelType.FIASP, InsulinModelType.LYUMJEV, InsulinModelType.EXPONENTIAL]:
        return ExponentialInsulinModel(action_duration)
    else:
        raise ValueError(f"Unknown insulin model type: {model_type}")
