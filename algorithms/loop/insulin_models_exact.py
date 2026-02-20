"""
Exact insulin action curve implementation from LoopKit.

Based on LoopKit's ExponentialInsulinModel.swift with exact formulas.
"""

import numpy as np
from enum import Enum


class InsulinType(Enum):
    """Insulin types with preset parameters from LoopKit."""
    RAPID_ACTING_ADULT = "rapid_acting_adult"  # Humalog/Novolog - adult curve
    RAPID_ACTING_CHILD = "rapid_acting_child"  # Child curve (faster)
    FIASP = "fiasp"                             # Ultra-rapid Fiasp
    LYUMJEV = "lyumjev"                         # Ultra-rapid Lyumjev
    AFREZZA = "afrezza"                         # Inhaled insulin


class ExponentialInsulinModelExact:
    """
    Exact exponential insulin model from LoopKit.

    Based on ExponentialInsulinModel.swift with the precise mathematical formulas.
    All parameters are configurable for experimentation.
    """

    def __init__(self,
                 action_duration_minutes: float = 360.0,
                 peak_activity_time_minutes: float = 75.0,
                 delay_minutes: float = 10.0):
        """
        Initialize exponential insulin model with exact LoopKit parameters.

        Args:
            action_duration_minutes: Total insulin action duration (default 360 min = 6 hours)
            peak_activity_time_minutes: Time when peak activity occurs (default 75 min)
            delay_minutes: Delay before insulin effect begins (default 10 min)
        """
        self.action_duration = action_duration_minutes
        self.peak_time = peak_activity_time_minutes
        self.delay = delay_minutes

        # Precompute constants (from LoopKit ExponentialInsulinModel.swift)
        # τ (tau) = peakActivityTime × (1 - peakActivityTime/actionDuration) /
        #           (1 - 2×peakActivityTime/actionDuration)
        self.tau = (peak_activity_time_minutes *
                   (1.0 - peak_activity_time_minutes / action_duration_minutes) /
                   (1.0 - 2.0 * peak_activity_time_minutes / action_duration_minutes))

        # a = 2×τ / actionDuration
        self.a = 2.0 * self.tau / action_duration_minutes

        # S = 1 / (1 - a + (1 + a)×exp(-actionDuration/τ))
        self.S = 1.0 / (1.0 - self.a + (1.0 + self.a) * np.exp(-action_duration_minutes / self.tau))

    def percent_effect_remaining(self, time_minutes: float) -> float:
        """
        Calculate percent of insulin effect remaining (IOB fraction).

        This is the EXACT formula from LoopKit's ExponentialInsulinModel.swift.

        Args:
            time_minutes: Time since insulin dose in minutes

        Returns:
            Fraction of effect remaining (0-1), where 1.0 = 100% remaining
        """
        # Case 1: Before delay
        if time_minutes <= self.delay:
            return 1.0

        # Adjust for delay
        t = time_minutes - self.delay

        # Case 2: After full action duration
        if t >= self.action_duration:
            return 0.0

        # Case 3: During active period - use exact LoopKit formula
        # percentEffectRemaining = 1 - S×(1-a)×((t²/(τ×actionDuration×(1-a)) - t/τ - 1)×exp(-t/τ) + 1)

        exp_term = np.exp(-t / self.tau)

        inner_term = (t * t / (self.tau * self.action_duration * (1.0 - self.a)) -
                     t / self.tau -
                     1.0)

        percent_remaining = 1.0 - self.S * (1.0 - self.a) * (inner_term * exp_term + 1.0)

        return max(0.0, min(1.0, percent_remaining))

    def percent_absorbed(self, time_minutes: float) -> float:
        """
        Calculate percent of insulin absorbed (inverse of effect remaining).

        Args:
            time_minutes: Time since dose in minutes

        Returns:
            Fraction absorbed (0-1)
        """
        return 1.0 - self.percent_effect_remaining(time_minutes)

    def activity_at_time(self, time_minutes: float) -> float:
        """
        Calculate instantaneous insulin activity (derivative of absorption).

        This is approximately the rate of change of absorption.

        Args:
            time_minutes: Time since dose in minutes

        Returns:
            Activity rate (fraction per minute)
        """
        if time_minutes <= self.delay or time_minutes >= self.delay + self.action_duration:
            return 0.0

        # Calculate activity as small finite difference
        dt = 0.1  # minutes
        absorption_now = self.percent_absorbed(time_minutes)
        absorption_next = self.percent_absorbed(time_minutes + dt)

        activity = (absorption_next - absorption_now) / dt

        return max(0.0, activity)

    def get_parameters(self) -> dict:
        """Return current model parameters."""
        return {
            'action_duration_minutes': self.action_duration,
            'peak_activity_time_minutes': self.peak_time,
            'delay_minutes': self.delay,
            'tau': self.tau,
            'a': self.a,
            'S': self.S
        }


def create_insulin_model(insulin_type: InsulinType = InsulinType.FIASP,
                        action_duration_hours: float = 6.0) -> ExponentialInsulinModelExact:
    """
    Factory function to create insulin model with LoopKit presets.

    Args:
        insulin_type: Type of insulin (determines peak time)
        action_duration_hours: Duration of insulin action in hours

    Returns:
        ExponentialInsulinModelExact instance
    """
    # Presets from LoopKit's ExponentialInsulinModelPreset.swift
    peak_times = {
        InsulinType.RAPID_ACTING_ADULT: 75.0,  # minutes
        InsulinType.RAPID_ACTING_CHILD: 65.0,
        InsulinType.FIASP: 55.0,
        InsulinType.LYUMJEV: 55.0,
        InsulinType.AFREZZA: 29.0,
    }

    # Afrezza has 5-hour duration by default
    if insulin_type == InsulinType.AFREZZA and action_duration_hours == 6.0:
        action_duration_hours = 5.0

    peak_time = peak_times.get(insulin_type, 75.0)

    return ExponentialInsulinModelExact(
        action_duration_minutes=action_duration_hours * 60.0,
        peak_activity_time_minutes=peak_time,
        delay_minutes=10.0  # Always 10 minutes in Loop
    )


def create_custom_insulin_model(action_duration_hours: float = 6.0,
                                peak_time_minutes: float = 75.0,
                                delay_minutes: float = 10.0) -> ExponentialInsulinModelExact:
    """
    Create insulin model with custom parameters.

    Args:
        action_duration_hours: Duration in hours
        peak_time_minutes: Peak activity time in minutes
        delay_minutes: Delay before effect begins

    Returns:
        ExponentialInsulinModelExact instance
    """
    return ExponentialInsulinModelExact(
        action_duration_minutes=action_duration_hours * 60.0,
        peak_activity_time_minutes=peak_time_minutes,
        delay_minutes=delay_minutes
    )
