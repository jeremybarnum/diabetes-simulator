"""
Carb absorption models - exact implementation from LoopKit.

Based on LoopKit's CarbMath.swift piecewise linear absorption model.
"""

from typing import Tuple
import numpy as np


class CarbAbsorptionModel:
    """Base class for carb absorption models."""

    def percent_absorbed_at_time(self, time_hours: float, absorption_time_hours: float) -> float:
        """Calculate percentage absorbed at given time."""
        raise NotImplementedError


class PiecewiseLinearCarbModel(CarbAbsorptionModel):
    """
    Piecewise linear carb absorption model from LoopKit.

    Three-phase absorption:
    1. Rise phase: Rate increases linearly from 0 to max
    2. Plateau phase: Rate stays at max (constant)
    3. Fall phase: Rate decreases linearly from max to 0

    All parameters are configurable for experimentation.
    """

    def __init__(self,
                 delay_minutes: float = 0.0,  # Delay is applied OUTSIDE model in Loop!
                 percent_end_of_rise: float = 0.15,
                 percent_start_of_fall: float = 0.5):
        """
        Initialize piecewise linear carb model.

        IMPORTANT: Loop applies the delay OUTSIDE the model (time - delay before calling).
        The delay parameter here should be 0.0 to match Loop's behavior.
        Apply delay in the calling code instead!

        Args:
            delay_minutes: Should be 0.0 (delay applied externally, not in model)
            percent_end_of_rise: Fraction of absorption time when peak rate reached (default 0.15)
            percent_start_of_fall: Fraction of absorption time when decline begins (default 0.5)
        """
        self.delay_minutes = delay_minutes
        self.percent_end_of_rise = percent_end_of_rise
        self.percent_start_of_fall = percent_start_of_fall

        # Calculate scale factor to ensure 100% absorption
        # From LoopKit: scale = 2.0 / (1.0 + percentStartOfFall - percentEndOfRise)
        self.scale = 2.0 / (1.0 + percent_start_of_fall - percent_end_of_rise)

    def absorption_rate_at_time(self, time_minutes: float, absorption_time_hours: float) -> float:
        """
        Calculate the instantaneous absorption rate (fraction per hour).

        This is the derivative of the absorption curve.

        Args:
            time_minutes: Time since carb entry in minutes
            absorption_time_hours: Total absorption time in hours

        Returns:
            Absorption rate as fraction per hour
        """
        if time_minutes < 0:
            return 0.0

        # Check if still in delay period
        if time_minutes < self.delay_minutes:
            return 0.0

        # Adjust for delay
        # The delay is INCLUDED in the absorption time, not added to it
        effective_time_minutes = time_minutes - self.delay_minutes
        total_absorption_minutes = absorption_time_hours * 60.0
        active_absorption_minutes = total_absorption_minutes - self.delay_minutes

        if effective_time_minutes >= active_absorption_minutes:
            return 0.0

        # Calculate progress through absorption (0 to 1)
        progress = effective_time_minutes / active_absorption_minutes

        # Determine which phase we're in
        if progress <= self.percent_end_of_rise:
            # Rise phase: linear ramp up
            # Rate goes from 0 to scale over this period
            phase_progress = progress / self.percent_end_of_rise
            rate_fraction_per_min = self.scale * phase_progress / active_absorption_minutes

        elif progress <= self.percent_start_of_fall:
            # Plateau phase: constant maximum rate
            rate_fraction_per_min = self.scale / active_absorption_minutes

        else:
            # Fall phase: linear ramp down
            # Rate goes from scale to 0 over this period
            phase_start = self.percent_start_of_fall
            phase_end = 1.0
            phase_duration = phase_end - phase_start
            phase_progress = (progress - phase_start) / phase_duration
            rate_fraction_per_min = self.scale * (1.0 - phase_progress) / active_absorption_minutes

        # Convert from per-minute to per-hour
        return rate_fraction_per_min * 60.0

    def percent_absorbed_at_time(self, time_hours: float, absorption_time_hours: float) -> float:
        """
        Calculate cumulative absorption percentage at given time.

        This is the integral of the absorption rate.

        Args:
            time_hours: Time since carb entry in hours
            absorption_time_hours: Total absorption time in hours

        Returns:
            Fraction absorbed (0-1)
        """
        time_minutes = time_hours * 60.0

        if time_minutes < 0:
            return 0.0

        # Check if still in delay period
        if time_minutes < self.delay_minutes:
            return 0.0

        # Adjust for delay
        # The delay is INCLUDED in the absorption time, not added to it
        effective_time_minutes = time_minutes - self.delay_minutes
        total_absorption_minutes = absorption_time_hours * 60.0
        active_absorption_minutes = total_absorption_minutes - self.delay_minutes

        if effective_time_minutes >= active_absorption_minutes:
            return 1.0

        # Calculate progress through absorption (0 to 1)
        progress = effective_time_minutes / active_absorption_minutes

        # Calculate absorbed amount based on phase
        # These are the areas under the rate curve

        rise_duration = self.percent_end_of_rise
        plateau_duration = self.percent_start_of_fall - self.percent_end_of_rise
        fall_duration = 1.0 - self.percent_start_of_fall

        if progress <= self.percent_end_of_rise:
            # In rise phase: area of triangle
            # Area = 0.5 * base * height
            # base = progress, height = scale * (progress/rise_duration)
            phase_progress = progress / rise_duration
            absorbed = 0.5 * progress * self.scale * phase_progress

        elif progress <= self.percent_start_of_fall:
            # In plateau phase: area of triangle + area of rectangle
            # Triangle area from rise phase
            rise_area = 0.5 * rise_duration * self.scale

            # Rectangle area in plateau
            plateau_progress = progress - self.percent_end_of_rise
            plateau_area = plateau_progress * self.scale

            absorbed = rise_area + plateau_area

        else:
            # In fall phase: all previous area + area of trapezoid
            # Triangle from rise
            rise_area = 0.5 * rise_duration * self.scale

            # Rectangle from plateau
            plateau_area = plateau_duration * self.scale

            # Trapezoid from fall phase
            # Area = 0.5 * (top + bottom) * height
            fall_progress = (progress - self.percent_start_of_fall) / fall_duration
            fall_height = progress - self.percent_start_of_fall
            fall_top = self.scale
            fall_bottom = self.scale * (1.0 - fall_progress)
            fall_area = 0.5 * (fall_top + fall_bottom) * fall_height

            absorbed = rise_area + plateau_area + fall_area

        return max(0.0, min(1.0, absorbed))

    def absorbedCarbs(self, of: float, atTime: float, absorptionTime: float) -> float:
        """
        Calculate grams absorbed at a given time.

        Args:
            of: Total grams of carbs
            atTime: Time since entry in hours
            absorptionTime: Total absorption time in hours

        Returns:
            Grams absorbed
        """
        percent = self.percent_absorbed_at_time(atTime, absorptionTime)
        return of * percent

    def unabsorbedCarbs(self, of: float, atTime: float, absorptionTime: float) -> float:
        """
        Calculate grams remaining (not yet absorbed) at a given time.

        Args:
            of: Total grams of carbs
            atTime: Time since entry in hours
            absorptionTime: Total absorption time in hours

        Returns:
            Grams remaining
        """
        percent_absorbed = self.percent_absorbed_at_time(atTime, absorptionTime)
        return of * (1.0 - percent_absorbed)

    def timeToAbsorb(self, forPercentAbsorbed: float, totalAbsorptionTime: float) -> float:
        """
        Calculate time needed to absorb a given percentage.

        This is the inverse of percent_absorbed_at_time.
        Uses binary search for now (analytical inverse needs debugging).

        Args:
            forPercentAbsorbed: Target percentage (0-1)
            totalAbsorptionTime: Total absorption time in hours

        Returns:
            Time in hours needed to reach target percentage
        """
        if forPercentAbsorbed <= 0:
            return 0.0
        if forPercentAbsorbed >= 1.0:
            return totalAbsorptionTime

        # Binary search to find the time
        low, high = 0.0, totalAbsorptionTime
        tolerance = 0.0001  # hours (~0.36 seconds) - tighter tolerance

        while high - low > tolerance:
            mid = (low + high) / 2.0
            percent = self.percent_absorbed_at_time(mid, totalAbsorptionTime)

            if percent < forPercentAbsorbed:
                low = mid
            else:
                high = mid

        return (low + high) / 2.0

    def absorptionRateAtTime(self, t: float, absorptionTime: float) -> float:
        """
        Calculate absorption rate at time t (for DCA).

        Args:
            t: Time since entry in hours
            absorptionTime: Total absorption time in hours

        Returns:
            Absorption rate as fraction per hour
        """
        return self.absorption_rate_at_time(t * 60.0, absorptionTime)

    def get_parameters(self) -> dict:
        """Return current model parameters."""
        return {
            'delay_minutes': self.delay_minutes,
            'percent_end_of_rise': self.percent_end_of_rise,
            'percent_start_of_fall': self.percent_start_of_fall,
            'scale': self.scale
        }


def create_loop_carb_model(delay_minutes: float = 0.0,
                           percent_end_of_rise: float = 0.15,
                           percent_start_of_fall: float = 0.5) -> PiecewiseLinearCarbModel:
    """
    Factory function to create Loop's piecewise linear carb model.

    IMPORTANT: delay_minutes should be 0.0 because Loop applies delay externally.
    The delay is applied in carb_math.py when calling the model.

    Args:
        delay_minutes: Should be 0.0 (delay applied externally)
        percent_end_of_rise: When peak rate is reached (default 0.15 = 15%)
        percent_start_of_fall: When decline begins (default 0.5 = 50%)

    Returns:
        PiecewiseLinearCarbModel instance
    """
    return PiecewiseLinearCarbModel(
        delay_minutes=delay_minutes,
        percent_end_of_rise=percent_end_of_rise,
        percent_start_of_fall=percent_start_of_fall
    )
