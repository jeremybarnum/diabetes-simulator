"""
Carb absorption and COB calculations.

Based on LoopKit's CarbMath.swift and carb absorption models.
"""

from typing import List, Tuple
import numpy as np
from algorithms.loop.carb_models import PiecewiseLinearCarbModel


class CarbAbsorptionModel:
    """Base class for carb absorption models."""

    def percent_absorbed_at_time(self, time_hours: float, absorption_time: float) -> float:
        """
        Calculate what percentage of carbs have been absorbed.

        Args:
            time_hours: Time since carb entry in hours
            absorption_time: Total absorption time in hours

        Returns:
            Fraction absorbed (0-1)
        """
        raise NotImplementedError


class ParabolicCarbModel(CarbAbsorptionModel):
    """
    Parabolic carb absorption model used by Loop.

    Features:
    - Absorption delay of 10 minutes (hardcoded in Loop)
    - Parabolic curve (slower at start/end, faster in middle)
    - More realistic than linear
    """

    ABSORPTION_DELAY_MINUTES = 10  # Loop hardcoded delay

    def percent_absorbed_at_time(self, time_hours: float, absorption_time: float) -> float:
        """
        Calculate absorption using parabolic model.

        Matches iOS Loop's CarbMath.swift ParabolicAbsorption:
          percentAbsorptionAtPercentTime(percentTime) where percentTime = time/absorptionTime

        NOTE: Delay is applied EXTERNALLY by the calling code (glucose_effect_of_carbs).
        This method receives time AFTER delay has been subtracted.
        DO NOT subtract delay again internally.

        The parabolic model from iOS Loop (Scheiner GI curve approximation):
        - For 0 ≤ t ≤ 0.5: absorbed = 2 * t²
        - For 0.5 < t < 1.0: absorbed = -1 + 2*t*(2-t)
        - For t ≥ 1.0: absorbed = 1.0

        Args:
            time_hours: Effective time (already delay-subtracted) in hours
            absorption_time: Total absorption time in hours

        Returns:
            Fraction absorbed (0-1)
        """
        if time_hours <= 0:
            return 0.0

        if time_hours >= absorption_time:
            return 1.0

        # Normalize time: iOS uses percentTime = time / absorptionTime
        t_normalized = time_hours / absorption_time

        # Parabolic absorption (from CarbMath.swift ParabolicAbsorption)
        if t_normalized <= 0.5:
            absorbed = 2.0 * t_normalized * t_normalized
        else:
            absorbed = -1.0 + 2.0 * t_normalized * (2.0 - t_normalized)

        return max(0.0, min(1.0, absorbed))

    # DCA-compatible interface methods (matching PiecewiseLinearCarbModel API)
    def absorptionRateAtTime(self, t: float, absorptionTime: float) -> float:
        """Absorption rate at time t (fraction per hour). t and absorptionTime in hours."""
        return self.absorption_rate_at_time(t, absorptionTime, 1.0)

    def absorbedCarbs(self, of: float, atTime: float, absorptionTime: float) -> float:
        """Grams absorbed at time (hours). of = total grams."""
        return of * self.percent_absorbed_at_time(atTime, absorptionTime)

    def unabsorbedCarbs(self, of: float, atTime: float, absorptionTime: float) -> float:
        """Grams remaining at time (hours). of = total grams."""
        return of * (1.0 - self.percent_absorbed_at_time(atTime, absorptionTime))

    def timeToAbsorb(self, forPercentAbsorbed: float, totalAbsorptionTime: float) -> float:
        """Inverse of percent_absorbed_at_time. Returns time in hours."""
        if forPercentAbsorbed <= 0:
            return 0.0
        if forPercentAbsorbed >= 1.0:
            return totalAbsorptionTime
        low, high = 0.0, totalAbsorptionTime
        for _ in range(50):
            mid = (low + high) / 2
            if self.percent_absorbed_at_time(mid, totalAbsorptionTime) < forPercentAbsorbed:
                low = mid
            else:
                high = mid
        return (low + high) / 2

    def absorption_rate_at_time(self, time_hours: float, absorption_time: float,
                                total_carbs: float) -> float:
        """
        Calculate the instantaneous absorption rate (g/hr).

        Derivative of parabolic absorption curve.
        NOTE: time_hours should already have delay subtracted by caller.

        Args:
            time_hours: Effective time (already delay-subtracted) in hours
            absorption_time: Total absorption time in hours
            total_carbs: Total grams of carbs

        Returns:
            Absorption rate in grams/hour
        """
        if time_hours <= 0 or time_hours >= absorption_time:
            return 0.0

        t_normalized = time_hours / absorption_time

        # Derivative of parabolic model
        if t_normalized <= 0.5:
            # d/dt[2*t^2] = 4*t, but need to account for normalization
            rate_normalized = 4.0 * t_normalized
        else:
            # d/dt[-1 + 2*t*(2-t)] = d/dt[-1 + 4t - 2t^2] = 4 - 4t
            rate_normalized = 4.0 * (1.0 - t_normalized)

        # Convert from per-normalized-time to per-hour
        return total_carbs * rate_normalized / absorption_time


class CarbMath:
    """Calculations for carbs on board and glucose effects."""

    def __init__(self, carb_ratio: float, insulin_sensitivity: float,
                 model: CarbAbsorptionModel = None):
        """
        Initialize carb math calculator.

        Args:
            carb_ratio: Carb ratio in g/U
            insulin_sensitivity: ISF in mg/dL per unit
            model: Carb absorption model (defaults to parabolic)
        """
        self.carb_ratio = carb_ratio
        self.isf = insulin_sensitivity
        self.model = model if model is not None else PiecewiseLinearCarbModel()

    def calculate_cob(self,
                     carb_entries: List[Tuple[int, float, float]],
                     current_time: int) -> float:
        """
        Calculate carbs on board.

        Args:
            carb_entries: List of (time_minutes, grams, absorption_hours)
            current_time: Current time in minutes

        Returns:
            COB in grams
        """
        total_cob = 0.0

        for entry_time, grams, absorption_hours in carb_entries:
            time_since_entry_minutes = current_time - entry_time

            if time_since_entry_minutes < 0:
                # Future entry
                continue

            time_since_entry_hours = time_since_entry_minutes / 60.0

            # Apply delay OUTSIDE the model (like Loop does)
            delay_hours = 10.0 / 60.0
            effective_time_hours = max(0.0, time_since_entry_hours - delay_hours)

            if effective_time_hours >= absorption_hours:
                # Fully absorbed
                continue

            # Calculate what's been absorbed
            absorbed_fraction = self.model.percent_absorbed_at_time(
                effective_time_hours, absorption_hours
            )

            # COB is what remains
            remaining_fraction = 1.0 - absorbed_fraction
            cob_from_entry = grams * remaining_fraction

            total_cob += cob_from_entry

        return total_cob

    def glucose_effect_of_carbs(self,
                               carb_entries: List[Tuple[int, float, float]],
                               start_time: int,
                               end_time: int,
                               time_step: int) -> List[Tuple[int, float]]:
        """
        Calculate the glucose raising effect of carbs over time.

        Args:
            carb_entries: List of (time_minutes, grams, absorption_hours)
            start_time: Start time in minutes
            end_time: End time in minutes
            time_step: Time step in minutes

        Returns:
            List of (time, glucose_effect) where effect is cumulative BG rise in mg/dL
        """
        times = range(int(start_time), int(end_time) + 1, int(time_step))
        effects = []

        for current_time in times:
            total_effect = 0.0

            for entry_time, grams, absorption_hours in carb_entries:
                time_since_entry_minutes = current_time - entry_time

                if time_since_entry_minutes < 0:
                    # Future entry
                    continue

                time_since_entry_hours = time_since_entry_minutes / 60.0

                # Apply delay OUTSIDE the model (like Loop does: time - delay)
                # Default delay is 10 minutes for carbs
                delay_hours = 10.0 / 60.0  # CarbMath.defaultEffectDelay
                effective_time_hours = max(0.0, time_since_entry_hours - delay_hours)

                if effective_time_hours >= absorption_hours:
                    # Use full absorption
                    effective_time_hours = absorption_hours

                # Calculate absorbed fraction
                # Pass the time AFTER applying delay (Loop does: absorbedCarbs(atTime: time - delay))
                absorbed_fraction = self.model.percent_absorbed_at_time(
                    effective_time_hours, absorption_hours
                )

                # Convert absorbed carbs to glucose effect
                # absorbed carbs -> insulin needed -> glucose effect
                absorbed_carbs = grams * absorbed_fraction
                insulin_equivalent = absorbed_carbs / self.carb_ratio
                effect_from_entry = insulin_equivalent * self.isf

                total_effect += effect_from_entry

            effects.append((current_time, total_effect))

        return effects

    def predict_glucose_from_carbs(self,
                                  current_glucose: float,
                                  carb_entries: List[Tuple[int, float, float]],
                                  current_time: int,
                                  prediction_horizon: int,
                                  time_step: int) -> List[Tuple[int, float]]:
        """
        Predict future glucose effects based on carb absorption.

        NOTE: Returns cumulative EFFECTS (deltas from baseline), not absolute glucose values.
        This matches LoopKit's architecture where effects are combined separately.

        Args:
            current_glucose: Current glucose in mg/dL (used for display/reference only)
            carb_entries: List of (time_minutes, grams, absorption_hours)
            current_time: Current time in minutes
            prediction_horizon: How far to predict in minutes
            time_step: Time step in minutes

        Returns:
            List of (time, cumulative_effect) tuples where effect is the total
            glucose raising from carbs (positive values)
        """
        end_time = current_time + prediction_horizon
        effects = self.glucose_effect_of_carbs(
            carb_entries=carb_entries,
            start_time=current_time,
            end_time=end_time,
            time_step=time_step
        )

        # Get baseline effect at current time
        baseline_effects = self.glucose_effect_of_carbs(
            carb_entries=carb_entries,
            start_time=current_time,
            end_time=current_time,
            time_step=time_step
        )
        baseline_effect = baseline_effects[0][1] if baseline_effects else 0.0

        # Return cumulative effects (relative to baseline at current_time)
        # This matches LoopKit's GlucoseEffect format
        effect_timeline = []
        for time, effect in effects:
            # Effect is cumulative from entry time, subtract baseline to get incremental from now
            cumulative_effect = effect - baseline_effect
            effect_timeline.append((time, cumulative_effect))

        return effect_timeline


def create_carb_math(carb_ratio: float, insulin_sensitivity: float,
                    delay_minutes: float = 0.0,
                    percent_end_of_rise: float = 0.15,
                    percent_start_of_fall: float = 0.5) -> CarbMath:
    """
    Factory function to create a CarbMath calculator with Loop's piecewise linear model.

    IMPORTANT: delay_minutes should be 0.0 because Loop applies the 10-minute delay
    EXTERNALLY in carb_math methods (via time - delay before calling model).

    Args:
        carb_ratio: Carb ratio in g/U
        insulin_sensitivity: ISF in mg/dL per unit
        delay_minutes: Should be 0.0 (delay applied externally in carb_math methods)
        percent_end_of_rise: When peak rate is reached (default 0.15)
        percent_start_of_fall: When decline begins (default 0.5)

    Returns:
        CarbMath instance
    """
    # PiecewiseLinear model - matching iOS Loop's PiecewiseLinearAbsorption
    model = PiecewiseLinearCarbModel(
        delay_minutes=delay_minutes,
        percent_end_of_rise=percent_end_of_rise,
        percent_start_of_fall=percent_start_of_fall
    )
    return CarbMath(
        carb_ratio=carb_ratio,
        insulin_sensitivity=insulin_sensitivity,
        model=model
    )
