"""
Exact insulin math using LoopKit's exponential model.

Based on exact ExponentialInsulinModel.swift formulas with Fiasp as default.
"""

from typing import List, Tuple
import numpy as np
from algorithms.loop.insulin_models_exact import create_insulin_model, InsulinType


class InsulinMathExact:
    """Calculations for insulin on board and glucose effects using exact LoopKit formulas."""

    def __init__(self,
                 insulin_sensitivity: float,
                 action_duration_hours: float = 6.0,
                 insulin_type: InsulinType = InsulinType.FIASP):
        """
        Initialize exact insulin math calculator.

        Args:
            insulin_sensitivity: ISF in mg/dL per unit
            action_duration_hours: Duration of insulin action in hours
            insulin_type: Type of insulin (default Fiasp)
        """
        self.isf = insulin_sensitivity
        self.insulin_model = create_insulin_model(insulin_type, action_duration_hours)

    def calculate_iob(self,
                      bolus_history: List[Tuple[int, float]],
                      current_time: int,
                      basal_rate: float = 0.0) -> Tuple[float, float]:
        """
        Calculate insulin on board using exact LoopKit model.

        Args:
            bolus_history: List of (time_minutes, units) tuples
            current_time: Current time in minutes
            basal_rate: Current basal rate in U/hr (for basal IOB tracking)

        Returns:
            Tuple of (IOB in units, current activity estimate)
        """
        total_iob = 0.0
        total_activity = 0.0

        for dose_time, units in bolus_history:
            time_since_dose_minutes = current_time - dose_time

            if time_since_dose_minutes < 0:
                continue  # Future dose

            # Use exact LoopKit formula for percent effect remaining
            percent_remaining = self.insulin_model.percent_effect_remaining(time_since_dose_minutes)
            iob_from_dose = units * percent_remaining

            # Estimate activity (absorption rate)
            activity_from_dose = self.insulin_model.activity_at_time(time_since_dose_minutes)
            activity_in_units_per_min = units * activity_from_dose

            total_iob += iob_from_dose
            total_activity += activity_in_units_per_min

        return total_iob, total_activity

    def glucose_effect_of_insulin(self,
                                  bolus_history: List[Tuple[int, float]],
                                  start_time: int,
                                  end_time: int,
                                  time_step: int,
                                  insulin_sensitivity: float) -> List[Tuple[int, float]]:
        """
        Calculate the glucose lowering effect of insulin over time.

        Args:
            bolus_history: List of (time_minutes, units) tuples
            start_time: Start time in minutes
            end_time: End time in minutes
            time_step: Time step in minutes (typically 5)
            insulin_sensitivity: ISF in mg/dL per unit

        Returns:
            List of (time, glucose_effect) tuples where glucose_effect is the
            cumulative glucose lowering in mg/dL (negative values)
        """
        times = range(int(start_time), int(end_time) + 1, int(time_step))
        effects = []

        for current_time in times:
            total_effect = 0.0

            for dose_time, units in bolus_history:
                time_since_dose_minutes = current_time - dose_time

                if time_since_dose_minutes < 0:
                    continue  # Future dose

                # Calculate how much insulin has been absorbed
                percent_absorbed = self.insulin_model.percent_absorbed(time_since_dose_minutes)

                # Convert absorbed insulin to glucose effect
                effect_from_dose = -units * insulin_sensitivity * percent_absorbed

                total_effect += effect_from_dose

            effects.append((current_time, total_effect))

        return effects

    def predict_glucose_from_insulin(self,
                                    current_glucose: float,
                                    bolus_history: List[Tuple[int, float]],
                                    current_time: int,
                                    prediction_horizon: int,
                                    time_step: int,
                                    insulin_sensitivity: float) -> List[Tuple[int, float]]:
        """
        Predict future glucose effects based on insulin action using exact LoopKit model.

        NOTE: Returns cumulative EFFECTS (deltas from baseline), not absolute glucose values.
        This matches LoopKit's architecture where effects are combined separately.

        Args:
            current_glucose: Current glucose in mg/dL (used for display/reference only)
            bolus_history: List of (time_minutes, units) tuples
            current_time: Current time in minutes
            prediction_horizon: How far to predict in minutes
            time_step: Time step in minutes
            insulin_sensitivity: ISF in mg/dL per unit

        Returns:
            List of (time, cumulative_effect) tuples where effect is the total
            glucose lowering from insulin (negative values)
        """
        end_time = current_time + prediction_horizon
        effects = self.glucose_effect_of_insulin(
            bolus_history=bolus_history,
            start_time=current_time,
            end_time=end_time,
            time_step=time_step,
            insulin_sensitivity=insulin_sensitivity
        )

        # Get baseline effect at current time
        baseline_effects = self.glucose_effect_of_insulin(
            bolus_history=bolus_history,
            start_time=current_time,
            end_time=current_time,
            time_step=time_step,
            insulin_sensitivity=insulin_sensitivity
        )
        baseline_effect = baseline_effects[0][1] if baseline_effects else 0.0

        # Return cumulative effects (relative to baseline at current_time)
        # This matches LoopKit's GlucoseEffect format
        effect_timeline = []
        for time, effect in effects:
            # Effect is cumulative from t=0, subtract baseline to get incremental from now
            cumulative_effect = effect - baseline_effect
            effect_timeline.append((time, cumulative_effect))

        return effect_timeline


def create_exact_insulin_math(insulin_sensitivity_factor: float,
                              duration_of_insulin_action: float = 6.0,
                              insulin_type: InsulinType = InsulinType.FIASP) -> InsulinMathExact:
    """
    Factory function to create an exact InsulinMath calculator.

    Args:
        insulin_sensitivity_factor: ISF in mg/dL per unit
        duration_of_insulin_action: DIA in hours (default 6.0)
        insulin_type: Type of insulin (default Fiasp)

    Returns:
        InsulinMathExact instance
    """
    return InsulinMathExact(
        insulin_sensitivity=insulin_sensitivity_factor,
        action_duration_hours=duration_of_insulin_action,
        insulin_type=insulin_type
    )
