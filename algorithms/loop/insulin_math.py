"""
Insulin on Board (IOB) calculations.

Based on LoopKit's InsulinMath.swift.
"""

from typing import List, Tuple
import numpy as np
from algorithms.loop.insulin_models import InsulinModel, ExponentialInsulinModel


class InsulinMath:
    """Calculations for insulin on board and insulin activity."""

    def __init__(self, insulin_model: InsulinModel):
        """
        Initialize insulin math calculator.

        Args:
            insulin_model: The insulin action model to use
        """
        self.insulin_model = insulin_model

    def calculate_iob(self,
                      bolus_history: List[Tuple[int, float]],
                      current_time: int,
                      basal_rate: float = 0.0) -> Tuple[float, float]:
        """
        Calculate insulin on board and current insulin activity.

        Args:
            bolus_history: List of (time_minutes, units) tuples
            current_time: Current time in minutes
            basal_rate: Current basal rate in U/hr (for basal IOB tracking)

        Returns:
            Tuple of (IOB in units, current activity in U/min)
        """
        total_iob = 0.0
        total_activity = 0.0

        for dose_time, units in bolus_history:
            time_since_dose_hours = (current_time - dose_time) / 60.0

            if time_since_dose_hours < 0:
                continue  # Future dose
            if time_since_dose_hours >= self.insulin_model.action_duration:
                continue  # Dose has fully absorbed

            # Calculate remaining IOB for this dose
            absorbed_fraction = self.insulin_model.percent_absorbed_at_time(time_since_dose_hours)
            remaining_fraction = 1.0 - absorbed_fraction
            iob_from_dose = units * remaining_fraction

            # Calculate current activity from this dose
            activity_fraction = self.insulin_model.percent_activity_at_time(time_since_dose_hours)
            # Activity is in units/hour, convert to units/minute
            activity_from_dose = units * activity_fraction / 60.0

            total_iob += iob_from_dose
            total_activity += activity_from_dose

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
                time_since_dose_hours = (current_time - dose_time) / 60.0

                if time_since_dose_hours < 0:
                    continue
                if time_since_dose_hours >= self.insulin_model.action_duration:
                    time_since_dose_hours = self.insulin_model.action_duration

                # Absorbed fraction determines how much glucose lowering has occurred
                absorbed_fraction = self.insulin_model.percent_absorbed_at_time(time_since_dose_hours)
                effect_from_dose = -units * insulin_sensitivity * absorbed_fraction

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
        Predict future glucose values based on insulin action.

        Args:
            current_glucose: Current glucose in mg/dL
            bolus_history: List of (time_minutes, units) tuples
            current_time: Current time in minutes
            prediction_horizon: How far to predict in minutes
            time_step: Time step in minutes
            insulin_sensitivity: ISF in mg/dL per unit

        Returns:
            List of (time, predicted_glucose) tuples
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

        # Calculate predictions relative to current glucose
        predictions = []
        for time, effect in effects:
            # Effect is cumulative, so we subtract baseline to get incremental change
            incremental_effect = effect - baseline_effect
            predicted_glucose = current_glucose + incremental_effect
            predictions.append((time, predicted_glucose))

        return predictions


def create_insulin_math(insulin_sensitivity_factor: float,
                       duration_of_insulin_action: float) -> InsulinMath:
    """
    Factory function to create an InsulinMath calculator.

    Args:
        insulin_sensitivity_factor: ISF in mg/dL per unit
        duration_of_insulin_action: DIA in hours

    Returns:
        InsulinMath instance
    """
    model = ExponentialInsulinModel(action_duration=duration_of_insulin_action)
    return InsulinMath(insulin_model=model)
