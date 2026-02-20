"""
Insulin Counteraction Effects (ICE) calculation.

Based on GlucoseMath.swift counteractionEffects() function.
"""

from typing import List, Tuple
import numpy as np


class InsulinCounteraction:
    """Calculates insulin counteraction effects from glucose readings and insulin effects."""

    def __init__(self):
        """Initialize counteraction calculator."""
        pass

    @staticmethod
    def _find_effect_gte(effects: list, target_time: float) -> float:
        """Find the first effect with time >= target_time (matching iOS >= logic)."""
        if not effects:
            return 0.0
        for t, v in effects:
            if t >= target_time:
                return v
        # If no effect >= target_time, use last value
        return effects[-1][1]

    @staticmethod
    def _interpolate_effect(effects: list, target_time: float) -> float:
        """Interpolate insulin effect at an arbitrary time point."""
        if not effects:
            return 0.0

        # Before first effect
        if target_time <= effects[0][0]:
            return effects[0][1]

        # After last effect - use last value (don't default to 0!)
        if target_time >= effects[-1][0]:
            return effects[-1][1]

        # Find bracketing effects and interpolate
        for i in range(len(effects) - 1):
            t0, v0 = effects[i]
            t1, v1 = effects[i + 1]
            if t0 <= target_time <= t1:
                if t1 == t0:
                    return v0
                frac = (target_time - t0) / (t1 - t0)
                return v0 + frac * (v1 - v0)

        return effects[-1][1]

    def calculate_counteraction_effects(
        self,
        glucose_samples: List[Tuple[float, float]],  # [(time_minutes, glucose_mg_dL), ...]
        insulin_effects: List[Tuple[float, float]]    # [(time_minutes, effect_mg_dL), ...]
    ) -> List[Tuple[float, float, float]]:
        """
        Calculate insulin counteraction effects (ICE).

        ICE represents the glucose change that is NOT explained by insulin effects.
        This captures ALL unmodeled effects: carbs, exercise, stress, hormones, etc.

        Args:
            glucose_samples: Sorted list of (time, glucose) readings
            insulin_effects: Sorted list of (time, cumulative_insulin_effect).
                            If empty or all zeros, ICE equals glucose change.

        Returns:
            List of (start_time, end_time, ice_velocity_mg_dL_per_min) tuples

            NOTE: Returns RAW ICE including negative values. Consumers should decide
            whether to clamp negatives:
            - DCA (carb absorption): clamp negatives to 0 (carbs can't be negative)
            - IRC (retrospective correction): keep negatives (detects over-prediction)

        Algorithm (from GlucoseMath.swift counteractionEffects):
        1. For each pair of glucose readings (at least 4 minutes apart)
        2. Calculate glucose change = end_glucose - start_glucose
        3. Find corresponding insulin effects at those times (or use 0 if none)
        4. Calculate effect change = end_effect - start_effect
        5. ICE = glucose_change - effect_change
        6. Convert to velocity (mg/dL per minute)
        """
        velocities = []

        if len(glucose_samples) < 2:
            return []

        # If no insulin effects, create zero-effect timeline
        if len(insulin_effects) == 0:
            insulin_effects = [(t, 0.0) for t, _ in glucose_samples]

        # Find first glucose sample that's after or at the first insulin effect
        start_glucose_idx = 0
        for i, (g_time, _) in enumerate(glucose_samples):
            if g_time >= insulin_effects[0][0]:
                start_glucose_idx = i
                break
        else:
            # If all glucose samples are before insulin effects, start from beginning
            start_glucose_idx = 0

        end_glucose_idx = start_glucose_idx + 1
        effect_idx = 0

        while end_glucose_idx < len(glucose_samples):
            start_time, start_glucose = glucose_samples[start_glucose_idx]
            end_time, end_glucose = glucose_samples[end_glucose_idx]

            time_interval = end_time - start_time

            # Require at least 4 minutes between readings (from LoopKit)
            if time_interval < 4.0:
                end_glucose_idx += 1
                continue

            # Calculate glucose change
            glucose_change = end_glucose - start_glucose

            # Find matching insulin effects using interpolation
            start_effect = self._interpolate_effect(insulin_effects, start_time)
            end_effect = self._interpolate_effect(insulin_effects, end_time)

            # Calculate insulin effect change
            effect_change = end_effect - start_effect

            # ICE = glucose change - insulin effect change
            # KEEP RAW VALUE (including negatives) for IRC
            # DCA will clamp to 0, but IRC needs negatives to detect over-prediction
            ice_raw = glucose_change - effect_change

            # Convert to velocity (mg/dL per minute)
            ice_velocity = ice_raw / time_interval

            velocities.append((start_time, end_time, ice_velocity))

            # Move to next glucose pair
            start_glucose_idx = end_glucose_idx
            end_glucose_idx += 1

        return velocities


def calculate_carb_sensitivity_factor(insulin_sensitivity: float, carb_ratio: float) -> float:
    """
    Calculate Carb Sensitivity Factor (CSF).

    CSF = ISF / CR

    Example:
        ISF = 100 mg/dL per unit
        CR = 9 g per unit
        CSF = 100 / 9 = 11.1 mg/dL per gram

    Args:
        insulin_sensitivity: ISF in mg/dL per unit
        carb_ratio: CR in grams per unit

    Returns:
        CSF in mg/dL per gram
    """
    return insulin_sensitivity / carb_ratio
