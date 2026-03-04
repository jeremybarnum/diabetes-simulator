"""
Enhanced patient model with time-varying sensitivity, basal mismatch, and stochastic events.

The patient model computes "true" BG based on actual physiology, which may differ from
what the algorithm believes. The key insight: algorithm settings represent the pump's
programmed values, but the patient's true sensitivity varies over time.

Sensitivity scalar S(t):
    - S(t) > 1: patient is more insulin resistant (needs more insulin)
    - S(t) < 1: patient is more insulin sensitive (needs less insulin)
    - true_ISF = settings.ISF / S(t)
    - true_basal_need = settings.basal * S(t)
    - true_IC = settings.IC / S(t)
    - true_CSF = true_ISF / true_IC = settings.CSF (invariant!)

Carb effect on BG is independent of sensitivity scalar (CSF cancels).
Only insulin effects and basal drift are affected.
"""

import math
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field

from algorithms.loop.insulin_models_exact import create_insulin_model, InsulinType


class SensitivityModel:
    """
    Persistent insulin sensitivity deviation.

    Draws a single sensitivity scalar at construction time from
    lognormal(0, sigma_day) and returns it for all time points. This models
    a patient whose true sensitivity is consistently different from their
    pump settings for the entire simulation run.

    The scalar represents the ratio of true insulin need to programmed insulin need.
    """

    def __init__(self, sigma_day: float = 0.15, rng: np.random.RandomState = None):
        """
        Args:
            sigma_day: Lognormal sigma for sensitivity deviation.
                       0.15 gives typical range ~0.75-1.35.
            rng: Random state for reproducibility.
        """
        self.sigma_day = sigma_day
        self.rng = rng or np.random.RandomState()

        if sigma_day > 0:
            self._scalar = float(np.exp(self.rng.normal(0, self.sigma_day)))
        else:
            self._scalar = 1.0

    def get_scalar(self, t_minutes: float) -> float:
        """
        Get sensitivity scalar (constant for all time points).

        Returns:
            S(t) sensitivity scalar.
        """
        return self._scalar

    def get_daily_shocks(self) -> List[float]:
        """Return the persistent scalar (for logging/debugging)."""
        return [self._scalar]


@dataclass
class ExerciseEvent:
    """
    Exercise with declared vs actual parameters.

    The patient declares a temp target to the algorithm, but the actual
    physiological effect may differ in magnitude and duration.
    """
    start_time_minutes: float  # When exercise starts (minutes from sim start)

    # What the patient tells the algorithm
    declared_scalar: float = 0.5       # e.g., 50% sensitivity increase
    declared_duration_hrs: float = 4.0

    # What actually happens to physiology
    actual_scalar: float = 0.5
    actual_duration_hrs: float = 7.0

    def get_actual_effect(self, t_minutes: float) -> float:
        """
        Get the exercise sensitivity multiplier at time t.

        Returns a scalar < 1 meaning more sensitive (needs less insulin).
        Decays exponentially from actual_scalar back to 1.0 over actual_duration.

        Returns:
            Multiplier on sensitivity (e.g., 0.5 = twice as sensitive).
            Returns 1.0 if exercise hasn't started or effect has fully decayed.
        """
        elapsed = t_minutes - self.start_time_minutes
        if elapsed < 0:
            return 1.0

        elapsed_hrs = elapsed / 60.0
        if elapsed_hrs >= self.actual_duration_hrs:
            return 1.0

        # Exponential decay: actual_scalar at start -> 1.0 at actual_duration
        # Use 3 half-lives over the duration so it's ~87.5% decayed
        decay_rate = 3.0 * math.log(2) / self.actual_duration_hrs
        effect = 1.0 + (self.actual_scalar - 1.0) * math.exp(-decay_rate * elapsed_hrs)
        return effect

    def is_declared_active(self, t_minutes: float) -> bool:
        """Whether the algorithm should see a temp target active."""
        elapsed = t_minutes - self.start_time_minutes
        return 0 <= elapsed < self.declared_duration_hrs * 60


@dataclass
class MealEvent:
    """
    Meal with actual vs declared carbs and absorption.

    The patient eats actual_carbs with actual_absorption_hrs,
    but tells the algorithm declared_carbs with declared_absorption_hrs.
    If undeclared=True, the patient eats but doesn't tell the pump at all.
    """
    time_minutes: float              # When meal is eaten
    actual_carbs: float              # True grams consumed
    actual_absorption_hrs: float     # True absorption time
    declared_carbs: float            # What patient tells pump (0 if undeclared)
    declared_absorption_hrs: float   # What patient tells pump
    undeclared: bool = False         # If True, patient eats but doesn't enter carbs


class EnhancedPatientModel:
    """
    Stochastic patient model with time-varying sensitivity.

    Computes BG(t) = starting_BG + insulin_effect(t) + carb_effect(t) + basal_drift(t)

    Key differences from deterministic PatientModel:
    - Insulin effects scaled by 1/S(t): true_ISF = settings.ISF / S(t)
    - Basal drift: pump delivers scheduled basal, but patient needs S(t) × scheduled
    - Carb effects: unaffected by S(t) since CSF = ISF/IC is invariant
    - Exercise events: additional sensitivity multiplier layered on top
    """

    def __init__(
        self,
        settings: Dict,
        starting_bg: float = 120.0,
        sensitivity_model: Optional[SensitivityModel] = None,
        exercise_events: Optional[List[ExerciseEvent]] = None,
    ):
        from algorithms.loop.carb_math import create_carb_math

        self.starting_bg = starting_bg
        self.settings_isf = settings['insulin_sensitivity_factor']
        self.settings_basal = settings['basal_rate']
        self.carb_ratio = settings['carb_ratio']

        self.insulin_model = create_insulin_model(
            InsulinType.FIASP,
            settings['duration_of_insulin_action'],
        )
        self.carb_math = create_carb_math(
            carb_ratio=self.carb_ratio,
            insulin_sensitivity=self.settings_isf,
        )

        self.sensitivity_model = sensitivity_model
        self.exercise_events = exercise_events or []

    def get_sensitivity_scalar(self, t_minutes: float) -> float:
        """
        Get combined sensitivity scalar at time t.

        Combines daily sensitivity model with any active exercise effects.
        """
        s = 1.0
        if self.sensitivity_model:
            s = self.sensitivity_model.get_scalar(t_minutes)

        for ex in self.exercise_events:
            s *= ex.get_actual_effect(t_minutes)

        return s

    def compute_bg(
        self,
        time: float,
        bolus_history: List[Tuple[float, float]],
        carb_entries: List[Tuple[float, float, float]],
        sim_start_time: float = 0.0,
    ) -> float:
        """
        Compute BG at given time.

        Args:
            time: Current time in minutes (absolute, matching bolus_history times)
            bolus_history: [(time_min, units), ...] — all insulin delivered
            carb_entries: [(time_min, grams, absorption_hrs), ...] — actual carbs
            sim_start_time: Simulation start time (for sensitivity model alignment)

        Returns:
            BG in mg/dL
        """
        # Time relative to simulation start (for sensitivity model)
        t_rel = time - sim_start_time

        # --- Insulin effect (scaled by sensitivity) ---
        insulin_effect = 0.0
        for dose_time, units in bolus_history:
            time_since = time - dose_time
            if time_since < 0:
                continue
            pct_absorbed = self.insulin_model.percent_absorbed(time_since)

            # True ISF at current time: settings.ISF / S(t)
            # When S > 1 (resistant): true_ISF is lower, insulin has less effect
            s = self.get_sensitivity_scalar(t_rel)
            true_isf = self.settings_isf / s

            insulin_effect += -units * true_isf * pct_absorbed

        # --- Carb effect (independent of sensitivity) ---
        carb_effect = 0.0
        if carb_entries:
            for carb_time, grams, abs_hrs in carb_entries:
                elapsed = time - carb_time
                if elapsed < 0:
                    continue
                preds = self.carb_math.predict_glucose_from_carbs(
                    current_glucose=0.0,
                    carb_entries=[(carb_time, grams, abs_hrs)],
                    current_time=carb_time,
                    prediction_horizon=max(elapsed + 5, 10),
                    time_step=5,
                )
                if preds:
                    best_effect = 0.0
                    best_dist = float('inf')
                    for pred_time, effect in preds:
                        dist = abs(pred_time - time)
                        if dist < best_dist:
                            best_dist = dist
                            best_effect = effect
                    carb_effect += best_effect

        # --- Basal drift ---
        # Pump delivers scheduled_basal continuously.
        # Patient needs scheduled_basal * S(t).
        # Deficit per 5 min = scheduled_basal * (1 - S(t)) / 12 units
        # BG effect of deficit = deficit * true_ISF = deficit * (ISF / S(t))
        # = [basal * (1 - S) / 12] * [ISF / S]
        # = basal * ISF * (1 - S) / (12 * S)
        #
        # Integrate over 5-min intervals from sim start to current time.
        basal_drift = 0.0
        if self.sensitivity_model or self.exercise_events:
            # Integrate in 5-min steps
            t_start_rel = 0.0
            t_end_rel = t_rel
            dt = 5.0  # minutes
            t = 0.0
            while t < t_end_rel:
                t_next = min(t + dt, t_end_rel)
                interval = t_next - t
                s = self.get_sensitivity_scalar(t)

                if abs(s - 1.0) > 1e-6:
                    # Basal deficit in units for this interval
                    deficit_units = self.settings_basal * (1.0 - s) * (interval / 60.0)
                    # The deficit insulin would have had effect by now
                    time_since_deficit = t_rel - t
                    pct_absorbed = self.insulin_model.percent_absorbed(time_since_deficit)
                    true_isf = self.settings_isf / s
                    # Negative deficit (S>1) means not enough insulin -> BG rises
                    # deficit_units is negative when S>1, so -deficit * ISF * pct is positive
                    basal_drift += -deficit_units * true_isf * pct_absorbed

                t = t_next

        bg = self.starting_bg + insulin_effect + carb_effect + basal_drift
        return max(39.0, min(400.0, bg))

    def compute_bg_delta(
        self,
        current_time: float,
        dt: float,
        bolus_history: List[Tuple[float, float]],
        basal_deficit_entries: List[Tuple[float, float]],
        carb_entries: List[Tuple[float, float, float]],
        sim_start_time: float = 0.0,
    ) -> float:
        """
        Compute BG change for one time step (delta-based).

        Instead of recomputing BG from scratch, this returns the change in BG
        over [current_time - dt, current_time]. The caller maintains a running BG.

        Args:
            current_time: Current time in minutes (absolute)
            dt: Time step in minutes (typically 5)
            bolus_history: [(time_min, units), ...] — real insulin doses
            basal_deficit_entries: [(time_min, units), ...] — virtual insulin from
                sensitivity mismatch (patient-only, algorithm doesn't see these)
            carb_entries: [(time_min, grams, absorption_hrs), ...] — actual carbs
            sim_start_time: Simulation start time (for sensitivity model)

        Returns:
            BG delta in mg/dL for this step
        """
        from itertools import chain

        t_rel = current_time - sim_start_time
        s_now = self.get_sensitivity_scalar(t_rel)
        dia_minutes = self.insulin_model.action_duration + 10  # buffer

        # --- Insulin delta (real doses + basal deficit virtual doses) ---
        insulin_delta = 0.0
        for dose_time, units in chain(bolus_history, basal_deficit_entries):
            elapsed = current_time - dose_time
            if elapsed < 0 or elapsed > dia_minutes:
                continue
            pct_now = self.insulin_model.percent_absorbed(elapsed)
            pct_prev = self.insulin_model.percent_absorbed(elapsed - dt)
            # true_ISF = settings_ISF / S(t)
            insulin_delta += -units * (self.settings_isf / s_now) * (pct_now - pct_prev)

        # --- Carb delta (independent of sensitivity — CSF is invariant) ---
        carb_delta = 0.0
        csf = self.settings_isf / self.carb_ratio  # mg/dL per gram, constant
        CARB_DELAY_MIN = 10.0
        carb_model = self.carb_math.model

        for carb_time, grams, abs_hrs in carb_entries:
            elapsed = current_time - carb_time
            max_duration = abs_hrs * 60 + CARB_DELAY_MIN + 10  # buffer
            if elapsed < 0 or elapsed > max_duration:
                continue
            # Apply delay externally (model expects time after delay)
            eff_now_hrs = max(0.0, (elapsed - CARB_DELAY_MIN) / 60.0)
            eff_prev_hrs = max(0.0, (elapsed - dt - CARB_DELAY_MIN) / 60.0)
            pct_now = carb_model.percent_absorbed_at_time(eff_now_hrs, abs_hrs)
            pct_prev = carb_model.percent_absorbed_at_time(eff_prev_hrs, abs_hrs)
            carb_delta += grams * csf * (pct_now - pct_prev)

        # TODO: BG-dependent sensitivity — S(t) could also depend on current BG
        # (e.g., insulin resistance increases at high BG). This would require
        # the caller to pass running_bg so we can compute S(t, BG). Also need
        # error estimation to understand how approximation errors accumulate
        # in the delta-based approach over multi-day simulations.

        return insulin_delta + carb_delta
