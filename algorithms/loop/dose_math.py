"""
Dosing logic for Loop algorithm.

Based on LoopKit's DoseMath.swift - simplified version for Phase 2.
Converts glucose predictions into temp basal recommendations.

Simplifications vs full DoseMath:
- No partial application
- No complex target scheduling
- Basic rounding
- Simpler temp basal optimization
"""

from typing import Optional, Dict
from dataclasses import dataclass


@dataclass
class TempBasalRecommendation:
    """
    Temp basal recommendation with rate, duration, and explanation.
    """
    rate: float  # U/hr
    duration: float  # minutes
    reason: str
    eventual_bg: Optional[float] = None  # For debugging
    correction_units: Optional[float] = None  # For debugging (same as insulin_req)
    insulin_req: Optional[float] = None  # Raw insulin requirement (can be negative)
    insulin_req_explanation: str = ""  # Detailed calculation

    # NEW: RealisticDoseMath fields
    insulin_per_5min: Optional[float] = None  # Actual insulin deliverable in 5-min period
    ideal_rate: Optional[float] = None  # Ideal rate before constraints
    was_capped: bool = False  # True if rate hit max_basal limit
    was_floored: bool = False  # True if rate hit zero floor


class SimplifiedDoseMath:
    """
    Simplified dosing logic for Loop.

    Converts glucose predictions into temp basal rate recommendations.
    """

    def __init__(self, settings: Dict):
        """
        Initialize dosing calculator.

        Expected settings:
            - insulin_sensitivity_factor: ISF in mg/dL per unit
            - target: Target BG in mg/dL (preferred) OR
            - target_range: (low, high) in mg/dL (legacy, will use midpoint)
            - basal_rate: Scheduled basal rate in U/hr
            - max_basal_rate: Maximum allowed basal rate in U/hr
            - suspend_threshold: BG threshold for suspending insulin (mg/dL)
        """
        self.isf = settings.get('insulin_sensitivity_factor', 50.0)

        # Support both single target and target_range for backward compatibility
        if 'target' in settings:
            # Single target (preferred)
            self.target = settings['target']
            self.target_low = self.target
            self.target_high = self.target
        else:
            # Legacy target_range - use midpoint
            target_range = settings.get('target_range', (100, 120))
            self.target_low = target_range[0]
            self.target_high = target_range[1]
            self.target = (self.target_low + self.target_high) / 2.0

        self.scheduled_basal = settings.get('basal_rate', 1.0)
        self.max_basal = settings.get('max_basal_rate', 3.0)
        self.suspend_threshold = settings.get('suspend_threshold', 65.0)

    def find_minimum_bg(self, predictions: list) -> float:
        """
        Find the minimum predicted BG in the prediction timeline.

        Args:
            predictions: List of predicted glucose values

        Returns:
            Minimum BG value
        """
        if not predictions:
            return float('inf')

        return min(predictions)

    def recommend_temp_basal(self,
                            eventual_bg: float,
                            prediction_values: list,
                            current_time: float,
                            iob: float,
                            duration_minutes: float = 30.0) -> TempBasalRecommendation:
        """
        Recommend temp basal based on glucose predictions.

        Simplified dosing logic:
        1. Check for suspend (min BG < threshold)
        2. Check if eventual BG is in range
        3. Calculate correction if above/below range
        4. Convert to temp basal rate

        Args:
            eventual_bg: Predicted eventual BG (typically 6 hours out)
            prediction_values: Full prediction timeline (for finding minimum)
            current_time: Current time in minutes
            iob: Current insulin on board (for reference)
            duration_minutes: Duration for temp basal (default 30 min)

        Returns:
            TempBasalRecommendation with rate, duration, and reason
        """
        # Find minimum predicted BG (for suspend check)
        min_bg = self.find_minimum_bg(prediction_values)

        # Suspend if predicted to go low
        if min_bg < self.suspend_threshold:
            # Calculate what the mathematical insulin requirement would be
            # (even though we'll suspend, this shows the negative insulin need)
            deviation = min_bg - self.target
            insulin_req = deviation / self.isf  # Will be negative
            explanation = f"(min_bg {min_bg:.0f} - target {self.target:.0f}) / ISF {self.isf:.0f} = {insulin_req:.2f}U"

            return TempBasalRecommendation(
                rate=0.0,
                duration=duration_minutes,
                reason=f"Suspend: min BG {min_bg:.0f} < {self.suspend_threshold:.0f}",
                eventual_bg=eventual_bg,
                correction_units=insulin_req,
                insulin_req=insulin_req,
                insulin_req_explanation=explanation
            )

        # Check if eventual BG is in range
        if self.target_low <= eventual_bg <= self.target_high:
            deviation = eventual_bg - self.target
            insulin_req = deviation / self.isf  # Small value, close to 0
            explanation = f"(eventual {eventual_bg:.0f} - target {self.target:.0f}) / ISF {self.isf:.0f} = {insulin_req:.2f}U"

            return TempBasalRecommendation(
                rate=self.scheduled_basal,
                duration=duration_minutes,
                reason=f"In range: eventual BG {eventual_bg:.0f} mg/dL",
                eventual_bg=eventual_bg,
                correction_units=insulin_req,
                insulin_req=insulin_req,
                insulin_req_explanation=explanation
            )

        # Calculate correction needed
        if eventual_bg > self.target_high:
            # Too high - need more insulin
            target_for_correction = self.target_high
            deviation = eventual_bg - target_for_correction
            correction_units = deviation / self.isf
            explanation = f"(eventual {eventual_bg:.0f} - target_high {target_for_correction:.0f}) / ISF {self.isf:.0f} = +{correction_units:.2f}U"

            # Convert to rate: units over duration (convert to U/hr)
            correction_rate = correction_units / (duration_minutes / 60.0)
            temp_rate = self.scheduled_basal + correction_rate

            # Cap at max basal
            temp_rate = min(temp_rate, self.max_basal)

            # Don't recommend temp lower than scheduled
            temp_rate = max(temp_rate, self.scheduled_basal)

            return TempBasalRecommendation(
                rate=temp_rate,
                duration=duration_minutes,
                reason=f"High: eventual {eventual_bg:.0f}, need {correction_units:+.2f}U",
                eventual_bg=eventual_bg,
                correction_units=correction_units,
                insulin_req=correction_units,
                insulin_req_explanation=explanation
            )
        else:
            # Too low - need less insulin (NEGATIVE insulin requirement)
            target_for_correction = self.target_low
            deviation = eventual_bg - target_for_correction  # Negative
            correction_units = deviation / self.isf  # Negative
            explanation = f"(eventual {eventual_bg:.0f} - target_low {target_for_correction:.0f}) / ISF {self.isf:.0f} = {correction_units:.2f}U (NEGATIVE)"

            # Convert to rate
            correction_rate = correction_units / (duration_minutes / 60.0)
            temp_rate = self.scheduled_basal + correction_rate  # This will reduce

            # Don't go below zero
            temp_rate = max(0.0, temp_rate)

            # Don't go above scheduled (we're trying to reduce)
            temp_rate = min(temp_rate, self.scheduled_basal)

            return TempBasalRecommendation(
                rate=temp_rate,
                duration=duration_minutes,
                reason=f"Low: eventual {eventual_bg:.0f}, reduce {correction_units:.2f}U",
                eventual_bg=eventual_bg,
                correction_units=correction_units,
                insulin_req=correction_units,
                insulin_req_explanation=explanation
            )

    def recommend_from_predictions(self,
                                  predictions: list,
                                  current_time: float,
                                  iob: float = 0.0,
                                  duration_minutes: float = 30.0) -> TempBasalRecommendation:
        """
        Convenience method that extracts eventual BG from predictions.

        Args:
            predictions: List of predicted glucose values
            current_time: Current time in minutes
            iob: Current IOB
            duration_minutes: Temp basal duration

        Returns:
            TempBasalRecommendation
        """
        if not predictions:
            # No predictions - maintain scheduled basal
            return TempBasalRecommendation(
                rate=self.scheduled_basal,
                duration=duration_minutes,
                reason="No predictions available",
                eventual_bg=None,
                correction_units=0.0
            )

        # Eventual BG is the last prediction
        eventual_bg = predictions[-1]

        return self.recommend_temp_basal(
            eventual_bg=eventual_bg,
            prediction_values=predictions,
            current_time=current_time,
            iob=iob,
            duration_minutes=duration_minutes
        )


class RealisticDoseMath:
    """
    Realistic dosing logic that accounts for temp basal constraints.

    Unlike SimplifiedDoseMath which calculates ideal insulin needs,
    RealisticDoseMath calculates the actual deliverable insulin given:
    - Temp basal rates are capped at max_basal_rate
    - Temp basal rates are floored at 0.0
    - Zero temp can only reduce by (scheduled_basal - 0)
    - Each 5-minute cycle tracks actual delivered insulin

    Implements Loop's haircut algorithm:
    - Uses time-graduated targets (suspend_threshold → correction_target)
    - Calculates minimum correction needed across all predictions
    - Reduces insulin to keep predictions above suspend threshold
    """

    def __init__(self, settings: Dict):
        """
        Initialize realistic dosing calculator.

        Expected settings:
            - insulin_sensitivity_factor: ISF in mg/dL per unit
            - target: Target BG in mg/dL (preferred) OR
            - target_range: (low, high) in mg/dL (legacy, will use midpoint)
            - basal_rate: Scheduled basal rate in U/hr
            - max_basal_rate: Maximum allowed basal rate in U/hr
            - suspend_threshold: BG threshold for suspending insulin (mg/dL)
            - duration_of_insulin_action: DIA in hours (default 6.0)
        """
        self.isf = settings.get('insulin_sensitivity_factor', 50.0)

        # Support both single target and target_range for backward compatibility
        if 'target' in settings:
            self.target = settings['target']
            self.target_low = self.target
            self.target_high = self.target
        else:
            target_range = settings.get('target_range', (100, 120))
            self.target_low = target_range[0]
            self.target_high = target_range[1]
            self.target = (self.target_low + self.target_high) / 2.0

        self.scheduled_basal = settings.get('basal_rate', 1.0)
        self.max_basal = settings.get('max_basal_rate', 3.0)
        self.suspend_threshold = settings.get('suspend_threshold', 65.0)
        self.dia_hours = settings.get('duration_of_insulin_action', 6.0)
        self.dia_minutes = self.dia_hours * 60.0

    def find_minimum_bg(self, predictions: list) -> float:
        """Find the minimum predicted BG in the prediction timeline."""
        if not predictions:
            return float('inf')
        return min(predictions)

    def target_glucose_value(self, percent_effect_duration: float, min_value: float, max_value: float) -> float:
        """
        Compute target glucose value as a function of time through insulin effect duration.

        Based on Loop's targetGlucoseValue() function in DoseMath.swift.

        For the first 50% of DIA, target stays at min_value (suspend_threshold).
        After 50%, target linearly transitions from min_value to max_value.
        At 100% (eventual), target equals max_value (correction_target).

        This implements the haircut: early predictions only need to stay above suspend threshold,
        while later predictions should reach the actual correction target.

        Args:
            percent_effect_duration: Fraction of DIA elapsed (0.0 to 1.0)
            min_value: Minimum target value (typically suspend_threshold)
            max_value: Maximum target value (typically correction target)

        Returns:
            Time-graduated target between min_value and max_value
        """
        # Use min_value for first 50% of insulin effect duration
        use_min_value_until_percent = 0.5

        if percent_effect_duration <= use_min_value_until_percent:
            return min_value

        if percent_effect_duration >= 1.0:
            return max_value

        # Linear transition from min to max after 50%
        slope = (max_value - min_value) / (1.0 - use_min_value_until_percent)
        return min_value + slope * (percent_effect_duration - use_min_value_until_percent)

    def _calculate_deliverable_insulin_per_5min(self, rate: float) -> float:
        """
        Calculate actual insulin delivered in 5-minute period.

        Args:
            rate: Temp basal rate in U/hr

        Returns:
            Insulin delivered in 5 minutes (units)
        """
        return rate * (5.0 / 60.0)

    def _explain_limitations(self,
                            eventual_bg: float,
                            correction_units: float,
                            ideal_rate: float,
                            actual_rate: float,
                            was_capped: bool,
                            was_floored: bool,
                            min_bg: float) -> str:
        """
        Generate detailed reason string explaining rate limitations.

        Args:
            eventual_bg: Predicted eventual BG
            correction_units: Calculated correction in units
            ideal_rate: Ideal rate before constraints
            actual_rate: Actual rate after constraints
            was_capped: True if hit max_basal limit
            was_floored: True if hit zero floor
            min_bg: Minimum predicted BG

        Returns:
            Detailed reason string
        """
        if min_bg < self.suspend_threshold:
            return f"Suspend: min BG {min_bg:.0f} < {self.suspend_threshold:.0f}"

        if self.target_low <= eventual_bg <= self.target_high:
            return f"In range: eventual BG {eventual_bg:.0f} mg/dL"

        base_reason = f"Eventual {eventual_bg:.0f}, need {correction_units:+.2f}U"

        if was_capped:
            return f"{base_reason}, CAPPED at {self.max_basal:.1f}U/hr (ideal {ideal_rate:.2f})"
        elif was_floored:
            return f"{base_reason}, FLOORED at 0.0U/hr (ideal {ideal_rate:.2f})"
        else:
            return f"{base_reason}, rate {actual_rate:.2f}U/hr"

    def recommend_temp_basal(self,
                            eventual_bg: float,
                            prediction_values: list,
                            current_time: float,
                            iob: float,
                            duration_minutes: float = 30.0) -> TempBasalRecommendation:
        """
        Recommend temp basal based on glucose predictions with Loop's haircut algorithm.

        Implements Loop's sophisticated haircut logic:
        1. For each prediction, calculate time-graduated target (suspend_threshold → correction_target)
        2. Calculate correction needed to bring each prediction to its target
        3. Take MINIMUM correction across all predictions (haircut)
        4. Convert to temp basal rate with realistic constraints

        Key differences from SimplifiedDoseMath:
        - Uses time-graduated targets instead of single target
        - Finds minimum correction needed (not just eventual correction)
        - Binary suspend if ANY prediction < suspend_threshold
        - Partial haircut if predictions are just above suspend_threshold

        Args:
            eventual_bg: Predicted eventual BG (typically 6 hours out)
            prediction_values: Full prediction timeline (5-minute intervals)
            current_time: Current time in minutes
            iob: Current insulin on board
            duration_minutes: Duration for temp basal (default 30 min)

        Returns:
            TempBasalRecommendation with realistic delivery tracking and haircut
        """
        # Track minimum correction needed across all predictions
        min_correction_units = None
        correcting_prediction_time = None
        correcting_prediction_bg = None
        correcting_prediction_target = None

        # Iterate through predictions (assume 5-minute intervals)
        for i, predicted_bg in enumerate(prediction_values):
            time_minutes = i * 5.0

            # Only consider predictions within DIA
            if time_minutes > self.dia_minutes:
                break

            # Binary suspend: if ANY prediction < suspend_threshold → rate = 0
            if predicted_bg < self.suspend_threshold:
                return TempBasalRecommendation(
                    rate=0.0,
                    duration=duration_minutes,
                    reason=f"Suspend: BG {predicted_bg:.0f} < {self.suspend_threshold:.0f} at t+{time_minutes:.0f}min",
                    eventual_bg=eventual_bg,
                    correction_units=0.0,
                    insulin_req=0.0,
                    insulin_req_explanation=f"Prediction at t+{time_minutes:.0f}min below suspend threshold",
                    insulin_per_5min=0.0,
                    ideal_rate=0.0,
                    was_capped=False,
                    was_floored=True  # Suspended = floored
                )

            # Calculate time-graduated target for this prediction
            percent_effect_duration = time_minutes / self.dia_minutes
            target = self.target_glucose_value(
                percent_effect_duration,
                min_value=self.suspend_threshold,
                max_value=self.target
            )

            # Calculate correction needed to bring this prediction to its target
            if predicted_bg > target:
                correction = (predicted_bg - target) / self.isf

                # Track minimum correction (haircut)
                if min_correction_units is None or correction < min_correction_units:
                    min_correction_units = correction
                    correcting_prediction_time = time_minutes
                    correcting_prediction_bg = predicted_bg
                    correcting_prediction_target = target

        # If no correction needed (all predictions at or below graduated targets)
        if min_correction_units is None or min_correction_units <= 0:
            insulin_per_5min = self._calculate_deliverable_insulin_per_5min(self.scheduled_basal)
            return TempBasalRecommendation(
                rate=self.scheduled_basal,
                duration=duration_minutes,
                reason=f"In range: eventual {eventual_bg:.0f} within graduated targets",
                eventual_bg=eventual_bg,
                correction_units=0.0,
                insulin_req=0.0,
                insulin_req_explanation="All predictions within time-graduated targets",
                insulin_per_5min=insulin_per_5min,
                ideal_rate=self.scheduled_basal,
                was_capped=False,
                was_floored=False
            )

        # Convert minimum correction to temp basal rate
        # Correction is delivered over duration_minutes
        correction_rate = min_correction_units / (duration_minutes / 60.0)
        ideal_rate = self.scheduled_basal + correction_rate

        # Apply constraints
        actual_rate = ideal_rate
        was_capped = actual_rate > self.max_basal
        was_floored = actual_rate < 0
        actual_rate = max(0.0, min(actual_rate, self.max_basal))

        insulin_per_5min = self._calculate_deliverable_insulin_per_5min(actual_rate)

        # Build explanation
        explanation = (f"Min correction {min_correction_units:.2f}U at t+{correcting_prediction_time:.0f}min "
                      f"(BG {correcting_prediction_bg:.0f} → target {correcting_prediction_target:.0f})")

        # Build reason string
        if was_capped:
            reason = f"Haircut (CAPPED): {explanation}, ideal {ideal_rate:.2f}, capped at {self.max_basal:.1f}U/hr"
        elif was_floored:
            reason = f"Haircut (FLOORED): {explanation}, ideal {ideal_rate:.2f}, floored at 0.0U/hr"
        else:
            reason = f"Haircut: {explanation}, rate {actual_rate:.2f}U/hr"

        return TempBasalRecommendation(
            rate=actual_rate,
            duration=duration_minutes,
            reason=reason,
            eventual_bg=eventual_bg,
            correction_units=min_correction_units,
            insulin_req=min_correction_units,
            insulin_req_explanation=explanation,
            insulin_per_5min=insulin_per_5min,
            ideal_rate=ideal_rate,
            was_capped=was_capped,
            was_floored=was_floored
        )

    def recommend_from_predictions(self,
                                  predictions: list,
                                  current_time: float,
                                  iob: float = 0.0,
                                  duration_minutes: float = 30.0) -> TempBasalRecommendation:
        """
        Convenience method that extracts eventual BG from predictions.

        Args:
            predictions: List of predicted glucose values
            current_time: Current time in minutes
            iob: Current IOB
            duration_minutes: Temp basal duration

        Returns:
            TempBasalRecommendation
        """
        if not predictions:
            return TempBasalRecommendation(
                rate=self.scheduled_basal,
                duration=duration_minutes,
                reason="No predictions available",
                eventual_bg=None,
                correction_units=0.0,
                insulin_per_5min=self._calculate_deliverable_insulin_per_5min(self.scheduled_basal),
                ideal_rate=self.scheduled_basal,
                was_capped=False,
                was_floored=False
            )

        eventual_bg = predictions[-1]

        return self.recommend_temp_basal(
            eventual_bg=eventual_bg,
            prediction_values=predictions,
            current_time=current_time,
            iob=iob,
            duration_minutes=duration_minutes
        )


class AutomaticBolusDoseMath:
    """
    Automatic Bolus dosing mode for Loop.

    Instead of adjusting temp basal rates, delivers a fraction of the
    recommended correction as a bolus each 5-minute cycle.

    Standard mode: 40% of correction per cycle.
    GBPA mode: 20% near target, scaling to 80% at BG >= 200 mg/dL.
    """

    def __init__(self, settings: Dict):
        self.isf = settings.get('insulin_sensitivity_factor', 50.0)

        if 'target' in settings:
            self.target = settings['target']
            self.target_low = self.target
            self.target_high = self.target
        else:
            target_range = settings.get('target_range', (100, 120))
            self.target_low = target_range[0]
            self.target_high = target_range[1]
            self.target = (self.target_low + self.target_high) / 2.0

        self.scheduled_basal = settings.get('basal_rate', 1.0)
        self.max_basal = settings.get('max_basal_rate', 3.0)
        self.max_bolus = settings.get('max_bolus', 5.0)
        self.suspend_threshold = settings.get('suspend_threshold', 65.0)
        self.dia_hours = settings.get('duration_of_insulin_action', 6.0)
        self.dia_minutes = self.dia_hours * 60.0

        # Automatic bolus settings
        self.default_application_factor = 0.4  # 40% standard
        self.enable_gbpa = settings.get('enable_gbpa', False)
        # GBPA: 20% at target, 80% at 200 mg/dL
        self.gbpa_min_factor = 0.2
        self.gbpa_max_factor = 0.8
        self.gbpa_max_bg = 200.0  # BG at which max factor applies

    def _application_factor(self, current_bg: float) -> float:
        """
        Calculate the application factor (fraction of correction to deliver).

        Standard: fixed 40%.
        GBPA: linear from 20% at correction range to 80% at 200 mg/dL.
        """
        if not self.enable_gbpa:
            return self.default_application_factor

        if current_bg <= self.target_high:
            return self.gbpa_min_factor

        if current_bg >= self.gbpa_max_bg:
            return self.gbpa_max_factor

        # Linear interpolation between target_high and gbpa_max_bg
        frac = (current_bg - self.target_high) / (self.gbpa_max_bg - self.target_high)
        return self.gbpa_min_factor + frac * (self.gbpa_max_factor - self.gbpa_min_factor)

    def recommend_from_predictions(self,
                                   predictions: list,
                                   current_time: float,
                                   iob: float = 0.0,
                                   duration_minutes: float = 30.0) -> TempBasalRecommendation:
        if not predictions:
            return TempBasalRecommendation(
                rate=self.scheduled_basal, duration=duration_minutes,
                reason="No predictions", eventual_bg=None, correction_units=0.0)

        eventual_bg = predictions[-1]
        min_bg = min(predictions)
        current_bg = predictions[0]

        # Suspend if predicted to go low
        if min_bg < self.suspend_threshold:
            rec = TempBasalRecommendation(
                rate=0.0, duration=duration_minutes,
                reason=f"Suspend: min BG {min_bg:.0f} < {self.suspend_threshold:.0f}",
                eventual_bg=eventual_bg, correction_units=0.0)
            rec.auto_bolus = 0.0
            rec.application_factor = 0.0
            return rec

        # If eventual BG below target, need LESS insulin
        if eventual_bg < self.target_low:
            correction_units = (eventual_bg - self.target) / self.isf  # negative
            # Reduce basal to deliver less insulin
            rate = self.scheduled_basal + (2 * correction_units) / (duration_minutes / 60.0)
            rate = max(0.0, rate)
            rec = TempBasalRecommendation(
                rate=rate, duration=duration_minutes,
                reason=f"Low eventual {eventual_bg:.0f}, reducing to {rate:.2f}",
                eventual_bg=eventual_bg, correction_units=correction_units)
            rec.auto_bolus = 0.0
            rec.application_factor = 0.0
            return rec

        # At or below target — no correction needed
        if eventual_bg <= self.target_high:
            rec = TempBasalRecommendation(
                rate=self.scheduled_basal, duration=duration_minutes,
                reason=f"In range: eventual {eventual_bg:.0f}",
                eventual_bg=eventual_bg, correction_units=0.0)
            rec.auto_bolus = 0.0
            rec.application_factor = 0.0
            return rec

        # Correction needed: (eventual - target) / ISF
        correction_units = (eventual_bg - self.target) / self.isf

        # Apply the application factor to determine bolus portion
        app_factor = self._application_factor(current_bg)
        bolus_units = correction_units * app_factor

        # Round to nearest 0.05U (pump increment)
        bolus_units = round(bolus_units * 20) / 20
        bolus_units = min(bolus_units, self.max_bolus)
        bolus_units = max(0.0, bolus_units)

        # Remaining correction delivered via temp basal over 30 min
        # rate = scheduled + 2 * remaining / (duration/60)
        remaining_units = correction_units - bolus_units
        temp_rate = self.scheduled_basal + (2 * remaining_units) / (duration_minutes / 60.0)
        temp_rate = max(0.0, min(temp_rate, self.max_basal))

        reason = (f"AutoBolus: eventual {eventual_bg:.0f}, "
                  f"corr {correction_units:.2f}U, "
                  f"bolus {bolus_units:.2f}U ({app_factor:.0%}), "
                  f"TB {temp_rate:.2f}")

        rec = TempBasalRecommendation(
            rate=temp_rate,
            duration=duration_minutes,
            reason=reason,
            eventual_bg=eventual_bg,
            correction_units=correction_units,
            insulin_req=correction_units,
        )
        rec.auto_bolus = bolus_units
        rec.application_factor = app_factor
        return rec


def create_dose_math(settings: Dict) -> SimplifiedDoseMath:
    """
    Factory function to create a SimplifiedDoseMath instance.

    Args:
        settings: Settings dictionary with ISF, target range, basal rates, etc.

    Returns:
        SimplifiedDoseMath instance
    """
    return SimplifiedDoseMath(settings)
