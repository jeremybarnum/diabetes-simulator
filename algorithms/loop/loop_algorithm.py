"""
Loop algorithm implementation with EXACT LoopKit models.

Uses exact exponential insulin model (Fiasp default) and piecewise linear carb model.
Phase 2: Includes momentum, retrospective correction, and dosing logic.
Phase 2.5: Integrated Dynamic Carb Absorption (DCA).
"""

from typing import Dict, List, Tuple, Optional
from algorithms.base import Algorithm, AlgorithmInput, AlgorithmOutput
from algorithms.loop.insulin_math_exact import create_exact_insulin_math
from algorithms.loop.carb_math import create_carb_math
from algorithms.loop.momentum import create_momentum_calculator
from algorithms.loop.integral_rc import create_integral_rc
from algorithms.loop.dose_math import create_dose_math, SimplifiedDoseMath, RealisticDoseMath, AutomaticBolusDoseMath
from algorithms.loop.dynamic_carb_absorption import DynamicCarbAbsorption
from algorithms.loop.insulin_counteraction import InsulinCounteraction


class LoopAlgorithm(Algorithm):
    """
    Loop algorithm implementation with EXACT LoopKit models.

    Phase 2 features:
    - Insulin: Exact exponential model with Fiasp (55-min peak) default
    - Carbs: Exact piecewise linear model (3 phases, 10-min delay)
    - Momentum: Linear regression glucose velocity prediction
    - Retrospective Correction: IRC for handling settings errors
    - Dosing: Simplified temp basal recommendations
    """

    def __init__(self, settings: Dict):
        """
        Initialize Loop algorithm.

        Expected settings:
            - insulin_sensitivity_factor: ISF in mg/dL per unit
            - duration_of_insulin_action: DIA in hours
            - basal_rate: Scheduled basal rate in U/hr
            - target_range: (low, high) in mg/dL
            - carb_ratio: Carb ratio in g/U
            - max_basal_rate: Maximum basal rate in U/hr
            - max_bolus: Maximum bolus in U
            - suspend_threshold: BG threshold for suspending (mg/dL)
            - enable_irc: Enable Integral Retrospective Correction (default True)
        """
        super().__init__(settings)

        # Extract settings with defaults
        self.isf = settings.get('insulin_sensitivity_factor', 50.0)
        self.dia = settings.get('duration_of_insulin_action', 6.0)
        self.basal_rate = settings.get('basal_rate', 1.0)
        self.target_low = settings.get('target_range', (100, 120))[0]
        self.target_high = settings.get('target_range', (100, 120))[1]
        self.target = (self.target_low + self.target_high) / 2
        self.carb_ratio = settings.get('carb_ratio', 10.0)
        self.max_basal = settings.get('max_basal_rate', 3.0)
        self.max_bolus = settings.get('max_bolus', 5.0)

        # Create insulin math calculator with configurable insulin type
        from algorithms.loop.insulin_models_exact import InsulinType
        insulin_type_name = settings.get('insulin_type', 'fiasp')
        insulin_type_map = {
            'fiasp': InsulinType.FIASP,
            'rapid_acting_adult': InsulinType.RAPID_ACTING_ADULT,
            'rapid_acting_child': InsulinType.RAPID_ACTING_CHILD,
            'lyumjev': InsulinType.LYUMJEV,
            'afrezza': InsulinType.AFREZZA,
        }
        insulin_type = insulin_type_map.get(insulin_type_name, InsulinType.FIASP)
        self.insulin_math = create_exact_insulin_math(
            insulin_sensitivity_factor=self.isf,
            duration_of_insulin_action=self.dia,
            insulin_type=insulin_type
        )

        # Create carb math calculator with piecewise linear model
        self.carb_math = create_carb_math(
            carb_ratio=self.carb_ratio,
            insulin_sensitivity=self.isf
        )

        # NEW: Phase 2 components
        self.momentum_calc = create_momentum_calculator()
        self.rc = create_integral_rc()

        # Select dosing strategy based on settings
        dosing_mode = settings.get('dosing_mode', 'temp_basal')
        if dosing_mode == 'automatic_bolus':
            self.dose_math = AutomaticBolusDoseMath(settings)
        elif settings.get('use_realistic_dosing', False):
            self.dose_math = RealisticDoseMath(settings)
        else:
            self.dose_math = SimplifiedDoseMath(settings)

        # NEW: Phase 2.5 - Dynamic Carb Absorption
        csf = self.isf / self.carb_ratio  # Carb Sensitivity Factor (mg/dL per gram)
        self.dca = DynamicCarbAbsorption(
            carb_sensitivity_factor=csf,
            absorption_model=self.carb_math.model  # Use same model as static carb math
        )
        self.ice_calculator = InsulinCounteraction()

        # IRC is now stateless - no stored predictions needed

    def get_name(self) -> str:
        """Return algorithm name."""
        return "Loop"

    def combine_effects(self,
                       current_glucose: float,
                       insulin_effect: List[Tuple[float, float]],
                       carb_effect: List[Tuple[float, float]],
                       momentum_effect: List[Tuple[float, float]],
                       rc_effect: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        Combine all prediction effects into single timeline.

        CORRECTED to match LoopMath.swift predictGlucose (lines 122-175):
        1. Convert cumulative effects to DELTAS
        2. Blend momentum deltas with effects deltas at momentum time points
        3. Build predictions SEQUENTIALLY from deltas: prediction[t] = prediction[t-1] + delta[t]

        This allows momentum to affect eventual BG through cascading.

        Args:
            current_glucose: Current glucose in mg/dL
            insulin_effect: Timeline of insulin effects (cumulative deltas from baseline)
            carb_effect: Timeline of carb effects (cumulative deltas from baseline)
            momentum_effect: Timeline of momentum effects (cumulative deltas from baseline)
            rc_effect: Timeline of retrospective correction effects (cumulative deltas)

        Returns:
            Combined prediction timeline as (time, glucose) tuples
        """
        # Step 1: Convert cumulative effects to deltas (LoopMath.swift lines 122-130)
        effect_deltas_at_time = {}  # Maps time -> delta

        # Process regular effects (insulin, carbs, RC)
        for effect_timeline in [insulin_effect, carb_effect, rc_effect]:
            if not effect_timeline:
                continue

            previous_value = 0.0
            for time, cumulative_value in effect_timeline:
                delta = cumulative_value - previous_value
                effect_deltas_at_time[time] = effect_deltas_at_time.get(time, 0.0) + delta
                previous_value = cumulative_value

        # Step 2: Blend momentum into the deltas (LoopMath.swift lines 132-160)
        if momentum_effect and len(momentum_effect) > 1:
            # Calculate blending parameters
            blend_count = len(momentum_effect) - 2
            blend_slope = 1.0 / blend_count if blend_count > 0 else 0.0

            # Calculate blend offset (LoopMath.swift lines 140-146)
            # This accounts for timing difference between first momentum point and current glucose
            if len(momentum_effect) >= 2:
                time_delta = momentum_effect[1][0] - momentum_effect[0][0]  # Time between momentum points
                # Assume current_glucose time equals the first momentum time (typical case)
                momentum_offset = 0.0  # current_glucose_time - momentum_effect[0][0]
                blend_offset = (momentum_offset / time_delta * blend_slope) if time_delta > 0 else 0.0
            else:
                blend_offset = 0.0

            # Build sorted list of effect times for nearest-match lookup
            effect_times_sorted = sorted(effect_deltas_at_time.keys())

            previous_momentum_value = 0.0

            for index, (mom_time, cumulative_momentum) in enumerate(momentum_effect):
                # Convert momentum to delta
                momentum_delta = cumulative_momentum - previous_momentum_value
                previous_momentum_value = cumulative_momentum

                # Calculate blend weight (split)
                if blend_count > 0:
                    split = (len(momentum_effect) - index) / blend_count - blend_slope + blend_offset
                    split = max(0.0, min(1.0, split))
                else:
                    split = 1.0 if index == 0 else 0.0

                # Find the nearest effect time point (within 3 minutes tolerance)
                # This handles timestamp misalignment between float momentum and int effect times
                matched_time = None
                if effect_times_sorted:
                    best_dist = float('inf')
                    for et in effect_times_sorted:
                        dist = abs(et - mom_time)
                        if dist < best_dist:
                            best_dist = dist
                            matched_time = et
                        elif dist > best_dist:
                            break  # Past the closest, stop

                    if best_dist > 3.0:  # More than 3 minutes away = no match
                        matched_time = None

                if matched_time is not None:
                    effects_delta = effect_deltas_at_time.get(matched_time, 0.0)
                    blended_delta = split * momentum_delta + (1.0 - split) * effects_delta
                    effect_deltas_at_time[matched_time] = blended_delta
                else:
                    # No matching effect time - add momentum at its own timestamp
                    blended_delta = split * momentum_delta
                    if abs(blended_delta) > 0.001:
                        effect_deltas_at_time[mom_time] = blended_delta

        # Step 3: Build predictions sequentially from deltas (LoopMath.swift lines 162-172)
        # prediction[t] = prediction[t-1] + delta[t]
        sorted_times = sorted(effect_deltas_at_time.keys())

        prediction = []
        current_prediction = current_glucose

        for time in sorted_times:
            delta = effect_deltas_at_time[time]
            current_prediction = current_prediction + delta
            prediction.append((time, current_prediction))

        return prediction

    def recommend(self, input_data: AlgorithmInput) -> AlgorithmOutput:
        """
        Generate Loop dosing recommendation using EXACT LoopKit models.

        Phase 2 Algorithm:
        1. Calculate IOB (exact Fiasp exponential model)
        2. Calculate COB (exact piecewise linear absorption)
        3. Calculate momentum effect (from recent CGM trend)
        4. Calculate retrospective correction (from past prediction errors)
        5. Predict glucose from insulin effects
        6. Predict glucose from carb effects
        7. Combine all effects (insulin + carbs + momentum + RC)
        8. Determine temp basal using dosing logic

        Args:
            input_data: Current state

        Returns:
            AlgorithmOutput with recommendation
        """
        # Calculate IOB
        iob, activity = self.insulin_math.calculate_iob(
            bolus_history=input_data.bolus_history,
            current_time=input_data.timestamp
        )

        # Predict glucose based on insulin and carbs
        prediction_horizon = 360  # 6 hours in minutes
        time_step = 5  # 5-minute intervals

        # Get insulin predictions
        insulin_predictions = self.insulin_math.predict_glucose_from_insulin(
            current_glucose=input_data.cgm_reading,
            bolus_history=input_data.bolus_history,
            current_time=input_data.timestamp,
            prediction_horizon=prediction_horizon,
            time_step=time_step,
            insulin_sensitivity=self.isf
        )

        # Calculate Insulin Counteraction Effects (ICE) for DCA and IRC
        # ICE = observed glucose change - expected insulin effect
        # iOS Loop only computes ICE when there are historical insulin effects (count > 0).
        # With no insulin doses at all, there are no insulin effects, so no ICE, no IRC.
        ice_observations = []
        if (input_data.cgm_history and len(input_data.cgm_history) >= 2
                and input_data.bolus_history):
            # For ICE, compute insulin effects at EXACT glucose sample timestamps
            # This avoids interpolation artifacts from the exponential absorption curve
            glucose_samples = input_data.cgm_history + [(input_data.timestamp, input_data.cgm_reading)]
            sample_times = sorted(set(t for t, _ in glucose_samples))

            historical_insulin_effects = []
            for t in sample_times:
                total_effect = 0.0
                for dose_time, units in input_data.bolus_history:
                    time_since_dose = t - dose_time
                    if time_since_dose < 0:
                        continue
                    pct = self.insulin_math.insulin_model.percent_absorbed(time_since_dose)
                    total_effect += -units * self.isf * pct
                historical_insulin_effects.append((t, total_effect))

            ice_observations = self.ice_calculator.calculate_counteraction_effects(
                glucose_samples=glucose_samples,
                insulin_effects=historical_insulin_effects
            )

        # Store for debugging
        self.last_ice_observations = ice_observations

        # Always process carbs through DCA (matching iOS Loop which always
        # uses the DCA/map() code path for carb status and effects)
        if input_data.carb_entries:
            carb_statuses = self.dca.process_carb_entries(
                carb_entries=input_data.carb_entries,
                ice_observations=ice_observations,  # May be empty if no insulin
                current_time=input_data.timestamp
            )
            self.last_carb_statuses = carb_statuses

            cob = self.dca.calculate_dynamic_cob(
                carb_statuses=carb_statuses,
                at_time=input_data.timestamp
            )
        else:
            self.last_carb_statuses = []
            cob = 0.0

        # Generate carb predictions: use DCA's dynamic effect calculation if available
        # This uses the carb statuses with dynamic COB and declared absorption rates
        # Always route carb predictions through DCA path (matching iOS Loop's
        # LoopAlgorithm.swift which always uses dynamicGlucoseEffects, even
        # when there are no ICE observations)
        if input_data.carb_entries and hasattr(self, 'last_carb_statuses') and self.last_carb_statuses:
            grid_start = int(input_data.timestamp)
            raw_effects = self.dca.calculate_dynamic_carb_effects(
                carb_statuses=self.last_carb_statuses,
                start_time=grid_start,
                end_time=grid_start + prediction_horizon,
                time_step=time_step
            )
            baseline_effect = raw_effects[0][1] if raw_effects else 0.0
            carb_predictions = [(t, effect - baseline_effect) for t, effect in raw_effects]
        elif input_data.carb_entries:
            # DCA not available (no carb statuses built yet) - use static
            carb_predictions = self.carb_math.predict_glucose_from_carbs(
                current_glucose=input_data.cgm_reading,
                carb_entries=input_data.carb_entries,
                current_time=input_data.timestamp,
                prediction_horizon=prediction_horizon,
                time_step=time_step
            )
        else:
            carb_predictions = []

        # NEW: Calculate momentum effect (if enabled)
        momentum_effect = []
        enable_momentum = self.settings.get('enable_momentum', True)
        if enable_momentum and input_data.cgm_history and len(input_data.cgm_history) >= 3:
            # Include current reading for velocity calculation (like ICE does)
            glucose_samples_for_momentum = input_data.cgm_history + [(input_data.timestamp, input_data.cgm_reading)]
            momentum_effect = self.momentum_calc.calculate(
                glucose_samples=glucose_samples_for_momentum,
                current_time=input_data.timestamp
            )

        # NEW: Calculate retrospective correction (if enabled)
        # IRC now uses ICE and carb effects (Loop's stateless approach)
        rc_effect = []
        irc_details = None
        enable_irc = self.settings.get('enable_irc', True)
        if enable_irc and ice_observations:
            # Calculate carb effects for the retrospection window (180 min back)
            # iOS Loop uses DCA-adjusted carb effects (dynamicGlucoseEffects),
            # NOT static declared effects. This was confirmed from LoopAlgorithm.swift
            # lines 106-117 where carbEffects use .dynamicGlucoseEffects().
            retrospection_start = input_data.timestamp - 180.0  # 3 hours
            # TODO(jeremy): Review this DCA fallback logic manually.
            # iOS's dynamicAbsorbedCarbs() falls back to the static carb model
            # (absorptionModel.absorbedCarbs) when observedTimeline is nil
            # (i.e., not enough observed absorption to meet min_predicted_grams).
            # This means IRC always gets carb effects — either from DCA observation
            # or from the declared/static model. See CarbStatus.swift lines 81-96.
            if input_data.carb_entries and hasattr(self, 'last_carb_statuses') and self.last_carb_statuses:
                # Use dynamicGlucoseEffects — matching iOS LoopAlgorithm.swift
                # lines 106-114 which calls .dynamicGlucoseEffects() on CarbStatus.
                # This uses dynamicAbsorbedCarbs() per entry, which handles:
                #   - No observed timeline → static model with dynamic absorption time
                #   - Within observation → sum observed timeline
                #   - After observation → project remaining via static model
                retrospection_carb_effects = self.dca.calculate_dynamic_carb_effects(
                    carb_statuses=self.last_carb_statuses,
                    start_time=int(retrospection_start),
                    end_time=int(input_data.timestamp) + 10,
                    time_step=5
                )
            elif input_data.carb_entries:
                # No DCA statuses at all — use static model
                retrospection_carb_effects = self.carb_math.glucose_effect_of_carbs(
                    carb_entries=input_data.carb_entries,
                    start_time=int(retrospection_start),
                    end_time=int(input_data.timestamp) + 10,
                    time_step=5
                )
            else:
                retrospection_carb_effects = []

            # Compute IRC effect from ICE and carb effects
            # Pass additional parameters for iOS-matching safety limits and scaling
            suspend_threshold = self.settings.get('suspend_threshold', 80.0)
            rc_effect, irc_details = self.rc.compute_effect(
                ice_timeline=ice_observations,
                carb_effects=retrospection_carb_effects,
                current_time=input_data.timestamp,
                prediction_horizon=prediction_horizon,
                current_glucose=input_data.cgm_reading,
                isf=self.isf,
                basal_rate=self.basal_rate,
                correction_range=(self.target_low, self.target_high)
            )

        # Combine all effects
        combined_prediction = self.combine_effects(
            current_glucose=input_data.cgm_reading,
            insulin_effect=insulin_predictions,
            carb_effect=carb_predictions,
            momentum_effect=momentum_effect,
            rc_effect=rc_effect
        )

        # Extract just the glucose values for dosing
        prediction_values = [bg for _, bg in combined_prediction]

        # IRC is now stateless - no need to save predictions

        # NEW: Use dosing logic to determine temp basal
        temp_basal_rec = self.dose_math.recommend_from_predictions(
            predictions=prediction_values,
            current_time=input_data.timestamp,
            iob=iob
        )

        # Calculate IRC contribution to next and eventual BG
        irc_next = 0.0
        irc_eventual = 0.0
        if rc_effect:
            # Next BG is typically at index 1 (5 minutes out)
            if len(rc_effect) > 1:
                irc_next = rc_effect[1][1]
            # Eventual BG is the last value
            irc_eventual = rc_effect[-1][1]

        # Calculate momentum contribution to eventual BG
        momentum_eventual = 0.0
        if momentum_effect:
            # Eventual is the last value
            momentum_eventual = momentum_effect[-1][1]

        # Extract IRC PID details
        irc_proportional = None
        irc_integral = None
        irc_differential = None
        irc_total = None
        irc_disc_count = None
        irc_discrepancy = None
        irc_discrepancies_timeline = None

        if irc_details:
            irc_proportional = irc_details.get('proportional')
            irc_integral = irc_details.get('integral')
            irc_differential = irc_details.get('differential')
            irc_total = irc_details.get('total')
            irc_disc_count = irc_details.get('discrepancies_count')
            irc_discrepancy = irc_details.get('most_recent_discrepancy')
            irc_discrepancies_timeline = irc_details.get('discrepancies_timeline')

        # Extract just glucose values from prediction timelines for separate predictions
        insulin_only_values = [bg for _, bg in insulin_predictions] if insulin_predictions else []
        carbs_only_values = [bg for _, bg in carb_predictions] if carb_predictions else []

        output = AlgorithmOutput(
            timestamp=input_data.timestamp,
            temp_basal_rate=temp_basal_rec.rate,
            temp_basal_duration=temp_basal_rec.duration,
            bolus=getattr(temp_basal_rec, 'auto_bolus', None),
            iob=iob,
            cob=cob,
            glucose_predictions={
                'main': prediction_values,
                'insulin_only': insulin_only_values,
                'carbs_only': carbs_only_values
            },
            reason=temp_basal_rec.reason,
            insulin_req=temp_basal_rec.insulin_req,
            insulin_req_explanation=temp_basal_rec.insulin_req_explanation,
            irc_effect_next=irc_next,
            irc_effect_eventual=irc_eventual,
            irc_discrepancy=irc_discrepancy,
            irc_proportional=irc_proportional,
            irc_integral=irc_integral,
            irc_differential=irc_differential,
            irc_total_correction=irc_total,
            irc_discrepancies_count=irc_disc_count,
            irc_discrepancies_timeline=irc_discrepancies_timeline,
            momentum_effect_eventual=momentum_eventual,
            momentum_effect_timeline=momentum_effect,  # Full timeline before blending
            irc_effect_timeline=rc_effect,  # Full timeline before blending
            # NEW: RealisticDoseMath fields
            insulin_per_5min=temp_basal_rec.insulin_per_5min,
            ideal_temp_basal_rate=temp_basal_rec.ideal_rate,
            temp_basal_was_capped=temp_basal_rec.was_capped,
            temp_basal_was_floored=temp_basal_rec.was_floored
        )

        # Add carb statuses if available (not part of base AlgorithmOutput)
        if hasattr(self, 'last_carb_statuses'):
            output.carb_statuses = self.last_carb_statuses

        return output
