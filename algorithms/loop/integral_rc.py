"""
Integral Retrospective Correction (IRC) implementation.

Based on LoopKit's IntegralRetrospectiveCorrection.swift.
Adjusted to EXACTLY match iOS Loop's PID controller with:
- Dynamic effect duration (grows with number of discrepancies)
- Scaled correction for extended duration
- Safety limits on integral correction
- Correct velocity and decay projection

IMPLEMENTATION APPROACH:
- Stateless: No stored predictions
- Discrepancies calculated from ICE (Insulin Counteraction Effects)
- Discrepancy = ICE - expected_carb_effects
- Represents glucose changes NOT explained by insulin OR carbs
"""

from typing import List, Tuple, Optional
import math


class IntegralRetrospectiveCorrection:
    """
    Integral Retrospective Correction - EXACT match to iOS Loop.

    PID controller that adjusts predictions based on past modeling errors.
    """

    # Constants from IntegralRetrospectiveCorrection.swift
    RETROSPECTION_INTERVAL = 180.0          # 3 hours (minutes)
    CURRENT_DISCREPANCY_GAIN = 1.0
    PERSISTENT_DISCREPANCY_GAIN = 2.0       # was 5.0 in older versions
    CORRECTION_TIME_CONSTANT = 60.0         # minutes (was 90.0)
    DIFFERENTIAL_GAIN = 2.0
    DELTA = 5.0                             # glucose sampling interval (minutes)
    MAXIMUM_CORRECTION_EFFECT_DURATION = 180.0  # minutes (was 240.0)

    # Derived parameters (matching Swift static let computations)
    INTEGRAL_FORGET = math.exp(-DELTA / CORRECTION_TIME_CONSTANT)
    INTEGRAL_GAIN = ((1 - INTEGRAL_FORGET) / INTEGRAL_FORGET) * \
                    (PERSISTENT_DISCREPANCY_GAIN - CURRENT_DISCREPANCY_GAIN)
    PROPORTIONAL_GAIN = CURRENT_DISCREPANCY_GAIN - INTEGRAL_GAIN

    def __init__(self,
                 effect_duration_minutes: float = 60.0,
                 grouping_interval_minutes: float = 30.0,
                 retrospection_interval_minutes: float = 180.0):
        self.effect_duration = effect_duration_minutes
        self.grouping_interval = grouping_interval_minutes
        self.retrospection_interval = retrospection_interval_minutes

    def calculate_discrepancies_from_ice(self,
                                        ice_timeline: List[Tuple[float, float, float]],
                                        carb_effects: List[Tuple[float, float]],
                                        current_time: float) -> List[dict]:
        """
        Calculate discrepancies from ICE and carb effects.

        Matches iOS Loop's subtracting() function (LoopMath.swift:279-334):
        1. Mutual time-trim between ICE and carb effects
        2. Iterate through carb effect grid (5-min steps), not ICE intervals
        3. Use fixed 5-min effectInterval for velocity→effect conversion

        Returns list of dicts with 'time', 'discrepancy', 'ice', 'carb_effect' keys,
        grouped into 30-minute sliding windows (matching Loop's combinedSums).
        """
        start_time = current_time - self.retrospection_interval
        effect_interval = self.DELTA  # 5 minutes, matching iOS defaultDelta

        # Filter ICE to retrospection window
        ice_filtered = [
            (s, e, v) for s, e, v in ice_timeline
            if e >= start_time and s <= current_time
        ]

        if not ice_filtered or not carb_effects:
            # No carb effects: process all ICE with zero carb change
            # (matching iOS subtracting() fallthrough loop)
            raw_discrepancies = []
            for ice_start, ice_end, ice_velocity in ice_filtered:
                ice_effect = ice_velocity * effect_interval
                midpoint_time = (ice_start + ice_end) / 2.0
                raw_discrepancies.append({
                    'time': midpoint_time,
                    'duration': ice_end - ice_start,
                    'discrepancy': ice_effect,
                    'ice': ice_effect,
                    'carb_effect': 0.0
                })
            if not raw_discrepancies:
                return []
            return self._combined_sums(raw_discrepancies)

        # Mutual trim matching iOS subtracting() lines 281-282:
        # 1. Trim carb effects to those with endDate >= first ICE endDate
        first_ice_end = ice_filtered[0][1]
        carb_trimmed = [(t, v) for t, v in carb_effects if t >= first_ice_end]

        if not carb_trimmed:
            return []

        # 2. Trim ICE to those with endDate >= first remaining carb startDate
        first_carb_time = carb_trimmed[0][0]
        ice_trimmed = [
            (s, e, v) for s, e, v in ice_filtered
            if e >= first_carb_time
        ]

        if not ice_trimmed:
            return []

        # Iterate through carb effect pairs (matching iOS subtracting loop)
        # iOS iterates otherEffects.dropFirst(), pairing each with an ICE interval
        raw_discrepancies = []
        previous_carb_value = carb_trimmed[0][1]
        ice_idx = 0

        for carb_t, carb_v in carb_trimmed[1:]:  # dropFirst()
            if ice_idx >= len(ice_trimmed):
                break

            carb_change = carb_v - previous_carb_value
            previous_carb_value = carb_v

            ice_start, ice_end, ice_velocity = ice_trimmed[ice_idx]

            # iOS guard: effect.endDate <= otherEffect.endDate
            # (carb_t serves as otherEffect.endDate since GlucoseEffect has endDate==startDate)
            if ice_end > carb_t:
                continue  # Skip this carb step, try next one

            ice_idx += 1

            # iOS: effectValue * effectInterval (fixed 5-min, NOT actual ICE duration)
            ice_effect = ice_velocity * effect_interval

            discrepancy = ice_effect - carb_change
            midpoint_time = (ice_start + ice_end) / 2.0
            raw_discrepancies.append({
                'time': midpoint_time,
                'duration': ice_end - ice_start,
                'discrepancy': discrepancy,
                'ice': ice_effect,
                'carb_effect': carb_change
            })

        # Process remaining ICE intervals with zero carb change
        # (matching iOS subtracting() fallthrough loop at line 320)
        for ice_start, ice_end, ice_velocity in ice_trimmed[ice_idx:]:
            ice_effect = ice_velocity * effect_interval
            midpoint_time = (ice_start + ice_end) / 2.0
            raw_discrepancies.append({
                'time': midpoint_time,
                'duration': ice_end - ice_start,
                'discrepancy': ice_effect,
                'ice': ice_effect,
                'carb_effect': 0.0
            })

        if not raw_discrepancies:
            return []

        return self._combined_sums(raw_discrepancies)

    def _combined_sums(self, raw_discrepancies: List[dict]) -> List[dict]:
        """
        Group raw discrepancies into sliding windows matching iOS combinedSums().
        iOS uses endDate for window comparisons and a 1.01 multiplier on the window.
        Each entry accumulates all earlier entries within the window duration.
        """
        window_duration = self.grouping_interval * 1.01  # Match iOS 1.01 multiplier

        sliding_window_discrepancies = []

        for i, current in enumerate(raw_discrepancies):
            # Use ICE interval END time (midpoint + half_duration)
            current_end = current['time'] + current['duration'] / 2.0

            window_discrepancy = 0.0
            window_ice = 0.0
            window_carb = 0.0

            # Include current + all earlier entries within window_duration
            for d in raw_discrepancies[:i+1]:
                d_end = d['time'] + d['duration'] / 2.0
                # iOS check: d_end + window_duration >= current_end
                if d_end + window_duration >= current_end:
                    window_discrepancy += d['discrepancy']
                    window_ice += d['ice']
                    window_carb += d['carb_effect']

            sliding_window_discrepancies.append({
                'time': current['time'],
                'discrepancy': window_discrepancy,
                'ice': window_ice,
                'carb_effect': window_carb
            })

        return sliding_window_discrepancies

    @staticmethod
    def _interpolate(effects: list, target_time: float) -> float:
        """Interpolate effect value at target_time."""
        if not effects:
            return 0.0
        if target_time <= effects[0][0]:
            return effects[0][1]
        if target_time >= effects[-1][0]:
            return effects[-1][1]
        for i in range(len(effects) - 1):
            t0, v0 = effects[i]
            t1, v1 = effects[i + 1]
            if t0 <= target_time <= t1:
                if t1 == t0:
                    return v0
                frac = (target_time - t0) / (t1 - t0)
                return v0 + frac * (v1 - v0)
        return effects[-1][1]

    def compute_pid_correction(self,
                              discrepancies: List[dict],
                              current_glucose: float = 100.0,
                              isf: float = 100.0,
                              basal_rate: float = 0.45,
                              correction_range_min: float = 100.0,
                              correction_range_max: float = 100.0,
                              recency_interval: float = 15.0
                              ) -> dict:
        """
        Compute PID correction from discrepancies.
        EXACTLY matches IntegralRetrospectiveCorrection.swift computeEffect().

        Returns dict with all IRC details.
        """
        result = {
            'total_correction': 0.0,
            'scaled_correction': 0.0,
            'proportional': 0.0,
            'integral': 0.0,
            'differential': 0.0,
            'effect_duration': self.effect_duration,
            'discrepancies_count': 0,
            'recent_discrepancy_values': [],
        }

        if not discrepancies:
            return result

        current_discrepancy = discrepancies[-1]['discrepancy']

        # Find contiguous same-sign discrepancies from the end
        # Matching Swift: checks sign AND recency (gap <= recencyInterval)
        current_sign = math.copysign(1, current_discrepancy)
        recent_values = []
        next_disc = discrepancies[-1]

        for i in range(len(discrepancies) - 1, -1, -1):
            d = discrepancies[i]
            d_value = d['discrepancy']
            d_sign = math.copysign(1, d_value)

            # Check same sign, recent enough, and above threshold
            time_gap = next_disc['time'] - d['time']
            if (d_sign == current_sign and
                time_gap <= recency_interval and
                abs(d_value) >= 0.1):
                recent_values.append(d_value)
                next_disc = d
            else:
                break

        recent_values = list(reversed(recent_values))
        result['recent_discrepancy_values'] = recent_values
        result['discrepancies_count'] = len(recent_values)

        if not recent_values:
            return result

        # Safety limits for integral correction (from Swift lines 146-151)
        glucose_error = current_glucose - correction_range_max
        zero_temp_effect = abs(isf * basal_rate)
        integral_positive_limit = min(max(glucose_error, 1.0 * zero_temp_effect),
                                      4.0 * zero_temp_effect)
        integral_negative_limit = -max(10.0, current_glucose - correction_range_min)

        # Dynamic effect duration (Swift lines 155, 160, 164)
        # Starts at effectDuration - 2*delta, grows by 2*delta per discrepancy
        effect_duration_minutes = self.effect_duration - 2.0 * self.DELTA
        for _ in recent_values:
            effect_duration_minutes += 2.0 * self.DELTA

        # Cap at maximum
        effect_duration_minutes = min(effect_duration_minutes,
                                      self.MAXIMUM_CORRECTION_EFFECT_DURATION)

        # Integral correction with exponential forgetting (Swift lines 156-161)
        integral_correction = 0.0
        for disc in recent_values:
            integral_correction = (self.INTEGRAL_FORGET * integral_correction +
                                   self.INTEGRAL_GAIN * disc)

        # Apply safety limits (Swift line 163)
        integral_correction = min(max(integral_correction, integral_negative_limit),
                                  integral_positive_limit)

        # Proportional correction (Swift line 174)
        proportional_correction = self.PROPORTIONAL_GAIN * current_discrepancy

        # Differential correction - only when negative (Swift lines 167-181)
        differential_correction = 0.0
        if len(recent_values) > 1:
            differential_discrepancy = current_discrepancy - recent_values[-2]
            if differential_discrepancy < 0.0:
                differential_correction = self.DIFFERENTIAL_GAIN * differential_discrepancy

        # Total correction (Swift line 183)
        total_correction = proportional_correction + integral_correction + differential_correction

        # Scaled correction for extended duration (Swift line 188)
        # scaledCorrection = totalCorrection * effectDuration / integralCorrectionEffectDuration
        scaled_correction = total_correction * self.effect_duration / effect_duration_minutes

        result['total_correction'] = total_correction
        result['scaled_correction'] = scaled_correction
        result['proportional'] = proportional_correction
        result['integral'] = integral_correction
        result['differential'] = differential_correction
        result['effect_duration'] = effect_duration_minutes

        return result

    def project_correction_forward(self,
                                   scaled_correction: float,
                                   discrepancy_time: float,
                                   effect_duration: float,
                                   current_time: float,
                                   end_time: float,
                                   time_step: float = 5.0) -> List[Tuple[float, float]]:
        """
        Project the correction effect forward using decayEffect.
        EXACTLY matches iOS Loop's velocity + decayEffect calculation.

        Args:
            scaled_correction: Scaled correction value (from PID, adjusted for duration)
            discrepancy_time: Time interval for velocity computation (seconds in iOS, minutes here)
            effect_duration: Dynamic effect duration (minutes)
            current_time: Current time in minutes
            end_time: End of prediction horizon in minutes
            time_step: Time step (default 5 min)
        """
        # Velocity = scaledCorrection / discrepancyTime (Swift line 193)
        velocity = scaled_correction / discrepancy_time if discrepancy_time > 0 else 0.0

        # decayEffect: linear velocity decay from velocity to 0 over effect_duration
        # Starting from delta minutes after current time
        # Formula from LoopMath.swift decayEffect():
        #   intercept = velocity (starting rate)
        #   slope = -velocity / (duration - delta)
        #   decayStartDate = startDate + delta
        #   value = lastValue + (intercept + slope * (date - decayStartDate)) * delta

        delta = time_step

        # Align to the same integer-minute grid as insulin/carb effects
        # This prevents doubling of prediction time points in combine_effects
        grid_start = int(current_time)
        decay_start = grid_start + int(delta)
        slope = -velocity / (effect_duration - delta) if effect_duration > delta else 0.0

        corrections = []
        corrections.append((grid_start, 0.0))

        date = decay_start
        last_value = 0.0

        while date <= grid_start + effect_duration:
            if date > end_time:
                break
            value = last_value + (velocity + slope * (date - decay_start)) * delta
            corrections.append((date, value))
            last_value = value
            date += int(delta)

        # After effect_duration, maintain the last value
        date = grid_start + int(effect_duration) + int(delta)
        while date <= end_time:
            corrections.append((date, last_value))
            date += int(delta)

        return corrections

    def compute_effect(self,
                      ice_timeline: List[Tuple[float, float, float]],
                      carb_effects: List[Tuple[float, float]],
                      current_time: float,
                      prediction_horizon: float = 360.0,
                      current_glucose: float = 100.0,
                      isf: float = 100.0,
                      basal_rate: float = 0.45,
                      correction_range: Tuple[float, float] = (100.0, 100.0)
                      ) -> Tuple[List[Tuple[float, float]], dict]:
        """
        Compute IRC effect for future predictions.
        Main method - orchestrates discrepancy calculation, PID, and projection.
        """
        # Calculate discrepancies
        discrepancies = self.calculate_discrepancies_from_ice(
            ice_timeline=ice_timeline,
            carb_effects=carb_effects,
            current_time=current_time
        )

        irc_details = {
            'proportional': 0.0, 'integral': 0.0, 'differential': 0.0,
            'total': 0.0, 'discrepancies_count': 0,
            'most_recent_discrepancy': 0.0, 'discrepancies_timeline': [],
        }

        if not discrepancies:
            return ([], irc_details)

        # Compute PID correction with safety limits
        pid_result = self.compute_pid_correction(
            discrepancies=discrepancies,
            current_glucose=current_glucose,
            isf=isf,
            basal_rate=basal_rate,
            correction_range_min=correction_range[0],
            correction_range_max=correction_range[1],
        )

        irc_details['proportional'] = pid_result['proportional']
        irc_details['integral'] = pid_result['integral']
        irc_details['differential'] = pid_result['differential']
        irc_details['total'] = pid_result['total_correction']
        irc_details['discrepancies_count'] = pid_result['discrepancies_count']
        irc_details['most_recent_discrepancy'] = discrepancies[-1]['discrepancy'] if discrepancies else 0.0
        irc_details['discrepancies_timeline'] = discrepancies

        if abs(pid_result['total_correction']) < 0.1:
            return ([], irc_details)

        # Compute discrepancy time for velocity calculation
        # Swift: max(retrospectionTimeInterval, retrospectiveCorrectionGroupingInterval)
        if discrepancies:
            last_disc = discrepancies[-1]
            # The retrospection interval is the grouping interval for summed discrepancies
            retrospection_time = self.grouping_interval
        else:
            retrospection_time = self.grouping_interval

        discrepancy_time = max(retrospection_time, self.grouping_interval)

        # Project forward with decay
        end_time = current_time + prediction_horizon
        correction_timeline = self.project_correction_forward(
            scaled_correction=pid_result['scaled_correction'],
            discrepancy_time=discrepancy_time,
            effect_duration=pid_result['effect_duration'],
            current_time=current_time,
            end_time=end_time,
        )

        return (correction_timeline, irc_details)


def create_integral_rc(effect_duration_minutes: float = 60.0,
                       **kwargs) -> IntegralRetrospectiveCorrection:
    """Factory function."""
    return IntegralRetrospectiveCorrection(
        effect_duration_minutes=effect_duration_minutes,
        **kwargs
    )
