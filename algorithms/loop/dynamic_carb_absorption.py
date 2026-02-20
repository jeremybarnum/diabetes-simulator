"""
Dynamic Carb Absorption (DCA) - Maps ICE observations to carb entries.

Based on CarbMath.swift map() function (lines 810-911).
"""

from typing import List, Tuple, Optional
import numpy as np
from algorithms.loop.carb_status import CarbStatusBuilder, CarbStatus
from algorithms.loop.insulin_counteraction import InsulinCounteraction


class DynamicCarbAbsorption:
    """
    Processes Insulin Counteraction Effects and attributes them to carb entries.

    Based on the map() function in CarbMath.swift.
    """

    def __init__(
        self,
        carb_sensitivity_factor: float,  # mg/dL per gram
        absorption_time_overrun: float = 1.5,
        delay_minutes: float = 10.0,
        absorption_model=None
    ):
        """
        Initialize DCA processor.

        Args:
            carb_sensitivity_factor: CSF in mg/dL per gram (ISF/CR)
            absorption_time_overrun: Max absorption multiplier (1.5 for carb expiry)
            delay_minutes: Carb effect delay
            absorption_model: Absorption model to use
        """
        self.csf = carb_sensitivity_factor
        self.absorption_time_overrun = absorption_time_overrun
        self.delay_minutes = delay_minutes
        self.absorption_model = absorption_model

        self.ice_calculator = InsulinCounteraction()

    def process_carb_entries(
        self,
        carb_entries: List[Tuple[float, float, float]],  # [(time, grams, absorption_hrs), ...]
        ice_observations: List[Tuple[float, float, float]],  # [(start, end, ice_velocity), ...]
        current_time: float
    ) -> List[CarbStatus]:
        """
        Map ICE observations to carb entries to determine dynamic absorption.

        Based on CarbMath.swift map() function lines 810-911.

        Algorithm:
        1. Create a CarbStatusBuilder for each carb entry
        2. For each ICE observation:
           - Find all "active" carb entries (overlapping in time)
           - Calculate instantaneous absorption rate for each active entry
           - Split ICE proportionally based on those rates
           - Handle overrun (remaining ICE goes to last entry)
        3. Return CarbStatus for each entry

        Args:
            carb_entries: List of (time, grams, absorption_hours) tuples
            ice_observations: List of (start_time, end_time, ice_mg_dL_per_min) tuples
            current_time: Current simulation time

        Returns:
            List of CarbStatus objects
        """
        if len(carb_entries) == 0:
            return []

        # Create builders for each carb entry
        # iOS uses effectVelocities.last?.endDate as lastEffectDate, NOT current_time.
        # This matters for dynamicAbsorptionTime and estimatedTimeRemaining calculations.
        last_ice_end = ice_observations[-1][1] if ice_observations else current_time
        builders: List[CarbStatusBuilder] = []
        for entry_time, grams, absorption_hrs in carb_entries:
            builder = CarbStatusBuilder(
                entry_time=entry_time,
                entry_grams=grams,
                declared_absorption_hours=absorption_hrs,
                carb_sensitivity_factor=self.csf,
                absorption_time_overrun=self.absorption_time_overrun,
                delay_minutes=self.delay_minutes,
                last_effect_time=last_ice_end,
                absorption_model=self.absorption_model
            )
            builders.append(builder)

        # Process each ICE observation
        for ice_start, ice_end, ice_velocity in ice_observations:
            if ice_end <= ice_start:
                continue

            # Clamp negative ICE to 0 for DCA (carb absorption can't be negative)
            # IRC uses raw ICE (including negatives), but DCA only cares about positive
            # Note: ICE calculator now returns raw values - consumers clamp as needed
            ice_velocity = max(0.0, ice_velocity)

            # Convert velocity to total effect for this interval
            interval_minutes = ice_end - ice_start
            total_effect_value = ice_velocity * interval_minutes  # mg/dL

            # Find active builders (carbs still absorbing during this ICE window)
            active_builders = [
                b for b in builders
                if ice_start < b.max_end_time and ice_start >= b.entry_time
            ]

            if len(active_builders) == 0:
                # No active entries - this is a "phantom meal"
                # (ICE with no declared carbs to explain it)
                continue

            # Calculate absorption rate for each active entry at this time
            # This determines how to split the ICE effect
            total_rate = 0.0
            builder_rates = []

            for builder in active_builders:
                effect_time = ice_start - builder.entry_time
                # iOS uses dynamicAbsorptionTime = observedDates.duration + estimatedTimeRemaining
                # observedDates.duration = lastEffectTime - entryTime (elapsed time)
                observed_duration = builder.last_effect_time - builder.entry_time
                dynamic_absorption_time = min(
                    observed_duration + builder.estimated_time_remaining,
                    builder.max_absorption_time
                )
                absorption_rate = builder.absorption_model.absorptionRateAtTime(
                    t=effect_time / 60.0,  # Convert to hours
                    absorptionTime=dynamic_absorption_time / 60.0  # Convert to hours
                )
                # Convert rate (fraction/hour) to grams/min
                rate_grams_per_min = absorption_rate * builder.entry_grams / 60.0
                builder_rates.append(rate_grams_per_min)
                total_rate += rate_grams_per_min

            # Distribute effect across active builders proportionally
            remaining_effect = total_effect_value

            for i, builder in enumerate(active_builders):
                if total_rate > 0:
                    # Proportional share based on absorption rate
                    # iOS: partialEffectValue = min(remainingEffect, (rate/totalRate) * effectValue)
                    # where effectValue is decremented each iteration (remaining_effect here)
                    rate_fraction = builder_rates[i] / total_rate
                    partial_effect = rate_fraction * remaining_effect

                    # Don't exceed remaining effect for this entry
                    partial_effect = min(partial_effect, builder.remaining_effect)
                else:
                    partial_effect = 0.0

                # Convert effect back to velocity for recording
                partial_velocity = partial_effect / interval_minutes
                builder.add_ice_observation(ice_start, ice_end, partial_velocity)

                remaining_effect -= partial_effect
                total_rate -= builder_rates[i]

                # If this is the last builder and there's still remaining effect,
                # attribute the overrun to it (lines 897-900)
                if remaining_effect > 1e-6 and builder is active_builders[-1]:
                    overrun_velocity = remaining_effect / interval_minutes
                    builder.add_ice_observation(ice_start, ice_end, overrun_velocity)
                    remaining_effect = 0.0

            # Any remaining effect with no active builders is a phantom meal
            # (TODO in LoopKit)

        # Convert builders to CarbStatus objects
        statuses = []
        for i, builder in enumerate(builders):
            entry_time, grams, absorption_hrs = carb_entries[i]
            absorption_info = builder.get_absorption_info()

            # Only include timeline if observed >= minimum predicted
            # OR if timeline shows 100% completion (prevents COB resurrection when ICE ages out)
            timeline = None
            if builder.observed_grams >= builder.min_predicted_grams:
                timeline = builder.observed_timeline
            elif builder.observed_timeline:
                # Check if timeline shows 100% absorption ever achieved
                total_observed = sum(cv.grams for cv in builder.observed_timeline)
                if total_observed >= grams - 0.1:  # Within tolerance
                    # Keep timeline - it shows historical completion
                    # This prevents COB resurrection when current observations are low
                    timeline = builder.observed_timeline

            status = CarbStatus(
                entry_time=entry_time,
                entry_grams=grams,
                declared_absorption_hours=absorption_hrs,
                absorption=absorption_info,
                observed_timeline=timeline
            )
            statuses.append(status)

        return statuses

    def calculate_dynamic_cob(
        self,
        carb_statuses: List[CarbStatus],
        at_time: float
    ) -> float:
        """
        Calculate total dynamic COB from all carb statuses.

        Args:
            carb_statuses: List of CarbStatus objects
            at_time: Time to calculate COB

        Returns:
            Total COB (grams)
        """
        total_cob = 0.0
        for status in carb_statuses:
            cob = status.dynamic_carbs_on_board(
                at_time=at_time,
                delay_minutes=self.delay_minutes,
                absorption_model=self.absorption_model
            )
            total_cob += cob

        return total_cob

    def calculate_dynamic_carb_effects(
        self,
        carb_statuses: List[CarbStatus],
        start_time: float,
        end_time: float,
        time_step: float = 5.0
    ) -> List[Tuple[float, float]]:
        """
        Calculate dynamic glucose effects from carbs over time.

        Uses observed absorption where available, predicted otherwise.

        Args:
            carb_statuses: List of CarbStatus objects
            start_time: Start time (minutes)
            end_time: End time (minutes)
            time_step: Time step (minutes)

        Returns:
            List of (time, cumulative_effect_mg_dL) tuples
        """
        times = np.arange(start_time, end_time + time_step, time_step)
        effects = []

        for t in times:
            # For each time point, calculate cumulative absorbed carbs
            total_absorbed = 0.0

            for status in carb_statuses:
                if t < status.entry_time:
                    continue

                # Calculate absorbed carbs at this time
                cob_now = status.dynamic_carbs_on_board(
                    at_time=t,
                    delay_minutes=self.delay_minutes,
                    absorption_model=self.absorption_model
                )
                absorbed = status.entry_grams - cob_now
                total_absorbed += absorbed

            # Convert to glucose effect
            effect = total_absorbed * self.csf
            effects.append((t, effect))

        return effects

    def calculate_observed_carb_effects(
        self,
        carb_statuses: List[CarbStatus],
        start_time: float,
        end_time: float,
        time_step: float = 5.0
    ) -> List[Tuple[float, float]]:
        """
        Calculate glucose effects from OBSERVED carb absorption (for IRC).

        This uses the observed_timeline when available, which can show absorption
        exceeding entry_grams (e.g., 30g observed when 15g declared).

        This is what IRC should use to properly detect unexplained carbs.

        Args:
            carb_statuses: List of CarbStatus objects
            start_time: Start time (minutes)
            end_time: End time (minutes)
            time_step: Time step (minutes)

        Returns:
            List of (time, cumulative_effect_mg_dL) tuples
        """
        times = np.arange(start_time, end_time + time_step, time_step)
        effects = []

        for t in times:
            # For each time point, calculate cumulative OBSERVED absorbed carbs
            total_absorbed = 0.0

            for status in carb_statuses:
                if t < status.entry_time:
                    continue

                # Use observed timeline if available (shows actual absorption via ICE)
                if status.observed_timeline is not None:
                    # Sum all observed carbs up to time t
                    observed_absorbed = 0.0
                    for carb_value in status.observed_timeline:
                        if carb_value.end_time <= t:
                            observed_absorbed += carb_value.grams
                        elif carb_value.start_time < t:
                            # Partial interval: interpolate
                            interval_duration = carb_value.end_time - carb_value.start_time
                            if interval_duration > 0:
                                fraction = (t - carb_value.start_time) / interval_duration
                                observed_absorbed += carb_value.grams * fraction
                    total_absorbed += observed_absorbed
                else:
                    # Fall back to dynamic COB calculation (capped at entry_grams)
                    cob_now = status.dynamic_carbs_on_board(
                        at_time=t,
                        delay_minutes=self.delay_minutes,
                        absorption_model=self.absorption_model
                    )
                    absorbed = status.entry_grams - cob_now
                    total_absorbed += absorbed

            # Convert to glucose effect
            effect = total_absorbed * self.csf
            effects.append((t, effect))

        return effects
