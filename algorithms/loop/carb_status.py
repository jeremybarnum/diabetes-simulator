"""
CarbStatus - Tracks observed vs predicted carb absorption.

Based on CarbStatus.swift and CarbStatusBuilder from CarbMath.swift.
"""

from typing import List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
import os


@dataclass
class CarbValue:
    """A carb absorption observation at a point in time."""
    start_time: float  # minutes
    end_time: float    # minutes
    grams: float       # grams absorbed in this interval


@dataclass
class AbsorptionInfo:
    """Detailed absorption information for a carb entry."""
    observed_grams: float           # Total grams observed from ICE
    clamped_grams: float            # Bounded by [min_predicted, entry_grams]
    total_grams: float              # Original declared grams
    remaining_grams: float          # total - clamped (never negative)
    observed_start_time: float      # When observation started
    observed_end_time: float        # When 100% observed or current time
    estimated_time_remaining: float # Minutes until absorption complete
    time_to_absorb_observed: float  # Minutes taken to absorb observed amount


class CarbStatusBuilder:
    """
    Builds CarbStatus by processing ICE observations.

    Based on CarbStatusBuilder from CarbMath.swift lines 548-791.
    """

    def __init__(
        self,
        entry_time: float,
        entry_grams: float,
        declared_absorption_hours: float,
        carb_sensitivity_factor: float,  # mg/dL per gram
        absorption_time_overrun: float,  # Multiplier for max time (typically 1.5)
        delay_minutes: float,
        last_effect_time: float,
        absorption_model  # CarbAbsorptionComputable
    ):
        """
        Initialize carb status builder.

        Args:
            entry_time: Time carbs were entered (minutes)
            entry_grams: Declared carb amount (grams)
            declared_absorption_hours: User-declared absorption time
            carb_sensitivity_factor: CSF in mg/dL per gram (ISF/CR)
            absorption_time_overrun: Max absorption multiplier (1.5)
            delay_minutes: Carb effect delay (typically 10 min)
            last_effect_time: Last observation time or "now"
            absorption_model: Absorption model for calculations
        """
        self.entry_time = entry_time
        self.entry_grams = entry_grams
        self.carb_sensitivity_factor = carb_sensitivity_factor
        self.absorption_model = absorption_model
        self.delay_minutes = delay_minutes

        # User-entered parameters
        self.initial_absorption_time = declared_absorption_hours * 60  # Convert to minutes
        self.entry_effect = entry_grams * carb_sensitivity_factor  # mg/dL

        # Carb expiry: maximum absorption time is 1.5x declared
        self.max_absorption_time = self.initial_absorption_time * absorption_time_overrun
        self.max_end_time = entry_time + self.max_absorption_time + delay_minutes

        # The last time we have observations
        self.last_effect_time = min(self.max_end_time, max(last_effect_time, entry_time))

        # Observed absorption tracking
        self.observed_effect = 0.0  # mg/dL
        self.observed_timeline: List[CarbValue] = []
        self.observed_completion_time: Optional[float] = None

    @property
    def observed_grams(self) -> float:
        """Grams absorbed based on observations."""
        return self.observed_effect / self.carb_sensitivity_factor

    @property
    def remaining_effect(self) -> float:
        """Effect remaining until 100% absorption observed."""
        return max(self.entry_effect - self.observed_effect, 0.0)

    @property
    def min_predicted_grams(self) -> float:
        """
        Minimum grams that MUST have absorbed by now (carb expiry).

        Based on minimum absorption rate (max absorption time).
        This prevents carbs from "hanging around forever."
        """
        time_since_entry = self.last_effect_time - self.entry_time - self.delay_minutes
        if time_since_entry <= 0:
            return 0.0

        return self.absorption_model.absorbedCarbs(
            of=self.entry_grams,
            atTime=time_since_entry / 60.0,  # Convert to hours
            absorptionTime=self.max_absorption_time / 60.0
        )

    @property
    def clamped_grams(self) -> float:
        """
        Grams absorbed, bounded by minimum and total.

        From CarbMath.swift line 642:
        return min(entryGrams, max(minPredictedGrams, observedGrams))
        """
        return min(self.entry_grams, max(self.min_predicted_grams, self.observed_grams))

    @property
    def percent_absorbed(self) -> float:
        """Fraction of entry absorbed."""
        return self.clamped_grams / self.entry_grams if self.entry_grams > 0 else 0.0

    @property
    def time_to_absorb_observed_carbs(self) -> float:
        """
        Time (minutes) needed to absorb the observed amount.

        When adaptive absorption rate is disabled (IRC off), this uses the MODEL
        to calculate how long the observed percentage should take, not the actual
        observation time. This maintains consistency with the original absorption curve.

        Based on CarbMath.swift lines 650-664.
        """
        time_since_entry = self.last_effect_time - self.entry_time - self.delay_minutes
        if time_since_entry <= 0:
            return 0.0

        # Use model-based time (adaptive rate disabled)
        # Ask the model: "How long should it take to absorb this percentage?"
        time_to_absorb = self.absorption_model.timeToAbsorb(
            forPercentAbsorbed=self.percent_absorbed,  # Use existing property
            totalAbsorptionTime=self.initial_absorption_time / 60.0  # hours
        ) * 60.0  # convert back to minutes

        return min(time_to_absorb, self.max_absorption_time)

    @property
    def estimated_time_remaining(self) -> float:
        """
        Estimated minutes until absorption complete.

        Based on static model (not adaptive rate).
        """
        time_since_entry = self.last_effect_time - self.entry_time - self.delay_minutes
        if time_since_entry <= 0:
            return self.initial_absorption_time

        # Maximum time we can still wait
        not_to_exceed = max(self.max_absorption_time - time_since_entry, 0.0)
        if not_to_exceed <= 0:
            return 0.0

        # Use static model (adaptive rate disabled)
        dynamic_time_remaining = self.initial_absorption_time - self.time_to_absorb_observed_carbs

        return min(dynamic_time_remaining, not_to_exceed)

    def add_ice_observation(self, start_time: float, end_time: float, ice_mg_dL_per_min: float):
        """
        Add an ICE observation to this carb entry.

        Args:
            start_time: Start of observation window (minutes)
            end_time: End of observation window (minutes)
            ice_mg_dL_per_min: ICE velocity (mg/dL per minute)
        """
        if start_time < self.entry_time:
            return

        # Convert velocity to total effect for this interval
        interval_minutes = end_time - start_time
        effect_value = ice_mg_dL_per_min * interval_minutes

        # Add to observed total
        self.observed_effect += effect_value

        # Track in timeline if not yet complete
        if self.observed_completion_time is None:
            grams_absorbed = effect_value / self.carb_sensitivity_factor
            self.observed_timeline.append(
                CarbValue(
                    start_time=start_time,
                    end_time=end_time,
                    grams=grams_absorbed
                )
            )

            # Check if 100% observed (with floating point tolerance)
            if self.observed_effect + 1e-6 >= self.entry_effect:
                self.observed_completion_time = end_time

    def get_absorption_info(self) -> AbsorptionInfo:
        """Get current absorption status."""
        observed_end = self.observed_completion_time or self.last_effect_time

        return AbsorptionInfo(
            observed_grams=self.observed_grams,
            clamped_grams=self.clamped_grams,
            total_grams=self.entry_grams,
            remaining_grams=max(self.entry_grams - self.clamped_grams, 0.0),
            observed_start_time=self.entry_time,
            observed_end_time=observed_end,
            estimated_time_remaining=self.estimated_time_remaining,
            time_to_absorb_observed=self.time_to_absorb_observed_carbs
        )


@dataclass
class CarbStatus:
    """
    Complete carb absorption status including observed vs predicted.

    Based on CarbStatus.swift.
    """
    entry_time: float
    entry_grams: float
    declared_absorption_hours: float
    absorption: Optional[AbsorptionInfo]
    observed_timeline: Optional[List[CarbValue]]  # None if observed < minimum

    def dynamic_carbs_on_board(
        self,
        at_time: float,
        delay_minutes: float,
        absorption_model,
        delta_minutes: float = 5.0,
        debug: bool = False
    ) -> float:
        """
        Calculate dynamic COB at a given time.

        Based on CarbStatus.swift dynamicCarbsOnBoard() lines 45-79.

        Three modes:
        1. Before/insufficient observation: Use model-based absorption
        2. During observation: Use actual observed timeline
        3. After observation: Project remaining carbs forward

        Args:
            at_time: Time to calculate COB (minutes)
            delay_minutes: Carb effect delay
            absorption_model: Absorption model
            delta_minutes: Calculation interval (tolerance for entry time)
            debug: Print debug info

        Returns:
            Carbs on board (grams)
        """
        # Before carb entry (minus delta tolerance), COB is 0 (carbs don't exist yet)
        # The delta creates a tolerance window for interval-based calculations
        if at_time < self.entry_time - delta_minutes:
            return 0.0

        # Stateless completion check: If carbs are fully absorbed, COB stays at 0
        # This prevents "COB resurrection" when ICE observations age out

        # Method 1: Check if observed timeline shows 100% absorption
        if self.absorption is not None and self.observed_timeline is not None:
            # Check if cumulative observed absorption reached 100%
            total_observed = sum(cv.grams for cv in self.observed_timeline)
            if total_observed >= self.entry_grams - 0.1:  # Within tolerance
                # Find when 100% was reached
                cumulative = 0.0
                completion_time = None
                for cv in self.observed_timeline:
                    cumulative += cv.grams
                    if cumulative >= self.entry_grams - 0.1:
                        completion_time = cv.end_time
                        break

                # If we're past completion, COB is 0 forever
                if completion_time is not None and at_time >= completion_time:
                    if debug or os.getenv('DEBUG_COB'):
                        print(f"    COMPLETION: Carbs fully absorbed at t={completion_time:.1f}min, COB=0")
                    return 0.0

        # Method 2: Check if clamped_grams shows 100% absorption (via observation or expiry)
        # This is the key mechanism - clamped_grams enforces minimum absorption via carb expiry
        if self.absorption is not None:
            if self.absorption.clamped_grams >= self.entry_grams - 0.1:
                # Carbs are 100% absorbed (either observed or forced by expiry)
                # Use observed_end_time as completion time
                completion_time = self.absorption.observed_end_time
                if at_time >= completion_time:
                    if debug or os.getenv('DEBUG_COB'):
                        print(f"    COMPLETION (clamped): Carbs fully absorbed, COB=0")
                    return 0.0

        if self.absorption is None:
            # No absorption info, use static model
            time_since_entry = (at_time - self.entry_time) / 60.0  # hours
            return absorption_model.unabsorbedCarbs(
                of=self.entry_grams,
                atTime=time_since_entry - delay_minutes / 60.0,
                absorptionTime=self.declared_absorption_hours
            )

        # Have absorption info
        if self.observed_timeline is None or len(self.observed_timeline) == 0:
            # Less than minimum observed or observation not yet started
            # Loop's exact formula from CarbStatus.swift lines 55-61
            time_since_entry = (at_time - self.entry_time) / 60.0  # hours

            # Use Loop's estimatedDate.duration formula (time-dependent by design!)
            # estimatedDate.duration = observedDate.duration + estimatedTimeRemaining
            # where observedDate.duration = lastEffectTime - entryTime
            observed_duration = (self.absorption.observed_end_time -
                               self.absorption.observed_start_time) / 60.0  # hours
            absorption_time = observed_duration + self.absorption.estimated_time_remaining / 60.0

            cob = absorption_model.unabsorbedCarbs(
                of=self.entry_grams,
                atTime=time_since_entry - delay_minutes / 60.0,
                absorptionTime=absorption_time
            )

            if debug or os.getenv('DEBUG_COB'):
                print(f"    MODE 1 (before/insufficient obs): COB={cob:.2f}g")
                print(f"      time_since_entry={time_since_entry*60:.1f}min, absorption_time={absorption_time*60:.1f}min")
                print(f"      observed_grams={self.absorption.observed_grams:.2f}g")

            return cob

        observation_end = self.observed_timeline[-1].end_time

        if at_time > observation_end:
            # After observation period - Loop's exact formula from lines 63-71
            # Predicted absorption for remaining carbs, post-observation

            time_since_observation_end = at_time - observation_end  # minutes
            effective_time = (time_since_observation_end +
                            self.absorption.time_to_absorb_observed) / 60.0  # hours

            # Total absorption time = model time for observed + estimated remaining
            effective_absorption_time = (self.absorption.time_to_absorb_observed +
                                       self.absorption.estimated_time_remaining) / 60.0  # hours

            # Use TOTAL grams (not remaining) with the effective times
            unabsorbed = absorption_model.unabsorbedCarbs(
                of=self.entry_grams,  # Use total, not remaining
                atTime=effective_time,
                absorptionTime=effective_absorption_time
            )
            cob = max(unabsorbed, 0.0)

            if debug or os.getenv('DEBUG_COB'):
                print(f"    MODE 3 (after obs): COB={cob:.2f}g")
                print(f"      observation_end={observation_end:.1f}min, time_since_obs_end={time_since_observation_end:.1f}min")
                print(f"      effective_time={effective_time*60:.1f}min, effective_absorption_time={effective_absorption_time*60:.1f}min")
                print(f"      time_to_absorb_observed={self.absorption.time_to_absorb_observed:.1f}min, estimated_time_remaining={self.absorption.estimated_time_remaining:.1f}min")
                print(f"      observed_grams={self.absorption.observed_grams:.2f}g")

            return cob

        # During observation period - use max of observed and minimum rate
        observed_absorbed = sum(
            cv.grams for cv in self.observed_timeline
            if cv.end_time <= at_time
        )

        # Calculate minimum absorption at this specific time (carb expiry enforcement)
        # Formula: COB = entry_grams - max(cumulative_observed, cumulative_minimum)
        time_since_entry = at_time - self.entry_time - delay_minutes
        if time_since_entry > 0:
            # Max absorption time is declared * 1.5 (carb expiry)
            max_absorption_hours = self.declared_absorption_hours * 1.5
            min_absorbed = absorption_model.absorbedCarbs(
                of=self.entry_grams,
                atTime=time_since_entry / 60.0,  # hours
                absorptionTime=max_absorption_hours  # declared * 1.5
            )
        else:
            min_absorbed = 0.0

        # Use max (enforce minimum absorption rate)
        total_absorbed = max(observed_absorbed, min_absorbed)
        cob = max(self.entry_grams - total_absorbed, 0.0)

        if debug or os.getenv('DEBUG_COB'):
            print(f"    MODE 2 (during obs): COB={cob:.2f}g")
            print(f"      observed_absorbed={observed_absorbed:.2f}g, min_absorbed={min_absorbed:.2f}g")
            print(f"      total_absorbed={total_absorbed:.2f}g (max of above)")
            print(f"      observation_end={observation_end:.1f}min")

        return cob
