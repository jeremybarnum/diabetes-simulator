"""
OpenAPS algorithm - Complete Phase 1 implementation.

Orchestrates all OpenAPS modules to generate dosing recommendations.
"""

from typing import Dict, List, Tuple, Any
from algorithms.base import Algorithm, AlgorithmInput, AlgorithmOutput

# Import all OpenAPS modules
from algorithms.openaps.iob import iob_total, create_treatment
from algorithms.openaps.cob import recent_carbs, create_carb_entry
from algorithms.openaps.glucose_stats import get_last_glucose, create_glucose_entry
from algorithms.openaps.profile import generate_profile
from algorithms.openaps.predictions import generate_predictions
from algorithms.openaps.determine_basal import determine_basal


class OpenAPSAlgorithm(Algorithm):
    """
    OpenAPS/oref0 algorithm - Phase 1 implementation.

    Integrates all OpenAPS modules:
    - Insulin math (exponential curves)
    - IOB calculation
    - COB calculation
    - Glucose statistics (deltas)
    - Profile/settings
    - BG predictions (IOB, COB)
    - Determine-basal (temp basal logic)
    """

    def __init__(self, settings: Dict):
        """
        Initialize OpenAPS algorithm.

        Expected settings:
            - insulin_sensitivity_factor: ISF in mg/dL per unit
            - duration_of_insulin_action: DIA in hours
            - basal_rate: Current basal rate in U/hr (or basal_schedule)
            - target_bg: Target BG in mg/dL
            - carb_ratio: Carb ratio in g/U
            - max_basal: Max basal rate (absolute or multiplier)
            - max_iob: Maximum IOB allowed
            - max_daily_basal: Max daily scheduled basal in U/hr
            - insulin_peak: Peak time in minutes (optional, default 75)
            - curve: Insulin curve type (optional, default 'rapid-acting')
        """
        super().__init__(settings)
        self.settings = settings

        # Cache the last profile to avoid rebuilding every time
        self._cached_profile = None

    def get_name(self) -> str:
        """Return algorithm name."""
        return "OpenAPS"

    def _build_profile(self, current_time_ms: int) -> Dict[str, Any]:
        """
        Build OpenAPS profile from settings.

        Args:
            current_time_ms: Current time in milliseconds

        Returns:
            Profile dict for OpenAPS modules
        """
        # If we have a cached profile and settings haven't changed, use it
        if self._cached_profile:
            return self._cached_profile

        # Extract settings
        isf = self.settings.get('insulin_sensitivity_factor', 50.0)
        dia = self.settings.get('duration_of_insulin_action', 6.0)
        target_bg = self.settings.get('target_bg', 100.0)
        carb_ratio = self.settings.get('carb_ratio', 10.0)
        max_iob = self.settings.get('max_iob', 5.0)

        # Basal settings
        basal_rate = self.settings.get('basal_rate', 1.0)
        max_daily_basal = self.settings.get('max_daily_basal', basal_rate)

        # Max basal can be absolute or multiplier
        max_basal = self.settings.get('max_basal', 4.0)
        if max_basal < 10:  # Assume it's a multiplier
            max_basal = max_daily_basal * max_basal

        # Insulin curve settings
        curve = self.settings.get('curve', 'rapid-acting')
        peak = self.settings.get('insulin_peak', 75)  # Default 75 min for rapid

        # Build simple profile (Phase 1 - no schedule lookups)
        profile = {
            'current_basal': basal_rate,
            'max_basal': max_basal,
            'max_daily_basal': max_daily_basal,
            'max_iob': max_iob,
            'sens': isf,
            'carb_ratio': carb_ratio,
            'dia': dia,
            'target_bg': target_bg,
            'min_bg': target_bg - 20,  # ±20 mg/dL range
            'max_bg': target_bg + 20,
            'curve': curve,
            'insulinPeakTime': peak,
            'maxCOB': 120,  # Default safety limit
            'out_units': 'mg/dL'
        }

        self._cached_profile = profile
        return profile

    def _prepare_treatments(
        self,
        bolus_history: List[Tuple[float, float]],
        current_time: float
    ) -> List[Dict[str, Any]]:
        """
        Convert bolus history to OpenAPS treatment format.

        Args:
            bolus_history: List of (time_minutes, units) tuples
            current_time: Current time in minutes

        Returns:
            List of treatment dicts
        """
        treatments = []
        for time_min, units in bolus_history:
            time_ms = time_min * 60 * 1000  # Convert to milliseconds
            treatments.append(create_treatment(
                insulin=units,
                time_ms=time_ms,
                treatment_type='bolus'
            ))
        return treatments

    def _prepare_carb_entries(
        self,
        carb_entries: List[Tuple[float, float, float]],
        current_time: float
    ) -> List[Dict[str, Any]]:
        """
        Convert carb entries to OpenAPS format.

        Args:
            carb_entries: List of (time_minutes, grams, absorption_hours) tuples
            current_time: Current time in minutes

        Returns:
            List of carb entry dicts
        """
        entries = []
        for time_min, grams, absorption_hours in carb_entries:
            time_ms = time_min * 60 * 1000  # Convert to milliseconds
            entries.append(create_carb_entry(
                carbs=grams,
                time_ms=time_ms,
                absorption_time_hours=absorption_hours
            ))
        return entries

    def _prepare_glucose_data(
        self,
        cgm_history: List[Tuple[float, float]],
        current_time: float
    ) -> List[Dict[str, Any]]:
        """
        Convert CGM history to OpenAPS glucose data format.

        Args:
            cgm_history: List of (time_minutes, glucose_mg_dL) tuples (oldest first)
            current_time: Current time in minutes

        Returns:
            List of glucose entry dicts (newest first for OpenAPS)
        """
        glucose_data = []
        for time_min, glucose in cgm_history:
            time_ms = time_min * 60 * 1000
            glucose_data.append(create_glucose_entry(
                glucose=glucose,
                time_ms=time_ms
            ))

        # OpenAPS expects newest first
        glucose_data.reverse()
        return glucose_data

    def recommend(self, input_data: AlgorithmInput) -> AlgorithmOutput:
        """
        Generate OpenAPS dosing recommendation using all Phase 1 modules.

        Args:
            input_data: Current state from simulator

        Returns:
            AlgorithmOutput with recommendation
        """
        current_time_ms = input_data.timestamp * 60 * 1000  # Convert min to ms

        # Build profile
        profile = self._build_profile(current_time_ms)

        # Prepare treatments (boluses)
        treatments = self._prepare_treatments(input_data.bolus_history, input_data.timestamp)

        # Calculate IOB
        iob_data = iob_total(
            treatments=treatments,
            time_ms=current_time_ms,
            profile=profile
        )

        # Prepare carb entries
        carb_entries = self._prepare_carb_entries(input_data.carb_entries, input_data.timestamp)

        # Calculate COB
        cob_data = recent_carbs(
            treatments=carb_entries,
            time_ms=current_time_ms,
            profile=profile
        )

        # Prepare glucose data (need recent history for deltas)
        glucose_data = self._prepare_glucose_data(input_data.cgm_history, input_data.timestamp)

        # Calculate glucose statistics
        glucose_status = get_last_glucose(glucose_data)

        # If no glucose status (empty data), use current reading
        if not glucose_status:
            glucose_status = {
                'glucose': input_data.cgm_reading,
                'delta': 0,
                'short_avgdelta': 0,
                'long_avgdelta': 0,
                'date': current_time_ms
            }

        # Generate predictions
        predictions = generate_predictions(
            bg=input_data.cgm_reading,
            iob_data=iob_data,
            profile=profile,
            glucose_status=glucose_status,
            meal_data=cob_data if cob_data else None
        )

        # Current temp basal (from simulator)
        if input_data.temp_basal:
            temp_rate, temp_duration = input_data.temp_basal
            current_temp = {'rate': temp_rate, 'duration': temp_duration}
        else:
            current_temp = {'rate': input_data.current_basal, 'duration': 0}

        # Determine basal rate
        recommendation = determine_basal(
            glucose_status=glucose_status,
            current_temp=current_temp,
            iob_data=iob_data,
            profile=profile,
            meal_data=cob_data,
            predictions=predictions
        )

        # Convert OpenAPS prediction format to simulator format
        glucose_predictions = {}

        if 'IOB' in predictions:
            glucose_predictions['iob'] = predictions['IOB']

        if 'COB' in predictions:
            glucose_predictions['cob'] = predictions['COB']
        else:
            # If no COB predictions, use IOB
            glucose_predictions['cob'] = predictions.get('IOB', [])

        # Placeholder for UAM and ZT (Phase 2)
        glucose_predictions['uam'] = predictions.get('IOB', [])
        glucose_predictions['zt'] = predictions.get('IOB', [])

        # Return AlgorithmOutput
        return AlgorithmOutput(
            timestamp=input_data.timestamp,
            temp_basal_rate=recommendation.get('rate'),
            temp_basal_duration=recommendation.get('duration', 30),
            bolus=None,  # SMB in Phase 2+
            iob=iob_data.get('iob', 0),
            cob=cob_data.get('mealCOB', 0) if cob_data else 0,
            glucose_predictions=glucose_predictions,
            reason=recommendation.get('reason', '')
        )
