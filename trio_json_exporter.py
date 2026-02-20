"""
Converts simulator scenarios to Trio/oref1 expected JSON formats.

Takes the internal scenario format (used by batch_validate.py) and produces
the JSON structures expected by trio_runner.js (which calls the actual JS).

Trio expects:
- clock: ISO 8601 timestamp
- profile: Full profile object with all settings
- glucose: Array of CGM readings (newest first, with dateString/sgv)
- history: Array of pump events (boluses, temp basals)
- carbs: Array of carb entries (with created_at, carbs fields)
- currenttemp: Current temp basal state
- basalprofile: Basal schedule array
- autosens: { ratio: 1.0 }
- oref2_variables: Trio-specific overrides
"""

from datetime import datetime, timezone
from typing import Dict, List, Any


class TrioJSONExporter:
    """Converts simulator scenarios to Trio JSON format."""

    def __init__(self, settings: Dict[str, Any]):
        """
        Initialize with therapy settings.

        Args:
            settings: Dict from settings.json
        """
        self.settings = settings

    def export_scenario(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert a full scenario to Trio runner input format.

        Args:
            scenario: Internal scenario dict with glucoseSamples, carbEntries,
                     insulinDoses, each with unix timestamp fields

        Returns:
            Dict ready to be JSON-serialized and piped to trio_runner.js
        """
        # Determine clock time from the most recent glucose reading
        glucose_samples = scenario.get('glucoseSamples', [])
        if not glucose_samples:
            raise ValueError("Scenario has no glucose samples")

        # Reference time = most recent glucose sample
        clock_unix = max(s['timestamp'] for s in glucose_samples)
        clock_iso = self._unix_to_iso(clock_unix)

        # Build profile
        profile = self.build_profile(clock_unix)

        # Build glucose array
        glucose = self.export_glucose(glucose_samples)

        # Build pump history (boluses + temp basals)
        insulin_doses = scenario.get('insulinDoses', [])
        history = self.export_pump_history(insulin_doses, clock_unix)

        # Build carb entries
        carb_entries = scenario.get('carbEntries', [])
        carbs = self.export_carbs(carb_entries)

        # Build basalprofile
        basalprofile = self.build_basal_profile()

        return {
            'clock': clock_iso,
            'profile': profile,
            'glucose': glucose,
            'history': history,
            'history24': [],
            'carbs': carbs,
            'currenttemp': {'rate': 0, 'duration': 0},
            'autosens': {'ratio': 1.0},
            'microBolusAllowed': True,
            'reservoir': 100,
            'basalprofile': basalprofile,
            'oref2_variables': {
                'useOverride': False,
                'overrideTarget': 0,
                'smbIsScheduledOff': False,
                'start': 0,
                'end': 0,
                'overridePercentage': 100,
                'useIsf': False,
                'isf': 0,
                'useCr': False,
                'cr': 0,
                'duration': 0,
                'unlimited': False,
            }
        }

    def build_profile(self, clock_unix: float) -> Dict[str, Any]:
        """Build a complete Trio profile from settings."""
        s = self.settings
        isf = s.get('insulin_sensitivity_factor', 100.0)
        cr = s.get('carb_ratio', 10.0)
        basal = s.get('basal_rate', 0.45)
        dia = s.get('duration_of_insulin_action', 6.0)
        target = s.get('target', 100.0)
        max_basal = s.get('max_basal_rate', 2.8)
        max_iob = s.get('max_iob', 3.5)
        suspend = s.get('suspend_threshold', 80.0)

        # Determine curve from insulin type
        insulin_type = s.get('insulin_type', 'fiasp')
        if insulin_type in ('fiasp', 'lyumjev'):
            curve = 'ultra-rapid'
            peak = 55
        else:
            curve = 'rapid-acting'
            peak = 75

        return {
            # Core settings
            'current_basal': basal,
            'max_basal': max_basal,
            'max_daily_basal': basal,  # Single-rate schedule
            'max_iob': max_iob,
            'sens': isf,
            'carb_ratio': cr,
            'dia': dia,
            'min_bg': target,
            'max_bg': target,
            'target_bg': target,

            # Insulin curve
            'curve': curve,
            'insulinPeakTime': peak,
            'useCustomPeakTime': False,

            # Basal profile
            'basalprofile': [{'i': 0, 'start': '00:00:00', 'rate': basal, 'minutes': 0}],

            # ISF profile (for cob.js lookups)
            'isfProfile': {
                'sensitivities': [{'i': 0, 'offset': 0, 'sensitivity': isf, 'x': 0}]
            },

            # Carb ratios
            'carb_ratios': {
                'schedule': [{'i': 0, 'offset': 0, 'ratio': cr, 'x': 0}]
            },

            # BG targets
            'bg_targets': {
                'user_preferred_units': 'mg/dL',
                'targets': [{
                    'i': 0, 'offset': 0,
                    'min_bg': target, 'max_bg': target,
                    'low': target, 'high': target,
                    'start': '00:00:00', 'x': 0
                }]
            },

            # Safety
            'max_daily_safety_multiplier': 3,
            'current_basal_safety_multiplier': 4,
            'autosens_max': 1.2,
            'autosens_min': 0.7,
            'maxCOB': 120,
            'out_units': 'mg/dL',

            # SMB settings
            'enableSMB_always': True,
            'enableSMB_with_COB': True,
            'enableSMB_after_carbs': True,
            'enableSMB_with_temptarget': False,
            'allowSMB_with_high_temptarget': False,
            'enableSMB_high_bg': False,
            'enableSMB_high_bg_target': 110,
            'enableUAM': True,
            'A52_risk_enable': False,
            'maxSMBBasalMinutes': 30,
            'maxUAMSMBBasalMinutes': 30,
            'SMBInterval': 3,
            'bolus_increment': 0.1,
            'maxDelta_bg_threshold': 0.2,
            'smb_delivery_ratio': 0.5,

            # Misc
            'min_5m_carbimpact': 8,
            'remainingCarbsFraction': 1.0,
            'remainingCarbsCap': 90,
            'carbsReqThreshold': 1,
            'temptargetSet': False,
            'skip_neutral_temps': True,
            'suspend_zeros_iob': True,
            'noisyCGMTargetMultiplier': 1.3,
            'exercise_mode': False,
            'half_basal_exercise_target': 160,
            'sensitivity_raises_target': False,
            'resistance_lowers_target': False,
            'high_temptarget_raises_sensitivity': False,
            'low_temptarget_lowers_sensitivity': False,
            'rewind_resets_autosens': True,
            'unsuspend_if_no_temp': False,

            # Dynamic ISF / Trio-specific (disabled by default)
            'useNewFormula': False,
            'enableDynamicCR': False,
            'sigmoid': False,
            'adjustmentFactor': 0.8,
            'adjustmentFactorSigmoid': 0.5,
            'weightPercentage': 0.65,
            'tddAdjBasal': False,
            'threshold_setting': suspend,
        }

    def export_glucose(self, glucose_samples: List[Dict]) -> List[Dict]:
        """
        Convert glucose samples to Trio format.

        Trio expects newest-first array with dateString and sgv fields.
        """
        entries = []
        for sample in glucose_samples:
            ts = sample['timestamp']
            bg = sample['value']
            entries.append({
                'sgv': bg,
                'glucose': bg,
                'dateString': self._unix_to_iso(ts),
                'date': int(ts * 1000),  # milliseconds
                'noise': 0,
                'type': 'sgv'
            })
        # Sort newest first
        entries.sort(key=lambda x: x['date'], reverse=True)
        return entries

    def export_pump_history(self, insulin_doses: List[Dict],
                           clock_unix: float) -> List[Dict]:
        """
        Convert insulin doses to Trio pump history format.

        Each bolus becomes a pump history event with _type: "Bolus".
        """
        events = []
        for dose in insulin_doses:
            ts = dose['timestamp']
            units = dose['units']
            iso = self._unix_to_iso(ts)
            events.append({
                '_type': 'Bolus',
                'timestamp': iso,
                'amount': units,
                'duration': 0,
                'type': 'normal'
            })
        # Sort newest first (matching Trio's expected order)
        events.sort(key=lambda x: x['timestamp'], reverse=True)
        return events

    def export_carbs(self, carb_entries: List[Dict]) -> List[Dict]:
        """
        Convert carb entries to Trio format.

        Trio expects carbs in the format used by meal/history.js:
        { created_at, carbs }
        """
        entries = []
        for entry in carb_entries:
            ts = entry['timestamp']
            grams = entry['grams']
            iso = self._unix_to_iso(ts)
            entries.append({
                'created_at': iso,
                'carbs': grams,
                'nsCarbs': grams,  # Classify as Nightscout carbs
            })
        return entries

    def build_basal_profile(self) -> List[Dict]:
        """Build basal profile schedule."""
        basal = self.settings.get('basal_rate', 0.45)
        return [{'i': 0, 'start': '00:00:00', 'rate': basal, 'minutes': 0}]

    @staticmethod
    def _unix_to_iso(unix_ts: float) -> str:
        """Convert unix timestamp to ISO 8601 string."""
        dt = datetime.fromtimestamp(unix_ts, tz=timezone.utc)
        return dt.strftime('%Y-%m-%dT%H:%M:%S.000Z')
