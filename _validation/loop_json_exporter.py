"""
Export Python simulator data to Loop JSON format.

This module converts AlgorithmInput and test scenarios to the JSON format
expected by iOS Loop's LoopPredictionInput.
"""

from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Any
import json

from algorithms.base import AlgorithmInput


class LoopJSONExporter:
    """
    Converts Python simulator data to Loop JSON format.

    Loop expects:
    - ISO8601 timestamps (e.g., "2024-01-01T12:00:00Z")
    - Quantities in specific units (mg/dL for glucose, grams for carbs, U for insulin)
    - Specific time windows (glucose: 10h, doses: 16h, carbs: 10h)
    """

    def __init__(self, reference_date: datetime = None):
        """
        Initialize exporter.

        Args:
            reference_date: Base date for converting minute offsets to absolute times.
                           Defaults to arbitrary fixed date for reproducibility.
        """
        self.reference_date = reference_date or datetime(2024, 1, 1, 12, 0, 0)

    def minutes_to_datetime(self, minutes: float) -> datetime:
        """
        Convert minutes offset to absolute datetime.

        Args:
            minutes: Minutes since reference (can be negative for history)

        Returns:
            Absolute datetime
        """
        return self.reference_date + timedelta(minutes=minutes)

    def format_iso8601(self, dt: datetime) -> str:
        """
        Format datetime as ISO8601 string for Loop.

        Args:
            dt: Datetime to format

        Returns:
            ISO8601 string (e.g., "2024-01-01T12:00:00Z")
        """
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    def export_glucose_sample(self, time_minutes: float, quantity_mgdl: float) -> Dict[str, Any]:
        """
        Export single glucose sample.

        Args:
            time_minutes: Time in minutes from reference
            quantity_mgdl: Glucose value in mg/dL

        Returns:
            Loop-format glucose sample dict
        """
        return {
            "quantity": quantity_mgdl,
            "startDate": self.format_iso8601(self.minutes_to_datetime(time_minutes)),
            "isDisplayOnly": False
        }

    def export_glucose_history(self, cgm_history: List[Tuple[float, float]]) -> List[Dict[str, Any]]:
        """
        Export glucose history.

        Args:
            cgm_history: List of (time_minutes, bg_value) tuples

        Returns:
            List of Loop-format glucose samples
        """
        return [
            self.export_glucose_sample(time, bg)
            for time, bg in cgm_history
        ]

    def export_bolus_dose(self, time_minutes: float, units: float) -> Dict[str, Any]:
        """
        Export bolus dose.

        Args:
            time_minutes: Time of bolus in minutes
            units: Insulin units

        Returns:
            Loop-format dose entry
        """
        start_time = self.minutes_to_datetime(time_minutes)
        # Boluses are assumed instantaneous, but Loop uses start/end dates
        # Use 4-second duration (typical pump delivery time)
        end_time = start_time + timedelta(seconds=4)

        return {
            "type": "bolus",
            "startDate": self.format_iso8601(start_time),
            "endDate": self.format_iso8601(end_time),
            "value": units,
            "unit": "U"
        }

    def export_basal_dose(self, start_minutes: float, duration_minutes: float,
                         rate_u_per_hr: float, is_temp: bool = False) -> Dict[str, Any]:
        """
        Export basal dose (scheduled or temp).

        Args:
            start_minutes: Start time in minutes
            duration_minutes: Duration of basal
            rate_u_per_hr: Basal rate in U/hr
            is_temp: True if temp basal, False if scheduled basal

        Returns:
            Loop-format dose entry
        """
        start_time = self.minutes_to_datetime(start_minutes)
        end_time = self.minutes_to_datetime(start_minutes + duration_minutes)

        # Calculate total units delivered
        total_units = rate_u_per_hr * (duration_minutes / 60.0)

        return {
            "type": "tempBasal" if is_temp else "basal",
            "startDate": self.format_iso8601(start_time),
            "endDate": self.format_iso8601(end_time),
            "value": total_units,  # Loop uses total units for basal, not rate
            "unit": "U"
        }

    def export_dose_history(self, bolus_history: List[Tuple[float, float]],
                           basal_rate: float = 1.0,
                           time_step: float = 5.0) -> List[Dict[str, Any]]:
        """
        Export dose history (boluses + basal).

        Args:
            bolus_history: List of (time_minutes, units) tuples
            basal_rate: Scheduled basal rate in U/hr
            time_step: Time step for basal segments (minutes)

        Returns:
            List of Loop-format dose entries, sorted by time
        """
        doses = []

        # Add boluses
        for time, units in bolus_history:
            doses.append(self.export_bolus_dose(time, units))

        # Add basal doses
        # For simplicity, create continuous basal segments at scheduled rate
        # In real Loop, this would include temp basals
        # TODO: Handle temp basals from simulator

        return sorted(doses, key=lambda d: d["startDate"])

    def export_carb_entry(self, time_minutes: float, grams: float,
                         absorption_hours: float = 3.0) -> Dict[str, Any]:
        """
        Export carb entry.

        Args:
            time_minutes: Time of carb entry in minutes
            grams: Carb amount in grams
            absorption_hours: Expected absorption time in hours

        Returns:
            Loop-format carb entry
        """
        return {
            "quantity": grams,
            "startDate": self.format_iso8601(self.minutes_to_datetime(time_minutes)),
            "absorptionTime": absorption_hours * 3600  # Convert to seconds
        }

    def export_carb_entries(self, carb_entries: List[Tuple[float, float, float]]) -> List[Dict[str, Any]]:
        """
        Export carb entries.

        Args:
            carb_entries: List of (time_minutes, grams, absorption_hours) tuples

        Returns:
            List of Loop-format carb entries
        """
        return [
            self.export_carb_entry(time, grams, absorption_hrs)
            for time, grams, absorption_hrs in carb_entries
        ]

    def export_settings(self, settings: Dict, duration_hours: float = 18) -> Dict[str, Any]:
        """
        Export algorithm settings to Loop format.

        Args:
            settings: Python simulator settings dict
            duration_hours: Duration for schedule coverage (default 18h to cover typical test windows)

        Returns:
            Loop-format settings with absolute schedules
        """
        isf = settings.get('insulin_sensitivity_factor', 50.0)
        carb_ratio = settings.get('carb_ratio', 10.0)
        basal = settings.get('basal_rate', 1.0)

        # Target can be a single value or a tuple (min, max)
        target = settings.get('target', 100.0)
        if isinstance(target, (tuple, list)):
            target_min, target_max = target[0], target[1]
        else:
            # Single value means tight range (same min and max)
            target_min = target_max = target

        # Create schedule that covers the test window
        # Use a window that starts before reference time to cover historical data
        start_time = self.minutes_to_datetime(-600)  # Start 10 hours before reference
        end_time = self.minutes_to_datetime(duration_hours * 60)  # End after duration

        start_iso = self.format_iso8601(start_time)
        end_iso = self.format_iso8601(end_time)

        return {
            "basal": [
                {
                    "startDate": start_iso,
                    "endDate": end_iso,
                    "value": basal
                }
            ],
            "sensitivity": [
                {
                    "startDate": start_iso,
                    "endDate": end_iso,
                    "value": isf
                }
            ],
            "carbRatio": [
                {
                    "startDate": start_iso,
                    "endDate": end_iso,
                    "value": carb_ratio
                }
            ],
            "target": [
                {
                    "startDate": start_iso,
                    "endDate": end_iso,
                    "value": {
                        "minValue": target_min,
                        "maxValue": target_max
                    }
                }
            ],
            "suspendThreshold": {
                "unit": "mg/dL",
                "value": 70.0
            },
            "useIntegralRetrospectiveCorrection": settings.get('enable_irc', False)
        }

    def export_algorithm_input(self, algorithm_input: AlgorithmInput) -> Dict[str, Any]:
        """
        Export complete AlgorithmInput to Loop JSON format.

        Args:
            algorithm_input: Python simulator AlgorithmInput

        Returns:
            Complete Loop LoopPredictionInput as dict
        """
        # Compute settings date range from data timestamps
        # Settings must cover earliest data to latest prediction
        all_times = [t for t, _ in algorithm_input.cgm_history]
        if algorithm_input.bolus_history:
            all_times.extend([t for t, _ in algorithm_input.bolus_history])
        if algorithm_input.carb_entries:
            all_times.extend([t for t, _, _ in algorithm_input.carb_entries])
        all_times.append(algorithm_input.timestamp)

        earliest = min(all_times)
        latest = max(all_times)

        # Override reference_date to ensure settings cover data range
        # Settings need to span from earliest data - 1h to latest + 7h
        saved_ref = self.reference_date
        from datetime import timezone
        self.reference_date = datetime(1970, 1, 1, tzinfo=None)

        settings_start = self.minutes_to_datetime(earliest - 60)
        settings_end = self.minutes_to_datetime(latest + 420)  # +7 hours

        # Patch export_settings to use correct dates
        settings = algorithm_input.settings.copy()
        result = {
            "glucoseHistory": self.export_glucose_history(algorithm_input.cgm_history),
            "doses": self.export_dose_history(
                algorithm_input.bolus_history,
                algorithm_input.current_basal
            ),
            "carbEntries": self.export_carb_entries(algorithm_input.carb_entries),
            "settings": self.export_settings(algorithm_input.settings)
        }

        # Fix settings date range to cover actual data
        start_iso = self.format_iso8601(settings_start)
        end_iso = self.format_iso8601(settings_end)
        for key in ['basal', 'sensitivity', 'carbRatio', 'target']:
            if key in result['settings']:
                for entry in result['settings'][key]:
                    entry['startDate'] = start_iso
                    entry['endDate'] = end_iso

        self.reference_date = saved_ref
        return result

    def export_to_file(self, algorithm_input: AlgorithmInput, filepath: str):
        """
        Export AlgorithmInput to JSON file.

        Args:
            algorithm_input: Input to export
            filepath: Path to output JSON file
        """
        loop_input = self.export_algorithm_input(algorithm_input)

        with open(filepath, 'w') as f:
            json.dump(loop_input, f, indent=2)

        print(f"Exported Loop test input to: {filepath}")


def create_test_export_example():
    """Example: Export a simple test case to Loop format."""
    from algorithms.base import AlgorithmInput

    # Create test input
    input_data = AlgorithmInput(
        cgm_reading=140.0,
        timestamp=60.0,  # Current time: 60 minutes from reference
        cgm_history=[
            (0.0, 120.0),
            (5.0, 125.0),
            (10.0, 130.0),
            (15.0, 135.0),
            (20.0, 138.0),
            (25.0, 140.0),
            (30.0, 142.0),
            (35.0, 144.0),
            (40.0, 145.0),
            (45.0, 144.0),
            (50.0, 143.0),
            (55.0, 141.0),
            (60.0, 140.0),
        ],
        current_basal=1.0,
        temp_basal=None,
        bolus_history=[
            (0.0, 5.0),  # 5U bolus at t=0
        ],
        carb_entries=[
            (0.0, 45.0, 3.0),  # 45g at t=0, 3hr absorption
        ],
        settings={
            'insulin_sensitivity_factor': 50.0,
            'carb_ratio': 10.0,
            'basal_rate': 1.0,
            'target_range': (100, 120),
        }
    )

    # Export
    exporter = LoopJSONExporter()
    exporter.export_to_file(input_data, "test_loop_input.json")

    print("\n✓ Created test_loop_input.json")
    print("  This file can be used with Loop's LoopAlgorithm.generatePrediction()")


if __name__ == "__main__":
    create_test_export_example()
