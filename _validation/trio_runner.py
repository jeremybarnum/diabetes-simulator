"""
Python wrapper for calling the Trio/oref1 algorithm via Node.js.

Analogous to loop_runner.py — calls the actual JS implementation via subprocess,
providing ground truth for validating the Python reimplementation.

Usage:
    from trio_runner import TrioRunner
    runner = TrioRunner()
    result = runner.run(trio_input_dict)
"""

import json
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional


class TrioRunner:
    """
    Wrapper for running Trio/oref1 algorithm from Python via Node.js.
    """

    def __init__(self, node_path: str = None, runner_js: str = None):
        """
        Initialize TrioRunner.

        Args:
            node_path: Path to node executable. Defaults to 'node' (in PATH).
            runner_js: Path to trio_runner.js. Defaults to ./trio_testing/trio_runner.js
        """
        self.node_path = node_path or 'node'

        if runner_js is None:
            base_dir = Path(__file__).parent
            self.runner_js = base_dir / "trio_testing" / "trio_runner.js"
        else:
            self.runner_js = Path(runner_js)

        if not self.runner_js.exists():
            raise FileNotFoundError(f"trio_runner.js not found at {self.runner_js}")

    def run(self, trio_input: Dict[str, Any], timeout: int = 30) -> Dict[str, Any]:
        """
        Run the full Trio pipeline via Node.js.

        Args:
            trio_input: Dict in the format expected by trio_runner.js
                       (clock, profile, glucose, history, carbs, etc.)
            timeout: Timeout in seconds

        Returns:
            Dict with result from Node.js:
            {
                "status": "success",
                "iob_data": [...],
                "meal_data": {...},
                "glucose_status": {...},
                "result": {...},       # determine_basal output
                "profile_used": {...}
            }

        Raises:
            subprocess.CalledProcessError: If Node.js execution fails
            json.JSONDecodeError: If output is not valid JSON
            RuntimeError: If Trio returns an error status
        """
        input_json = json.dumps(trio_input)

        result = subprocess.run(
            [self.node_path, str(self.runner_js)],
            input=input_json,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False
        )

        # Parse stdout (the result JSON)
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()

        if not stdout:
            raise RuntimeError(
                f"Trio runner produced no output.\n"
                f"Exit code: {result.returncode}\n"
                f"STDERR:\n{stderr}"
            )

        try:
            output = json.loads(stdout)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"Failed to parse Trio runner output as JSON: {e}\n"
                f"STDOUT:\n{stdout}\n"
                f"STDERR:\n{stderr}"
            )

        if output.get('status') == 'error':
            raise RuntimeError(
                f"Trio runner error: {output.get('error', 'unknown')}\n"
                f"STDERR:\n{stderr}"
            )

        # Attach stderr for debugging (JS console.error goes here)
        output['_stderr'] = stderr

        return output

    def get_iob(self, result: Dict[str, Any]) -> float:
        """Extract current IOB from Trio result."""
        iob_data = result.get('iob_data', [])
        if iob_data:
            return iob_data[0].get('iob', 0.0)
        return 0.0

    def get_cob(self, result: Dict[str, Any]) -> float:
        """Extract COB from Trio result."""
        return result.get('meal_data', {}).get('mealCOB', 0.0)

    def get_eventual_bg(self, result: Dict[str, Any]) -> Optional[float]:
        """Extract eventualBG from determine_basal result."""
        db = result.get('result', {})
        return db.get('eventualBG')

    def get_temp_basal(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract temp basal recommendation."""
        db = result.get('result', {})
        return {
            'rate': db.get('rate'),
            'duration': db.get('duration'),
            'reason': db.get('reason', ''),
        }

    def get_smb(self, result: Dict[str, Any]) -> Optional[float]:
        """Extract SMB recommendation (units)."""
        db = result.get('result', {})
        return db.get('units')

    def get_predictions(self, result: Dict[str, Any]) -> Dict[str, List[float]]:
        """Extract prediction arrays from determine_basal result."""
        db = result.get('result', {})
        pred_bgs = db.get('predBGs', {})
        return {
            'IOB': pred_bgs.get('IOB', []),
            'COB': pred_bgs.get('COB', []),
            'UAM': pred_bgs.get('UAM', []),
            'ZT': pred_bgs.get('ZT', []),
        }

    def get_reason(self, result: Dict[str, Any]) -> str:
        """Extract reason string from determine_basal result."""
        db = result.get('result', {})
        return db.get('reason', '')


class TrioComparison:
    """Compare Trio JS results vs Python implementation results."""

    @staticmethod
    def compare_results(js_result: Dict, py_result: Dict,
                        tolerance: float = 0.1) -> Dict[str, Any]:
        """
        Compare JS ground truth vs Python implementation.

        Args:
            js_result: Output from TrioRunner.run()
            py_result: Output from Python OpenAPS algorithm
            tolerance: Acceptable difference in mg/dL

        Returns:
            Comparison dict with pass/fail and detailed diffs
        """
        diffs = {}
        js_db = js_result.get('result', {})
        py_db = py_result  # Depends on Python output format

        # Compare eventualBG
        js_ebg = js_db.get('eventualBG')
        py_ebg = py_db.get('eventualBG')
        if js_ebg is not None and py_ebg is not None:
            diffs['eventualBG'] = py_ebg - js_ebg

        # Compare IOB
        js_iob = js_result.get('iob_data', [{}])[0].get('iob', 0)
        py_iob = py_db.get('IOB', 0)
        diffs['iob'] = py_iob - js_iob

        # Compare COB
        js_cob = js_result.get('meal_data', {}).get('mealCOB', 0)
        py_cob = py_db.get('COB', 0)
        diffs['cob'] = py_cob - js_cob

        # Compare temp basal rate
        js_rate = js_db.get('rate')
        py_rate = py_db.get('rate')
        if js_rate is not None and py_rate is not None:
            diffs['rate'] = py_rate - js_rate

        # Compare SMB
        js_units = js_db.get('units')
        py_units = py_db.get('units')
        if js_units is not None and py_units is not None:
            diffs['units'] = py_units - js_units

        # Compare predictions
        js_preds = js_db.get('predBGs', {})
        py_preds = py_db.get('predBGs', {})
        for pred_type in ['IOB', 'COB', 'UAM', 'ZT']:
            js_arr = js_preds.get(pred_type, [])
            py_arr = py_preds.get(pred_type, [])
            if js_arr and py_arr:
                min_len = min(len(js_arr), len(py_arr))
                pred_diffs = [py_arr[i] - js_arr[i] for i in range(min_len)]
                if pred_diffs:
                    diffs[f'pred_{pred_type}_max'] = max(abs(d) for d in pred_diffs)
                    diffs[f'pred_{pred_type}_mean'] = sum(pred_diffs) / len(pred_diffs)

        # Determine pass/fail
        ebg_diff = abs(diffs.get('eventualBG', 0))
        passed = ebg_diff <= tolerance

        return {
            'passed': passed,
            'diffs': diffs,
            'tolerance': tolerance,
        }

    @staticmethod
    def print_comparison(name: str, comparison: Dict[str, Any], verbose: bool = False):
        """Pretty-print comparison results."""
        status = "PASS" if comparison['passed'] else "FAIL"
        diffs = comparison['diffs']

        print(f"  [{status}] {name}")
        if 'eventualBG' in diffs:
            print(f"    eventualBG diff: {diffs['eventualBG']:+.2f} mg/dL")
        if 'iob' in diffs:
            print(f"    IOB diff: {diffs['iob']:+.4f} U")
        if 'cob' in diffs:
            print(f"    COB diff: {diffs['cob']:+.1f} g")
        if 'rate' in diffs:
            print(f"    Rate diff: {diffs['rate']:+.3f} U/hr")

        if verbose:
            for key in sorted(diffs.keys()):
                if key.startswith('pred_'):
                    print(f"    {key}: {diffs[key]:+.2f}")


# Quick test
if __name__ == "__main__":
    import sys

    print("TrioRunner - Python interface to Trio/oref1 algorithm")
    print("=" * 50)

    try:
        runner = TrioRunner()
        print(f"Found trio_runner.js: {runner.runner_js}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Test with a minimal scenario
    from trio_json_exporter import TrioJSONExporter
    import time

    settings = {
        'insulin_sensitivity_factor': 100.0,
        'carb_ratio': 10.0,
        'basal_rate': 0.45,
        'duration_of_insulin_action': 6.0,
        'target': 100.0,
        'max_basal_rate': 2.8,
        'max_iob': 3.5,
        'suspend_threshold': 80.0,
        'insulin_type': 'fiasp',
    }

    exporter = TrioJSONExporter(settings)
    now = time.time()

    # Simple flat BG scenario
    scenario = {
        'name': 'Flat BG Test',
        'description': 'Simple flat BG at 120',
        'glucoseSamples': [
            {'timestamp': now - 15 * 60, 'value': 120.0},
            {'timestamp': now - 10 * 60, 'value': 120.0},
            {'timestamp': now - 5 * 60, 'value': 120.0},
        ],
        'carbEntries': [],
        'insulinDoses': [],
    }

    trio_input = exporter.export_scenario(scenario)
    print(f"\nRunning flat BG scenario...")

    try:
        result = runner.run(trio_input)
        print(f"Status: {result['status']}")
        print(f"IOB: {runner.get_iob(result):.3f} U")
        print(f"COB: {runner.get_cob(result):.0f} g")
        ebg = runner.get_eventual_bg(result)
        if ebg:
            print(f"Eventual BG: {ebg:.1f} mg/dL")
        tb = runner.get_temp_basal(result)
        print(f"Temp basal: {tb['rate']} U/hr for {tb['duration']} min")
        print(f"Reason: {tb['reason'][:100]}...")
    except Exception as e:
        print(f"Error running Trio: {e}")
        sys.exit(1)
