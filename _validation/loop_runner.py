"""
Python wrapper for calling the Loop algorithm via LoopTestRunner executable.

This module provides a simple interface to:
1. Convert Python AlgorithmInput to Loop JSON format
2. Call the Loop executable
3. Parse Loop's prediction output
4. Compare Loop vs Python simulator results
"""

import json
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Any
from datetime import datetime

from loop_json_exporter import LoopJSONExporter


class LoopRunner:
    """
    Wrapper for running iOS Loop algorithm from Python.
    """

    def __init__(self, executable_path: str = None):
        """
        Initialize LoopRunner.

        Args:
            executable_path: Path to LoopTestRunner executable.
                           Defaults to ./loop_testing/LoopTestRunner/.build/debug/LoopTestRunner
        """
        if executable_path is None:
            # Default path relative to this file
            base_dir = Path(__file__).parent
            executable_path = base_dir / "loop_testing/LoopTestRunner/.build/debug/LoopTestRunner"

        self.executable_path = Path(executable_path)
        if not self.executable_path.exists():
            raise FileNotFoundError(f"LoopTestRunner not found at {self.executable_path}")

    def run_loop_prediction(self, loop_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run Loop algorithm with given input.

        Args:
            loop_input: Loop-format JSON input (from LoopJSONExporter)

        Returns:
            Dictionary with Loop's prediction results:
            {
                "status": "success",
                "glucose": [{"date": "ISO8601", "value": mg/dL}, ...],
                "eventualBG": float,
                "predictionCount": int,
                "effects": {
                    "insulin": int,
                    "carbs": int,
                    "momentum": int,
                    "rc": int,
                    "ice": int
                }
            }

        Raises:
            subprocess.CalledProcessError: If Loop executable fails
            json.JSONDecodeError: If Loop output is not valid JSON
        """
        # Convert input dict to JSON string
        input_json = json.dumps(loop_input)

        # Run Loop executable with input via stdin
        result = subprocess.run(
            [str(self.executable_path)],
            input=input_json,
            capture_output=True,
            text=True,
            check=False  # Don't raise on non-zero exit, we'll check manually
        )

        # Check for errors
        if result.returncode != 0:
            error_msg = f"Loop executable failed with exit code {result.returncode}\n"
            error_msg += f"STDERR:\n{result.stderr}\n"
            error_msg += f"STDOUT:\n{result.stdout}"
            raise subprocess.CalledProcessError(result.returncode, str(self.executable_path), error_msg)

        # Parse JSON output from stdout
        try:
            output = json.loads(result.stdout)
            return output
        except json.JSONDecodeError as e:
            error_msg = f"Failed to parse Loop output as JSON: {e}\n"
            error_msg += f"Output was:\n{result.stdout}"
            raise json.JSONDecodeError(error_msg, result.stdout, 0)

    def extract_glucose_predictions(self, loop_output: Dict[str, Any]) -> List[Tuple[datetime, float]]:
        """
        Extract glucose predictions from Loop output.

        Args:
            loop_output: Output from run_loop_prediction()

        Returns:
            List of (datetime, glucose_mg_dl) tuples
        """
        predictions = []
        for pred in loop_output.get("glucose", []):
            date = datetime.fromisoformat(pred["date"].replace("Z", "+00:00"))
            value = pred["value"]
            predictions.append((date, value))
        return predictions

    def get_eventual_bg(self, loop_output: Dict[str, Any]) -> float:
        """
        Get eventual BG from Loop output.

        Args:
            loop_output: Output from run_loop_prediction()

        Returns:
            Eventual BG in mg/dL
        """
        return loop_output.get("eventualBG", 0.0)


class LoopComparison:
    """
    Compare Loop predictions vs Python simulator predictions.
    """

    @staticmethod
    def compare_predictions(
        loop_predictions: List[Tuple[datetime, float]],
        python_predictions: List[Tuple[float, float]],
        reference_time: datetime
    ) -> Dict[str, Any]:
        """
        Compare Loop vs Python predictions.

        Args:
            loop_predictions: List of (datetime, glucose_mg_dl) from Loop
            python_predictions: List of (minutes, glucose_mg_dl) from Python simulator
            reference_time: Reference datetime for converting Python minutes to absolute time

        Returns:
            Comparison dictionary with:
            {
                "matched_points": int,
                "mean_difference": float,
                "max_difference": float,
                "rmse": float,
                "differences": [(time, loop_value, python_value, diff), ...]
            }
        """
        from datetime import timedelta

        # Convert Python predictions to absolute times
        python_abs = [
            (reference_time + timedelta(minutes=minutes), bg)
            for minutes, bg in python_predictions
        ]

        # Find matching time points (within 1 second tolerance)
        differences = []
        for loop_time, loop_bg in loop_predictions:
            # Find closest Python prediction
            closest = min(python_abs, key=lambda p: abs((p[0] - loop_time).total_seconds()))
            time_diff = abs((closest[0] - loop_time).total_seconds())

            if time_diff <= 1.0:  # Within 1 second
                python_bg = closest[1]
                diff = loop_bg - python_bg
                differences.append((loop_time, loop_bg, python_bg, diff))

        if not differences:
            return {
                "matched_points": 0,
                "mean_difference": 0.0,
                "max_difference": 0.0,
                "rmse": 0.0,
                "differences": []
            }

        # Calculate statistics
        diffs = [d[3] for d in differences]
        mean_diff = sum(diffs) / len(diffs)
        max_diff = max(abs(d) for d in diffs)
        rmse = (sum(d**2 for d in diffs) / len(diffs)) ** 0.5

        return {
            "matched_points": len(differences),
            "mean_difference": mean_diff,
            "max_difference": max_diff,
            "rmse": rmse,
            "differences": differences
        }

    @staticmethod
    def print_comparison(comparison: Dict[str, Any], verbose: bool = False):
        """
        Print comparison results.

        Args:
            comparison: Result from compare_predictions()
            verbose: If True, print all difference points
        """
        print(f"\n=== Loop vs Python Comparison ===")
        print(f"Matched time points: {comparison['matched_points']}")
        print(f"Mean difference: {comparison['mean_difference']:.2f} mg/dL")
        print(f"Max difference: {comparison['max_difference']:.2f} mg/dL")
        print(f"RMSE: {comparison['rmse']:.2f} mg/dL")

        if verbose and comparison['differences']:
            print(f"\nDetailed differences:")
            print(f"{'Time':<20} {'Loop':>8} {'Python':>8} {'Diff':>8}")
            print("-" * 50)
            for time, loop_val, py_val, diff in comparison['differences']:
                print(f"{time.strftime('%H:%M:%S'):<20} {loop_val:>8.1f} {py_val:>8.1f} {diff:>+8.1f}")


# Example usage
if __name__ == "__main__":
    print("LoopRunner - Python interface to iOS Loop algorithm")
    print("=" * 50)

    # Initialize runner
    runner = LoopRunner()
    print(f"✓ Found Loop executable: {runner.executable_path}")

    # Load test input
    test_file = Path(__file__).parent / "test_loop_input_corrected.json"
    if test_file.exists():
        print(f"✓ Loading test input: {test_file}")

        with open(test_file) as f:
            test_input = json.load(f)

        # Run Loop prediction
        print("\nRunning Loop prediction...")
        output = runner.run_loop_prediction(test_input)

        print(f"✓ Loop prediction complete")
        print(f"  Predictions: {output['predictionCount']}")
        print(f"  Eventual BG: {output['eventualBG']:.1f} mg/dL")
        print(f"  Effects: insulin={output['effects']['insulin']}, "
              f"carbs={output['effects']['carbs']}, "
              f"momentum={output['effects']['momentum']}, "
              f"rc={output['effects']['rc']}, "
              f"ice={output['effects']['ice']}")

        # Show first few predictions
        predictions = runner.extract_glucose_predictions(output)
        print(f"\nFirst 5 predictions:")
        for dt, bg in predictions[:5]:
            print(f"  {dt.strftime('%Y-%m-%d %H:%M:%S')}: {bg:.1f} mg/dL")
    else:
        print(f"✗ Test file not found: {test_file}")
