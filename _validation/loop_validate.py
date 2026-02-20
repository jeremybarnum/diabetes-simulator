#!/usr/bin/env python3
"""
Loop Validation CLI - Unified tool to validate Python Loop against iOS Loop

This tool orchestrates the complete validation workflow:
1. Clears HealthKit data in simulator
2. Injects test scenario into simulator
3. Waits for iOS Loop to run prediction
4. Extracts Loop logs with momentum/IRC impacts
5. Runs Python Loop with same inputs
6. Compares results and reports differences

Usage:
    python loop_validate.py test_scenarios/rising_bg_momentum.json
    python loop_validate.py test_scenarios/meal_with_irc.json --clear --wait 10
"""

import sys
import argparse
import json
import subprocess
import time
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from loop_testing.extract_simulator_logs import SimulatorLogExtractor


class HealthKitInjector:
    """Wrapper for HealthKitInjectorApp to inject/clear data via command files."""

    def __init__(self, simulator_id: str = None, app_bundle_id: str = "com.test.healthkitinjector"):
        self.app_bundle_id = app_bundle_id

        # Get simulator ID
        if simulator_id is None:
            simulator_id = self._get_booted_simulator()
        self.simulator_id = simulator_id

        if not self.simulator_id:
            raise RuntimeError("No booted simulator found")

        # Get app container paths
        self.app_container = self._get_app_container()
        if not self.app_container:
            raise RuntimeError(f"HealthKitInjectorApp ({app_bundle_id}) not found in simulator")

        self.commands_dir = self.app_container / "Documents" / "commands"
        self.scenarios_dir = self.app_container / "Documents" / "scenarios"

        # Ensure directories exist
        self.commands_dir.mkdir(parents=True, exist_ok=True)
        self.scenarios_dir.mkdir(parents=True, exist_ok=True)

    def _get_booted_simulator(self) -> Optional[str]:
        """Get the device ID of the currently booted simulator."""
        result = subprocess.run(
            ['xcrun', 'simctl', 'list', 'devices', '-j'],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            return None

        devices = json.loads(result.stdout)
        for runtime, device_list in devices['devices'].items():
            for device in device_list:
                if device.get('state') == 'Booted':
                    return device['udid']
        return None

    def _get_app_container(self) -> Optional[Path]:
        """Get the app's container path in the simulator."""
        result = subprocess.run(
            ['xcrun', 'simctl', 'get_app_container', self.simulator_id, self.app_bundle_id, 'data'],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            return None

        return Path(result.stdout.strip())

    def _send_command(self, command: str, wait_seconds: float = 2.0) -> bool:
        """Send a command to the HealthKitInjectorApp via command file."""
        command_data = {
            "command": command,
            "timestamp": time.time()
        }

        # Create unique command file name
        command_file = self.commands_dir / f"command_{int(time.time() * 1000)}.json"

        try:
            command_file.write_text(json.dumps(command_data, indent=2))
            print(f"📝 Sent command: {command}")

            # Wait for command to be processed (app deletes the file)
            start = time.time()
            while command_file.exists() and (time.time() - start) < wait_seconds:
                time.sleep(0.1)

            if command_file.exists():
                print(f"⚠️  Command file still exists after {wait_seconds}s - app may not be running")
                command_file.unlink()  # Clean up
                return False

            return True
        except Exception as e:
            print(f"❌ Error sending command: {e}")
            return False

    def clear_data(self) -> bool:
        """Clear all HealthKit data from simulator."""
        print("🗑️  Clearing HealthKit data...")
        if self._send_command("clear_all", wait_seconds=3.0):
            print("✅ Cleared HealthKit data")
            return True
        else:
            print("❌ Failed to clear data - ensure HealthKitInjectorApp is running")
            return False

    def inject_scenario(self, scenario_path: Path) -> bool:
        """Inject scenario into simulator HealthKit."""
        print(f"💉 Injecting scenario: {scenario_path.name}")

        # Copy scenario file to app's scenarios directory
        dest_path = self.scenarios_dir / scenario_path.name
        try:
            import shutil
            shutil.copy2(scenario_path, dest_path)
            print(f"📂 Copied scenario to: {dest_path}")
        except Exception as e:
            print(f"❌ Error copying scenario: {e}")
            return False

        # Send inject command
        if self._send_command(f"inject:{scenario_path.name}", wait_seconds=3.0):
            print("✅ Injected scenario")
            return True
        else:
            print("❌ Failed to inject scenario - ensure HealthKitInjectorApp is running")
            return False


class LoopLogParser:
    """Parse Loop prediction logs including MOMENTUM and IRC impacts."""

    @staticmethod
    def parse_impact_analysis(log_lines: List[str]) -> Dict:
        """
        Parse MOMENTUM IMPACT and IRC IMPACT sections from Loop logs.

        Expected format:
        === MOMENTUM IMPACT ANALYSIS ===
        Momentum net impact on eventual BG: +12.3 mg/dL
        ...
        === IRC IMPACT ANALYSIS ===
        IRC net impact on eventual BG: -5.2 mg/dL
        Eventual BG WITH IRC: 145.0 mg/dL
        """
        result = {
            'momentum_impact': None,
            'irc_impact': None,
            'eventual_bg': None,
            'eventual_bg_without_irc': None
        }

        for i, line in enumerate(log_lines):
            # Parse momentum impact
            if '=== MOMENTUM IMPACT ANALYSIS ===' in line:
                for j in range(i+1, min(i+10, len(log_lines))):
                    if 'Momentum net impact on eventual BG:' in log_lines[j]:
                        # Extract: "Momentum net impact on eventual BG: +12.3 mg/dL"
                        match = re.search(r'([+-]?\d+\.?\d*)\s*mg/dL', log_lines[j])
                        if match:
                            result['momentum_impact'] = float(match.group(1))

            # Parse IRC impact
            if '=== IRC IMPACT ANALYSIS ===' in line:
                for j in range(i+1, min(i+10, len(log_lines))):
                    if 'IRC net impact on eventual BG:' in log_lines[j]:
                        match = re.search(r'([+-]?\d+\.?\d*)\s*mg/dL', log_lines[j])
                        if match:
                            result['irc_impact'] = float(match.group(1))

                    if 'Eventual BG WITHOUT IRC:' in log_lines[j]:
                        match = re.search(r'(\d+\.?\d*)\s*mg/dL', log_lines[j])
                        if match:
                            result['eventual_bg_without_irc'] = float(match.group(1))

                    if 'Eventual BG WITH IRC:' in log_lines[j]:
                        match = re.search(r'(\d+\.?\d*)\s*mg/dL', log_lines[j])
                        if match:
                            result['eventual_bg'] = float(match.group(1))

        return result

    @staticmethod
    def extract_from_simulator(bundle_id: str = 'com.Exercise.Loop') -> Tuple[Dict, List[str]]:
        """Extract and parse logs from running simulator."""
        extractor = SimulatorLogExtractor(bundle_id=bundle_id)

        if not extractor.simulator_id:
            print("❌ No booted simulator found")
            return {}, []

        # Get recent logs (last 2 minutes, filter for ##LOOP##)
        log_lines = extractor.get_recent_logs(minutes=2, filter_pattern='##LOOP##')

        if not log_lines:
            print("⚠️  No Loop prediction logs found in last 2 minutes")
            return {}, []

        # Parse impact analysis
        parsed = LoopLogParser.parse_impact_analysis(log_lines)

        return parsed, log_lines


class PythonLoopRunner:
    """Run Python Loop algorithm with scenario data."""

    def run_scenario(self, scenario: Dict) -> Dict:
        """Run Python Loop algorithm and extract impacts."""
        from algorithms.loop.loop_algorithm import LoopAlgorithm
        from algorithms.base import AlgorithmInput

        # Extract or use default settings
        settings = scenario.get('settings', {
            'insulin_sensitivity_factor': 50.0,
            'carb_ratio': 10.0,
            'basal_rate': 1.0,
            'duration_of_insulin_action': 6.0,
            'target_range': (100, 120),
            'enable_momentum': True,
            'enable_irc': True,
            'enable_dca': True
        })

        loop = LoopAlgorithm(settings)

        # Convert scenario to AlgorithmInput
        glucose_samples = scenario['glucoseSamples']
        current_bg = glucose_samples[-1]['value']
        current_time = glucose_samples[-1]['timestamp']

        cgm_history = [(s['timestamp'], s['value']) for s in glucose_samples]
        bolus_history = [(d['timestamp'], d['units']) for d in scenario.get('insulinDoses', [])]
        carb_entries = [(c['timestamp'], c['grams'], c.get('absorptionHours', 3.0))
                       for c in scenario.get('carbEntries', [])]

        algorithm_input = AlgorithmInput(
            cgm_reading=current_bg,
            timestamp=current_time,
            cgm_history=cgm_history,
            current_basal=settings['basal_rate'],
            temp_basal=None,
            bolus_history=bolus_history,
            carb_entries=carb_entries,
            settings=settings
        )

        # Run prediction
        output = loop.recommend(algorithm_input)

        return {
            'eventual_bg': output.glucose_predictions['main'][-1] if output.glucose_predictions.get('main') else None,
            'momentum_impact': getattr(output, 'momentum_effect_eventual', None),
            'irc_impact': getattr(output, 'irc_effect_eventual', None),
            'predictions': output.glucose_predictions.get('main', [])
        }


class ValidationComparator:
    """Compare Loop vs Python results."""

    @staticmethod
    def compare(loop_result: Dict, python_result: Dict, tolerance: float = 1.0) -> Dict:
        """Compare results and report differences."""
        comparison = {
            'eventual_bg_diff': None,
            'momentum_diff': None,
            'irc_diff': None,
            'passed': True,
            'errors': []
        }

        # Compare eventual BG
        if loop_result.get('eventual_bg') is not None and python_result.get('eventual_bg') is not None:
            comparison['eventual_bg_diff'] = python_result['eventual_bg'] - loop_result['eventual_bg']
            if abs(comparison['eventual_bg_diff']) > tolerance:
                comparison['passed'] = False
                comparison['errors'].append(f"Eventual BG difference {comparison['eventual_bg_diff']:.1f} exceeds tolerance {tolerance}")
        elif loop_result.get('eventual_bg') is None:
            comparison['errors'].append("Loop eventual BG not found in logs")
            comparison['passed'] = False
        elif python_result.get('eventual_bg') is None:
            comparison['errors'].append("Python Loop did not produce eventual BG")
            comparison['passed'] = False

        # Compare momentum impact
        if loop_result.get('momentum_impact') is not None and python_result.get('momentum_impact') is not None:
            comparison['momentum_diff'] = python_result['momentum_impact'] - loop_result['momentum_impact']
            if abs(comparison['momentum_diff']) > tolerance:
                comparison['passed'] = False
                comparison['errors'].append(f"Momentum difference {comparison['momentum_diff']:.1f} exceeds tolerance {tolerance}")

        # Compare IRC impact
        if loop_result.get('irc_impact') is not None and python_result.get('irc_impact') is not None:
            comparison['irc_diff'] = python_result['irc_impact'] - loop_result['irc_impact']
            if abs(comparison['irc_diff']) > tolerance:
                comparison['passed'] = False
                comparison['errors'].append(f"IRC difference {comparison['irc_diff']:.1f} exceeds tolerance {tolerance}")

        return comparison

    @staticmethod
    def print_report(loop_result: Dict, python_result: Dict, comparison: Dict):
        """Print detailed comparison report."""
        print("\n" + "="*80)
        print("VALIDATION RESULTS")
        print("="*80)

        print("\n📊 Eventual BG:")
        loop_ebg = loop_result.get('eventual_bg')
        python_ebg = python_result.get('eventual_bg')
        print(f"  Loop:   {loop_ebg:.1f} mg/dL" if loop_ebg is not None else "  Loop:   N/A")
        print(f"  Python: {python_ebg:.1f} mg/dL" if python_ebg is not None else "  Python: N/A")
        if comparison['eventual_bg_diff'] is not None:
            print(f"  Δ:      {comparison['eventual_bg_diff']:+.1f} mg/dL")

        print("\n🚀 Momentum Impact:")
        loop_mom = loop_result.get('momentum_impact')
        python_mom = python_result.get('momentum_impact')
        print(f"  Loop:   {loop_mom:+.1f} mg/dL" if loop_mom is not None else "  Loop:   N/A")
        print(f"  Python: {python_mom:+.1f} mg/dL" if python_mom is not None else "  Python: N/A")
        if comparison['momentum_diff'] is not None:
            print(f"  Δ:      {comparison['momentum_diff']:+.1f} mg/dL")

        print("\n🔄 IRC Impact:")
        loop_irc = loop_result.get('irc_impact')
        python_irc = python_result.get('irc_impact')
        print(f"  Loop:   {loop_irc:+.1f} mg/dL" if loop_irc is not None else "  Loop:   N/A")
        print(f"  Python: {python_irc:+.1f} mg/dL" if python_irc is not None else "  Python: N/A")
        if comparison['irc_diff'] is not None:
            print(f"  Δ:      {comparison['irc_diff']:+.1f} mg/dL")

        print("\n" + "="*80)
        if comparison['passed']:
            print("✅ VALIDATION PASSED")
        else:
            print("❌ VALIDATION FAILED")
            if comparison['errors']:
                print("\nErrors:")
                for error in comparison['errors']:
                    print(f"  • {error}")
        print("="*80 + "\n")


def load_scenario(scenario_path: Path) -> Dict:
    """Load test scenario from JSON file."""
    if not scenario_path.exists():
        raise FileNotFoundError(f"Scenario file not found: {scenario_path}")

    scenario = json.loads(scenario_path.read_text())

    # Validate required fields
    required = ['glucoseSamples']
    for field in required:
        if field not in scenario:
            raise ValueError(f"Missing required field: {field}")

    return scenario


def launch_app_foreground(simulator_id: str, bundle_id: str, app_name: str) -> bool:
    """Launch an app in the foreground in the simulator."""
    print(f"🚀 Launching {app_name} in foreground...")
    result = subprocess.run(
        ['xcrun', 'simctl', 'launch', simulator_id, bundle_id],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print(f"❌ Failed to launch {app_name}: {result.stderr}")
        return False
    print(f"✅ {app_name} launched")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Validate Python Loop against iOS Loop",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python loop_validate.py test_scenarios/rising_bg_momentum.json
  python loop_validate.py test_scenarios/meal_with_irc.json --clear --wait 10
  python loop_validate.py test_scenarios/flat_bg_baseline.json --tolerance 0.5
        """
    )
    parser.add_argument('scenario', type=Path, help='Scenario JSON file')
    parser.add_argument('--clear', action='store_true', help='Clear HealthKit data before injection')
    parser.add_argument('--wait', type=int, default=10, help='Seconds to wait for Loop prediction (default: 10)')
    parser.add_argument('--tolerance', type=float, default=1.0, help='Acceptable difference in mg/dL (default: 1.0)')
    parser.add_argument('--bundle-id', type=str, default='com.Exercise.Loop', help='Loop app bundle ID (default: com.Exercise.Loop)')
    parser.add_argument('--injector-bundle-id', type=str, default='com.test.healthkitinjector', help='HealthKitInjectorApp bundle ID')
    parser.add_argument('--no-inject', action='store_true', help='Skip injection, only extract and compare logs')
    parser.add_argument('--no-orchestrate', action='store_true', help='Skip automatic app launching (assume apps already open)')
    args = parser.parse_args()

    print("="*80)
    print("iOS LOOP VALIDATION TOOL")
    print("="*80)

    # Step 1: Load scenario
    print(f"\n📂 Loading scenario: {args.scenario}")
    try:
        scenario = load_scenario(args.scenario)
    except Exception as e:
        print(f"❌ Failed to load scenario: {e}")
        return 1

    print(f"  • Glucose samples: {len(scenario['glucoseSamples'])}")
    print(f"  • Carb entries: {len(scenario.get('carbEntries', []))}")
    print(f"  • Insulin doses: {len(scenario.get('insulinDoses', []))}")

    if not args.no_inject:
        # Step 2: HealthKit injection
        try:
            injector = HealthKitInjector()
        except Exception as e:
            print(f"❌ {e}")
            return 1

        # Step 2a: Launch HealthKitInjectorApp in foreground (if orchestrating)
        if not args.no_orchestrate:
            if not launch_app_foreground(injector.simulator_id, args.injector_bundle_id, "HealthKitInjectorApp"):
                print("⚠️  Failed to launch HealthKitInjectorApp - assuming already running")
            time.sleep(2)  # Wait for app to initialize

        if args.clear:
            if not injector.clear_data():
                return 1
            # Wait for HealthKit deletion to complete and propagate
            print("⏳ Waiting for HealthKit clear to propagate...")
            time.sleep(3)

        if not injector.inject_scenario(args.scenario):
            return 1

        # Step 2b: Kill and relaunch Loop to refresh pump data (if orchestrating)
        if not args.no_orchestrate:
            print(f"\n📱 Restarting Loop app (to refresh pump data)...")
            # Kill Loop if running
            subprocess.run(
                ['xcrun', 'simctl', 'terminate', injector.simulator_id, args.bundle_id],
                capture_output=True
            )
            time.sleep(1)

            # Launch Loop fresh
            if not launch_app_foreground(injector.simulator_id, args.bundle_id, "Loop"):
                print("⚠️  Failed to launch Loop - assuming already running")
            time.sleep(3)  # Wait for app to initialize with fresh pump data

        # Step 3: Wait for Loop prediction
        print(f"\n⏳ Waiting {args.wait}s for Loop to run prediction...")
        time.sleep(args.wait)

    # Step 4: Extract Loop logs
    print("\n📋 Extracting Loop prediction logs...")
    loop_result, log_lines = LoopLogParser.extract_from_simulator(bundle_id=args.bundle_id)

    if not loop_result.get('eventual_bg'):
        print("❌ Failed to extract Loop prediction from logs")
        print("\nTip: Make sure Loop app is running in simulator and has executed a prediction cycle.")
        print(f"     Try increasing --wait time or check that bundle ID is correct: {args.bundle_id}")
        return 1

    print(f"✅ Extracted Loop results:")
    print(f"  • Eventual BG: {loop_result.get('eventual_bg', 'N/A')}")
    print(f"  • Momentum impact: {loop_result.get('momentum_impact', 'N/A')}")
    print(f"  • IRC impact: {loop_result.get('irc_impact', 'N/A')}")

    # Step 5: Run Python Loop
    print("\n🐍 Running Python Loop...")
    runner = PythonLoopRunner()
    try:
        python_result = runner.run_scenario(scenario)
        print(f"✅ Python Loop completed")
    except Exception as e:
        print(f"❌ Python Loop failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Step 6: Compare and report
    comparison = ValidationComparator.compare(loop_result, python_result, args.tolerance)
    ValidationComparator.print_report(loop_result, python_result, comparison)

    return 0 if comparison['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
