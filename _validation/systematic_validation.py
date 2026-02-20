#!/usr/bin/env python3
"""
Systematic validation of Python Loop against iOS Loop.

This script runs test scenarios through both implementations and compares results:
1. Create test scenario
2. Run through iOS Loop (HealthKit injection + log extraction)
3. Run through Python Loop
4. Compare results
5. Document discrepancies
"""

import json
import time
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from algorithms.loop.loop_algorithm import LoopAlgorithm
from algorithms.base import AlgorithmInput
from healthkit_inject import clear_healthkit_data, inject_scenario, terminate_app, launch_app

# Configuration
SIMULATOR_UDID = "4A0939FF-02C0-4000-843E-EEAE8BC727CC"
LOOP_BUNDLE_ID = "com.Exercise.Loop"
VALIDATION_DIR = Path(__file__).parent / "validation_results"
VALIDATION_DIR.mkdir(exist_ok=True)


class ValidationTest:
    """A single validation test case."""

    def __init__(self, name: str, description: str, scenario: Dict):
        self.name = name
        self.description = description
        self.scenario = scenario
        self.ios_results = None
        self.python_results = None
        self.comparison = None


class SystematicValidator:
    """Runs systematic validation tests comparing Python Loop to iOS Loop."""

    def __init__(self):
        self.tests = []
        self.results = []

    def add_test(self, test: ValidationTest):
        """Add a test to the validation suite."""
        self.tests.append(test)

    def force_quit_loop(self):
        """Force quit Loop app."""
        print("🛑 Force quitting Loop...")
        try:
            subprocess.run(
                ['xcrun', 'simctl', 'terminate', SIMULATOR_UDID, LOOP_BUNDLE_ID],
                capture_output=True,
                timeout=5
            )
            time.sleep(1)
        except:
            pass

    def open_loop(self):
        """Open Loop app."""
        print("📱 Opening Loop...")
        result = subprocess.run(
            ['xcrun', 'simctl', 'launch', SIMULATOR_UDID, LOOP_BUNDLE_ID],
            capture_output=True,
            text=True,
            timeout=10
        )
        time.sleep(3)  # Wait for Loop to initialize
        return result.returncode == 0

    def extract_loop_logs(self, wait_seconds: int = 5) -> List[str]:
        """Extract Loop logs from simulator using xcrun simctl spawn."""
        print(f"📋 Extracting Loop logs (waiting {wait_seconds}s)...")

        # Wait for Loop to process
        time.sleep(wait_seconds)

        # Get logs from last 2 minutes with ##LOOP## marker
        result = subprocess.run(
            [
                'xcrun', 'simctl', 'spawn', SIMULATOR_UDID,
                'log', 'show',
                '--predicate', 'process == "Loop"',
                '--style', 'compact',
                '--last', '2m'
            ],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode != 0:
            print(f"⚠️  Warning: log show returned {result.returncode}")
            return []

        # Filter for ##LOOP## lines
        lines = [line for line in result.stdout.split('\n') if '##LOOP##' in line]
        print(f"✅ Found {len(lines)} ##LOOP## log lines")
        return lines

    def parse_loop_logs(self, log_lines: List[str]) -> Optional[Dict]:
        """Parse ##LOOP## logs to extract eventual BG and effects."""
        result = {
            'eventual_bg': None,
            'momentum_impact': None,
            'irc_impact': None,
            'raw_logs': log_lines
        }

        for line in log_lines:
            # Extract eventual BG
            if 'Eventual BG:' in line:
                import re
                match = re.search(r'Eventual BG:\s*([\d.]+)', line)
                if match:
                    result['eventual_bg'] = float(match.group(1))

            # Extract momentum impact
            if 'Momentum effect:' in line or 'momentum' in line.lower():
                import re
                match = re.search(r'([\d.]+)\s*mg/dL', line)
                if match:
                    result['momentum_impact'] = float(match.group(1))

            # Extract IRC impact
            if 'IRC effect:' in line or 'retrospective' in line.lower():
                import re
                match = re.search(r'([\d.]+)\s*mg/dL', line)
                if match:
                    result['irc_impact'] = float(match.group(1))

        return result if result['eventual_bg'] is not None else None

    def run_ios_loop(self, scenario: Dict) -> Optional[Dict]:
        """Run scenario through iOS Loop and extract results."""
        print("\n" + "="*60)
        print("RUNNING iOS LOOP")
        print("="*60)

        # Step 1: Force quit Loop
        self.force_quit_loop()

        # Step 2: Clear HealthKit (using healthkit_inject.py)
        print("🗑️  Clearing HealthKit...")
        clear_healthkit_data()
        time.sleep(2)

        # Step 3: Inject scenario
        print(f"💉 Injecting scenario: {scenario.get('name', 'Unknown')}")
        inject_scenario(scenario)
        time.sleep(2)

        # Step 4: Open Loop
        if not self.open_loop():
            print("❌ Failed to open Loop")
            return None

        # Step 5: Extract logs
        log_lines = self.extract_loop_logs(wait_seconds=8)

        if not log_lines:
            print("❌ No logs extracted")
            return None

        # Step 6: Parse logs
        results = self.parse_loop_logs(log_lines)

        if results:
            print(f"✅ iOS Loop Results:")
            print(f"   Eventual BG: {results['eventual_bg']} mg/dL")
            print(f"   Momentum: {results['momentum_impact']} mg/dL")
            print(f"   IRC: {results['irc_impact']} mg/dL")
        else:
            print("❌ Failed to parse logs")

        return results

    def run_python_loop(self, scenario: Dict) -> Dict:
        """Run scenario through Python Loop."""
        print("\n" + "="*60)
        print("RUNNING PYTHON LOOP")
        print("="*60)

        # Extract settings from scenario
        settings = scenario.get('settings', {})

        # Create Loop algorithm
        loop = LoopAlgorithm(settings)

        # Convert scenario to AlgorithmInput
        glucose_samples = scenario.get('glucoseSamples', [])
        carb_entries = scenario.get('carbEntries', [])
        insulin_doses = scenario.get('insulinDoses', [])

        if not glucose_samples:
            print("❌ No glucose samples in scenario")
            return None

        # Use last glucose sample as current reading
        current_sample = glucose_samples[-1]
        current_time = current_sample['timestamp']
        current_bg = current_sample['value']

        # Build history (all samples except last)
        cgm_history = [(s['timestamp'], s['value']) for s in glucose_samples[:-1]]

        # Convert carb entries
        carb_history = [(c['timestamp'], c['grams']) for c in carb_entries]

        # Convert insulin doses
        bolus_history = [(d['timestamp'], d['units']) for d in insulin_doses]

        # Get basal rate from settings
        basal_rate = settings.get('basal_rate', 1.0)

        # Create input
        input_data = AlgorithmInput(
            timestamp=current_time,
            cgm_reading=current_bg,
            cgm_history=cgm_history,
            current_basal=basal_rate,
            carb_entries=carb_history,
            bolus_history=bolus_history
        )

        # Run prediction
        output = loop.recommend(input_data)

        # Extract results
        predictions = output.glucose_predictions['main']
        eventual_bg = predictions[-1] if predictions else current_bg

        results = {
            'eventual_bg': eventual_bg,
            'momentum_impact': output.momentum_effect_eventual or 0.0,
            'irc_impact': output.irc_effect_eventual or 0.0,
            'full_output': output
        }

        print(f"✅ Python Loop Results:")
        print(f"   Eventual BG: {results['eventual_bg']:.1f} mg/dL")
        print(f"   Momentum: {results['momentum_impact']:.1f} mg/dL")
        print(f"   IRC: {results['irc_impact']:.1f} mg/dL")

        return results

    def compare_results(self, ios_results: Dict, python_results: Dict) -> Dict:
        """Compare iOS and Python results."""
        print("\n" + "="*60)
        print("COMPARISON")
        print("="*60)

        comparison = {
            'eventual_bg_diff': None,
            'momentum_diff': None,
            'irc_diff': None,
            'passes': False
        }

        if ios_results['eventual_bg'] is not None and python_results['eventual_bg'] is not None:
            diff = abs(ios_results['eventual_bg'] - python_results['eventual_bg'])
            comparison['eventual_bg_diff'] = diff
            print(f"Eventual BG difference: {diff:.2f} mg/dL")

            if diff <= 1.0:
                print("✅ PASS: Within 1 mg/dL tolerance")
                comparison['passes'] = True
            else:
                print(f"❌ FAIL: Exceeds 1 mg/dL tolerance")

        if ios_results['momentum_impact'] is not None and python_results['momentum_impact'] is not None:
            diff = abs(ios_results['momentum_impact'] - python_results['momentum_impact'])
            comparison['momentum_diff'] = diff
            print(f"Momentum difference: {diff:.2f} mg/dL")

        if ios_results['irc_impact'] is not None and python_results['irc_impact'] is not None:
            diff = abs(ios_results['irc_impact'] - python_results['irc_impact'])
            comparison['irc_diff'] = diff
            print(f"IRC difference: {diff:.2f} mg/dL")

        return comparison

    def run_test(self, test: ValidationTest) -> Dict:
        """Run a single validation test."""
        print("\n" + "="*80)
        print(f"TEST: {test.name}")
        print(f"DESCRIPTION: {test.description}")
        print("="*80)

        # Run iOS Loop
        ios_results = self.run_ios_loop(test.scenario)
        if not ios_results:
            print("❌ iOS Loop test failed")
            return None

        test.ios_results = ios_results

        # Run Python Loop
        python_results = self.run_python_loop(test.scenario)
        if not python_results:
            print("❌ Python Loop test failed")
            return None

        test.python_results = python_results

        # Compare
        comparison = self.compare_results(ios_results, python_results)
        test.comparison = comparison

        # Save results
        self.save_test_results(test)

        return comparison

    def save_test_results(self, test: ValidationTest):
        """Save test results to file."""
        results = {
            'test_name': test.name,
            'description': test.description,
            'scenario': test.scenario,
            'ios_results': test.ios_results,
            'python_results': {
                'eventual_bg': test.python_results['eventual_bg'],
                'momentum_impact': test.python_results['momentum_impact'],
                'irc_impact': test.python_results['irc_impact']
            },
            'comparison': test.comparison,
            'timestamp': datetime.now().isoformat()
        }

        # Sanitize filename - remove special characters
        safe_name = test.name.lower().replace(' ', '_').replace('/', '_').replace('\\', '_')
        filename = f"test_{safe_name}.json"
        filepath = VALIDATION_DIR / filename

        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n💾 Saved results to: {filepath}")

    def run_all_tests(self):
        """Run all validation tests."""
        print("\n" + "="*80)
        print(f"STARTING SYSTEMATIC VALIDATION - {len(self.tests)} tests")
        print("="*80)

        passed = 0
        failed = 0

        for i, test in enumerate(self.tests, 1):
            print(f"\n\n>>> Test {i}/{len(self.tests)}")
            comparison = self.run_test(test)

            if comparison and comparison['passes']:
                passed += 1
            else:
                failed += 1

        # Summary
        print("\n" + "="*80)
        print("VALIDATION SUMMARY")
        print("="*80)
        print(f"Total tests: {len(self.tests)}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Results saved to: {VALIDATION_DIR}")


def create_test_1_flat_bg() -> ValidationTest:
    """Test #1: Flat BG baseline."""
    now = time.time()

    scenario = {
        "name": "Test 1: Flat BG",
        "description": "Three flat glucose readings at 120 mg/dL - baseline test",
        "glucoseSamples": [
            {"timestamp": now - 600, "value": 120.0},  # 10 min ago
            {"timestamp": now - 300, "value": 120.0},  # 5 min ago
            {"timestamp": now, "value": 120.0}          # now
        ],
        "carbEntries": [],
        "insulinDoses": [],
        "settings": {
            "insulin_sensitivity_factor": 50.0,
            "carb_ratio": 10.0,
            "basal_rate": 1.0,
            "duration_of_insulin_action": 6.0,
            "target_range": [100, 120],
            "max_basal_rate": 3.0,
            "max_bolus": 5.0,
            "enable_momentum": True,
            "enable_irc": True,
            "enable_dca": True
        }
    }

    return ValidationTest(
        name="Flat BG",
        description="Baseline test with flat glucose - should have minimal momentum and no IRC",
        scenario=scenario
    )


def create_test_2_rising_1mg() -> ValidationTest:
    """Test #2: Rising BG at 1 mg/dL/min."""
    now = time.time()

    scenario = {
        "name": "Test 2: Rising 1 mg/dL/min",
        "description": "BG rising at exactly 1 mg/dL/min - test momentum calculation",
        "glucoseSamples": [
            {"timestamp": now - 600, "value": 110.0},  # 10 min ago
            {"timestamp": now - 300, "value": 115.0},  # 5 min ago
            {"timestamp": now, "value": 120.0}          # now
        ],
        "carbEntries": [],
        "insulinDoses": [],
        "settings": {
            "insulin_sensitivity_factor": 50.0,
            "carb_ratio": 10.0,
            "basal_rate": 1.0,
            "duration_of_insulin_action": 6.0,
            "target_range": [100, 120],
            "max_basal_rate": 3.0,
            "max_bolus": 5.0,
            "enable_momentum": True,
            "enable_irc": True,
            "enable_dca": True
        }
    }

    return ValidationTest(
        name="Rising 1mg/min",
        description="Rising BG - should show ~11 mg/dL momentum impact over 15 minutes",
        scenario=scenario
    )


def main():
    """Main entry point."""
    validator = SystematicValidator()

    # Add tests (start with Test #1)
    validator.add_test(create_test_1_flat_bg())
    validator.add_test(create_test_2_rising_1mg())  # Test #2

    # Run all tests
    validator.run_all_tests()


if __name__ == '__main__':
    main()
