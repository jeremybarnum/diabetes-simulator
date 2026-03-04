#!/usr/bin/env python3
"""
Trio batch validation harness: Runs scenarios through both Node.js (ground truth)
and Python implementation, compares results.

Similar to batch_validate.py for Loop, but using Node.js instead of iOS simulator.

Usage:
    python3 trio_batch_validate.py                  # Run all tests
    python3 trio_batch_validate.py --test 01_flat_bg # Run single test
    python3 trio_batch_validate.py --js-only         # Only run JS (ground truth)
    python3 trio_batch_validate.py --verbose          # Show full output
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from trio_runner import TrioRunner, TrioComparison
from trio_json_exporter import TrioJSONExporter
from algorithms.openaps.iob import generate_iob_array
from algorithms.openaps.cob import recent_carbs
from algorithms.openaps.glucose_stats import get_last_glucose
from algorithms.openaps.determine_basal import determine_basal


# Fixed reference time for deterministic results
# Use :30 past the hour to avoid oref1's "cancel temp near top of hour" guard
# (oref1 cancels temps if deliverAt.getMinutes() >= 55 and skip_neutral_temps is true)
REFERENCE_TIME = 1739622600.0  # 2025-02-15T12:30:00Z


def load_settings() -> Dict:
    """Load settings from settings.json."""
    settings_path = Path(__file__).parent.parent / "settings.json"
    with open(settings_path) as f:
        return json.load(f)


def create_scenario(name: str, description: str,
                    bg_readings: List[Tuple[float, float]],
                    carb_entries: List[Dict] = None,
                    insulin_doses: List[Dict] = None,
                    reference_time: float = None) -> Dict:
    """
    Create a scenario dict with relative timestamps.

    bg_readings: [(minutes_ago, value), ...]
    carb_entries: [{'minutes_ago': float, 'grams': float, 'absorptionHours': float}, ...]
    insulin_doses: [{'minutes_ago': float, 'units': float}, ...]
    reference_time: fixed unix timestamp
    """
    now = reference_time if reference_time is not None else time.time()

    scenario = {
        'name': name,
        'description': description,
        'glucoseSamples': [
            {'timestamp': now - mins_ago * 60, 'value': value}
            for mins_ago, value in bg_readings
        ],
        'carbEntries': [],
        'insulinDoses': []
    }

    if carb_entries:
        for c in carb_entries:
            scenario['carbEntries'].append({
                'timestamp': now - c['minutes_ago'] * 60,
                'grams': c['grams'],
                'absorptionHours': c.get('absorptionHours', 3.0)
            })

    if insulin_doses:
        for d in insulin_doses:
            scenario['insulinDoses'].append({
                'timestamp': now - d['minutes_ago'] * 60,
                'units': d['units']
            })

    return scenario


def define_trio_test_scenarios(reference_time: float = None) -> List[Tuple[str, Dict]]:
    """Define test scenarios for Trio validation."""
    scenarios = []
    rt = reference_time

    # Test 1: Slightly rising BG baseline (oref1 exits early if deltas are exactly 0)
    scenarios.append(("01_slight_rise", create_scenario(
        "Slight Rise Baseline",
        "BG rising slightly from 118 to 121, no carbs, no insulin",
        bg_readings=[
            (30, 118.0), (25, 118.5), (20, 119.0), (15, 119.5),
            (10, 120.0), (5, 120.5), (0.5, 121.0)
        ],
        reference_time=rt,
    )))

    # Test 2: High BG with IOB (bolus already working)
    scenarios.append(("02_high_bg_with_iob", create_scenario(
        "High BG + IOB",
        "BG slowly falling from 155 to 148, 2U bolus 60min ago",
        bg_readings=[
            (30, 155.0), (25, 154.0), (20, 153.0), (15, 151.0),
            (10, 150.0), (5, 149.0), (0.5, 148.0)
        ],
        insulin_doses=[{'minutes_ago': 60, 'units': 2.0}],
        reference_time=rt,
    )))

    # Test 3: Rising BG, no declared cause (deviation/UAM test)
    scenarios.append(("03_rising_bg_no_cause", create_scenario(
        "Rising BG (UAM)",
        "BG rising from 100 to 130, no carbs or insulin declared",
        bg_readings=[
            (30, 100.0), (25, 105.0), (20, 110.0), (15, 115.0),
            (10, 120.0), (5, 125.0), (0.5, 130.0)
        ],
        reference_time=rt,
    )))

    # Test 4: Rising BG + COB (carb effect, no bolus)
    scenarios.append(("04_rising_with_cob", create_scenario(
        "Rising BG + COB",
        "BG rising from 100 to 115, 30g carbs 60min ago, no bolus",
        bg_readings=[
            (30, 100.0), (25, 102.0), (20, 105.0), (15, 108.0),
            (10, 111.0), (5, 113.0), (0.5, 115.0)
        ],
        carb_entries=[{'minutes_ago': 60, 'grams': 30.0}],
        reference_time=rt,
    )))

    # Test 5: Meal scenario (carbs + bolus)
    scenarios.append(("05_meal_bolus", create_scenario(
        "Meal + Bolus",
        "30g carbs + 3U bolus 30min ago, BG rising",
        bg_readings=[
            (45, 100.0), (40, 100.0), (35, 102.0), (30, 105.0),
            (25, 110.0), (20, 118.0), (15, 125.0), (10, 130.0),
            (5, 133.0), (0.5, 135.0)
        ],
        carb_entries=[{'minutes_ago': 30, 'grams': 30.0}],
        insulin_doses=[{'minutes_ago': 30, 'units': 3.0}],
        reference_time=rt,
    )))

    # Test 6: High BG correction (SMB test) - slight rise to avoid "unchanged" guard
    scenarios.append(("06_high_bg_correction", create_scenario(
        "High BG Correction",
        "BG at 220 slightly rising, needs correction",
        bg_readings=[
            (30, 218.0), (25, 218.5), (20, 219.0), (15, 219.5),
            (10, 220.0), (5, 220.5), (0.5, 221.0)
        ],
        reference_time=rt,
    )))

    # Test 7: Going low (zero temp / suspend)
    scenarios.append(("07_going_low", create_scenario(
        "Going Low",
        "BG falling from 90 to 75, near suspend threshold",
        bg_readings=[
            (20, 90.0), (15, 85.0), (10, 82.0), (5, 78.0), (0.5, 75.0)
        ],
        reference_time=rt,
    )))

    # Test 8: Multiple carb entries
    scenarios.append(("08_multiple_carbs", create_scenario(
        "Multiple Carb Entries",
        "20g at 90min + 30g at 30min, no insulin",
        bg_readings=[
            (15, 110.0), (10, 115.0), (5, 120.0), (0.5, 125.0)
        ],
        carb_entries=[
            {'minutes_ago': 90, 'grams': 20.0},
            {'minutes_ago': 30, 'grams': 30.0},
        ],
        reference_time=rt,
    )))

    # Test 9: Correction in progress
    scenarios.append(("09_correction_in_progress", create_scenario(
        "Correction In Progress",
        "BG falling from 240 after 3U correction 45min ago",
        bg_readings=[
            (60, 240.0), (55, 238.0), (50, 235.0), (45, 232.0),
            (40, 228.0), (35, 224.0), (30, 220.0), (25, 216.0),
            (20, 213.0), (15, 210.0), (10, 208.0), (5, 206.0), (0.5, 204.0)
        ],
        insulin_doses=[{'minutes_ago': 45, 'units': 3.0}],
        reference_time=rt,
    )))

    # Test 10: UAM detection (BG rising, no carbs entered)
    scenarios.append(("10_uam_detection", create_scenario(
        "UAM Detection",
        "BG rising rapidly from 100 to 160, no carbs entered",
        bg_readings=[
            (30, 100.0), (25, 110.0), (20, 120.0), (15, 132.0),
            (10, 144.0), (5, 154.0), (0.5, 160.0)
        ],
        reference_time=rt,
    )))

    return scenarios


def run_js_trio(scenario: Dict, exporter: TrioJSONExporter,
                runner: TrioRunner) -> Optional[Dict]:
    """Run scenario through JS (ground truth)."""
    try:
        trio_input = exporter.export_scenario(scenario)
        result = runner.run(trio_input)
        return result
    except Exception as e:
        print(f"    JS error: {e}")
        return None


def run_python_trio(scenario: Dict, settings: Dict) -> Optional[Dict]:
    """Run scenario through Python OpenAPS implementation.

    Uses the same direct pipeline as simulation.py's get_trio_recommendation():
    generate_iob_array → recent_carbs → get_last_glucose → determine_basal.
    """
    try:
        exporter = TrioJSONExporter(settings)

        glucose_samples = sorted(scenario['glucoseSamples'], key=lambda x: x['timestamp'])
        most_recent = glucose_samples[-1]
        current_bg = most_recent['value']
        current_time_unix = most_recent['timestamp']
        now_ms = int(current_time_unix * 1000)

        profile = exporter.build_profile(int(current_time_unix))
        if 'max_iob' not in profile or profile['max_iob'] is None:
            profile['max_iob'] = 3.5

        # Build pump history (boluses as ISO-timestamped dicts)
        pump_history = []
        for d in scenario.get('insulinDoses', []):
            iso = datetime.fromtimestamp(d['timestamp'], tz=timezone.utc).strftime(
                '%Y-%m-%dT%H:%M:%S.000Z')
            pump_history.append({'_type': 'Bolus', 'timestamp': iso, 'amount': d['units']})

        # Build glucose data (reverse-chronological dicts)
        glucose_data = []
        for s in glucose_samples:
            glucose_data.append({
                'glucose': s['value'], 'sgv': s['value'],
                'date': int(s['timestamp'] * 1000),
                'dateString': datetime.fromtimestamp(
                    s['timestamp'], tz=timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.000Z'),
            })
        glucose_data.sort(key=lambda x: x['date'], reverse=True)

        # Build carb treatments
        carb_treatments = []
        for c in scenario.get('carbEntries', []):
            iso = datetime.fromtimestamp(c['timestamp'], tz=timezone.utc).strftime(
                '%Y-%m-%dT%H:%M:%S.000Z')
            carb_treatments.append({
                'carbs': c['grams'], 'nsCarbs': c['grams'],
                'timestamp': iso, 'created_at': iso,
            })

        bp = exporter.build_basal_profile()
        iob_array = generate_iob_array(pump_history, profile, now_ms)
        iob_data = iob_array[0] if iob_array else {'iob': 0, 'activity': 0}

        meal_data = recent_carbs(
            treatments=carb_treatments, time_ms=now_ms, profile=profile,
            glucose_data=glucose_data, pump_history=pump_history, basalprofile=bp,
        )

        glucose_status = get_last_glucose(glucose_data)
        if not glucose_status:
            glucose_status = {
                'glucose': current_bg, 'delta': 0,
                'short_avgdelta': 0, 'long_avgdelta': 0, 'date': now_ms,
            }

        result = determine_basal(
            glucose_status=glucose_status,
            currenttemp={'rate': 0, 'duration': 0},
            iob_data=iob_data,
            profile=profile,
            meal_data=meal_data,
            iob_array=iob_array,
            micro_bolus_allowed=True,
            clock_ms=now_ms,
        )

        return {
            'eventualBG': result.get('eventualBG'),
            'IOB': iob_data.get('iob', 0),
            'COB': meal_data.get('mealCOB', 0) if meal_data else 0,
            'rate': result.get('rate'),
            'duration': result.get('duration'),
            'units': result.get('units'),
            'reason': result.get('reason', ''),
            'predBGs': result.get('predBGs', {}),
        }
    except Exception as e:
        print(f"    Python error: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_test(name: str, scenario: Dict, settings: Dict,
             exporter: TrioJSONExporter, runner: TrioRunner,
             results_log: List[Dict],
             js_only: bool = False, verbose: bool = False) -> Dict:
    """Run a single test through both JS and Python, compare."""
    print(f"\n{'='*70}")
    print(f"TEST: {name} - {scenario.get('description', '')}")
    print(f"{'='*70}")

    # Run JS (ground truth)
    print("  Running JS (ground truth)...")
    js_result = run_js_trio(scenario, exporter, runner)

    if js_result is None:
        print("  FAILED: JS runner returned no result")
        result = {'name': name, 'status': 'JS_FAILED'}
        results_log.append(result)
        return result

    # Extract key values
    js_db = js_result.get('result', {})
    js_ebg = js_db.get('eventualBG')
    js_iob = js_result.get('iob_data', [{}])[0].get('iob', 0)
    js_cob = js_result.get('meal_data', {}).get('mealCOB', 0)
    js_rate = js_db.get('rate')
    js_duration = js_db.get('duration')
    js_smb = js_db.get('units')

    print(f"  JS results:")
    print(f"    eventualBG: {js_ebg}")
    print(f"    IOB: {js_iob:.3f} U")
    print(f"    COB: {js_cob:.0f} g")
    if js_rate is not None:
        print(f"    Temp basal: {js_rate:.2f} U/hr for {js_duration}min")
    if js_smb:
        print(f"    SMB: {js_smb:.2f} U")

    # Show predictions
    pred_bgs = js_db.get('predBGs', {})
    for pred_type in ['IOB', 'COB', 'UAM', 'ZT']:
        arr = pred_bgs.get(pred_type, [])
        if arr:
            print(f"    predBGs.{pred_type}: {len(arr)} points, "
                  f"first={arr[0]}, last={arr[-1]}")

    if verbose:
        reason = js_db.get('reason', '')
        print(f"    Reason: {reason[:200]}")
        # Show stderr (JS console.error)
        stderr = js_result.get('_stderr', '')
        if stderr:
            print(f"    JS debug output:")
            for line in stderr.split('\n')[:20]:
                print(f"      {line}")

    if js_only:
        result = {
            'name': name, 'status': 'JS_ONLY',
            'js_eventualBG': js_ebg,
            'js_iob': js_iob,
            'js_cob': js_cob,
            'js_rate': js_rate,
            'js_smb': js_smb,
        }
        results_log.append(result)
        return result

    # Run Python
    print("  Running Python...")
    py_result = run_python_trio(scenario, settings)

    if py_result is None:
        print("  FAILED: Python returned no result")
        result = {'name': name, 'status': 'PYTHON_FAILED', 'js_eventualBG': js_ebg}
        results_log.append(result)
        return result

    print(f"  Python results:")
    print(f"    IOB: {py_result.get('IOB', 0):.3f} U")
    print(f"    COB: {py_result.get('COB', 0):.0f} g")
    if py_result.get('rate') is not None:
        print(f"    Temp basal: {py_result['rate']:.2f} U/hr for {py_result.get('duration', 30)}min")

    # Compare
    comparison = TrioComparison.compare_results(js_result, py_result)
    TrioComparison.print_comparison(name, comparison, verbose=verbose)

    result = {
        'name': name,
        'status': 'PASS' if comparison['passed'] else 'FAIL',
        'js_eventualBG': js_ebg,
        'js_iob': js_iob,
        'js_cob': js_cob,
        'diffs': comparison['diffs'],
    }
    results_log.append(result)
    return result


def main():
    js_only = '--js-only' in sys.argv
    verbose = '--verbose' in sys.argv or '-v' in sys.argv
    single_test = None

    for i, arg in enumerate(sys.argv):
        if arg == '--test' and i + 1 < len(sys.argv):
            single_test = sys.argv[i + 1]

    settings = load_settings()

    # Initialize runner
    try:
        runner = TrioRunner()
    except FileNotFoundError:
        print("ERROR: trio_runner.js not found")
        sys.exit(1)

    exporter = TrioJSONExporter(settings)

    # Define scenarios
    scenarios = define_trio_test_scenarios(REFERENCE_TIME)

    if single_test:
        scenarios = [(n, s) for n, s in scenarios if single_test in n]
        if not scenarios:
            print(f"No scenario matching '{single_test}'")
            sys.exit(1)

    results_log = []

    print(f"\nTrio Batch Validation")
    print(f"{'='*70}")
    print(f"Reference time: {datetime.fromtimestamp(REFERENCE_TIME, tz=timezone.utc).isoformat()}")
    print(f"Scenarios: {len(scenarios)}")
    print(f"Mode: {'JS only' if js_only else 'JS + Python comparison'}")

    for name, scenario in scenarios:
        run_test(name, scenario, settings, exporter, runner,
                 results_log, js_only=js_only, verbose=verbose)

    # Summary
    print(f"\n\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")

    passed = sum(1 for r in results_log if r['status'] == 'PASS')
    failed = sum(1 for r in results_log if r['status'] == 'FAIL')
    js_only_count = sum(1 for r in results_log if r['status'] == 'JS_ONLY')
    errors = sum(1 for r in results_log if r['status'] in ('JS_FAILED', 'PYTHON_FAILED'))

    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    if js_only_count:
        print(f"JS only: {js_only_count}")
    if errors:
        print(f"Errors: {errors}")

    for r in results_log:
        status = r['status']
        name = r['name']
        if status == 'JS_ONLY':
            js_ebg = r.get('js_eventualBG', '?')
            print(f"  [{status}] {name} — eventualBG={js_ebg}")
        elif status in ('JS_FAILED', 'PYTHON_FAILED'):
            print(f"  [{status}] {name}")
        else:
            ebg_diff = r.get('diffs', {}).get('eventualBG', 0)
            print(f"  [{status}] {name} — eventualBG diff: {ebg_diff:+.2f}")

    # Save results
    results_file = Path(__file__).parent / "trio_validation_results.json"
    with open(results_file, 'w') as f:
        # Convert non-serializable types
        clean_results = json.loads(json.dumps(results_log, default=str))
        json.dump(clean_results, f, indent=2)
    print(f"\nResults saved to {results_file}")

    return 1 if failed > 0 or errors > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
