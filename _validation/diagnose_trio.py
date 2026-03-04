#!/usr/bin/env python3
"""
Diagnostic script for Trio eventualBG mismatches between JS and Python.

Runs test scenarios through both JS (ground truth) and Python pipelines,
printing all intermediate values side-by-side to pinpoint divergence.

Usage:
    python3 _validation/diagnose_trio.py                    # Run test 02 only
    python3 _validation/diagnose_trio.py --test 02          # Run specific test
    python3 _validation/diagnose_trio.py --all              # Run all 10 tests
    python3 _validation/diagnose_trio.py --test 02 --curves # Show full UAM curve
"""

import sys
import json
import re
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from trio_runner import TrioRunner
from trio_json_exporter import TrioJSONExporter
from trio_batch_validate import (
    define_trio_test_scenarios, load_settings, REFERENCE_TIME
)
from algorithms.openaps.iob import generate_iob_array
from algorithms.openaps.cob import recent_carbs
from algorithms.openaps.glucose_stats import get_last_glucose
from algorithms.openaps.determine_basal import determine_basal
from algorithms.openaps.predictions import generate_predictions


def run_python_with_intermediates(scenario: Dict, settings: Dict) -> Optional[Dict]:
    """Run Python pipeline and capture all intermediate values."""
    exporter = TrioJSONExporter(settings)

    glucose_samples = sorted(scenario['glucoseSamples'], key=lambda x: x['timestamp'])
    most_recent = glucose_samples[-1]
    current_bg = most_recent['value']
    current_time_unix = most_recent['timestamp']
    now_ms = int(current_time_unix * 1000)

    profile = exporter.build_profile(int(current_time_unix))
    if 'max_iob' not in profile or profile['max_iob'] is None:
        profile['max_iob'] = 3.5

    # Build pump history
    pump_history = []
    for d in scenario.get('insulinDoses', []):
        iso = datetime.fromtimestamp(d['timestamp'], tz=timezone.utc).strftime(
            '%Y-%m-%dT%H:%M:%S.000Z')
        pump_history.append({'_type': 'Bolus', 'timestamp': iso, 'amount': d['units']})

    # Build glucose data (reverse-chronological)
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

    # Run generate_predictions directly to get all intermediates
    pred_result = generate_predictions(
        bg=current_bg,
        iob_array=iob_array,
        profile=profile,
        glucose_status=glucose_status,
        meal_data=meal_data,
        iob_data=iob_data,
        clock_ms=now_ms,
    )

    # Also run determine_basal for the final result
    db_result = determine_basal(
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
        'glucose_status': glucose_status,
        'iob_data': iob_data,
        'meal_data': meal_data,
        'pred_result': pred_result,
        'db_result': db_result,
        'iob_array': iob_array,
    }


def parse_js_reason(reason: str) -> Dict[str, Any]:
    """Extract numeric values from JS reason string."""
    parsed = {}
    patterns = {
        'Dev': r'Dev:\s*(-?\d+(?:\.\d+)?)',
        'BGI': r'BGI:\s*(-?\d+(?:\.\d+)?)',
        'CR': r'CR:\s*(-?\d+(?:\.\d+)?)',
        'Target': r'Target:\s*(-?\d+(?:\.\d+)?)',
        'minPredBG': r'minPredBG\s+(-?\d+(?:\.\d+)?)',
        'minGuardBG': r'minGuardBG\s+(-?\d+(?:\.\d+)?)',
        'IOBpredBG': r'IOBpredBG\s+(-?\d+(?:\.\d+)?)',
        'COBpredBG': r'COBpredBG\s+(-?\d+(?:\.\d+)?)',
        'UAMpredBG': r'UAMpredBG\s+(-?\d+(?:\.\d+)?)',
        'insulinReq': r'insulinReq\s+(-?\d+(?:\.\d+)?)',
    }
    for key, pat in patterns.items():
        m = re.search(pat, reason)
        if m:
            parsed[key] = float(m.group(1))
    return parsed


def fmt(val, width=12, decimals=None):
    """Format a value for the comparison table."""
    if val is None:
        return str(val).rjust(width)
    if isinstance(val, float):
        if decimals is not None:
            return f"{val:.{decimals}f}".rjust(width)
        # Auto-detect reasonable precision
        if abs(val) < 0.01:
            return f"{val:.6f}".rjust(width)
        elif abs(val) < 1:
            return f"{val:.4f}".rjust(width)
        else:
            return f"{val:.2f}".rjust(width)
    return str(val).rjust(width)


def fmt_diff(js_val, py_val, width=12):
    """Format the diff column with a marker for significant diffs."""
    if js_val is None or py_val is None:
        return "N/A".rjust(width)
    diff = py_val - js_val
    marker = ""
    if abs(diff) > 1.0:
        marker = " <--"
    elif abs(diff) > 0.1:
        marker = " <-"
    return f"{diff:+.4f}".rjust(width) + marker


def print_row(label, js_val, py_val, decimals=None):
    """Print a single comparison row."""
    label_w = 24
    val_w = 14
    print(f"  {label:<{label_w}}{fmt(js_val, val_w, decimals)}{fmt(py_val, val_w, decimals)}{fmt_diff(js_val, py_val, val_w)}")


def print_header():
    """Print the comparison table header."""
    label_w = 24
    val_w = 14
    print(f"  {'':>{label_w}}{'JS':>{val_w}}{'Python':>{val_w}}{'Diff':>{val_w}}")
    print(f"  {'-'*label_w}{'-'*val_w}{'-'*val_w}{'-'*val_w}")


def diagnose_test(name: str, scenario: Dict, settings: Dict,
                  exporter: TrioJSONExporter, runner: TrioRunner,
                  show_curves: bool = False):
    """Run diagnosis on a single test."""
    print(f"\n{'='*72}")
    print(f"  TEST: {name}")
    print(f"  {scenario.get('description', '')}")
    print(f"{'='*72}")

    # --- Run JS ---
    try:
        trio_input = exporter.export_scenario(scenario)
        js_full = runner.run(trio_input)
    except Exception as e:
        print(f"  JS ERROR: {e}")
        return

    js_db = js_full.get('result', {})
    js_gs = js_full.get('glucose_status', {})
    js_iob = js_full.get('iob_data', [{}])
    js_iob0 = js_iob[0] if js_iob else {}
    js_meal = js_full.get('meal_data', {})
    js_reason = js_db.get('reason', '')
    js_parsed = parse_js_reason(js_reason)
    js_preds = js_db.get('predBGs', {})

    # --- Run Python ---
    try:
        py = run_python_with_intermediates(scenario, settings)
    except Exception as e:
        print(f"  PYTHON ERROR: {e}")
        import traceback
        traceback.print_exc()
        return

    py_gs = py['glucose_status']
    py_iob0 = py['iob_data']
    py_meal = py['meal_data']
    py_pred = py['pred_result']
    py_db = py['db_result']
    py_preds = py_pred.get('predBGs', {})

    # --- 1. Glucose Status ---
    print(f"\n  glucose_status:")
    print_header()
    print_row('glucose', js_gs.get('glucose'), py_gs.get('glucose'))
    print_row('delta', js_gs.get('delta'), py_gs.get('delta'))
    print_row('short_avgdelta', js_gs.get('short_avgdelta'), py_gs.get('short_avgdelta'))
    print_row('long_avgdelta', js_gs.get('long_avgdelta'), py_gs.get('long_avgdelta'))

    # --- 2. IOB ---
    print(f"\n  IOB:")
    print_header()
    print_row('iob', js_iob0.get('iob'), py_iob0.get('iob'))
    print_row('activity', js_iob0.get('activity'), py_iob0.get('activity'))
    print_row('lastBolusTime', js_iob0.get('lastBolusTime'), py_iob0.get('lastBolusTime'))

    # --- 3. Meal Data / Deviation Slopes ---
    print(f"\n  meal_data (slopes):")
    print_header()
    print_row('mealCOB', js_meal.get('mealCOB'), py_meal.get('mealCOB'))
    print_row('carbs', js_meal.get('carbs'), py_meal.get('carbs'))
    print_row('lastCarbTime', js_meal.get('lastCarbTime'), py_meal.get('lastCarbTime'))
    print_row('currentDeviation', js_meal.get('currentDeviation'), py_meal.get('currentDeviation'))
    print_row('maxDeviation', js_meal.get('maxDeviation'), py_meal.get('maxDeviation'))
    print_row('minDeviation', js_meal.get('minDeviation'), py_meal.get('minDeviation'))
    print_row('slopeFromMaxDev', js_meal.get('slopeFromMaxDeviation'), py_meal.get('slopeFromMaxDeviation'))
    print_row('slopeFromMinDev', js_meal.get('slopeFromMinDeviation'), py_meal.get('slopeFromMinDeviation'))

    # Deviations array comparison
    js_devs = js_meal.get('allDeviations', [])
    py_devs = py_meal.get('allDeviations', [])
    print(f"\n  allDeviations:")
    print(f"    JS  ({len(js_devs)}): {js_devs}")
    print(f"    Py  ({len(py_devs)}): {py_devs}")
    if js_devs and py_devs:
        min_len = min(len(js_devs), len(py_devs))
        diffs = [py_devs[i] - js_devs[i] for i in range(min_len)]
        print(f"    Diff({min_len}): {diffs}")

    # --- 4. Prediction Inputs ---
    print(f"\n  prediction inputs:")
    print_header()
    # From generate_predictions return
    print_row('bgi', js_parsed.get('BGI'), py_pred.get('bgi'))
    print_row('deviation', js_parsed.get('Dev'), py_pred.get('deviation'))
    print_row('ci', None, py_pred.get('ci'))  # JS doesn't expose ci directly
    print_row('uci', None, py_pred.get('uci'))
    print_row('csf', None, py_pred.get('csf'))
    print_row('cid', None, py_pred.get('cid'))

    # Slope values used in prediction loop
    slope_max = js_meal.get('slopeFromMaxDeviation', 0)
    slope_min = js_meal.get('slopeFromMinDeviation', 999)
    js_slope_from_devs = min(slope_max, -slope_min / 3) if slope_max is not None else None
    py_slope_max = py_meal.get('slopeFromMaxDeviation', 0)
    py_slope_min = py_meal.get('slopeFromMinDeviation', 999)
    py_slope_from_devs = min(py_slope_max, -py_slope_min / 3)
    print_row('slopeFromDevs', js_slope_from_devs, py_slope_from_devs)

    print_row('naive_eventualBG', None, py_pred.get('naive_eventualBG'))

    # --- 5. Final Results ---
    print(f"\n  final results:")
    print_header()
    print_row('eventualBG', js_db.get('eventualBG'), py_db.get('eventualBG'))
    print_row('minPredBG', js_parsed.get('minPredBG'), py_pred.get('minPredBG'))
    print_row('minGuardBG', js_parsed.get('minGuardBG'), py_pred.get('minGuardBG'))
    print_row('rate', js_db.get('rate'), py_db.get('rate'))
    print_row('duration', js_db.get('duration'), py_db.get('duration'))
    print_row('units (SMB)', js_db.get('units'), py_db.get('units'))
    print_row('insulinReq', js_parsed.get('insulinReq'), py_db.get('insulinReq'))

    # --- 6. Prediction Curves ---
    print(f"\n  prediction curves:")
    label_w = 24
    val_w = 14
    print(f"  {'':>{label_w}}{'JS':>{val_w}}{'Python':>{val_w}}{'Last diff':>{val_w}}")
    print(f"  {'-'*label_w}{'-'*val_w}{'-'*val_w}{'-'*val_w}")
    for curve_name in ['IOB', 'ZT', 'COB', 'UAM']:
        js_arr = js_preds.get(curve_name, [])
        py_arr = py_preds.get(curve_name, [])
        if not js_arr and not py_arr:
            continue
        js_summary = f"n={len(js_arr)},last={js_arr[-1]}" if js_arr else "absent"
        py_summary = f"n={len(py_arr)},last={py_arr[-1]}" if py_arr else "absent"
        last_diff = ""
        if js_arr and py_arr:
            diff = py_arr[-1] - js_arr[-1]
            marker = " <--" if abs(diff) > 1.0 else ""
            last_diff = f"{diff:+.0f}{marker}"
        print(f"  {curve_name:<{label_w}}{js_summary:>{val_w}}{py_summary:>{val_w}}{last_diff:>{val_w}}")

    # --- 7. Step-by-step UAM curve (if requested or large diff) ---
    js_uam = js_preds.get('UAM', [])
    py_uam = py_preds.get('UAM', [])
    uam_last_diff = abs(py_uam[-1] - js_uam[-1]) if js_uam and py_uam else 0

    if show_curves or uam_last_diff > 2:
        print(f"\n  UAM curve step-by-step (diff={uam_last_diff:.0f}):")
        print(f"  {'step':>6} {'JS':>10} {'Python':>10} {'diff':>10}")
        print(f"  {'-'*6} {'-'*10} {'-'*10} {'-'*10}")
        max_len = max(len(js_uam), len(py_uam))
        for i in range(max_len):
            js_v = js_uam[i] if i < len(js_uam) else None
            py_v = py_uam[i] if i < len(py_uam) else None
            diff_str = ""
            if js_v is not None and py_v is not None:
                d = py_v - js_v
                diff_str = f"{d:+.1f}"
            js_str = f"{js_v}" if js_v is not None else "-"
            py_str = f"{py_v}" if py_v is not None else "-"
            print(f"  {i:>6} {js_str:>10} {py_str:>10} {diff_str:>10}")

        # Also show IOB curve if it diverges
        js_iob_c = js_preds.get('IOB', [])
        py_iob_c = py_preds.get('IOB', [])
        iob_last_diff = abs(py_iob_c[-1] - js_iob_c[-1]) if js_iob_c and py_iob_c else 0
        if iob_last_diff > 2:
            print(f"\n  IOB curve step-by-step (diff={iob_last_diff:.0f}):")
            print(f"  {'step':>6} {'JS':>10} {'Python':>10} {'diff':>10}")
            print(f"  {'-'*6} {'-'*10} {'-'*10} {'-'*10}")
            max_len = max(len(js_iob_c), len(py_iob_c))
            for i in range(max_len):
                js_v = js_iob_c[i] if i < len(js_iob_c) else None
                py_v = py_iob_c[i] if i < len(py_iob_c) else None
                diff_str = ""
                if js_v is not None and py_v is not None:
                    d = py_v - js_v
                    diff_str = f"{d:+.1f}"
                js_str = f"{js_v}" if js_v is not None else "-"
                py_str = f"{py_v}" if py_v is not None else "-"
                print(f"  {i:>6} {js_str:>10} {py_str:>10} {diff_str:>10}")

    # --- JS reason for reference ---
    print(f"\n  JS reason (first 300 chars):")
    print(f"    {js_reason[:300]}")

    return {
        'name': name,
        'eventualBG_diff': (py_db.get('eventualBG') or 0) - (js_db.get('eventualBG') or 0),
    }


def main():
    show_curves = '--curves' in sys.argv
    run_all = '--all' in sys.argv
    single_test = None

    for i, arg in enumerate(sys.argv):
        if arg == '--test' and i + 1 < len(sys.argv):
            single_test = sys.argv[i + 1]

    # Default to test 02 if no flags
    if not run_all and not single_test:
        single_test = '02'

    settings = load_settings()

    # trio_runner.js lives at repo_root/trio_testing/trio_runner.js
    repo_root = Path(__file__).parent.parent
    runner_js = repo_root / "trio_testing" / "trio_runner.js"
    try:
        runner = TrioRunner(runner_js=str(runner_js))
    except FileNotFoundError:
        print(f"ERROR: trio_runner.js not found at {runner_js}")
        sys.exit(1)

    exporter = TrioJSONExporter(settings)
    scenarios = define_trio_test_scenarios(REFERENCE_TIME)

    if single_test:
        scenarios = [(n, s) for n, s in scenarios if single_test in n]
        if not scenarios:
            print(f"No scenario matching '{single_test}'")
            sys.exit(1)

    print(f"Trio Diagnostic: {len(scenarios)} test(s)")
    print(f"Reference time: {datetime.fromtimestamp(REFERENCE_TIME, tz=timezone.utc).isoformat()}")

    results = []
    for name, scenario in scenarios:
        r = diagnose_test(name, scenario, settings, exporter, runner,
                         show_curves=show_curves)
        if r:
            results.append(r)

    # Summary
    if len(results) > 1:
        print(f"\n{'='*72}")
        print(f"  SUMMARY")
        print(f"{'='*72}")
        for r in results:
            diff = r['eventualBG_diff']
            marker = " ***" if abs(diff) > 5 else ""
            print(f"  {r['name']:<30} eventualBG diff: {diff:+.0f}{marker}")


if __name__ == "__main__":
    sys.exit(main() or 0)
