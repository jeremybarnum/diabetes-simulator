#!/usr/bin/env python3
"""
Trio regression test suite: Fast Python-only tests validated against JS ground truth.

Two-tier approach (same as Loop's test_regression.py):
1. JS-validated baselines locked in trio_regression_baselines.json
2. Python tests compare against those baselines

Usage:
    python3 trio_test_regression.py              # Run regression tests
    python3 trio_test_regression.py --all        # Show backlog status too
    python3 trio_test_regression.py --generate   # Generate baselines from JS
    python3 trio_test_regression.py --promote TEST_NAME  # Promote test to regression
    python3 trio_test_regression.py --verbose    # Show detailed output
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Set

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from trio_batch_validate import (
    define_trio_test_scenarios, REFERENCE_TIME, load_settings,
    run_js_trio, run_python_trio
)
from trio_runner import TrioRunner
from trio_json_exporter import TrioJSONExporter

BASELINES_FILE = Path(__file__).parent / "trio_regression_baselines.json"

# Tests that have been validated against JS ground truth
# Tolerance: the Python result must match JS within this tolerance
JS_VALIDATED: Set[str] = set()  # Will be populated as tests pass

# Default tolerance for eventualBG comparison
TOLERANCE = 1.0  # mg/dL


def load_baselines() -> Dict:
    """Load regression baselines."""
    if BASELINES_FILE.exists():
        with open(BASELINES_FILE) as f:
            return json.load(f)
    return {}


def save_baselines(baselines: Dict):
    """Save regression baselines."""
    with open(BASELINES_FILE, 'w') as f:
        json.dump(baselines, f, indent=2, sort_keys=True)


def generate_baselines():
    """Generate baselines from JS ground truth for all scenarios."""
    print("Generating JS baselines for all scenarios...")

    settings = load_settings()
    runner = TrioRunner()
    exporter = TrioJSONExporter(settings)
    scenarios = define_trio_test_scenarios(REFERENCE_TIME)

    baselines = {}
    for name, scenario in scenarios:
        print(f"  Running JS for {name}...", end=" ")
        js_result = run_js_trio(scenario, exporter, runner)

        if js_result is None or js_result.get('status') != 'success':
            print("FAILED")
            continue

        db = js_result.get('result', {})
        iob_data = js_result.get('iob_data', [{}])
        meal_data = js_result.get('meal_data', {})

        baseline = {
            'eventualBG': db.get('eventualBG'),
            'iob': iob_data[0].get('iob', 0) if iob_data else 0,
            'cob': meal_data.get('mealCOB', 0),
            'rate': db.get('rate'),
            'duration': db.get('duration'),
            'units': db.get('units'),
            'reason_prefix': db.get('reason', '')[:80],
            'predBGs': {},
        }

        # Save prediction arrays
        pred_bgs = db.get('predBGs', {})
        for pred_type in ['IOB', 'COB', 'UAM', 'ZT']:
            arr = pred_bgs.get(pred_type, [])
            if arr:
                baseline['predBGs'][pred_type] = {
                    'count': len(arr),
                    'first': arr[0],
                    'last': arr[-1],
                }

        baselines[name] = baseline
        ebg = baseline['eventualBG']
        print(f"eventualBG={ebg}, IOB={baseline['iob']:.3f}, "
              f"COB={baseline['cob']:.0f}")

    save_baselines(baselines)
    print(f"\nSaved {len(baselines)} baselines to {BASELINES_FILE}")
    return baselines


def run_regression_tests(verbose: bool = False, show_all: bool = False) -> int:
    """
    Run regression tests comparing Python vs JS baselines.

    Returns exit code (0 = all pass, 1 = failures).
    """
    baselines = load_baselines()
    if not baselines:
        print("No baselines found. Run with --generate first.")
        return 1

    settings = load_settings()
    scenarios = define_trio_test_scenarios(REFERENCE_TIME)

    passed = 0
    failed = 0
    skipped = 0
    backlog = 0

    print(f"\nTrio Regression Tests")
    print(f"{'='*60}")
    print(f"Baselines: {len(baselines)}")
    print(f"Validated: {len(JS_VALIDATED)}")
    print(f"Tolerance: {TOLERANCE} mg/dL")
    print()

    for name, scenario in scenarios:
        if name not in baselines:
            if show_all:
                print(f"  [SKIP] {name} — no baseline")
            skipped += 1
            continue

        baseline = baselines[name]
        js_ebg = baseline.get('eventualBG')

        if name not in JS_VALIDATED:
            # Still in backlog — just show status
            if show_all:
                print(f"  [BACKLOG] {name} — JS eventualBG={js_ebg}")
            backlog += 1
            continue

        # Run Python
        py_result = run_python_trio(scenario, settings)

        if py_result is None:
            print(f"  [FAIL] {name} — Python returned None")
            failed += 1
            continue

        # Compare eventualBG
        py_ebg = py_result.get('eventualBG')

        if js_ebg is None:
            # JS didn't compute eventualBG (early return)
            # Compare other fields instead
            py_iob = py_result.get('IOB', 0)
            js_iob = baseline.get('iob', 0)
            iob_diff = abs(py_iob - js_iob)

            if iob_diff <= 0.01:
                print(f"  [PASS] {name} — IOB diff={iob_diff:.4f}")
                passed += 1
            else:
                print(f"  [FAIL] {name} — IOB diff={iob_diff:.4f} (>{0.01})")
                failed += 1
        else:
            if py_ebg is None:
                print(f"  [FAIL] {name} — Python eventualBG=None, JS={js_ebg}")
                failed += 1
                continue

            ebg_diff = py_ebg - js_ebg
            if abs(ebg_diff) <= TOLERANCE:
                print(f"  [PASS] {name} — eventualBG diff={ebg_diff:+.2f}")
                passed += 1
            else:
                print(f"  [FAIL] {name} — eventualBG diff={ebg_diff:+.2f} "
                      f"(Python={py_ebg}, JS={js_ebg})")
                failed += 1

        if verbose and py_result:
            print(f"         Python: IOB={py_result.get('IOB', 0):.3f}, "
                  f"COB={py_result.get('COB', 0):.0f}, "
                  f"rate={py_result.get('rate')}")

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed, "
          f"{backlog} backlog, {skipped} skipped")

    return 1 if failed > 0 else 0


def promote_test(test_name: str):
    """Promote a test from backlog to validated regression suite."""
    baselines = load_baselines()
    if test_name not in baselines:
        print(f"No baseline for '{test_name}'. Run --generate first.")
        return

    print(f"Promoting {test_name} to regression suite.")
    print(f"Add '{test_name}' to JS_VALIDATED set in trio_test_regression.py")
    print(f"Baseline: {json.dumps(baselines[test_name], indent=2)}")


def main():
    if '--generate' in sys.argv:
        generate_baselines()
        return 0
    elif '--promote' in sys.argv:
        idx = sys.argv.index('--promote')
        if idx + 1 < len(sys.argv):
            promote_test(sys.argv[idx + 1])
        else:
            print("Usage: --promote TEST_NAME")
        return 0
    else:
        verbose = '--verbose' in sys.argv or '-v' in sys.argv
        show_all = '--all' in sys.argv
        return run_regression_tests(verbose=verbose, show_all=show_all)


if __name__ == "__main__":
    sys.exit(main())
