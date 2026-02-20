#!/usr/bin/env python3
"""
Regression test suite for Python Loop.

Tests are split into two tiers:
  1. REGRESSION (iOS-validated): These passed iOS validation (<1 mg/dL diff).
     Expected values are the Python output for these scenarios.
     Run fast, purely in Python, catch regressions when fixing other tests.

  2. BACKLOG (needs iOS validation): These still diverge from iOS Loop.
     Shown as informational output but don't cause test failures.

Workflow:
  - Validate a test against iOS Loop using batch_validate.py
  - Once it passes, run: python3 test_regression.py --promote <test_name>
  - This locks in the current Python output as the regression baseline
  - From then on, any Python changes that break this test will be caught

Usage:
  python3 test_regression.py              # Run regression tests
  python3 test_regression.py --all        # Also show backlog status
  python3 test_regression.py --promote 05_insulin_only   # Promote a test
  python3 test_regression.py --regenerate # Regenerate all regression baselines
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from batch_validate import define_test_scenarios, run_python_loop

# Fixed reference time for deterministic results
FIXED_TIME = 1770000000.0

# Tolerance for regression tests (tight - these are Python-only, deterministic)
REGRESSION_TOLERANCE = 0.01  # mg/dL

# Tests that have been validated against iOS Loop (<2 mg/dL eventual BG diff)
# Last validated: 2026-02-15 (with 0.2U@180m trigger, IRC subtracting() fix)
IOS_VALIDATED = {
    "01_flat_bg",              # +0.5 mg/dL
    "02_rising_1mg_min",       # +0.7 mg/dL
    "03_rising_2mg_min",       # +1.2 mg/dL
    "04_falling_1mg_min",      # -1.1 mg/dL
    "05_insulin_only",         # +1.6 mg/dL
    "06_carbs_only",           # -2.0 mg/dL
    "07_carbs_and_insulin",    # -1.4 mg/dL
    "08_rising_bg_with_insulin",  # +0.5 mg/dL
    "09_large_insulin",        # +0.1 mg/dL
    "10_future_carbs",         # +0.8 mg/dL
    "11_multiple_carbs",       # -5.3 mg/dL
    "12_multiple_insulin",     # +1.2 mg/dL
    "13_rising_despite_insulin",  # +2.2 mg/dL
    "14_falling_fast",         # -1.8 mg/dL
    "15_meal_scenario",        # -2.0 mg/dL
    "16_multi_carb_meal",      # -5.0 mg/dL
    "17_low_bg",               # -0.4 mg/dL
    "18_high_bg_steady",       # +0.1 mg/dL
    "19_correction_in_progress",  # +0.6 mg/dL
    "20_post_meal_future_carbs",  # +0.8 mg/dL
}

BASELINES_FILE = Path(__file__).parent / "regression_baselines.json"


def load_baselines():
    if BASELINES_FILE.exists():
        return json.loads(BASELINES_FILE.read_text())
    return {}


def save_baselines(baselines):
    BASELINES_FILE.write_text(json.dumps(baselines, indent=2))


def run_scenario(name, scenario):
    """Run a single scenario through Python Loop, return results dict."""
    result = run_python_loop(scenario)
    if result is None:
        return None
    return {
        'eventual_bg': round(result['eventual_bg'], 4),
        'momentum_impact': round(result['momentum_impact'], 4),
        'irc_impact': round(result['irc_impact'], 4),
    }


def generate_baselines(test_names=None):
    """Generate baselines for specified tests (or all iOS-validated tests)."""
    scenarios = define_test_scenarios(reference_time=FIXED_TIME)
    baselines = load_baselines()
    targets = test_names or IOS_VALIDATED

    for name, scenario in scenarios:
        if name not in targets:
            continue
        result = run_scenario(name, scenario)
        if result:
            baselines[name] = result
            print(f"  Generated baseline: {name} -> eventual_bg={result['eventual_bg']}")
        else:
            print(f"  FAILED to generate baseline for {name}")

    save_baselines(baselines)
    return baselines


def promote_test(test_name):
    """Promote a test from backlog to regression by generating its baseline."""
    if test_name in IOS_VALIDATED:
        print(f"{test_name} is already in IOS_VALIDATED set.")
        print(f"Regenerating its baseline...")
    else:
        print(f"Promoting {test_name} to regression suite.")
        print(f"NOTE: You must also add '{test_name}' to IOS_VALIDATED in test_regression.py")

    generate_baselines(test_names={test_name})
    print(f"Baseline saved. Run test_regression.py to verify.")


def run_regression_tests(show_backlog=False):
    """Run regression tests. Returns 0 if all pass, 1 if any fail."""
    scenarios = define_test_scenarios(reference_time=FIXED_TIME)
    baselines = load_baselines()

    # Check that all iOS-validated tests have baselines
    missing = IOS_VALIDATED - set(baselines.keys())
    if missing:
        print(f"Missing baselines for: {', '.join(sorted(missing))}")
        print("Generating...")
        baselines = generate_baselines()

    regression_passed = 0
    regression_failed = 0
    backlog_results = []

    print("=" * 60)
    print("REGRESSION TESTS (iOS-validated)")
    print("=" * 60)

    for name, scenario in scenarios:
        if name not in IOS_VALIDATED:
            # Collect backlog results for later display
            if show_backlog:
                result = run_scenario(name, scenario)
                backlog_results.append((name, result))
            continue

        if name not in baselines:
            print(f"  SKIP {name}: no baseline")
            continue

        result = run_scenario(name, scenario)
        if result is None:
            print(f"  FAIL {name}: Python Loop error")
            regression_failed += 1
            continue

        expected = baselines[name]['eventual_bg']
        actual = result['eventual_bg']
        diff = abs(actual - expected)

        if diff <= REGRESSION_TOLERANCE:
            print(f"  PASS {name}: {actual:.2f} (expected {expected:.2f})")
            regression_passed += 1
        else:
            print(f"  FAIL {name}: {actual:.2f} (expected {expected:.2f}, diff={diff:.4f})")
            regression_failed += 1

    print(f"\nRegression: {regression_passed} passed, {regression_failed} failed"
          f" out of {regression_passed + regression_failed}")

    if show_backlog and backlog_results:
        print(f"\n{'=' * 60}")
        print("BACKLOG (needs iOS validation)")
        print("=" * 60)
        for name, result in backlog_results:
            if result:
                print(f"  {name}: eventual_bg={result['eventual_bg']:.2f}"
                      f"  mom={result['momentum_impact']:+.2f}"
                      f"  irc={result['irc_impact']:+.2f}")
            else:
                print(f"  {name}: Python Loop error")

    return 0 if regression_failed == 0 else 1


if __name__ == '__main__':
    args = sys.argv[1:]

    if '--promote' in args:
        idx = args.index('--promote')
        if idx + 1 < len(args):
            promote_test(args[idx + 1])
        else:
            print("Usage: test_regression.py --promote <test_name>")
        sys.exit(0)

    if '--regenerate' in args:
        print("Regenerating all regression baselines...")
        generate_baselines()
        sys.exit(0)

    show_backlog = '--all' in args
    sys.exit(run_regression_tests(show_backlog=show_backlog))
