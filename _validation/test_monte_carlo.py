#!/usr/bin/env python3
"""
Monte Carlo directional correctness tests.

Validates that the simulation produces expected directional relationships:
higher carb uncertainty → worse TIR, carb bias → shifted mean BG, etc.

Each test creates two profiles differing in one parameter, runs 30 paths
with a fixed seed, and asserts the expected relationship on median metrics.

Usage:
    python3 test_monte_carlo.py           # Run all tests
    python3 test_monte_carlo.py --verbose  # Show metric values
"""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from simulation import PatientProfile, MealSpec
from monte_carlo import run_monte_carlo

# ─── Configuration ────────────────────────────────────────────────────────────

ALGO = 'loop_ab40'
N_PATHS = 20
N_DAYS = 3
SEED = 42
MAX_WORKERS = os.cpu_count() or 4


# ─── Helpers ──────────────────────────────────────────────────────────────────

def make_base_profile(**overrides):
    """Create a reproducible base profile for directional tests."""
    # Per-meal carb_count_sigma: extract from overrides or default to 0.15
    meal_sigma = overrides.pop('carb_count_sigma', 0.15)
    defaults = dict(
        meals_rest=[MealSpec(time_of_day_minutes=180, carbs_mean=50, carbs_sd=5,
                        absorption_hrs=3.0, carb_count_sigma=meal_sigma)],
        carb_count_bias=0.0,
        absorption_sigma=0.15,
        undeclared_meal_prob=0.0,
        undeclared_meals_rest=[],
        sensitivity_sigma=0.15,
        starting_bg=120.0,
        rescue_carbs_enabled=True,
        rescue_threshold=65.0,
        rescue_carbs_grams=8.0,
        rescue_cooldown_min=15.0,
        rescue_carbs_declared_pct=0.0,
    )
    defaults.update(overrides)
    return PatientProfile(**defaults)


def run_comparison(base_profile, modified_profile):
    """Run MC on two profiles, return (base_summary, modified_summary)."""
    base_results = run_monte_carlo(
        profile=base_profile,
        algorithms=[ALGO],
        n_paths=N_PATHS,
        n_days=N_DAYS,
        seed=SEED,
        max_workers=MAX_WORKERS,
    )
    mod_results = run_monte_carlo(
        profile=modified_profile,
        algorithms=[ALGO],
        n_paths=N_PATHS,
        n_days=N_DAYS,
        seed=SEED,
        max_workers=MAX_WORKERS,
    )
    return base_results[ALGO].summary(), mod_results[ALGO].summary()


def get_tir_sd(mc_results_obj):
    """Get the standard deviation of TIR across paths."""
    import numpy as np
    tirs = [m.time_in_range for m in mc_results_obj.all_metrics]
    return float(np.std(tirs))


# ─── Tests ────────────────────────────────────────────────────────────────────

def test_higher_carb_sigma_lowers_tir(verbose=False):
    """Higher carb counting error → lower TIR."""
    base = make_base_profile(carb_count_sigma=0.05)
    modified = make_base_profile(carb_count_sigma=0.40)

    base_s, mod_s = run_comparison(base, modified)

    base_tir = base_s['time_in_range']['median']
    mod_tir = mod_s['time_in_range']['median']

    if verbose:
        print(f"    Base TIR median: {base_tir:.1f}%  |  High sigma TIR median: {mod_tir:.1f}%")

    assert mod_tir < base_tir, (
        f"Expected higher carb_sigma to lower TIR: {mod_tir:.1f} >= {base_tir:.1f}")


def test_higher_sensitivity_sigma_lowers_tir(verbose=False):
    """Higher sensitivity variation → lower TIR."""
    base = make_base_profile(sensitivity_sigma=0.05)
    modified = make_base_profile(sensitivity_sigma=0.40)

    base_s, mod_s = run_comparison(base, modified)

    base_tir = base_s['time_in_range']['median']
    mod_tir = mod_s['time_in_range']['median']

    if verbose:
        print(f"    Base TIR median: {base_tir:.1f}%  |  High sigma TIR median: {mod_tir:.1f}%")

    assert mod_tir < base_tir, (
        f"Expected higher sensitivity_sigma to lower TIR: {mod_tir:.1f} >= {base_tir:.1f}")


def test_higher_carb_sigma_increases_tir_spread(verbose=False):
    """Higher carb counting error → greater spread (SD) of TIR across paths."""
    base = make_base_profile(carb_count_sigma=0.05)
    modified = make_base_profile(carb_count_sigma=0.40)

    base_results = run_monte_carlo(
        profile=base, algorithms=[ALGO],
        n_paths=N_PATHS, n_days=N_DAYS, seed=SEED, max_workers=MAX_WORKERS,
    )
    mod_results = run_monte_carlo(
        profile=modified, algorithms=[ALGO],
        n_paths=N_PATHS, n_days=N_DAYS, seed=SEED, max_workers=MAX_WORKERS,
    )

    base_sd = get_tir_sd(base_results[ALGO])
    mod_sd = get_tir_sd(mod_results[ALGO])

    if verbose:
        print(f"    Base TIR SD: {base_sd:.2f}%  |  High sigma TIR SD: {mod_sd:.2f}%")

    assert mod_sd > base_sd, (
        f"Expected higher carb_sigma to increase TIR spread: {mod_sd:.2f} <= {base_sd:.2f}")


def test_positive_carb_bias_raises_mean_bg(verbose=False):
    """Under-declaring carbs (positive bias) → higher mean BG."""
    base = make_base_profile(carb_count_bias=0.0)
    modified = make_base_profile(carb_count_bias=0.3)

    base_s, mod_s = run_comparison(base, modified)

    base_bg = base_s['mean_bg']['median']
    mod_bg = mod_s['mean_bg']['median']

    if verbose:
        print(f"    Base mean BG median: {base_bg:.1f}  |  Under-declare mean BG median: {mod_bg:.1f}")

    assert mod_bg > base_bg, (
        f"Expected positive carb_bias to raise mean BG: {mod_bg:.1f} <= {base_bg:.1f}")


def test_negative_carb_bias_lowers_mean_bg(verbose=False):
    """Over-declaring carbs (negative bias) → lower mean BG."""
    base = make_base_profile(carb_count_bias=0.0)
    modified = make_base_profile(carb_count_bias=-0.3)

    base_s, mod_s = run_comparison(base, modified)

    base_bg = base_s['mean_bg']['median']
    mod_bg = mod_s['mean_bg']['median']

    if verbose:
        print(f"    Base mean BG median: {base_bg:.1f}  |  Over-declare mean BG median: {mod_bg:.1f}")

    assert mod_bg < base_bg, (
        f"Expected negative carb_bias to lower mean BG: {mod_bg:.1f} >= {base_bg:.1f}")


def test_rescue_carbs_reduce_severe_hypos(verbose=False):
    """Rescue carbs enabled → less time below 54 mg/dL."""
    base = make_base_profile(rescue_carbs_enabled=True)
    modified = make_base_profile(rescue_carbs_enabled=False)

    base_s, mod_s = run_comparison(base, modified)

    base_hypo = base_s['time_below_54']['median']
    mod_hypo = mod_s['time_below_54']['median']

    if verbose:
        print(f"    Rescue ON time<54 median: {base_hypo:.2f}%  |  "
              f"Rescue OFF time<54 median: {mod_hypo:.2f}%")

    assert base_hypo <= mod_hypo, (
        f"Expected rescue carbs to reduce severe hypos: "
        f"rescue_on={base_hypo:.2f}% > rescue_off={mod_hypo:.2f}%")


def test_undeclared_meals_lower_tir(verbose=False):
    """Undeclared meals → lower TIR."""
    base = make_base_profile(undeclared_meal_prob=0.0, undeclared_meals_rest=[])
    modified = make_base_profile(
        undeclared_meal_prob=0.5,
        undeclared_meals_rest=[
            MealSpec(time_of_day_minutes=480, carbs_mean=30, carbs_sd=5,
                     absorption_hrs=3.0),
        ],
    )

    base_s, mod_s = run_comparison(base, modified)

    base_tir = base_s['time_in_range']['median']
    mod_tir = mod_s['time_in_range']['median']

    if verbose:
        print(f"    Base TIR median: {base_tir:.1f}%  |  Undeclared meals TIR median: {mod_tir:.1f}%")

    assert mod_tir < base_tir, (
        f"Expected undeclared meals to lower TIR: {mod_tir:.1f} >= {base_tir:.1f}")


# ─── Runner ──────────────────────────────────────────────────────────────────

ALL_TESTS = [
    ("test_higher_carb_sigma_lowers_tir", test_higher_carb_sigma_lowers_tir),
    ("test_higher_sensitivity_sigma_lowers_tir", test_higher_sensitivity_sigma_lowers_tir),
    ("test_higher_carb_sigma_increases_tir_spread", test_higher_carb_sigma_increases_tir_spread),
    ("test_positive_carb_bias_raises_mean_bg", test_positive_carb_bias_raises_mean_bg),
    ("test_negative_carb_bias_lowers_mean_bg", test_negative_carb_bias_lowers_mean_bg),
    ("test_rescue_carbs_reduce_severe_hypos", test_rescue_carbs_reduce_severe_hypos),
    ("test_undeclared_meals_lower_tir", test_undeclared_meals_lower_tir),
]


def main():
    verbose = '--verbose' in sys.argv or '-v' in sys.argv

    print(f"\nMonte Carlo Directional Tests")
    print(f"{'='*60}")
    print(f"Algorithm: {ALGO}")
    print(f"Paths: {N_PATHS}, Days: {N_DAYS}, Seed: {SEED}")
    print()

    passed = 0
    failed = 0
    t_total = time.time()

    for name, test_fn in ALL_TESTS:
        print(f"  Running {name}...", end=" ", flush=True)
        t0 = time.time()
        try:
            test_fn(verbose=verbose)
            elapsed = time.time() - t0
            print(f"PASS ({elapsed:.1f}s)")
            passed += 1
        except AssertionError as e:
            elapsed = time.time() - t0
            print(f"FAIL ({elapsed:.1f}s)")
            print(f"    {e}")
            failed += 1
        except Exception as e:
            elapsed = time.time() - t0
            print(f"ERROR ({elapsed:.1f}s)")
            print(f"    {e}")
            failed += 1

    total_elapsed = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed ({total_elapsed:.0f}s total)")

    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
