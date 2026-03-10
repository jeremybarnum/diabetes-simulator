#!/usr/bin/env python3
"""
Build a canonical postprandial curve lookup table.

Runs single-meal sims across a grid of (carbs, absorption_hrs) to build a
precise lookup of postprandial BG responses with Trio + current settings.

The table records (peak_rise, time_to_peak) for each combo, enabling
inverse lookup: given an observed BG rise and time-to-peak, find the
(carbs, absorption) that produced it.

Usage:
    python3 build_canonical_curves.py
    python3 build_canonical_curves.py --profile patient_profiles/real_patient.json
    python3 build_canonical_curves.py --output my_curves.json
"""

import json
import time
import argparse
import os
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from simulation import PatientProfile, MealSpec, SimulationRun


CARB_GRID = [2, 3, 5, 8, 10, 12, 15, 18, 20, 25, 30, 35, 40]
ABSORPTION_GRID = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]


def run_single_meal_sim(carbs, absorption_hrs, settings, starting_bg=100.0,
                        algo="trio"):
    """Run a single-meal sim and return the BG trace.

    Meal at t=60min (1h after 7am start), flat BG start.
    Returns list of (minutes_from_7am, bg) tuples.
    """
    profile = PatientProfile(
        meals_rest=[MealSpec(
            time_of_day_minutes=60,
            carbs_mean=carbs,
            carbs_sd=0.0,  # deterministic
            absorption_hrs=absorption_hrs,
            declared=True,
            carb_count_sigma=0.0,  # perfect counting
        )],
        carb_count_bias=0.0,
        absorption_sigma=0.0,  # deterministic absorption
        sensitivity_sigma=0.0,  # no sensitivity variation
        starting_bg=starting_bg,
        rescue_carbs_enabled=False,
        algorithm_settings=settings,
    )

    rng = np.random.RandomState(42)
    sim = SimulationRun(profile, algo, n_days=1, rng=rng)
    result = sim.run()

    return result.days[0].bg_trace  # [(t_rel, bg), ...]


def extract_peak(trace, meal_time_min=60):
    """Extract peak rise and time-to-peak from a BG trace.

    Args:
        trace: [(t_rel_minutes, bg), ...]
        meal_time_min: when the meal was eaten (minutes from day start)

    Returns:
        (peak_rise_mg_dl, time_to_peak_min, pre_meal_bg)
    """
    # Find pre-meal BG (value at or just before meal time)
    pre_meal_bg = None
    for t, bg in trace:
        if t <= meal_time_min:
            pre_meal_bg = bg
        else:
            break

    if pre_meal_bg is None:
        pre_meal_bg = trace[0][1] if trace else 100.0

    # Find peak BG after meal
    peak_bg = pre_meal_bg
    peak_time = meal_time_min
    for t, bg in trace:
        if t > meal_time_min and bg > peak_bg:
            peak_bg = bg
            peak_time = t

    peak_rise = peak_bg - pre_meal_bg
    time_to_peak = peak_time - meal_time_min

    return peak_rise, time_to_peak, pre_meal_bg


def _run_one_combo(args):
    """Top-level worker for parallel execution (must be picklable)."""
    carbs, absorption_hrs, settings, starting_bg, algo = args
    trace = run_single_meal_sim(carbs, absorption_hrs, settings, starting_bg, algo)
    peak_rise, time_to_peak, pre_meal_bg = extract_peak(trace)
    return {
        "carbs": carbs,
        "absorption_hrs": absorption_hrs,
        "peak_rise": round(peak_rise, 2),
        "time_to_peak": round(time_to_peak, 1),
        "pre_meal_bg": round(pre_meal_bg, 1),
    }


def build_canonical_table(settings, algo="trio", starting_bg=100.0):
    """Build the full canonical curve lookup table (parallelized).

    Returns:
        dict with metadata and "curves" list of
        {carbs, absorption_hrs, peak_rise, time_to_peak, pre_meal_bg}
    """
    total = len(CARB_GRID) * len(ABSORPTION_GRID)
    t_start = time.time()

    work_items = [
        (carbs, absorption_hrs, settings, starting_bg, algo)
        for carbs in CARB_GRID
        for absorption_hrs in ABSORPTION_GRID
    ]

    max_workers = os.cpu_count() or 4
    curves = []
    done = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_run_one_combo, item): item
                   for item in work_items}
        for future in as_completed(futures):
            curves.append(future.result())
            done += 1
            if done % 10 == 0:
                elapsed = time.time() - t_start
                rate = elapsed / done
                eta = rate * (total - done)
                print(f"  {done}/{total} ({elapsed:.0f}s elapsed, "
                      f"~{eta:.0f}s remaining)")

    # Sort by (carbs, absorption) for stable output
    curves.sort(key=lambda c: (c["carbs"], c["absorption_hrs"]))

    elapsed = time.time() - t_start
    print(f"\nCompleted {total} sims in {elapsed:.0f}s "
          f"({elapsed/total:.1f}s/sim, {max_workers} workers)")

    return {
        "metadata": {
            "algorithm": algo,
            "starting_bg": starting_bg,
            "isf": settings["insulin_sensitivity_factor"],
            "cr": settings["carb_ratio"],
            "insulin_type": settings.get("insulin_type", "fiasp"),
            "carb_grid": CARB_GRID,
            "absorption_grid": ABSORPTION_GRID,
        },
        "curves": curves,
    }


def load_canonical_table(path=None):
    """Load a canonical curves table from JSON.

    Returns the full dict with metadata and curves.
    """
    if path is None:
        path = Path(__file__).parent / "canonical_curves.json"
    with open(path) as f:
        return json.load(f)


def lookup_meal(target_rise, target_time_to_peak, table,
                isf_ratio=1.0, cr_ratio=1.0):
    """Find the best (carbs, absorption_hrs) match for an observed rise.

    The canonical table was built at a reference ISF and CR. To use with
    different settings, scale the target rise:
      canonical_rise = observed_rise * (ref_ISF / actual_ISF)

    Args:
        target_rise: observed BG rise in mg/dL
        target_time_to_peak: observed time to peak in minutes
        table: canonical curves dict (from load_canonical_table)
        isf_ratio: actual_ISF / table_ISF (>1 means less sensitive)
        cr_ratio: actual_CR / table_CR (>1 means more carbs per unit)

    Returns:
        (carbs, absorption_hrs, peak_rise, time_to_peak, distance)
    """
    # Scale target rise to canonical space
    # Higher ISF → smaller rise for same carbs → scale up to find equivalent
    # Higher CR → less insulin per carb → larger rise → scale down
    canonical_rise = target_rise / isf_ratio * cr_ratio

    best = None
    best_dist = float("inf")

    for c in table["curves"]:
        # Weighted distance: rise matters more than timing
        rise_diff = (c["peak_rise"] - canonical_rise) / max(canonical_rise, 1)
        time_diff = (c["time_to_peak"] - target_time_to_peak) / max(
            target_time_to_peak, 30)
        dist = rise_diff**2 + 0.3 * time_diff**2

        if dist < best_dist:
            best_dist = dist
            best = c

    if best is None:
        return None

    return (best["carbs"], best["absorption_hrs"],
            best["peak_rise"], best["time_to_peak"], best_dist)


def main():
    parser = argparse.ArgumentParser(
        description="Build canonical postprandial curve lookup table")
    parser.add_argument("--profile", type=str, default=None,
                        help="Patient profile to get settings from")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: canonical_curves.json)")
    parser.add_argument("--algo", type=str, default="trio",
                        help="Algorithm (default: trio)")
    parser.add_argument("--starting-bg", type=float, default=100.0,
                        help="Starting BG (default: 100)")
    args = parser.parse_args()

    # Load settings
    if args.profile:
        profile = PatientProfile.from_json(args.profile)
        settings = profile.get_settings()
    else:
        settings_path = Path(__file__).parent / "settings.json"
        with open(settings_path) as f:
            settings = json.load(f)

    print(f"Building canonical curves: {len(CARB_GRID)} carbs × "
          f"{len(ABSORPTION_GRID)} absorptions = "
          f"{len(CARB_GRID) * len(ABSORPTION_GRID)} sims")
    print(f"Settings: ISF={settings['insulin_sensitivity_factor']}, "
          f"CR={settings['carb_ratio']}, "
          f"insulin={settings.get('insulin_type', 'fiasp')}")
    print(f"Algorithm: {args.algo}, starting BG: {args.starting_bg}")
    print()

    table = build_canonical_table(settings, args.algo, args.starting_bg)

    # Save
    output_path = args.output or str(
        Path(__file__).parent / "canonical_curves.json")
    with open(output_path, "w") as f:
        json.dump(table, f, indent=2)
    print(f"\nSaved to {output_path}")

    # Print summary
    print(f"\n{'='*70}")
    print("Summary: Peak Rise (mg/dL) by Carbs × Absorption")
    print(f"{'='*70}")
    header = f"{'Carbs':>8}"
    for a in ABSORPTION_GRID:
        header += f" {a:>6.1f}h"
    print(header)
    print("-" * len(header))

    for carbs in CARB_GRID:
        row = f"{carbs:>6}g "
        for absorption in ABSORPTION_GRID:
            match = [c for c in table["curves"]
                     if c["carbs"] == carbs and c["absorption_hrs"] == absorption]
            if match:
                row += f" {match[0]['peak_rise']:>6.1f}"
            else:
                row += "      ?"
        print(row)


if __name__ == "__main__":
    main()
