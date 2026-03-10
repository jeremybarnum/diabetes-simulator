#!/usr/bin/env python3
"""
Infer meal schedule from a Nightscout median daily BG trace.

Detects postprandial peaks in the median trace, then uses the canonical
curve lookup table to solve for (carbs, absorption_hrs) per meal.

Usage:
    python3 infer_meals_from_trace.py --profile patient_profiles/real_patient.json
    python3 infer_meals_from_trace.py --profile patient_profiles/real_patient.json --day-type rest
    python3 infer_meals_from_trace.py --profile patient_profiles/real_patient.json --day-type exercise
    python3 infer_meals_from_trace.py --profile patient_profiles/real_patient.json --update
"""

import json
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from build_canonical_curves import load_canonical_table, lookup_meal


def load_median_trace(profile_path: str,
                      day_type: str = "rest") -> List[Tuple[float, float]]:
    """Load median daily BG trace from a profile's ns_reference_stats.

    Args:
        profile_path: path to patient profile JSON
        day_type: "rest", "exercise", or "all"

    Returns:
        List of (hour_from_7am, median_bg) tuples, sorted by time.
    """
    with open(profile_path) as f:
        profile = json.load(f)

    ref = profile.get("ns_reference_stats", {})
    if not ref:
        raise ValueError(f"No ns_reference_stats in {profile_path}")

    key_map = {
        "rest": "median_trace_rest",
        "exercise": "median_trace_exercise",
        "all": "median_trace",
    }
    key = key_map.get(day_type, "median_trace_rest")
    trace = ref.get(key, ref.get("median_trace", []))

    if not trace:
        raise ValueError(f"No {key} in ns_reference_stats")

    # Ensure sorted by time
    trace = sorted(trace, key=lambda p: p[0])
    return trace


def smooth_trace(trace: List[Tuple[float, float]],
                 window_min: int = 15) -> List[Tuple[float, float]]:
    """Smooth a BG trace with a moving average.

    Args:
        trace: [(hour_from_7am, bg), ...]
        window_min: window size in minutes

    Returns:
        Smoothed trace (same format).
    """
    if len(trace) < 3:
        return trace

    times = np.array([t for t, _ in trace])
    values = np.array([v for _, v in trace])

    # Convert window to number of 5-min buckets
    bucket_size = 5  # minutes
    window = max(1, window_min // bucket_size)

    # Pad edges for convolution
    kernel = np.ones(window) / window
    smoothed = np.convolve(values, kernel, mode='same')

    return list(zip(times.tolist(), smoothed.tolist()))


def detect_peaks(trace: List[Tuple[float, float]],
                 min_rise: float = 5.0,
                 min_gap_hours: float = 1.5) -> List[Dict]:
    """Detect postprandial peaks in a median BG trace.

    Algorithm:
    1. Find all local maxima
    2. For each maximum, find preceding local minimum (meal start)
    3. Filter by minimum rise threshold
    4. Merge peaks that are too close together

    Args:
        trace: [(hour_from_7am, bg), ...] sorted by time
        min_rise: minimum BG rise to count as a meal (mg/dL)
        min_gap_hours: minimum gap between detected meals

    Returns:
        List of dicts: {peak_time, peak_bg, start_time, start_bg,
                        rise, time_to_peak_min}
    """
    if len(trace) < 5:
        return []

    times = np.array([t for t, _ in trace])
    values = np.array([v for _, v in trace])

    # Find local maxima (higher than both neighbors)
    maxima = []
    for i in range(1, len(values) - 1):
        if values[i] > values[i-1] and values[i] >= values[i+1]:
            maxima.append(i)

    # For each maximum, find the preceding local minimum
    peaks = []
    for peak_idx in maxima:
        # Walk backward to find the preceding minimum
        min_idx = peak_idx
        for j in range(peak_idx - 1, -1, -1):
            if values[j] < values[min_idx]:
                min_idx = j
            elif values[j] > values[min_idx] + 3.0:
                # BG started rising before this point, stop
                break

        rise = values[peak_idx] - values[min_idx]
        if rise >= min_rise:
            time_to_peak_min = (times[peak_idx] - times[min_idx]) * 60
            peaks.append({
                "peak_time": float(times[peak_idx]),
                "peak_bg": float(values[peak_idx]),
                "start_time": float(times[min_idx]),
                "start_bg": float(values[min_idx]),
                "rise": float(rise),
                "time_to_peak_min": float(time_to_peak_min),
            })

    # Merge peaks that are too close (keep the larger one)
    if not peaks:
        return []

    merged = [peaks[0]]
    for p in peaks[1:]:
        prev = merged[-1]
        gap = p["start_time"] - prev["peak_time"]
        if gap < min_gap_hours:
            # Merge: keep whichever has the larger rise
            if p["rise"] > prev["rise"]:
                merged[-1] = p
        else:
            merged.append(p)

    return merged


def infer_meals(trace: List[Tuple[float, float]],
                canonical_table: dict,
                settings: dict,
                min_rise: float = 5.0,
                min_gap_hours: float = 1.5,
                default_carb_count_sigma: float = 0.15) -> List[Dict]:
    """Infer meal schedule from a median daily BG trace.

    Args:
        trace: [(hour_from_7am, bg), ...]
        canonical_table: from build_canonical_curves
        settings: algorithm settings (for ISF/CR scaling)
        min_rise: minimum BG rise to detect as a meal
        min_gap_hours: minimum gap between meals
        default_carb_count_sigma: default carb counting error

    Returns:
        List of meal spec dicts ready for PatientProfile JSON.
    """
    # Smooth first
    smoothed = smooth_trace(trace, window_min=15)

    # Detect peaks
    peaks = detect_peaks(smoothed, min_rise=min_rise,
                         min_gap_hours=min_gap_hours)

    if not peaks:
        print("  No peaks detected in trace!")
        return []

    # Compute ISF/CR ratio relative to canonical table
    table_isf = canonical_table["metadata"]["isf"]
    table_cr = canonical_table["metadata"]["cr"]
    actual_isf = settings["insulin_sensitivity_factor"]
    actual_cr = settings["carb_ratio"]
    isf_ratio = actual_isf / table_isf
    cr_ratio = actual_cr / table_cr

    meals = []
    for i, peak in enumerate(peaks):
        # Convert start_time (hours from 7am) to minutes from 7am
        meal_time_min = int(round(peak["start_time"] * 60))

        # Lookup in canonical table
        result = lookup_meal(
            peak["rise"], peak["time_to_peak_min"],
            canonical_table, isf_ratio, cr_ratio,
        )

        if result is None:
            print(f"  Peak {i+1}: no match for rise={peak['rise']:.0f}, "
                  f"ttp={peak['time_to_peak_min']:.0f}min")
            continue

        carbs, absorption_hrs, canon_rise, canon_ttp, dist = result

        # Estimate carbs_sd as ~30% of mean for substantial meals, less for small
        carbs_sd = max(1.0, round(carbs * 0.3, 1))
        if carbs <= 8:
            carbs_sd = max(0.5, round(carbs * 0.15, 1))

        # Small meals (<10g) get lower sigma (coffee, snacks — well-known)
        sigma = default_carb_count_sigma
        if carbs <= 8:
            sigma = 0.0
        elif carbs <= 15:
            sigma = max(0.1, default_carb_count_sigma * 0.7)

        # Small meals (<10g) with short absorption are likely coffee
        if carbs <= 8 and absorption_hrs <= 1.0:
            absorption_hrs = 1.0

        meal = {
            "time_of_day_minutes": meal_time_min,
            "carbs_mean": round(carbs, 1),
            "carbs_sd": carbs_sd,
            "absorption_hrs": absorption_hrs,
            "declared": True,
            "carb_count_sigma": sigma,
        }
        meals.append(meal)

        # Print details
        clock_h = 7 + meal_time_min // 60
        clock_m = meal_time_min % 60
        print(f"  Meal {i+1}: {clock_h:02d}:{clock_m:02d} | "
              f"{carbs}g carbs, {absorption_hrs}h abs | "
              f"rise={peak['rise']:.0f}→canon {canon_rise:.0f} mg/dL, "
              f"ttp={peak['time_to_peak_min']:.0f}→canon {canon_ttp:.0f}min "
              f"(dist={dist:.3f})")

    return meals


def update_profile_meals(profile_path: str, meals: List[Dict],
                         day_type: str = "rest"):
    """Update a profile's meal schedule with inferred meals.

    Args:
        profile_path: path to patient profile JSON
        meals: list of meal spec dicts
        day_type: "rest" or "exercise"
    """
    with open(profile_path) as f:
        profile = json.load(f)

    key = "meals_rest" if day_type == "rest" else "meals_exercise"
    profile[key] = meals

    with open(profile_path, "w") as f:
        json.dump(profile, f, indent=4)

    print(f"\nUpdated {key} in {profile_path} ({len(meals)} meals)")


def main():
    parser = argparse.ArgumentParser(
        description="Infer meal schedule from NS median trace")
    parser.add_argument("--profile", type=str, required=True,
                        help="Patient profile JSON path")
    parser.add_argument("--day-type", type=str, default="rest",
                        choices=["rest", "exercise", "all"],
                        help="Which trace to use (default: rest)")
    parser.add_argument("--canonical", type=str, default=None,
                        help="Canonical curves JSON (default: canonical_curves.json)")
    parser.add_argument("--min-rise", type=float, default=5.0,
                        help="Minimum BG rise to detect as meal (mg/dL)")
    parser.add_argument("--min-gap", type=float, default=1.5,
                        help="Minimum gap between meals (hours)")
    parser.add_argument("--update", action="store_true",
                        help="Update the profile with inferred meals")
    parser.add_argument("--both", action="store_true",
                        help="Infer both rest and exercise meals")
    args = parser.parse_args()

    # Load canonical table
    canonical = load_canonical_table(args.canonical)
    print(f"Canonical table: ISF={canonical['metadata']['isf']}, "
          f"CR={canonical['metadata']['cr']}, "
          f"insulin={canonical['metadata']['insulin_type']}")

    # Load settings from profile
    with open(args.profile) as f:
        profile_data = json.load(f)
    settings = profile_data.get("algorithm_settings", {})
    if not settings:
        settings_path = Path(__file__).parent / "settings.json"
        with open(settings_path) as f:
            settings = json.load(f)

    day_types = ["rest", "exercise"] if args.both else [args.day_type]

    for dt in day_types:
        print(f"\n{'='*60}")
        print(f"Inferring {dt}-day meals from median trace")
        print(f"{'='*60}")

        try:
            trace = load_median_trace(args.profile, dt)
        except ValueError as e:
            print(f"  Skipping: {e}")
            continue

        print(f"  Trace: {len(trace)} points, "
              f"range {trace[0][0]:.1f}h-{trace[-1][0]:.1f}h")

        meals = infer_meals(
            trace, canonical, settings,
            min_rise=args.min_rise,
            min_gap_hours=args.min_gap,
        )

        if not meals:
            print("  No meals inferred.")
            continue

        print(f"\n  Inferred {len(meals)} meals:")
        total_carbs = sum(m["carbs_mean"] for m in meals)
        print(f"  Total daily carbs: {total_carbs:.0f}g")

        if args.update:
            update_profile_meals(args.profile, meals, dt)

    if not args.update:
        print(f"\nDry run — use --update to write meals to profile")


if __name__ == "__main__":
    main()
