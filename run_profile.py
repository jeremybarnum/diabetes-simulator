#!/usr/bin/env python3
"""Run a profile from JSON file. Usage: python3 run_profile.py <profile.json> [algo] [paths] [days]"""
import sys, json, numpy as np
from simulation import PatientProfile
from monte_carlo import run_monte_carlo

def main():
    profile_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/profile_test.json"
    algo = sys.argv[2] if len(sys.argv) > 2 else "trio"
    n_paths = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    n_days = int(sys.argv[4]) if len(sys.argv) > 4 else 7

    profile = PatientProfile.from_json(profile_path)
    mc = run_monte_carlo(profile, [algo], n_paths=n_paths, n_days=n_days, seed=42)

    for label, mc_r in mc.items():
        metrics = mc_r.all_metrics
        print(f"\n=== {label}: {n_paths} paths x {n_days} days ===")
        print(f"Mean BG:          {np.mean([m.mean_bg for m in metrics]):.1f}")
        print(f"SD BG:            {np.mean([m.sd_bg for m in metrics]):.1f}")
        print(f"TIR 70-180:       {np.mean([m.time_in_range for m in metrics]):.1f}%")
        print(f"Time <70:         {np.mean([m.time_below_70 for m in metrics]):.2f}%")
        print(f"Time <54:         {np.mean([m.time_below_54 for m in metrics]):.3f}%")
        print(f"Time >180:        {np.mean([m.time_above_180 for m in metrics]):.1f}%")
        print(f"Time >250:        {np.mean([m.time_above_250 for m in metrics]):.2f}%")
        print(f"CV:               {np.mean([m.cv for m in metrics]):.1f}%")
        print(f"GMI:              {np.mean([m.gmi for m in metrics]):.2f}%")
        print(f"Hypo events/path: {np.mean([m.hypo_events for m in metrics]):.2f}")
        print(f"Rescue events:    {np.mean([m.rescue_carb_events for m in metrics]):.1f}/path")
        print(f"Rescue carbs:     {np.mean([m.rescue_carb_grams_total for m in metrics]):.0f}g/path")

if __name__ == '__main__':
    main()
