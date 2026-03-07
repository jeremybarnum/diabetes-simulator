#!/usr/bin/env python3
"""Quick targeted parameter sweep. Usage: python3 targeted_sweep.py <profile.json>"""
import sys, json, copy, numpy as np
from simulation import PatientProfile
from monte_carlo import run_monte_carlo

def run(profile_dict, algo="trio", n_paths=20, n_days=7, seed=42):
    with open("/tmp/_sweep.json", "w") as f:
        json.dump(profile_dict, f)
    p = PatientProfile.from_json("/tmp/_sweep.json")
    mc = run_monte_carlo(p, [algo], n_paths=n_paths, n_days=n_days, seed=seed)
    m = list(mc.values())[0].all_metrics
    return {
        "mean": np.mean([x.mean_bg for x in m]),
        "sd": np.mean([x.sd_bg for x in m]),
        "tir": np.mean([x.time_in_range for x in m]),
        "<70": np.mean([x.time_below_70 for x in m]),
        ">180": np.mean([x.time_above_180 for x in m]),
        "hypos": np.mean([x.hypo_events for x in m]),
        "rescues": np.mean([x.rescue_carb_events for x in m]),
    }

def main():
    profile_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/calibrated_profile.json"
    with open(profile_path) as f:
        base = json.load(f)

    variants = []

    # Baseline
    variants.append(("baseline", base))

    # ISF perturbations
    for isf in [90, 95]:
        v = copy.deepcopy(base)
        v["algorithm_settings"]["insulin_sensitivity_factor"] = isf
        variants.append((f"ISF={isf}", v))

    # Target perturbations
    for tgt in [105, 110]:
        v = copy.deepcopy(base)
        v["algorithm_settings"]["target"] = tgt
        variants.append((f"target={tgt}", v))

    # Bias perturbations
    for bias in [0.1, 0.2]:
        v = copy.deepcopy(base)
        v["carb_count_bias"] = bias
        variants.append((f"bias={bias}", v))

    # Combos
    for isf, tgt, bias in [(90, 105, 0.0), (95, 105, 0.1), (90, 105, 0.1)]:
        v = copy.deepcopy(base)
        v["algorithm_settings"]["insulin_sensitivity_factor"] = isf
        v["algorithm_settings"]["target"] = tgt
        v["carb_count_bias"] = bias
        variants.append((f"ISF={isf}/tgt={tgt}/b={bias}", v))

    print(f"{'Config':<30} {'Mean':>6} {'SD':>6} {'TIR':>6} {'<70':>6} {'>180':>6} {'Hypos':>6} {'Resc':>6}")
    print("-" * 78)

    for label, profile_dict in variants:
        r = run(profile_dict)
        print(f"{label:<30} {r['mean']:>5.1f}  {r['sd']:>5.1f}  {r['tir']:>5.1f}  {r['<70']:>5.2f}  {r['>180']:>5.1f}  {r['hypos']:>5.2f}  {r['rescues']:>5.1f}")

    print(f"\n{'NS Target':<30} {'129.0':>6} {'29.0':>6} {'94.5':>6} {'0.60':>6} {'4.90':>6}")

if __name__ == "__main__":
    main()
