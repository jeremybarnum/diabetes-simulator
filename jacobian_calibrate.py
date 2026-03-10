#!/usr/bin/env python3
"""
Jacobian-based calibration: compute sensitivity of sim outputs to input parameters.

Runs baseline + perturbed simulations, builds a Jacobian matrix, and suggests
parameter adjustments to match NS reference targets.

Targets can be loaded from the profile's ns_reference_stats (--from-profile)
or specified manually. Supports weighted least-squares so event counts and
percentages contribute equally.

Usage:
    python3 jacobian_calibrate.py patient_profiles/real_patient.json
    python3 jacobian_calibrate.py patient_profiles/real_patient.json trio 40 7
    python3 jacobian_calibrate.py patient_profiles/real_patient.json --from-profile
"""
import sys, json, copy, argparse, numpy as np
from simulation import PatientProfile
from monte_carlo import run_monte_carlo


def run_sim(profile_dict, algo="trio", n_paths=20, n_days=7, seed=42):
    """Run simulation and return key metrics."""
    path = "/tmp/_jacobian_temp.json"
    with open(path, "w") as f:
        json.dump(profile_dict, f)
    profile = PatientProfile.from_json(path)
    mc = run_monte_carlo(profile, [algo], n_paths=n_paths, n_days=n_days, seed=seed)
    metrics = list(mc.values())[0].all_metrics
    n_days_actual = n_days
    mean_bg = np.mean([m.mean_bg for m in metrics])
    return {
        "mean_bg": mean_bg,
        "sd_bg": np.mean([m.sd_bg for m in metrics]),
        "tir": np.mean([m.time_in_range for m in metrics]),
        "time_below_70": np.mean([m.time_below_70 for m in metrics]),
        "time_below_54": np.mean([m.time_below_54 for m in metrics]),
        "time_above_180": np.mean([m.time_above_180 for m in metrics]),
        "time_above_250": np.mean([m.time_above_250 for m in metrics]),
        "gmi": 3.31 + 0.02392 * mean_bg,
        "hypo_events": np.mean([m.hypo_events for m in metrics]),
        "hypo_events_concerning": np.mean([m.hypo_events_concerning for m in metrics]),
        "rescue_events": np.mean([m.rescue_carb_events for m in metrics]),
        "rescue_grams": np.mean([m.rescue_carb_grams_total for m in metrics]),
        # Per-week rates (sim is n_days, scale to 7-day week)
        "hypo_events_per_week": np.mean([m.hypo_events for m in metrics]) / n_days_actual * 7,
        "hypo_concerning_per_week": np.mean([m.hypo_events_concerning for m in metrics]) / n_days_actual * 7,
        "rescue_events_per_week": np.mean([m.rescue_carb_events for m in metrics]) / n_days_actual * 7,
        "rescue_carbs_per_week": np.mean([m.rescue_carb_grams_total for m in metrics]) / n_days_actual * 7,
    }


def perturb(profile_dict, param_name, delta):
    """Return a copy of profile_dict with one parameter perturbed."""
    p = copy.deepcopy(profile_dict)
    if param_name == "meal_scale":
        for m in p.get("meals_rest", p.get("meals", [])):
            m["carbs_mean"] *= (1.0 + delta)
            m["carbs_sd"] *= (1.0 + delta)
    elif param_name == "carb_count_bias":
        p["carb_count_bias"] = p.get("carb_count_bias", 0) + delta
    elif param_name == "sensitivity_sigma":
        p["sensitivity_sigma"] = max(0.01, p.get("sensitivity_sigma", 0.15) + delta)
    elif param_name == "absorption_sigma":
        p["absorption_sigma"] = max(0.01, p.get("absorption_sigma", 0.15) + delta)
    elif param_name == "isf":
        p["algorithm_settings"]["insulin_sensitivity_factor"] += delta
    elif param_name == "basal_rate":
        p["algorithm_settings"]["basal_rate"] = max(0.1, p["algorithm_settings"]["basal_rate"] + delta)
    elif param_name == "carb_ratio":
        p["algorithm_settings"]["carb_ratio"] = max(3.0, p["algorithm_settings"]["carb_ratio"] + delta)
    elif param_name == "target":
        p["algorithm_settings"]["target"] += delta
    elif param_name == "suspend_threshold":
        p["algorithm_settings"]["suspend_threshold"] = max(50, p["algorithm_settings"].get("suspend_threshold", 80) + delta)
    return p


def load_targets_from_profile(profile_dict):
    """Load NS reference targets from profile's ns_reference_stats.

    Returns dict of metric_name → target_value.
    """
    ref = profile_dict.get("ns_reference_stats", {})
    if not ref:
        return None

    targets = {}
    # Standard BG metrics
    for key in ["mean_bg", "sd_bg", "tir", "time_below_70", "time_below_54",
                "time_above_180", "time_above_250", "gmi"]:
        if key in ref:
            targets[key] = ref[key]

    # Event rates (already per-week in NS reference)
    for key in ["hypo_events_per_week", "hypo_concerning_per_week",
                "rescue_events_per_week", "rescue_carbs_per_week"]:
        if key in ref:
            targets[key] = ref[key]

    return targets


def compute_weights(targets):
    """Compute weights for least-squares so all metrics contribute equally.

    Uses fractional error normalization:
    - Percentages (tir, time_below, etc.): weight = 1/max(|target|, 1)
    - Event counts: weight = 1/max(|target|, 0.1)
    - BG values: weight = 1/max(|target|, 10)

    Returns weight vector aligned with target keys.
    """
    weights = {}
    for key, val in targets.items():
        abs_val = abs(val)
        if key.startswith("hypo") or key.startswith("rescue"):
            # Event counts — normalize by target (or 0.1 if near zero)
            weights[key] = 1.0 / max(abs_val, 0.1)
        elif key in ("mean_bg", "sd_bg"):
            weights[key] = 1.0 / max(abs_val, 10.0)
        elif key == "gmi":
            weights[key] = 1.0 / max(abs_val, 0.1)
        else:
            # Percentages
            weights[key] = 1.0 / max(abs_val, 1.0)
    return weights


def _get_current(profile, param_name):
    if param_name == "meal_scale":
        return 1.0  # scale factor, 1.0 = no change
    elif param_name == "carb_count_bias":
        return profile.get("carb_count_bias", 0)
    elif param_name == "sensitivity_sigma":
        return profile.get("sensitivity_sigma", 0.15)
    elif param_name == "absorption_sigma":
        return profile.get("absorption_sigma", 0.15)
    elif param_name == "isf":
        return profile["algorithm_settings"]["insulin_sensitivity_factor"]
    elif param_name == "basal_rate":
        return profile["algorithm_settings"]["basal_rate"]
    elif param_name == "carb_ratio":
        return profile["algorithm_settings"]["carb_ratio"]
    elif param_name == "target":
        return profile["algorithm_settings"]["target"]
    elif param_name == "suspend_threshold":
        return profile["algorithm_settings"].get("suspend_threshold", 80)
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Jacobian-based calibration to match NS reference targets")
    parser.add_argument("profile", type=str,
                        help="Patient profile JSON path")
    parser.add_argument("algo", type=str, nargs="?", default="trio",
                        help="Algorithm (default: trio)")
    parser.add_argument("n_paths", type=int, nargs="?", default=40,
                        help="Number of simulation paths (default: 40)")
    parser.add_argument("n_days", type=int, nargs="?", default=7,
                        help="Days per path (default: 7)")
    parser.add_argument("--from-profile", action="store_true",
                        help="Load targets from profile's ns_reference_stats")
    parser.add_argument("--params", type=str, nargs="+",
                        default=["meal_scale", "carb_count_bias",
                                 "sensitivity_sigma", "absorption_sigma"],
                        help="Parameters to calibrate")
    parser.add_argument("--targets", type=str, default=None,
                        help="JSON string of manual targets (overrides --from-profile)")
    args = parser.parse_args()

    profile_path = args.profile
    algo = args.algo
    n_paths = args.n_paths
    n_days = args.n_days

    with open(profile_path) as f:
        base_profile = json.load(f)

    # Load targets
    if args.targets:
        targets = json.loads(args.targets)
    elif args.from_profile:
        targets = load_targets_from_profile(base_profile)
        if not targets:
            print("ERROR: No ns_reference_stats in profile. Cannot use --from-profile.")
            sys.exit(1)
    else:
        # Default hardcoded targets (backward compat)
        targets = {
            "mean_bg": 129.0,
            "sd_bg": 29.0,
            "tir": 94.5,
            "time_below_70": 0.6,
            "time_above_180": 4.9,
        }

    output_names = list(targets.keys())

    # Parameters to perturb and their step sizes
    param_steps = {
        "meal_scale": 0.10,
        "carb_count_bias": 0.10,
        "sensitivity_sigma": 0.05,
        "absorption_sigma": 0.05,
        "isf": 5.0,
        "basal_rate": 0.05,
        "carb_ratio": 1.0,
        "target": 5.0,
        "suspend_threshold": 5.0,
    }
    params = [(p, param_steps.get(p, 0.05)) for p in args.params]

    print(f"Profile: {profile_path}")
    print(f"Algorithm: {algo}, {n_paths} paths x {n_days} days")
    print(f"Parameters: {[p[0] for p in params]}")
    print(f"\nTargets:")
    for k, v in targets.items():
        print(f"  {k:>30}: {v}")

    # Compute weights
    weights = compute_weights(targets)

    # Run baseline
    print(f"\n--- Running baseline ---")
    base_results = run_sim(base_profile, algo, n_paths, n_days)
    print(f"Baseline:")
    for k in output_names:
        v = base_results.get(k, float('nan'))
        print(f"  {k:>30}: {v:.3f}  (target: {targets[k]:.3f})")

    # Compute gaps
    gaps = {k: targets[k] - base_results.get(k, 0) for k in output_names}
    print(f"\nGaps:")
    for k, v in gaps.items():
        print(f"  {k:>30}: {v:+.3f}")

    # Run perturbations and build Jacobian
    n_outputs = len(output_names)
    n_params = len(params)
    J = np.zeros((n_outputs, n_params))

    for j, (param_name, delta) in enumerate(params):
        print(f"\n--- Perturbing {param_name} by {delta:+.3f} ---")
        perturbed_profile = perturb(base_profile, param_name, delta)
        perturbed_results = run_sim(perturbed_profile, algo, n_paths, n_days)

        for i, out_name in enumerate(output_names):
            J[i, j] = (perturbed_results.get(out_name, 0) - base_results.get(out_name, 0)) / delta

        print(f"  Sensitivities:")
        for i, out_name in enumerate(output_names):
            print(f"    d{out_name}/d{param_name} = {J[i,j]:.4f}")

    # Print Jacobian
    print(f"\n{'='*80}")
    print("JACOBIAN MATRIX (d_output / d_param)")
    print(f"{'='*80}")
    header = f"{'':>30}" + "".join(f"{p[0]:>18}" for p in params)
    print(header)
    for i, out_name in enumerate(output_names):
        row = f"{out_name:>30}" + "".join(f"{J[i,j]:>18.4f}" for j in range(n_params))
        print(row)

    # Weighted least-squares solve: W @ J @ delta_params ≈ W @ gaps
    gap_vec = np.array([gaps[k] for k in output_names])
    W = np.diag([weights[k] for k in output_names])

    WJ = W @ J
    Wgaps = W @ gap_vec

    delta_params, residuals, rank, sv = np.linalg.lstsq(WJ, Wgaps, rcond=None)

    print(f"\n{'='*80}")
    print("SUGGESTED PARAMETER ADJUSTMENTS (weighted least-squares)")
    print(f"{'='*80}")
    for j, (param_name, step) in enumerate(params):
        current_val = _get_current(base_profile, param_name)
        new_val = current_val + delta_params[j]
        print(f"  {param_name:>25}: {current_val:>8.3f} → {new_val:>8.3f} (Δ={delta_params[j]:>+8.4f})")

    # Predict what the adjusted outputs would be
    predicted_outputs = np.array([base_results.get(k, 0) for k in output_names]) + J @ delta_params
    print(f"\nPredicted outputs after adjustment:")
    for i, out_name in enumerate(output_names):
        print(f"  {out_name:>30}: {predicted_outputs[i]:>8.3f}  (target: {targets[out_name]:>8.3f})")

    # Save adjusted profile
    adjusted = copy.deepcopy(base_profile)
    for j, (param_name, _) in enumerate(params):
        adjusted = perturb(adjusted, param_name, delta_params[j])

    adjusted_path = profile_path.replace(".json", "_adjusted.json")
    with open(adjusted_path, "w") as f:
        json.dump(adjusted, f, indent=2)
    print(f"\nAdjusted profile saved to {adjusted_path}")

    # Verify by running the adjusted profile
    print(f"\n--- Verifying adjusted profile ---")
    adjusted_results = run_sim(adjusted, algo, n_paths, n_days)

    print(f"\n{'='*80}")
    print("COMPARISON")
    print(f"{'='*80}")
    print(f"{'Metric':>30} {'Baseline':>10} {'Adjusted':>10} {'Target':>10} {'Gap (adj)':>10}")
    for k in output_names:
        gap_adj = targets[k] - adjusted_results.get(k, 0)
        print(f"{k:>30} {base_results.get(k, 0):>10.3f} {adjusted_results.get(k, 0):>10.3f} {targets[k]:>10.3f} {gap_adj:>+10.3f}")


if __name__ == "__main__":
    main()
