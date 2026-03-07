#!/usr/bin/env python3
"""
Jacobian-based calibration: compute sensitivity of sim outputs to input parameters.

Runs baseline + perturbed simulations, builds a Jacobian matrix, and suggests
parameter adjustments to match NS reference targets.
"""
import sys, json, copy, numpy as np
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
    return {
        "mean_bg": np.mean([m.mean_bg for m in metrics]),
        "sd_bg": np.mean([m.sd_bg for m in metrics]),
        "tir": np.mean([m.time_in_range for m in metrics]),
        "time_below_70": np.mean([m.time_below_70 for m in metrics]),
        "time_above_180": np.mean([m.time_above_180 for m in metrics]),
        "hypo_events": np.mean([m.hypo_events for m in metrics]),
        "rescue_events": np.mean([m.rescue_carb_events for m in metrics]),
    }

def perturb(profile_dict, param_name, delta):
    """Return a copy of profile_dict with one parameter perturbed."""
    p = copy.deepcopy(profile_dict)
    if param_name == "meal_scale":
        for m in p["meals"]:
            m["carbs_mean"] *= (1.0 + delta)
            m["carbs_sd"] *= (1.0 + delta)
    elif param_name == "carb_count_bias":
        p["carb_count_bias"] = p.get("carb_count_bias", 0) + delta
    elif param_name == "sensitivity_sigma":
        p["sensitivity_sigma"] = max(0.01, p.get("sensitivity_sigma", 0.15) + delta)
    elif param_name == "absorption_sigma":
        p["absorption_sigma"] = max(0.01, p.get("absorption_sigma", 0.15) + delta)
    elif param_name == "undeclared_meal_prob":
        p["undeclared_meal_prob"] = max(0.0, min(1.0, p.get("undeclared_meal_prob", 0) + delta))
    elif param_name == "isf":
        p["algorithm_settings"]["insulin_sensitivity_factor"] += delta
    elif param_name == "basal_rate":
        p["algorithm_settings"]["basal_rate"] = max(0.1, p["algorithm_settings"]["basal_rate"] + delta)
    elif param_name == "carb_ratio":
        p["algorithm_settings"]["carb_ratio"] = max(3.0, p["algorithm_settings"]["carb_ratio"] + delta)
    elif param_name == "target":
        p["algorithm_settings"]["target"] += delta
    return p

def main():
    profile_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/calibrated_profile.json"
    algo = sys.argv[2] if len(sys.argv) > 2 else "trio"
    n_paths = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    n_days = int(sys.argv[4]) if len(sys.argv) > 4 else 7

    with open(profile_path) as f:
        base_profile = json.load(f)

    # NS reference targets
    targets = {
        "mean_bg": 129.0,
        "sd_bg": 29.0,
        "tir": 94.5,
        "time_below_70": 0.6,
        "time_above_180": 4.9,
    }
    output_names = list(targets.keys())

    # Parameters to perturb and their step sizes
    params = [
        ("meal_scale",          0.10),   # +10% meal sizes
        ("carb_count_bias",     0.10),   # +0.1 bias
        ("sensitivity_sigma",   0.05),   # +0.05 sigma
        ("absorption_sigma",    0.05),   # +0.05 sigma
        ("undeclared_meal_prob", 0.05),  # +5% undeclared probability
    ]

    print(f"Profile: {profile_path}")
    print(f"Algorithm: {algo}, {n_paths} paths x {n_days} days")
    print(f"\nTargets: {targets}")

    # Run baseline
    print(f"\n--- Running baseline ---")
    base_results = run_sim(base_profile, algo, n_paths, n_days)
    print(f"Baseline: {', '.join(f'{k}={v:.2f}' for k, v in base_results.items() if k in output_names)}")

    # Compute gaps
    gaps = {k: targets[k] - base_results[k] for k in output_names}
    print(f"Gaps:     {', '.join(f'{k}={v:+.2f}' for k, v in gaps.items())}")

    # Run perturbations and build Jacobian
    n_outputs = len(output_names)
    n_params = len(params)
    J = np.zeros((n_outputs, n_params))

    for j, (param_name, delta) in enumerate(params):
        print(f"\n--- Perturbing {param_name} by {delta:+.3f} ---")
        perturbed_profile = perturb(base_profile, param_name, delta)
        perturbed_results = run_sim(perturbed_profile, algo, n_paths, n_days)

        for i, out_name in enumerate(output_names):
            J[i, j] = (perturbed_results[out_name] - base_results[out_name]) / delta

        print(f"  Results: {', '.join(f'{k}={perturbed_results[k]:.2f}' for k in output_names)}")
        print(f"  Sensitivities: {', '.join(f'd{output_names[i]}/d{param_name}={J[i,j]:.2f}' for i in range(n_outputs))}")

    # Print Jacobian
    print(f"\n{'='*80}")
    print("JACOBIAN MATRIX (d_output / d_param)")
    print(f"{'='*80}")
    header = f"{'':>20}" + "".join(f"{p[0]:>18}" for p in params)
    print(header)
    for i, out_name in enumerate(output_names):
        row = f"{out_name:>20}" + "".join(f"{J[i,j]:>18.3f}" for j in range(n_params))
        print(row)

    # Least-squares solve: J @ delta_params ≈ gaps
    gap_vec = np.array([gaps[k] for k in output_names])

    # Use pseudo-inverse (handles over/under-determined)
    delta_params, residuals, rank, sv = np.linalg.lstsq(J, gap_vec, rcond=None)

    print(f"\n{'='*80}")
    print("SUGGESTED PARAMETER ADJUSTMENTS (least-squares)")
    print(f"{'='*80}")
    for j, (param_name, step) in enumerate(params):
        current_val = _get_current(base_profile, param_name)
        new_val = current_val + delta_params[j]
        print(f"  {param_name:>25}: {current_val:>8.3f} → {new_val:>8.3f} (Δ={delta_params[j]:>+8.4f})")

    # Predict what the adjusted outputs would be
    predicted_outputs = np.array([base_results[k] for k in output_names]) + J @ delta_params
    print(f"\nPredicted outputs after adjustment:")
    for i, out_name in enumerate(output_names):
        print(f"  {out_name:>20}: {predicted_outputs[i]:>8.2f}  (target: {targets[out_name]:>8.2f})")

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
    print(f"Adjusted: {', '.join(f'{k}={v:.2f}' for k in output_names for v in [adjusted_results[k]])}")

    print(f"\n{'='*80}")
    print("COMPARISON")
    print(f"{'='*80}")
    print(f"{'Metric':>20} {'Baseline':>10} {'Adjusted':>10} {'Target':>10} {'Gap (adj)':>10}")
    for k in output_names:
        gap_adj = targets[k] - adjusted_results[k]
        print(f"{k:>20} {base_results[k]:>10.2f} {adjusted_results[k]:>10.2f} {targets[k]:>10.2f} {gap_adj:>+10.2f}")


def _get_current(profile, param_name):
    if param_name == "meal_scale":
        return 1.0  # scale factor, 1.0 = no change
    elif param_name == "carb_count_bias":
        return profile.get("carb_count_bias", 0)
    elif param_name == "sensitivity_sigma":
        return profile.get("sensitivity_sigma", 0.15)
    elif param_name == "absorption_sigma":
        return profile.get("absorption_sigma", 0.15)
    elif param_name == "undeclared_meal_prob":
        return profile.get("undeclared_meal_prob", 0)
    elif param_name == "isf":
        return profile["algorithm_settings"]["insulin_sensitivity_factor"]
    elif param_name == "basal_rate":
        return profile["algorithm_settings"]["basal_rate"]
    elif param_name == "carb_ratio":
        return profile["algorithm_settings"]["carb_ratio"]
    elif param_name == "target":
        return profile["algorithm_settings"]["target"]
    return 0


if __name__ == "__main__":
    main()
