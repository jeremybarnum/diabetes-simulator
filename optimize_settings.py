#!/usr/bin/env python3
"""
Jacobian-based optimizer: find algorithm settings that minimize concerning hypos
at minimum GMI cost.

Uses the same Jacobian approach as jacobian_calibrate.py but with a different
objective: instead of matching NS reference, minimize a safety metric subject
to glycemic control constraints.

Usage:
    python3 optimize_settings.py patient_profiles/real_patient.json
    python3 optimize_settings.py patient_profiles/real_patient.json --params target suspend_threshold
    python3 optimize_settings.py patient_profiles/real_patient.json --verify --paths 100
"""

import json
import copy
import argparse
import numpy as np
from pathlib import Path

from jacobian_calibrate import run_sim, perturb, _get_current


def compute_jacobian(base_profile, base_results, params, algo, n_paths, n_days,
                     output_names):
    """Compute Jacobian matrix of outputs w.r.t. parameters."""
    n_outputs = len(output_names)
    n_params = len(params)
    J = np.zeros((n_outputs, n_params))

    for j, (param_name, delta) in enumerate(params):
        print(f"  Perturbing {param_name} by {delta:+.3f}...")
        perturbed_profile = perturb(base_profile, param_name, delta)
        perturbed_results = run_sim(perturbed_profile, algo, n_paths, n_days)

        for i, out_name in enumerate(output_names):
            J[i, j] = (perturbed_results.get(out_name, 0) -
                        base_results.get(out_name, 0)) / delta

    return J


def optimize_step(base_profile, base_results, params, algo, n_paths, n_days,
                  gmi_budget=0.1, tir_floor=1.0):
    """One iteration of Jacobian-based optimization.

    Objective: minimize concerning_hypos_per_week
    Constraints:
      - GMI within ±gmi_budget of baseline
      - TIR >= baseline - tir_floor

    Returns:
        (suggested_deltas, predicted_results, J)
    """
    # Metrics we need sensitivities for
    output_names = ["hypo_concerning_per_week", "gmi", "tir",
                    "hypo_events_per_week", "rescue_events_per_week",
                    "mean_bg", "time_below_70"]

    # Only use outputs that exist in base_results
    output_names = [k for k in output_names if k in base_results]

    print(f"\nComputing Jacobian ({len(params)} params × {len(output_names)} outputs)...")
    J = compute_jacobian(base_profile, base_results, params, algo, n_paths,
                         n_days, output_names)

    # Print Jacobian
    print(f"\n{'='*70}")
    print("Jacobian:")
    header = f"{'':>28}" + "".join(f"{p[0]:>18}" for p in params)
    print(header)
    for i, out_name in enumerate(output_names):
        row = f"{out_name:>28}" + "".join(f"{J[i,j]:>18.4f}" for j in range(len(params)))
        print(row)

    # Extract rows for optimization
    hypo_idx = output_names.index("hypo_concerning_per_week")
    hypo_grad = J[hypo_idx, :]  # gradient of hypos w.r.t. params

    # Simple gradient descent on hypos, clamped by GMI constraint
    # Step = -alpha * grad(hypos), where alpha is chosen so GMI stays within budget
    baseline_gmi = base_results.get("gmi", 0)
    baseline_tir = base_results.get("tir", 0)

    gmi_idx = output_names.index("gmi") if "gmi" in output_names else None
    tir_idx = output_names.index("tir") if "tir" in output_names else None

    # Normalize gradient
    grad_norm = np.linalg.norm(hypo_grad)
    if grad_norm < 1e-8:
        print("  Hypo gradient is near-zero — no optimization possible.")
        return np.zeros(len(params)), base_results, J

    direction = -hypo_grad / grad_norm  # unit step that reduces hypos

    # Find maximum step size that satisfies constraints
    # Predicted change: delta_output = J @ (alpha * direction)
    alpha_max = 100.0  # generous initial bound
    alpha_step = 0.5

    best_alpha = 0
    best_hypo = base_results.get("hypo_concerning_per_week", 0)

    for trial in range(20):
        alpha = alpha_step * (trial + 1)
        delta_params = alpha * direction
        predicted = np.array([base_results.get(k, 0) for k in output_names]) + J @ delta_params

        pred_hypo = predicted[hypo_idx]
        pred_gmi = predicted[gmi_idx] if gmi_idx is not None else baseline_gmi
        pred_tir = predicted[tir_idx] if tir_idx is not None else baseline_tir

        gmi_ok = abs(pred_gmi - baseline_gmi) <= gmi_budget
        tir_ok = pred_tir >= baseline_tir - tir_floor

        if gmi_ok and tir_ok and pred_hypo < best_hypo:
            best_alpha = alpha
            best_hypo = pred_hypo

        if not gmi_ok or not tir_ok:
            break  # past constraint boundary

    if best_alpha == 0:
        print("  No improvement found within constraints.")
        return np.zeros(len(params)), base_results, J

    delta_params = best_alpha * direction
    predicted = {k: base_results.get(k, 0) + float(v)
                 for k, v in zip(output_names, J @ delta_params)}

    print(f"\n  Optimal step: alpha={best_alpha:.1f}")
    print(f"  Predicted hypos: {base_results.get('hypo_concerning_per_week', 0):.2f} → {predicted['hypo_concerning_per_week']:.2f}")
    if gmi_idx is not None:
        print(f"  Predicted GMI: {baseline_gmi:.2f} → {predicted['gmi']:.2f} (budget: ±{gmi_budget})")
    if tir_idx is not None:
        print(f"  Predicted TIR: {baseline_tir:.1f} → {predicted['tir']:.1f} (floor: -{tir_floor})")

    return delta_params, predicted, J


def main():
    parser = argparse.ArgumentParser(
        description="Optimize algorithm settings to minimize concerning hypos")
    parser.add_argument("profile", type=str,
                        help="Patient profile JSON path")
    parser.add_argument("--algo", type=str, default="trio",
                        help="Algorithm (default: trio)")
    parser.add_argument("--paths", type=int, default=40,
                        help="Simulation paths (default: 40)")
    parser.add_argument("--days", type=int, default=7,
                        help="Days per path (default: 7)")
    parser.add_argument("--params", type=str, nargs="+",
                        default=["target", "suspend_threshold"],
                        help="Parameters to optimize (default: target suspend_threshold)")
    parser.add_argument("--gmi-budget", type=float, default=0.1,
                        help="Max GMI change from baseline (default: 0.1%%)")
    parser.add_argument("--tir-floor", type=float, default=1.0,
                        help="Max TIR loss from baseline (default: 1.0%%)")
    parser.add_argument("--iterations", type=int, default=3,
                        help="Number of optimization iterations (default: 3)")
    parser.add_argument("--verify", action="store_true",
                        help="Run verification sim after optimization")
    parser.add_argument("--verify-paths", type=int, default=100,
                        help="Paths for verification run (default: 100)")
    args = parser.parse_args()

    with open(args.profile) as f:
        base_profile = json.load(f)

    # Parameter step sizes for Jacobian
    param_steps = {
        "target": 5.0,
        "suspend_threshold": 5.0,
        "isf": 5.0,
        "max_basal_rate": 0.2,
        "basal_rate": 0.05,
        "carb_ratio": 1.0,
    }
    params = [(p, param_steps.get(p, 1.0)) for p in args.params]

    print(f"Optimizing: {args.profile}")
    print(f"Algorithm: {args.algo}, {args.paths} paths × {args.days} days")
    print(f"Free parameters: {[p[0] for p in params]}")
    print(f"Constraints: GMI ±{args.gmi_budget}%, TIR floor -{args.tir_floor}%")

    current_profile = copy.deepcopy(base_profile)

    for iteration in range(args.iterations):
        print(f"\n{'='*70}")
        print(f"ITERATION {iteration + 1}/{args.iterations}")
        print(f"{'='*70}")

        # Current parameter values
        print("\nCurrent settings:")
        for param_name, _ in params:
            val = _get_current(current_profile, param_name)
            print(f"  {param_name}: {val}")

        # Run baseline
        print("\nRunning baseline...")
        results = run_sim(current_profile, args.algo, args.paths, args.days)

        print(f"\nBaseline metrics:")
        for k in ["hypo_concerning_per_week", "hypo_events_per_week",
                   "rescue_events_per_week", "gmi", "tir", "mean_bg",
                   "time_below_70"]:
            if k in results:
                print(f"  {k:>30}: {results[k]:.3f}")

        # Optimize
        delta_params, predicted, J = optimize_step(
            current_profile, results, params, args.algo,
            args.paths, args.days, args.gmi_budget, args.tir_floor,
        )

        # Apply deltas
        if np.linalg.norm(delta_params) < 1e-6:
            print("\nConverged — no further improvement found.")
            break

        for j, (param_name, _) in enumerate(params):
            current_profile = perturb(current_profile, param_name, delta_params[j])

        print(f"\nUpdated settings:")
        for param_name, _ in params:
            val = _get_current(current_profile, param_name)
            print(f"  {param_name}: {val:.1f}")

    # Save optimized profile
    optimized_path = args.profile.replace(".json", "_optimized.json")
    with open(optimized_path, "w") as f:
        json.dump(current_profile, f, indent=2)
    print(f"\nOptimized profile saved to {optimized_path}")

    # Print final comparison
    print(f"\n{'='*70}")
    print("FINAL SETTINGS COMPARISON")
    print(f"{'='*70}")
    print(f"{'Parameter':>25} {'Original':>12} {'Optimized':>12} {'Change':>12}")
    for param_name, _ in params:
        orig = _get_current(base_profile, param_name)
        opt = _get_current(current_profile, param_name)
        print(f"{param_name:>25} {orig:>12.1f} {opt:>12.1f} {opt - orig:>+12.1f}")

    # Verification run
    if args.verify:
        print(f"\n{'='*70}")
        print(f"VERIFICATION ({args.verify_paths} paths)")
        print(f"{'='*70}")

        print("\nRunning original profile...")
        orig_results = run_sim(base_profile, args.algo, args.verify_paths, args.days)

        print("\nRunning optimized profile...")
        opt_results = run_sim(current_profile, args.algo, args.verify_paths, args.days)

        print(f"\n{'Metric':>30} {'Original':>12} {'Optimized':>12} {'Change':>12}")
        for k in ["hypo_concerning_per_week", "hypo_events_per_week",
                   "rescue_events_per_week", "gmi", "tir", "mean_bg",
                   "time_below_70", "time_below_54", "time_above_180"]:
            if k in orig_results and k in opt_results:
                orig_v = orig_results[k]
                opt_v = opt_results[k]
                print(f"{k:>30} {orig_v:>12.3f} {opt_v:>12.3f} {opt_v - orig_v:>+12.3f}")


if __name__ == "__main__":
    main()
