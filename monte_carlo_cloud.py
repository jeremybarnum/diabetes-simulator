"""
Modal cloud runner for Monte Carlo simulations.

Runs simulation paths on Modal's serverless infrastructure for massive parallelism.
Reuses all logic from monte_carlo.py — no algorithm code is duplicated.

Usage:
    modal run monte_carlo_cloud.py --paths 100 --days 7
    modal run monte_carlo_cloud.py --paths 500 --days 7 --algorithms "loop_ab40,trio"
    modal run monte_carlo_cloud.py --profile patient_profiles/variable_sensitivity.json
"""

import pathlib
import modal

app = modal.App("diabetes-monte-carlo")

# Directories/patterns to exclude from the cloud image
_EXCLUDE_DIRS = {
    "venv", ".git", "sim_results", "outputs", "ios_logs",
    "loop_testing", "loop_ios_testing", "loop_ios_testing_v2",
    "trio_testing", "_dev", "_validation", "docs",
    "node_modules", "__pycache__", ".mypy_cache", ".pytest_cache",
}


def _should_ignore(pth: pathlib.Path) -> bool:
    """Filter for add_local_dir ignore: return True for paths to EXCLUDE."""
    parts = pth.parts
    if any(part in _EXCLUDE_DIRS for part in parts):
        return True
    if pth.suffix == ".pyc":
        return True
    return False


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy>=1.21.0", "scipy>=1.7.0")
    .add_local_dir(".", "/root/project", ignore=_should_ignore)
)


@app.function(image=image, cpu=1.0, memory=512, timeout=600, retries=1)
def run_path_remote(path_seed, variants, n_days):
    """Run a single Monte Carlo path on a remote worker."""
    import sys
    sys.path.insert(0, "/root/project")
    from monte_carlo import _run_single_path
    return _run_single_path((path_seed, variants, n_days))


@app.local_entrypoint()
def main(
    paths: int = 10,
    days: int = 1,
    algorithms: str = "loop_ab40,trio",
    profile: str = "",
    profiles: str = "",
    seed: int = 42,
):
    import time
    from pathlib import Path
    from monte_carlo import (
        _profile_to_dict, default_profile, build_variants, MonteCarloResults,
        print_comparison,
    )
    from simulation import PatientProfile, SimulationRunResult, save_batch_results

    algo_list = [a.strip() for a in algorithms.split(",")]

    # Build profile list: --profiles takes precedence over --profile
    if profiles:
        profile_paths = [p.strip() for p in profiles.split(",")]
        profile_pairs = [
            (Path(p).stem, _profile_to_dict(PatientProfile.from_json(p)))
            for p in profile_paths
        ]
    elif profile:
        p = PatientProfile.from_json(profile)
        profile_pairs = [(Path(profile).stem, _profile_to_dict(p))]
    else:
        profile_pairs = [("default", _profile_to_dict(default_profile()))]

    variants = build_variants(algo_list, profile_pairs)
    variant_labels = [v[0] for v in variants]

    # Build per-path argument tuples (same seeding as local runner)
    args_list = []
    for i in range(paths):
        path_seed = seed + i * 1000
        args_list.append((path_seed, variants, days))

    print(f"Monte Carlo Cloud: {paths} paths x {days} days")
    print(f"Algorithms: {', '.join(algo_list)}")
    if len(profile_pairs) > 1:
        print(f"Profiles: {', '.join(label for label, _ in profile_pairs)}")
    print(f"Variants: {', '.join(variant_labels)}")
    print(f"Launching {paths} remote workers...")
    print()

    mc_results = {label: MonteCarloResults(label, paths, days)
                  for label in variant_labels}

    t_start = time.time()
    completed = 0
    errors = 0

    for result in run_path_remote.starmap(
        args_list, order_outputs=False, return_exceptions=True,
        wrap_returned_exceptions=False,
    ):
        if isinstance(result, Exception):
            errors += 1
            print(f"  ERROR: {result}")
            continue

        for label, data in result.items():
            mc_results[label].all_metrics.append(data["metrics"])
            mc_results[label].all_run_results.append(
                SimulationRunResult.from_dict(data["result"])
            )

        completed += 1
        if completed % max(1, paths // 10) == 0:
            elapsed = time.time() - t_start
            print(f"  {completed}/{paths} paths ({elapsed:.1f}s)")

    elapsed = time.time() - t_start
    print(
        f"\nDone: {completed}/{paths} paths in {elapsed:.1f}s"
        + (f" ({errors} errors)" if errors else "")
    )

    print_comparison(mc_results)

    # Save results
    for variant_label, mc in mc_results.items():
        safe_label = variant_label.replace("/", "_")
        label = f"{safe_label}_{paths}paths_{days}d_s{seed}_cloud"
        metadata = {
            "variant": variant_label,
            "n_paths": paths,
            "n_days": days,
            "seed": seed,
            "runner": "modal_cloud",
        }
        save_batch_results(mc.all_run_results, label, metadata)
