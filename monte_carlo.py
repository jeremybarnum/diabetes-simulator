"""
Monte Carlo framework for algorithm comparison.

Runs N simulation paths with different random seeds, collects glycemic metrics,
and produces summary statistics and head-to-head comparisons.

Usage:
    python3 monte_carlo.py                                    # defaults
    python3 monte_carlo.py --paths 100 --days 7               # 100 paths, 7 days each
    python3 monte_carlo.py --profile patient_profiles/poor_carb_counter.json
    python3 monte_carlo.py --algorithms loop_ab40 trio
"""

import time
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, as_completed

from simulation import (PatientProfile, SimulationRun, SimulationRunResult,
                        save_batch_results)


# ─── Metrics ──────────────────────────────────────────────────────────────────

@dataclass
class GlycemicMetrics:
    """Glycemic control metrics from a single simulation path."""
    mean_bg: float = 0.0
    sd_bg: float = 0.0
    cv: float = 0.0               # Coefficient of variation (%)
    gmi: float = 0.0              # Glucose Management Indicator (%)

    time_in_range: float = 0.0    # % time 70-180 mg/dL
    time_below_70: float = 0.0    # % time < 70
    time_below_54: float = 0.0    # % time < 54 (severe hypo)
    time_above_180: float = 0.0   # % time > 180
    time_above_250: float = 0.0   # % time > 250

    min_bg: float = 0.0
    max_bg: float = 0.0

    hypo_events: int = 0          # BG < 70 for >= 15 min (3 consecutive readings)

    rescue_carb_events: int = 0
    rescue_carb_grams_total: float = 0.0

    n_readings: int = 0


def compute_metrics(result: SimulationRunResult) -> GlycemicMetrics:
    """Compute glycemic metrics from a simulation run."""
    bgs = result.all_bg_values()
    if not bgs:
        return GlycemicMetrics()

    arr = np.array(bgs)
    n = len(arr)

    mean_bg = float(np.mean(arr))
    sd_bg = float(np.std(arr))
    cv = (sd_bg / mean_bg * 100) if mean_bg > 0 else 0
    # GMI formula: GMI(%) = 3.31 + 0.02392 × mean glucose (mg/dL)
    gmi = 3.31 + 0.02392 * mean_bg

    tir = float(np.sum((arr >= 70) & (arr <= 180)) / n * 100)
    tb70 = float(np.sum(arr < 70) / n * 100)
    tb54 = float(np.sum(arr < 54) / n * 100)
    ta180 = float(np.sum(arr > 180) / n * 100)
    ta250 = float(np.sum(arr > 250) / n * 100)

    # Count hypo events: consecutive runs of BG < 70 lasting >= 15 min (3 readings)
    hypo_events = 0
    consecutive_low = 0
    for bg in bgs:
        if bg < 70:
            consecutive_low += 1
        else:
            if consecutive_low >= 3:
                hypo_events += 1
            consecutive_low = 0
    if consecutive_low >= 3:
        hypo_events += 1

    rescue_events = sum(d.rescue_carb_events for d in result.days)
    rescue_grams = sum(d.rescue_carb_grams_total for d in result.days)

    return GlycemicMetrics(
        mean_bg=mean_bg,
        sd_bg=sd_bg,
        cv=cv,
        gmi=gmi,
        time_in_range=tir,
        time_below_70=tb70,
        time_below_54=tb54,
        time_above_180=ta180,
        time_above_250=ta250,
        min_bg=float(np.min(arr)),
        max_bg=float(np.max(arr)),
        hypo_events=hypo_events,
        rescue_carb_events=rescue_events,
        rescue_carb_grams_total=rescue_grams,
        n_readings=n,
    )


# ─── Seeding ─────────────────────────────────────────────────────────────────

def _algo_seed_offset(algo_name: str) -> int:
    """Deterministic seed offset from algorithm name.

    TODO(jeremy): Python's built-in hash() is randomized per-process
    (PYTHONHASHSEED), so using hash(algo_name) for RNG seeding made results
    non-reproducible across runs, and different between local and Modal cloud
    workers. This uses md5 instead for a stable, deterministic mapping.
    """
    import hashlib
    return int(hashlib.md5(algo_name.encode()).hexdigest(), 16) % (2**31)


# ─── Variant Labels ──────────────────────────────────────────────────────────

def _variant_label(algo_name: str, profile_label: str, n_profiles: int) -> str:
    """Build a variant label encoding (algorithm, profile).

    When there's only one profile, returns just the algo name (backward compat).
    With multiple profiles, returns 'algo_name/profile_label'.
    """
    if n_profiles <= 1:
        return algo_name
    return f"{algo_name}/{profile_label}"


def build_variants(
    algorithms: List[str],
    profiles: List[Tuple[str, Dict]],
) -> List[Tuple[str, Dict, str]]:
    """Build list of (variant_label, profile_dict, algo_name) tuples."""
    n_profiles = len(profiles)
    variants = []
    for algo_name in algorithms:
        for profile_label, profile_dict in profiles:
            label = _variant_label(algo_name, profile_label, n_profiles)
            variants.append((label, profile_dict, algo_name))
    return variants


# ─── Single Path Runner (top-level for pickling) ─────────────────────────────

def _run_single_path(args) -> Dict[str, dict]:
    """Run one path for all variants. Must be top-level for multiprocessing.

    Accepts two formats for backward compatibility:
    - New: (path_seed, variants, n_days)
      where variants is [(variant_label, profile_dict, algo_name), ...]
    - Legacy: (path_seed, profile_dict, algorithms, n_days)

    Returns dict mapping variant_label -> {'metrics': GlycemicMetrics, 'result': dict}
    """
    # Detect format: new has 3 elements, legacy has 4
    if len(args) == 3:
        path_seed, variants, n_days = args
    else:
        # Legacy format
        path_seed, profile_dict, algorithms, n_days = args
        variants = [(algo, profile_dict, algo) for algo in algorithms]

    results = {}
    for variant_label, profile_dict, algo_name in variants:
        profile = _profile_from_dict(profile_dict)
        # Use algo_name for seeding when it equals the variant label (single profile),
        # otherwise use variant_label (multi-profile)
        seed_key = algo_name if variant_label == algo_name else variant_label
        rng = np.random.RandomState(path_seed + _algo_seed_offset(seed_key))
        sim = SimulationRun(
            profile=profile,
            algorithm_name=algo_name,
            n_days=n_days,
            rng=rng,
        )
        run_result = sim.run()
        results[variant_label] = {
            'metrics': compute_metrics(run_result),
            'result': run_result.to_dict(),
        }

    return results


def _profile_from_dict(d: Dict) -> PatientProfile:
    """Reconstruct PatientProfile from a serializable dict."""
    from simulation import MealSpec, ExerciseSpec
    # Backward compat: 'meals' key treated as 'meals_rest'
    meals_rest = [MealSpec(**m) for m in d.get('meals_rest', d.get('meals', []))]
    meals_exercise = [MealSpec(**m) for m in d.get('meals_exercise', [])]
    exercise_spec = None
    if d.get('exercise_spec'):
        exercise_spec = ExerciseSpec(**d['exercise_spec'])
    undeclared_meals_rest = [MealSpec(**m) for m in
                             d.get('undeclared_meals_rest', d.get('undeclared_meals', []))]
    undeclared_meals_exercise = [MealSpec(**m) for m in d.get('undeclared_meals_exercise', [])]
    return PatientProfile(
        meals_rest=meals_rest,
        meals_exercise=meals_exercise,
        carb_count_sigma=d.get('carb_count_sigma', 0.15),
        carb_count_bias=d.get('carb_count_bias', 0.0),
        absorption_sigma=d.get('absorption_sigma', 0.15),
        undeclared_meal_prob=d.get('undeclared_meal_prob', 0.0),
        undeclared_meals_rest=undeclared_meals_rest,
        undeclared_meals_exercise=undeclared_meals_exercise,
        sensitivity_sigma=d.get('sensitivity_sigma', 0.15),
        exercise_days=d.get('exercise_days', []),
        exercise_spec=exercise_spec,
        starting_bg=d.get('starting_bg', 100.0),
        rescue_carbs_enabled=d.get('rescue_carbs_enabled', True),
        rescue_threshold=d.get('rescue_threshold', 65.0),
        rescue_carbs_grams=d.get('rescue_carbs_grams', 8.0),
        rescue_absorption_hrs=d.get('rescue_absorption_hrs', 1.0),
        rescue_cooldown_min=d.get('rescue_cooldown_min', 15.0),
        rescue_carbs_declared_pct=d.get('rescue_carbs_declared_pct',
                                          1.0 if d.get('rescue_carbs_declared') else 0.0),
        algorithm_settings=d.get('algorithm_settings'),
    )


def _profile_to_dict(p: PatientProfile) -> Dict:
    """Serialize PatientProfile to a picklable dict."""
    def _meal_list(specs):
        return [{'time_of_day_minutes': m.time_of_day_minutes,
                 'carbs_mean': m.carbs_mean, 'carbs_sd': m.carbs_sd,
                 'absorption_hrs': m.absorption_hrs} for m in specs]
    ex = None
    if p.exercise_spec:
        ex = {
            'time_of_day_minutes': p.exercise_spec.time_of_day_minutes,
            'declared_scalar': p.exercise_spec.declared_scalar,
            'declared_duration_hrs': p.exercise_spec.declared_duration_hrs,
            'actual_scalar_mean': p.exercise_spec.actual_scalar_mean,
            'actual_scalar_sigma': p.exercise_spec.actual_scalar_sigma,
            'actual_duration_hrs_mean': p.exercise_spec.actual_duration_hrs_mean,
            'actual_duration_hrs_sigma': p.exercise_spec.actual_duration_hrs_sigma,
        }
    return {
        'meals_rest': _meal_list(p.meals_rest),
        'meals_exercise': _meal_list(p.meals_exercise),
        'carb_count_sigma': p.carb_count_sigma,
        'carb_count_bias': p.carb_count_bias,
        'absorption_sigma': p.absorption_sigma,
        'undeclared_meal_prob': p.undeclared_meal_prob,
        'undeclared_meals_rest': _meal_list(p.undeclared_meals_rest),
        'undeclared_meals_exercise': _meal_list(p.undeclared_meals_exercise),
        'sensitivity_sigma': p.sensitivity_sigma,
        'exercise_days': p.exercise_days,
        'exercise_spec': ex,
        'starting_bg': p.starting_bg,
        'rescue_carbs_enabled': p.rescue_carbs_enabled,
        'rescue_threshold': p.rescue_threshold,
        'rescue_carbs_grams': p.rescue_carbs_grams,
        'rescue_absorption_hrs': p.rescue_absorption_hrs,
        'rescue_cooldown_min': p.rescue_cooldown_min,
        'rescue_carbs_declared_pct': p.rescue_carbs_declared_pct,
        'algorithm_settings': p.get_settings(),
    }


# ─── Monte Carlo Runner ──────────────────────────────────────────────────────

@dataclass
class MonteCarloResults:
    """Aggregated results from Monte Carlo simulation."""
    algorithm_name: str
    n_paths: int
    n_days: int
    all_metrics: List[GlycemicMetrics] = field(default_factory=list)
    all_run_results: List[SimulationRunResult] = field(default_factory=list)

    def _values(self, attr: str) -> np.ndarray:
        return np.array([getattr(m, attr) for m in self.all_metrics])

    def summary(self) -> Dict[str, Dict[str, float]]:
        """Summary statistics for each metric."""
        attrs = ['mean_bg', 'sd_bg', 'cv', 'gmi',
                 'time_in_range', 'time_below_70', 'time_below_54',
                 'time_above_180', 'time_above_250', 'hypo_events']
        out = {}
        for attr in attrs:
            vals = self._values(attr)
            out[attr] = {
                'mean': float(np.mean(vals)),
                'median': float(np.median(vals)),
                'p5': float(np.percentile(vals, 5)),
                'p25': float(np.percentile(vals, 25)),
                'p75': float(np.percentile(vals, 75)),
                'p95': float(np.percentile(vals, 95)),
                'sd': float(np.std(vals)),
            }
        return out


def run_monte_carlo(
    profile: PatientProfile,
    algorithms: List[str],
    n_paths: int = 100,
    n_days: int = 7,
    seed: int = 42,
    max_workers: int = 0,
    profiles: Optional[List[Tuple[str, PatientProfile]]] = None,
) -> Dict[str, MonteCarloResults]:
    """
    Run Monte Carlo comparison of algorithms.

    Args:
        profile: Patient profile configuration (used when profiles is None)
        algorithms: List of algorithm names (e.g., ['loop_ab40', 'trio'])
        n_paths: Number of simulation paths
        n_days: Days per path
        seed: Base random seed
        max_workers: Number of parallel workers (1 = sequential)
        profiles: Optional list of (label, PatientProfile) pairs for multi-profile comparison.
                  When provided, overrides the single `profile` arg.

    Returns:
        Dict mapping variant label to MonteCarloResults
    """
    # Build profile list
    if profiles is None:
        profile_pairs = [("default", _profile_to_dict(profile))]
    else:
        profile_pairs = [(label, _profile_to_dict(p)) for label, p in profiles]

    variants = build_variants(algorithms, profile_pairs)
    variant_labels = [v[0] for v in variants]

    # Prepare per-path args
    path_args = []
    for i in range(n_paths):
        path_seed = seed + i * 1000
        path_args.append((path_seed, variants, n_days))

    # Collect results
    mc_results = {label: MonteCarloResults(label, n_paths, n_days)
                  for label in variant_labels}

    t_start = time.time()

    def _collect_path(path_data):
        for name, data in path_data.items():
            mc_results[name].all_metrics.append(data['metrics'])
            mc_results[name].all_run_results.append(
                SimulationRunResult.from_dict(data['result']))

    # Resolve 0 → cpu_count (auto-parallel)
    if max_workers <= 0:
        import os
        max_workers = os.cpu_count() or 4

    if max_workers <= 1:
        # Sequential
        for i, args in enumerate(path_args):
            path_data = _run_single_path(args)
            _collect_path(path_data)
            if (i + 1) % max(1, n_paths // 10) == 0:
                elapsed = time.time() - t_start
                print(f"  Path {i+1}/{n_paths} ({elapsed:.1f}s)")
    else:
        # Parallel
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_run_single_path, args): i
                       for i, args in enumerate(path_args)}
            completed = 0
            for future in as_completed(futures):
                path_data = future.result()
                _collect_path(path_data)
                completed += 1
                if completed % max(1, n_paths // 10) == 0:
                    elapsed = time.time() - t_start
                    print(f"  Path {completed}/{n_paths} ({elapsed:.1f}s)")

    elapsed = time.time() - t_start
    print(f"  Done: {n_paths} paths x {n_days} days in {elapsed:.1f}s")

    return mc_results


# ─── Reporting ────────────────────────────────────────────────────────────────

def print_comparison(mc_results: Dict[str, MonteCarloResults]):
    """Print a summary comparison table."""
    names = list(mc_results.keys())
    if not names:
        return

    summaries = {name: mc_results[name].summary() for name in names}

    print(f"\n{'='*80}")
    print(f"Monte Carlo Comparison: {mc_results[names[0]].n_paths} paths x "
          f"{mc_results[names[0]].n_days} days")
    print(f"{'='*80}")

    # Header
    col_w = 14
    header = f"{'Metric':<22}"
    for name in names:
        header += f" {'[' + name + ']':>{col_w}}"
    print(header)
    print("-" * (22 + (col_w + 1) * len(names)))

    # Rows
    display_metrics = [
        ('Mean BG (mg/dL)', 'mean_bg', 'mean', '.1f'),
        ('SD BG (mg/dL)', 'sd_bg', 'mean', '.1f'),
        ('CV (%)', 'cv', 'mean', '.1f'),
        ('GMI (%)', 'gmi', 'mean', '.2f'),
        ('TIR 70-180 (%)', 'time_in_range', 'mean', '.1f'),
        ('  p5-p95', 'time_in_range', 'range', '.1f'),
        ('Time <70 (%)', 'time_below_70', 'mean', '.2f'),
        ('Time <54 (%)', 'time_below_54', 'mean', '.3f'),
        ('Time >180 (%)', 'time_above_180', 'mean', '.1f'),
        ('Time >250 (%)', 'time_above_250', 'mean', '.2f'),
        ('Hypo events/path', 'hypo_events', 'mean', '.2f'),
    ]

    for label, metric, stat, fmt in display_metrics:
        row = f"{label:<22}"
        for name in names:
            s = summaries[name][metric]
            if stat == 'mean':
                val = f"{s['mean']:{fmt}}"
            elif stat == 'range':
                val = f"{s['p5']:{fmt}}-{s['p95']:{fmt}}"
            row += f" {val:>{col_w}}"
        print(row)

    # Head-to-head comparison (if exactly 2 algorithms)
    if len(names) == 2:
        a, b = names
        ma = mc_results[a].all_metrics
        mb = mc_results[b].all_metrics
        n = min(len(ma), len(mb))

        tir_a = [m.time_in_range for m in ma[:n]]
        tir_b = [m.time_in_range for m in mb[:n]]
        a_wins_tir = sum(1 for x, y in zip(tir_a, tir_b) if x > y)
        b_wins_tir = sum(1 for x, y in zip(tir_a, tir_b) if y > x)

        hypo_a = [m.time_below_70 for m in ma[:n]]
        hypo_b = [m.time_below_70 for m in mb[:n]]
        a_wins_hypo = sum(1 for x, y in zip(hypo_a, hypo_b) if x < y)
        b_wins_hypo = sum(1 for x, y in zip(hypo_a, hypo_b) if y < x)

        print(f"\n{'─'*60}")
        print(f"Head-to-head ({n} paired paths):")
        print(f"  TIR: {a} wins {a_wins_tir}, {b} wins {b_wins_tir}, "
              f"tied {n - a_wins_tir - b_wins_tir}")
        print(f"  Less hypo: {a} wins {a_wins_hypo}, {b} wins {b_wins_hypo}, "
              f"tied {n - a_wins_hypo - b_wins_hypo}")

    print()


# ─── CLI ──────────────────────────────────────────────────────────────────────

def default_profile() -> PatientProfile:
    """A reasonable default patient profile for quick testing."""
    from simulation import MealSpec
    settings_path = Path(__file__).parent / 'settings.json'
    with open(settings_path) as f:
        settings = json.load(f)

    return PatientProfile(
        meals_rest=[
            MealSpec(time_of_day_minutes=30, carbs_mean=30, carbs_sd=10,
                     absorption_hrs=3.0),   # Breakfast at 7:30am
            MealSpec(time_of_day_minutes=330, carbs_mean=50, carbs_sd=15,
                     absorption_hrs=3.0),   # Lunch at 12:30pm
            MealSpec(time_of_day_minutes=690, carbs_mean=60, carbs_sd=20,
                     absorption_hrs=3.5),   # Dinner at 6:30pm
        ],
        carb_count_sigma=0.15,
        absorption_sigma=0.15,
        sensitivity_sigma=0.15,
        starting_bg=100.0,
        algorithm_settings=settings,
    )


def _load_profiles(args) -> Optional[List[Tuple[str, PatientProfile]]]:
    """Load profile list from CLI args. Returns None for single-profile mode."""
    if hasattr(args, 'profiles') and args.profiles:
        profile_paths = [p.strip() for p in args.profiles.split(',')]
        return [(Path(p).stem, PatientProfile.from_json(p)) for p in profile_paths]
    if args.profile:
        return [(Path(args.profile).stem, PatientProfile.from_json(args.profile))]
    return None


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Monte Carlo algorithm comparison')
    parser.add_argument('--paths', type=int, default=10,
                        help='Number of simulation paths')
    parser.add_argument('--days', type=int, default=1,
                        help='Days per path')
    parser.add_argument('--algorithms', nargs='+', default=['loop_ab40', 'trio'],
                        help='Algorithms to compare')
    parser.add_argument('--profile', type=str, default=None,
                        help='Path to patient profile JSON (single profile)')
    parser.add_argument('--profiles', type=str, default=None,
                        help='Comma-separated profile JSON paths for multi-profile comparison')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--workers', type=int, default=0,
                        help='Parallel workers (0=auto/cpu_count, 1=sequential)')
    args = parser.parse_args()

    profiles_list = _load_profiles(args)
    if profiles_list:
        profile = profiles_list[0][1]  # for display only
    else:
        profile = default_profile()

    print(f"Monte Carlo: {args.paths} paths x {args.days} days")
    print(f"Algorithms: {', '.join(args.algorithms)}")
    if profiles_list and len(profiles_list) > 1:
        print(f"Profiles: {', '.join(label for label, _ in profiles_list)}")
    print(f"Sensitivity sigma: {profile.sensitivity_sigma}")
    print(f"Carb count sigma: {profile.carb_count_sigma}")
    print(f"Meals/day (rest): {len(profile.meals_rest)}")
    print()

    workers = args.workers
    if workers <= 0:
        import os
        workers = os.cpu_count() or 4

    results = run_monte_carlo(
        profile=profile,
        algorithms=args.algorithms,
        n_paths=args.paths,
        n_days=args.days,
        seed=args.seed,
        max_workers=workers,
        profiles=profiles_list,
    )

    print_comparison(results)

    # Auto-save results
    for variant_label, mc in results.items():
        safe_label = variant_label.replace('/', '_')
        label = f'{safe_label}_{args.paths}paths_{args.days}d_s{args.seed}'
        metadata = {
            'variant': variant_label,
            'n_paths': args.paths,
            'n_days': args.days,
            'seed': args.seed,
        }
        save_batch_results(mc.all_run_results, label, metadata)


if __name__ == '__main__':
    main()
