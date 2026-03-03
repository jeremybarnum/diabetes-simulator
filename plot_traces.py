#!/usr/bin/env python3
"""
Spaghetti plot: BG traces for Loop vs Trio over 24 hours.

Each simulated path is a line. Loop in one color, Trio in another.
X axis runs from 7am to 7am.

Usage:
    python3 plot_traces.py
    python3 plot_traces.py --paths 50 --profile patient_profiles/real_patient.json
    python3 plot_traces.py --no-carbs  # undeclared meals variant
    python3 plot_traces.py --from-results sim_results/20260219_*.json  # re-plot saved
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from simulation import (PatientProfile, SimulationRun, SimulationRunResult,
                        save_batch_results, load_batch_results)
from monte_carlo import _algo_seed_offset


def run_paths(profile, algorithm_name, n_paths, seed):
    """Run n_paths and return (traces, SimulationRunResults).

    traces: list of [(hour, bg), ...] for plotting
    results: list of SimulationRunResult for saving
    """
    traces = []
    results = []
    for i in range(n_paths):
        path_seed = seed + i * 1000 + _algo_seed_offset(algorithm_name)
        rng = np.random.RandomState(path_seed)
        sim = SimulationRun(profile=profile, algorithm_name=algorithm_name,
                            n_days=1, rng=rng)
        result = sim.run()
        results.append(result)
        day = result.days[0]
        # Convert to hours from 7am
        trace = [(t_rel / 60, bg) for t_rel, bg in day.bg_trace]
        traces.append(trace)
    return traces, results


def traces_from_results(results):
    """Extract plot traces from loaded SimulationRunResults."""
    traces = []
    for result in results:
        day = result.days[0]
        trace = [(t_rel / 60, bg) for t_rel, bg in day.bg_trace]
        traces.append(trace)
    return traces


def plot_comparison(loop_traces, trio_traces, title="Loop vs Trio — 24h BG Traces",
                    output_path=None, ns_trace=None):
    """Plot spaghetti chart with optional Nightscout overlay."""
    fig, ax = plt.subplots(figsize=(14, 7))

    # Plot each trace
    loop_color = '#2563eb'  # blue
    trio_color = '#dc2626'  # red

    for trace in loop_traces:
        hours = [h for h, _ in trace]
        bgs = [bg for _, bg in trace]
        ax.plot(hours, bgs, color=loop_color, alpha=0.15, linewidth=0.8)

    for trace in trio_traces:
        hours = [h for h, _ in trace]
        bgs = [bg for _, bg in trace]
        ax.plot(hours, bgs, color=trio_color, alpha=0.15, linewidth=0.8)

    # Plot medians
    def compute_median_trace(traces):
        # All traces have same time points
        hours = [h for h, _ in traces[0]]
        bg_matrix = np.array([[bg for _, bg in t] for t in traces])
        medians = np.median(bg_matrix, axis=0)
        p25 = np.percentile(bg_matrix, 25, axis=0)
        p75 = np.percentile(bg_matrix, 75, axis=0)
        return hours, medians, p25, p75

    lh, lm, lp25, lp75 = compute_median_trace(loop_traces)
    th, tm, tp25, tp75 = compute_median_trace(trio_traces)

    ax.plot(lh, lm, color=loop_color, linewidth=2.5, label='Loop AB40 (median)')
    ax.plot(th, tm, color=trio_color, linewidth=2.5, label='Trio (median)')

    # Nightscout overlay
    if ns_trace:
        ns_hours = [h for h, _ in ns_trace]
        ns_bgs = [bg for _, bg in ns_trace]
        ax.plot(ns_hours, ns_bgs, color='#16a34a', linewidth=2.5, linestyle='-',
                label='Nightscout (median 30d)', zorder=5)

    # Target range shading
    ax.axhspan(70, 180, color='#22c55e', alpha=0.08, zorder=0)
    ax.axhline(70, color='#dc2626', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.axhline(180, color='#f59e0b', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.axhline(100, color='gray', linewidth=0.5, linestyle=':', alpha=0.4)

    # X axis: hours from 7am
    ax.set_xlim(0, 24)
    hour_labels = []
    hour_ticks = []
    for h in range(0, 25, 2):
        clock_hr = (7 + h) % 24
        ampm = 'am' if clock_hr < 12 else 'pm'
        display = clock_hr if clock_hr <= 12 else clock_hr - 12
        if display == 0:
            display = 12
        hour_labels.append(f'{display}{ampm}')
        hour_ticks.append(h)
    ax.set_xticks(hour_ticks)
    ax.set_xticklabels(hour_labels)

    ax.set_xlabel('Time of Day')
    ax.set_ylabel('Blood Glucose (mg/dL)')
    ax.set_title(title)
    ax.legend(loc='upper left')
    ax.set_ylim(40, 300)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f'Saved plot to {output_path}')
    else:
        plt.show()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Plot BG spaghetti traces')
    parser.add_argument('--paths', type=int, default=50)
    parser.add_argument('--profile', type=str, default='patient_profiles/real_patient.json')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no-carbs', action='store_true',
                        help='Use no-carbs-declared variant')
    parser.add_argument('--output', type=str, default=None,
                        help='Save plot to file instead of showing')
    parser.add_argument('--from-results', nargs='+', default=None,
                        help='Load from saved result files (2 files: loop then trio)')
    parser.add_argument('--nightscout', action='store_true',
                        help='Fetch Nightscout CGM data and overlay median')
    parser.add_argument('--ns-data', type=str, default=None,
                        help='Load previously saved Nightscout trace JSON')
    parser.add_argument('--ns-days', type=int, default=30,
                        help='Days of Nightscout data to fetch (default 30)')
    args = parser.parse_args()

    # --- Load or fetch Nightscout trace ---
    ns_trace = None
    if args.ns_data:
        from nightscout_query import load_nightscout_trace
        ns_trace = load_nightscout_trace(args.ns_data)
        print(f'Loaded Nightscout trace from {args.ns_data} ({len(ns_trace)} points)')
    elif args.nightscout:
        from nightscout_query import fetch_and_save_nightscout_trace
        ns_trace, ns_path = fetch_and_save_nightscout_trace(days=args.ns_days)

    # --- Load or run sim traces ---
    if args.from_results:
        # Load from saved results
        if len(args.from_results) != 2:
            print('Error: --from-results expects exactly 2 files (loop, trio)')
            return
        loop_results, loop_meta = load_batch_results(args.from_results[0])
        trio_results, trio_meta = load_batch_results(args.from_results[1])
        loop_traces = traces_from_results(loop_results)
        trio_traces = traces_from_results(trio_results)
        n = len(loop_traces)
        carb_mode = "Loaded Results"
        title = f"Loop vs Trio — {n} paths, {carb_mode}"
    else:
        # Run simulations
        if args.no_carbs:
            profile_path = args.profile.replace('.json', '_no_carbs.json')
        else:
            profile_path = args.profile

        profile = PatientProfile.from_json(profile_path)
        n = args.paths

        print(f'Running {n} paths for Loop AB40...')
        loop_traces, loop_results = run_paths(profile, 'loop_ab40', n, args.seed)
        print(f'Running {n} paths for Trio...')
        trio_traces, trio_results = run_paths(profile, 'trio', n, args.seed)

        # Auto-save results
        profile_label = Path(profile_path).stem
        for algo_name, results in [('loop_ab40', loop_results), ('trio', trio_results)]:
            label = f'{algo_name}_{n}paths_1d_{profile_label}_s{args.seed}'
            metadata = {
                'algorithm': algo_name,
                'n_paths': n,
                'n_days': 1,
                'seed': args.seed,
                'profile': profile_path,
                'source': 'plot_traces',
            }
            save_batch_results(results, label, metadata)

        carb_mode = "No Carbs Declared" if args.no_carbs else "Carbs Declared"
        title = f"Loop vs Trio — {n} paths, {carb_mode}"

    plot_comparison(loop_traces, trio_traces, title=title, output_path=args.output,
                    ns_trace=ns_trace)


if __name__ == '__main__':
    main()
