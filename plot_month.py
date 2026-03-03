#!/usr/bin/env python3
"""
Plot full month of Nightscout real data vs simulated Trio trace.

Usage:
    python3 plot_month.py
    python3 plot_month.py --ns-data sim_results/..._nightscout_30d_raw.json
    python3 plot_month.py --sim-data sim_results/..._trio_30d.json
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta, timezone

from simulation import (PatientProfile, SimulationRun, SimulationRunResult,
                        save_batch_results, load_batch_results)
from monte_carlo import _algo_seed_offset
from nightscout_query import (NIGHTSCOUT_URL, fetch_entries)


def fetch_and_save_raw_entries(days=30, base_url=NIGHTSCOUT_URL, token=None,
                               utc_offset_hours=-5.0):
    """Fetch raw Nightscout CGM entries and save them."""
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=days)

    print(f"Fetching CGM entries from {base_url} ({days} days)...")
    entries = fetch_entries(base_url, start_date, end_date, token=token)
    print(f"  Got {len(entries)} CGM readings")

    # Convert to simple (day_offset_hours, bg) trace
    # Day 0 = first reading's date at 7am local
    tz_offset = timedelta(hours=utc_offset_hours)
    points = []
    for entry in entries:
        sgv = entry.get("sgv")
        date_ms = entry.get("date")
        if sgv is None or date_ms is None or sgv <= 0:
            continue
        dt_utc = datetime.fromtimestamp(date_ms / 1000, tz=timezone.utc)
        dt_local = dt_utc + tz_offset
        points.append((dt_local, sgv))

    points.sort(key=lambda x: x[0])

    # Anchor to first day's 7am
    first_day = points[0][0].replace(hour=7, minute=0, second=0, microsecond=0)
    trace = []
    for dt, bg in points:
        hours_from_start = (dt - first_day).total_seconds() / 3600
        if hours_from_start >= 0:
            trace.append((hours_from_start, bg))

    # Save
    results_dir = Path(__file__).parent / "sim_results"
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = results_dir / f"{timestamp}_nightscout_{days}d_raw.json"

    data = {
        "metadata": {
            "source": "nightscout_raw",
            "days": days,
            "n_entries": len(trace),
            "utc_offset_hours": utc_offset_hours,
        },
        "trace": trace,
    }
    with open(path, "w") as f:
        json.dump(data, f)
    print(f"  Saved raw trace to {path}")
    return trace, str(path)


def load_raw_trace(path):
    with open(path) as f:
        data = json.load(f)
    return [tuple(p) for p in data["trace"]]


def run_independent_days(profile, algorithm_name, n_days=30, seed=42):
    """Run n_days independent 1-day simulations, each starting fresh.

    Returns (stitched_trace, list_of_SimulationRunResult).
    trace: [(hours_from_day0_start, bg), ...] for plotting as a month.
    """
    trace = []
    results = []
    for day in range(n_days):
        path_seed = seed + day * 1000 + _algo_seed_offset(algorithm_name)
        rng = np.random.RandomState(path_seed)
        sim = SimulationRun(profile=profile, algorithm_name=algorithm_name,
                            n_days=1, rng=rng)
        result = sim.run()
        results.append(result)

        # Stitch: offset by day index
        day_offset_hrs = day * 24
        for t_rel_min, bg in result.days[0].bg_trace:
            trace.append((day_offset_hrs + t_rel_min / 60, bg))

        if (day + 1) % 10 == 0:
            print(f"  Day {day+1}/{n_days}")

    return trace, results


def plot_month_comparison(ns_trace, sim_trace, output_path=None):
    """Plot full-month Nightscout vs Trio traces."""
    fig, axes = plt.subplots(2, 1, figsize=(18, 10), sharex=True, sharey=True)

    # Determine total days from data
    ns_max_hrs = max(h for h, _ in ns_trace) if ns_trace else 720
    sim_max_hrs = max(h for h, _ in sim_trace) if sim_trace else 720
    total_days = int(max(ns_max_hrs, sim_max_hrs) / 24) + 1

    for ax, trace, label, color in [
        (axes[0], ns_trace, 'Nightscout (Real Data)', '#16a34a'),
        (axes[1], sim_trace, 'Trio Simulation', '#dc2626'),
    ]:
        hours = [h for h, _ in trace]
        bgs = [bg for _, bg in trace]
        ax.plot(hours, bgs, color=color, alpha=0.7, linewidth=0.5)

        # Target range
        ax.axhspan(70, 180, color='#22c55e', alpha=0.06, zorder=0)
        ax.axhline(70, color='#dc2626', linewidth=0.5, linestyle='--', alpha=0.4)
        ax.axhline(180, color='#f59e0b', linewidth=0.5, linestyle='--', alpha=0.4)

        ax.set_ylabel('BG (mg/dL)')
        ax.set_title(label)
        ax.set_ylim(39, 350)
        ax.grid(True, alpha=0.15)

        # Shade nighttime (11pm-7am)
        for d in range(total_days):
            night_start = d * 24 + 16  # 11pm = 16 hours from 7am
            night_end = (d + 1) * 24    # 7am next day
            ax.axvspan(night_start, night_end, color='#1e293b', alpha=0.04, zorder=0)

    # X axis: day labels
    day_ticks = [d * 24 for d in range(total_days + 1)]
    day_labels = [f'Day {d+1}' for d in range(total_days + 1)]
    axes[1].set_xticks(day_ticks[::2])
    axes[1].set_xticklabels(day_labels[::2], rotation=45, ha='right')
    axes[1].set_xlabel('Day')

    # Compute summary stats
    for ax, trace, label in [
        (axes[0], ns_trace, 'NS'),
        (axes[1], sim_trace, 'Sim'),
    ]:
        bgs = np.array([bg for _, bg in trace])
        tir = np.sum((bgs >= 70) & (bgs <= 180)) / len(bgs) * 100
        tb70 = np.sum(bgs < 70) / len(bgs) * 100
        ta180 = np.sum(bgs > 180) / len(bgs) * 100
        mean_bg = np.mean(bgs)
        stats = f'Mean: {mean_bg:.0f}  TIR: {tir:.0f}%  <70: {tb70:.1f}%  >180: {ta180:.1f}%'
        ax.text(0.99, 0.97, stats, transform=ax.transAxes,
                ha='right', va='top', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    fig.suptitle('30-Day Comparison: Real Data vs Trio Simulation', fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f'Saved to {output_path}')
    else:
        plt.show()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Month comparison: Nightscout vs Trio')
    parser.add_argument('--ns-data', type=str, default=None,
                        help='Load saved raw Nightscout trace')
    parser.add_argument('--sim-data', type=str, default=None,
                        help='Load saved sim results')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--profile', type=str, default='patient_profiles/real_patient.json')
    parser.add_argument('--output', type=str, default='month_comparison.png')
    parser.add_argument('--ns-days', type=int, default=30)
    args = parser.parse_args()

    # --- Nightscout data ---
    if args.ns_data:
        ns_trace = load_raw_trace(args.ns_data)
        print(f'Loaded {len(ns_trace)} NS points from {args.ns_data}')
    else:
        ns_trace, ns_path = fetch_and_save_raw_entries(days=args.ns_days)

    # --- Trio simulation (30 independent days) ---
    if args.sim_data:
        results, meta = load_batch_results(args.sim_data)
        # Reconstruct stitched trace from independent day results
        sim_trace = []
        for day_idx, result in enumerate(results):
            day_offset_hrs = day_idx * 24
            for t_rel_min, bg in result.days[0].bg_trace:
                sim_trace.append((day_offset_hrs + t_rel_min / 60, bg))
        print(f'Loaded sim trace ({len(sim_trace)} points, {len(results)} days) from {args.sim_data}')
    else:
        profile = PatientProfile.from_json(args.profile)
        print(f'Running 30 independent Trio days...')
        sim_trace, results = run_independent_days(
            profile, 'trio', n_days=args.ns_days, seed=args.seed)

        # Save
        profile_label = Path(args.profile).stem
        label = f'trio_{args.ns_days}days_indep_{profile_label}_s{args.seed}'
        save_batch_results(results, label, {
            'algorithm': 'trio', 'n_paths': args.ns_days, 'n_days': 1,
            'seed': args.seed, 'profile': args.profile,
            'mode': 'independent_days',
        })

    plot_month_comparison(ns_trace, sim_trace, output_path=args.output)


if __name__ == '__main__':
    main()
