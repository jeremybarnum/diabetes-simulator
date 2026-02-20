#!/usr/bin/env python3
"""
Multi-cycle closed-loop simulation: runs Python Loop and iOS Loop in parallel,
feeding BG readings every 5 minutes, executing dosing logic, watching IOB/COB evolve.

Usage:
    python3 multi_cycle_sim.py --variant exact    # 30g carbs + 3U exact
    python3 multi_cycle_sim.py --variant under    # 30g carbs + 2U underdose
    python3 multi_cycle_sim.py --variant over     # 30g carbs + 4U overdose
"""

import sys
import json
import time
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

sys.path.insert(0, str(Path(__file__).parent))

from algorithms.loop.loop_algorithm import LoopAlgorithm
from algorithms.base import AlgorithmInput


# iOS Loop interaction
SIMULATOR_ID = "4A0939FF-02C0-4000-843E-EEAE8BC727CC"
HEALTHKIT_BUNDLE = "com.test.healthkitinjector"
LOOP_BUNDLE = "com.Exercise.Loop"

SETTINGS = {
    'insulin_sensitivity_factor': 100.0,
    'carb_ratio': 10.0,
    'basal_rate': 0.45,
    'duration_of_insulin_action': 6.0,
    'target_range': (100, 100),
    'suspend_threshold': 80.0,
    'insulin_type': 'fiasp',
    'enable_momentum': True,
    'enable_irc': True,
    'enable_dca': True,
}


class SimulationState:
    """Track the evolving state of the simulation using simulated time."""

    def __init__(self, starting_bg: float, carb_grams: float, bolus_units: float):
        self.starting_bg = starting_bg
        self.sim_start = time.time()  # Real wall clock at sim start
        self.sim_time = self.sim_start  # Current simulated time (unix seconds)
        self.bg_history: List[Tuple[float, float]] = []  # [(sim_unix_seconds, bg)]
        self.carb_entries: List[Dict] = [{'timestamp': self.sim_time, 'grams': carb_grams, 'absorptionHours': 3.0}]
        self.insulin_doses: List[Dict] = [{'timestamp': self.sim_time, 'units': bolus_units}]
        self.temp_basals: List[Dict] = []
        self.cycle_results: List[Dict] = []

        # Add initial BG reading
        self.bg_history.append((self.sim_time, starting_bg))

    def advance_time(self, minutes: float = 5.0):
        """Advance simulated time by given minutes."""
        self.sim_time += minutes * 60  # Convert to seconds

    def add_bg(self, bg_value: float):
        """Add a new BG reading at current simulated time."""
        self.bg_history.append((self.sim_time, bg_value))

    def add_temp_basal(self, rate: float, duration_min: float = 5.0):
        """Record a temp basal action at simulated time."""
        self.temp_basals.append({
            'timestamp': self.sim_time,
            'rate': rate,
            'duration_min': duration_min,
        })
        # Convert to bolus equivalent for insulin tracking
        net_rate = rate - SETTINGS['basal_rate']
        net_units = net_rate * duration_min / 60.0
        if abs(net_units) > 0.001:
            self.insulin_doses.append({
                'timestamp': self.sim_time,
                'units': net_units,
            })


def run_python_cycle(state: SimulationState) -> Dict:
    """Run one cycle of Python Loop and return prediction + recommendation."""
    loop = LoopAlgorithm(SETTINGS)

    # Convert to minutes for Python Loop
    def to_min(ts):
        return ts / 60.0

    current_bg = state.bg_history[-1][1]
    current_time = to_min(state.bg_history[-1][0])

    cgm_history = [(to_min(ts), bg) for ts, bg in state.bg_history]
    bolus_history = [(to_min(d['timestamp']), d['units']) for d in state.insulin_doses]
    carb_entries = [(to_min(c['timestamp']), c['grams'], c.get('absorptionHours', 3.0))
                   for c in state.carb_entries]

    inp = AlgorithmInput(
        cgm_reading=current_bg,
        timestamp=current_time,
        cgm_history=cgm_history,
        current_basal=SETTINGS['basal_rate'],
        temp_basal=None,
        bolus_history=bolus_history,
        carb_entries=carb_entries,
        settings=SETTINGS,
    )

    output = loop.recommend(inp)

    predictions = output.glucose_predictions.get('main', [])

    return {
        'eventual_bg': predictions[-1] if predictions else current_bg,
        'predicted_bg_5min': predictions[1] if len(predictions) > 1 else current_bg,
        'temp_basal_rate': output.temp_basal_rate,
        'temp_basal_duration': output.temp_basal_duration,
        'iob': output.iob,
        'cob': output.cob,
        'predictions': predictions[:13],  # First hour
        'irc_effect': output.irc_effect_eventual,
        'momentum_effect': output.momentum_effect_eventual,
    }


def run_ios_cycle(state: SimulationState) -> Optional[Dict]:
    """
    Run one cycle of iOS Loop by injecting current state into HealthKit.
    Returns parsed prediction or None on failure.
    """
    import re

    # Get app container
    result = subprocess.run(
        ['xcrun', 'simctl', 'get_app_container', SIMULATOR_ID, HEALTHKIT_BUNDLE, 'data'],
        capture_output=True, text=True, check=True
    )
    container = Path(result.stdout.strip())
    cd = container / "Documents" / "commands"
    sd = container / "Documents" / "scenarios"
    cd.mkdir(parents=True, exist_ok=True)
    sd.mkdir(parents=True, exist_ok=True)

    # Create scenario from current state
    scenario = {
        'name': 'sim_cycle',
        'glucoseSamples': [{'timestamp': ts, 'value': bg} for ts, bg in state.bg_history],
        'carbEntries': state.carb_entries,
        'insulinDoses': state.insulin_doses,
    }

    (sd / 'sim_cycle.json').write_text(json.dumps(scenario))
    cmd = cd / f'i_{int(time.time()*1000)}.json'
    cmd.write_text(json.dumps({'command': 'inject:sim_cycle.json', 'timestamp': time.time()}))
    time.sleep(3)

    # Restart Loop to pick up new data
    subprocess.run(['xcrun', 'simctl', 'terminate', SIMULATOR_ID, LOOP_BUNDLE], stderr=subprocess.DEVNULL)
    time.sleep(1)
    subprocess.run(['xcrun', 'simctl', 'launch', SIMULATOR_ID, LOOP_BUNDLE], check=True)
    time.sleep(6)

    # Extract prediction
    result = subprocess.run([
        'xcrun', 'simctl', 'spawn', SIMULATOR_ID, 'log', 'show',
        '--predicate', f'process == "Loop"', '--style', 'compact', '--last', '15s'
    ], capture_output=True, text=True, timeout=30)

    ios_result = {'eventual_bg': None, 'momentum_impact': None, 'irc_impact': None}
    for line in result.stdout.split('\n'):
        if '##LOOP##' not in line:
            continue
        if 'Eventual BG WITH IRC:' in line:
            m = re.search(r'(-?\d+\.?\d*) mg/dL', line)
            if m:
                ios_result['eventual_bg'] = float(m.group(1))

    return ios_result


def run_simulation(variant: str = 'exact', duration_hours: float = 4.0, use_ios: bool = False):
    """
    Run the multi-cycle simulation.

    Args:
        variant: 'exact' (3U for 30g), 'under' (2U), 'over' (4U)
        duration_hours: How long to simulate
        use_ios: Whether to also run iOS Loop (slower)
    """
    # Determine dosing
    carb_grams = 30.0
    if variant == 'exact':
        bolus_units = carb_grams / SETTINGS['carb_ratio']  # 3.0U
    elif variant == 'under':
        bolus_units = 2.0  # Underdose
    elif variant == 'over':
        bolus_units = 4.0  # Overdose
    else:
        raise ValueError(f"Unknown variant: {variant}")

    starting_bg = 100.0
    state = SimulationState(starting_bg, carb_grams, bolus_units)

    total_cycles = int(duration_hours * 60 / 5)

    print(f"{'='*80}")
    print(f"MULTI-CYCLE SIMULATION: {variant} dosing")
    print(f"  30g carbs + {bolus_units}U insulin at T=0")
    print(f"  Starting BG: {starting_bg} mg/dL")
    print(f"  Duration: {duration_hours} hours ({total_cycles} cycles)")
    print(f"  iOS Loop: {'enabled' if use_ios else 'disabled'}")
    print(f"{'='*80}")
    print()

    header = f"{'Cycle':>5} {'Time':>6} {'BG':>6} {'Py eBG':>7} {'IOB':>5} {'COB':>5} {'TempBR':>6}"
    if use_ios:
        header += f" {'iOS eBG':>8} {'Diff':>6}"
    print(header)
    print("-" * len(header))

    for cycle in range(total_cycles):
        cycle_time = cycle * 5  # minutes since start

        # Run Python Loop
        py_result = run_python_cycle(state)

        # Run iOS Loop (if enabled)
        ios_result = None
        if use_ios:
            ios_result = run_ios_cycle(state)

        # Record cycle data
        cycle_data = {
            'cycle': cycle,
            'time_min': cycle_time,
            'bg': state.bg_history[-1][1],
            'python': py_result,
            'ios': ios_result,
        }
        state.cycle_results.append(cycle_data)

        # Print cycle summary
        bg = state.bg_history[-1][1]
        temp_rate = py_result.get('temp_basal_rate', SETTINGS['basal_rate'])
        line = f"{cycle:5d} {cycle_time:5d}m {bg:6.1f} {py_result['eventual_bg']:7.1f} {py_result['iob']:5.2f} {py_result['cob']:5.1f} {temp_rate:6.2f}"
        if use_ios and ios_result and ios_result.get('eventual_bg'):
            diff = py_result['eventual_bg'] - ios_result['eventual_bg']
            line += f" {ios_result['eventual_bg']:8.1f} {diff:+6.1f}"
        print(line)

        # Execute dosing action (apply temp basal)
        if temp_rate is not None:
            state.add_temp_basal(temp_rate, 5.0)

        # Evolve BG using simple patient model at SIMULATED time
        from algorithms.loop.insulin_models_exact import create_insulin_model, InsulinType
        fiasp_model = create_insulin_model(InsulinType.FIASP, 6.0)
        from algorithms.loop.carb_models import PiecewiseLinearCarbModel
        carb_model = PiecewiseLinearCarbModel()

        isf = SETTINGS['insulin_sensitivity_factor']
        csf = isf / SETTINGS['carb_ratio']
        sim_now = state.sim_time

        # Carb effect: sum of absorption changes over this 5-min interval
        carb_bg_change = 0.0
        for c in state.carb_entries:
            t_since_entry_min = (sim_now - c['timestamp']) / 60.0
            t_hours = t_since_entry_min / 60.0
            t_hours_next = (t_since_entry_min + 5) / 60.0
            abs_time = max(0, t_hours - 10.0/60.0)  # 10-min delay
            abs_time_next = max(0, t_hours_next - 10.0/60.0)
            pct_now = carb_model.percent_absorbed_at_time(abs_time, c.get('absorptionHours', 3.0))
            pct_next = carb_model.percent_absorbed_at_time(abs_time_next, c.get('absorptionHours', 3.0))
            carb_bg_change += c['grams'] * csf * (pct_next - pct_now)

        # Insulin effect: sum of absorption changes over this 5-min interval
        insulin_bg_change = 0.0
        for d in state.insulin_doses:
            t_since_dose_min = (sim_now - d['timestamp']) / 60.0
            if t_since_dose_min < 0:
                continue
            pct_now = fiasp_model.percent_absorbed(t_since_dose_min)
            pct_next = fiasp_model.percent_absorbed(t_since_dose_min + 5)
            insulin_bg_change += -d['units'] * isf * (pct_next - pct_now)

        next_bg = bg + carb_bg_change + insulin_bg_change
        next_bg = max(40.0, min(400.0, next_bg))

        # Advance simulated time by 5 minutes
        state.advance_time(5.0)
        state.add_bg(next_bg)

    # Save results
    results_file = Path(__file__).parent / f"sim_results_{variant}.json"
    json_results = []
    for c in state.cycle_results:
        jr = {
            'cycle': c['cycle'],
            'time_min': c['time_min'],
            'bg': c['bg'],
            'python_eventual': c['python']['eventual_bg'],
            'iob': c['python']['iob'],
            'cob': c['python']['cob'],
            'temp_basal': c['python'].get('temp_basal_rate'),
        }
        if c['ios'] and c['ios'].get('eventual_bg'):
            jr['ios_eventual'] = c['ios']['eventual_bg']
        json_results.append(jr)

    results_file.write_text(json.dumps(json_results, indent=2))
    print(f"\nResults saved to: {results_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Multi-cycle Loop simulation")
    parser.add_argument('--variant', choices=['exact', 'under', 'over'], default='exact')
    parser.add_argument('--hours', type=float, default=4.0)
    parser.add_argument('--ios', action='store_true', help='Also run iOS Loop (slower)')
    args = parser.parse_args()

    run_simulation(variant=args.variant, duration_hours=args.hours, use_ios=args.ios)
