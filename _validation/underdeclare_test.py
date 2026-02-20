#!/usr/bin/env python3
"""
Under-declaration test: Declare 20g carbs but reality is 30g carbs.
Both systems get 2U insulin. BG evolves according to 30g reality.

This tests how Loop handles the mismatch between declared and actual carbs,
including DCA detection, IRC correction, and dosing response.
"""

import sys
import json
import time
import subprocess
import re
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from algorithms.loop.loop_algorithm import LoopAlgorithm
from algorithms.loop.insulin_models_exact import create_insulin_model, InsulinType
from algorithms.loop.carb_models import PiecewiseLinearCarbModel
from algorithms.base import AlgorithmInput

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

ISF = SETTINGS['insulin_sensitivity_factor']
CR = SETTINGS['carb_ratio']
CSF = ISF / CR


def generate_reality_bg(starting_bg, actual_carbs_g, insulin_units, duration_min=300):
    """
    Generate the 'reality' BG trajectory based on actual carbs and insulin.
    Uses the patient model (carb absorption + insulin absorption).
    """
    fiasp = create_insulin_model(InsulinType.FIASP, 6.0)
    carb_model = PiecewiseLinearCarbModel()

    bg_trajectory = [starting_bg]
    current_bg = starting_bg

    for t in range(0, duration_min, 5):
        # Carb effect this interval
        t_hours = t / 60.0
        t_next_hours = (t + 5) / 60.0
        abs_time = max(0, t_hours - 10.0 / 60.0)  # 10-min delay
        abs_time_next = max(0, t_next_hours - 10.0 / 60.0)
        pct_now = carb_model.percent_absorbed_at_time(abs_time, 3.0)
        pct_next = carb_model.percent_absorbed_at_time(abs_time_next, 3.0)
        carb_change = actual_carbs_g * CSF * (pct_next - pct_now)

        # Insulin effect this interval
        pct_ins_now = fiasp.percent_absorbed(t)
        pct_ins_next = fiasp.percent_absorbed(t + 5)
        ins_change = -insulin_units * ISF * (pct_ins_next - pct_ins_now)

        current_bg += carb_change + ins_change
        current_bg = max(40.0, min(400.0, current_bg))
        bg_trajectory.append(current_bg)

    return bg_trajectory


def run_python_cycle(bg_history_min, declared_carbs, insulin_doses):
    """Run one Python Loop cycle and return results."""
    loop = LoopAlgorithm(SETTINGS)

    current_bg = bg_history_min[-1][1]
    current_time = bg_history_min[-1][0]

    inp = AlgorithmInput(
        cgm_reading=current_bg,
        timestamp=current_time,
        cgm_history=bg_history_min,
        current_basal=SETTINGS['basal_rate'],
        bolus_history=insulin_doses,
        carb_entries=declared_carbs,
        settings=SETTINGS,
    )

    output = loop.recommend(inp)
    predictions = output.glucose_predictions.get('main', [])

    return {
        'eventual_bg': predictions[-1] if predictions else current_bg,
        'temp_basal': output.temp_basal_rate,
        'iob': output.iob,
        'cob': output.cob,
        'irc_total': output.irc_total_correction or 0,
        'momentum': output.momentum_effect_eventual or 0,
    }


def inject_to_ios(bg_history_sec, declared_carbs_sec, insulin_doses_sec, cycle_num):
    """Inject current state into iOS Loop via HealthKit."""
    result = subprocess.run(
        ['xcrun', 'simctl', 'get_app_container', SIMULATOR_ID, HEALTHKIT_BUNDLE, 'data'],
        capture_output=True, text=True, check=True)
    container = Path(result.stdout.strip())
    cd = container / "Documents" / "commands"
    sd = container / "Documents" / "scenarios"
    cd.mkdir(parents=True, exist_ok=True)
    sd.mkdir(parents=True, exist_ok=True)

    scenario = {
        'name': f'cycle_{cycle_num}',
        'glucoseSamples': [{'timestamp': ts, 'value': bg} for ts, bg in bg_history_sec],
        'carbEntries': declared_carbs_sec,
        'insulinDoses': insulin_doses_sec,
    }

    (sd / f'cycle_{cycle_num}.json').write_text(json.dumps(scenario))
    cmd = cd / f'i_{int(time.time() * 1000)}.json'
    cmd.write_text(json.dumps({'command': f'inject:cycle_{cycle_num}.json', 'timestamp': time.time()}))
    time.sleep(3)


def extract_ios_result():
    """Extract iOS Loop prediction from logs."""
    # Restart Loop to pick up new data
    subprocess.run(['xcrun', 'simctl', 'terminate', SIMULATOR_ID, LOOP_BUNDLE], stderr=subprocess.DEVNULL)
    time.sleep(1)
    subprocess.run(['xcrun', 'simctl', 'launch', SIMULATOR_ID, LOOP_BUNDLE], check=True)
    time.sleep(8)

    result = subprocess.run([
        'xcrun', 'simctl', 'spawn', SIMULATOR_ID, 'log', 'show',
        '--predicate', f'process == "Loop"', '--style', 'compact', '--last', '15s'
    ], capture_output=True, text=True, timeout=30)

    ios = {'eventual_bg': None, 'irc_impact': None, 'momentum_impact': None}
    for line in result.stdout.split('\n'):
        if '##LOOP##' not in line:
            continue
        if 'Eventual BG WITH IRC:' in line:
            m = re.search(r'(-?\d+\.?\d*) mg/dL', line)
            if m:
                ios['eventual_bg'] = float(m.group(1))
        # Get DCA COB
        if 'DCA remainingGrams' in line:
            m = re.search(r'(\d+\.?\d*)', line.split('remainingGrams')[1])
            if m:
                ios['cob'] = float(m.group(1))
    return ios


def main():
    starting_bg = 100.0
    actual_carbs = 30.0  # Reality
    declared_carbs = 20.0  # What Loop thinks
    insulin_units = 2.0
    duration_min = 300  # 5 hours

    print("=" * 80)
    print("UNDER-DECLARATION TEST")
    print(f"  Reality: {actual_carbs}g carbs + {insulin_units}U insulin")
    print(f"  Declared: {declared_carbs}g carbs + {insulin_units}U insulin")
    print(f"  Starting BG: {starting_bg}")
    print(f"  Duration: {duration_min // 60} hours")
    print("=" * 80)

    # Step 1: Generate reality BG trajectory
    print("\nGenerating reality BG trajectory (30g + 2U)...")
    reality_bg = generate_reality_bg(starting_bg, actual_carbs, insulin_units, duration_min)
    print(f"  BG at T+0: {reality_bg[0]:.0f}")
    print(f"  BG peak: {max(reality_bg):.0f} at T+{reality_bg.index(max(reality_bg)) * 5}m")
    print(f"  BG at T+{duration_min}: {reality_bg[-1]:.0f}")

    # Step 2: Setup iOS Loop
    print("\nSetting up iOS Loop...")
    subprocess.run(['xcrun', 'simctl', 'terminate', SIMULATOR_ID, LOOP_BUNDLE], stderr=subprocess.DEVNULL)
    subprocess.run(['xcrun', 'simctl', 'terminate', SIMULATOR_ID, HEALTHKIT_BUNDLE], stderr=subprocess.DEVNULL)
    time.sleep(2)
    subprocess.run(['xcrun', 'simctl', 'launch', SIMULATOR_ID, HEALTHKIT_BUNDLE], check=True)
    time.sleep(3)

    # Clear HealthKit
    result = subprocess.run(
        ['xcrun', 'simctl', 'get_app_container', SIMULATOR_ID, HEALTHKIT_BUNDLE, 'data'],
        capture_output=True, text=True, check=True)
    container = Path(result.stdout.strip())
    cd = container / "Documents" / "commands"
    cd.mkdir(parents=True, exist_ok=True)
    for _ in range(2):
        cmd = cd / f'c_{int(time.time() * 1000)}.json'
        cmd.write_text(json.dumps({'command': 'clear_all', 'timestamp': time.time()}))
        time.sleep(2)

    # Step 3: Run cycles
    # Use simulated time (5 min per cycle)
    sim_start = time.time()
    cycle_results = []

    # Build timestamps
    bg_history_sec = []  # [(unix_seconds, bg)] for iOS injection
    bg_history_min = []  # [(minutes, bg)] for Python Loop

    # Carb and insulin entries (fixed at T=0)
    carb_ts_sec = sim_start
    ins_ts_sec = sim_start
    declared_carbs_sec = [{'timestamp': carb_ts_sec, 'grams': declared_carbs, 'absorptionHours': 3.0}]
    insulin_doses_sec = [{'timestamp': ins_ts_sec, 'units': insulin_units}]

    carb_ts_min = sim_start / 60.0
    declared_carbs_min = [(carb_ts_min, declared_carbs, 3.0)]
    insulin_doses_min = [(carb_ts_min, insulin_units)]

    total_cycles = duration_min // 5

    header = f"{'Cyc':>3} {'Time':>5} {'BG':>6} {'Py eBG':>7} {'Py COB':>7} {'Py IOB':>6} {'Py IRC':>7} {'iOS eBG':>8} {'iOS COB':>8}"
    print(f"\n{header}")
    print("-" * len(header))

    for cycle in range(total_cycles):
        cycle_time_min = cycle * 5  # minutes since start

        # Get reality BG for this cycle
        bg_value = reality_bg[cycle]

        # Add to history (simulated time = sim_start + cycle*5min)
        sim_time_sec = sim_start + cycle_time_min * 60
        sim_time_min = sim_time_sec / 60.0

        bg_history_sec.append((sim_time_sec, bg_value))
        bg_history_min.append((sim_time_min, bg_value))

        # Only run prediction if we have at least 3 BG readings
        if len(bg_history_min) < 3:
            print(f"{cycle:3d} {cycle_time_min:4d}m {bg_value:6.1f}   (building history...)")
            continue

        # Run Python Loop
        py_result = run_python_cycle(bg_history_min, declared_carbs_min, insulin_doses_min)

        # Run iOS Loop every cycle (every 5 min, matching real Loop behavior)
        ios_result = {'eventual_bg': None, 'cob': None}
        if cycle >= 3:  # Need at least 3 BG readings
            inject_to_ios(bg_history_sec, declared_carbs_sec, insulin_doses_sec, cycle)
            ios_result = extract_ios_result()

        # Record
        cycle_data = {
            'cycle': cycle,
            'time_min': cycle_time_min,
            'bg': bg_value,
            'python': py_result,
            'ios': ios_result,
        }
        cycle_results.append(cycle_data)

        # Print
        ios_ebg_s = f"{ios_result['eventual_bg']:.1f}" if ios_result.get('eventual_bg') else "   ---"
        ios_cob_s = f"{ios_result['cob']:.1f}" if ios_result.get('cob') else "   ---"
        print(f"{cycle:3d} {cycle_time_min:4d}m {bg_value:6.1f} {py_result['eventual_bg']:7.1f} {py_result['cob']:7.1f} {py_result['iob']:6.2f} {py_result['irc_total']:+7.1f} {ios_ebg_s:>8} {ios_cob_s:>8}")

    # Save results
    results_file = Path(__file__).parent / "underdeclare_results.json"
    json_results = []
    for c in cycle_results:
        jr = {
            'cycle': c['cycle'],
            'time_min': c['time_min'],
            'bg': c['bg'],
            'python_eventual': c['python']['eventual_bg'],
            'python_cob': c['python']['cob'],
            'python_iob': c['python']['iob'],
            'python_irc': c['python']['irc_total'],
            'python_temp_basal': c['python']['temp_basal'],
        }
        if c['ios'].get('eventual_bg'):
            jr['ios_eventual'] = c['ios']['eventual_bg']
        if c['ios'].get('cob'):
            jr['ios_cob'] = c['ios']['cob']
        json_results.append(jr)

    results_file.write_text(json.dumps(json_results, indent=2))
    print(f"\nResults saved to: {results_file}")


if __name__ == '__main__':
    main()
