#!/usr/bin/env python3
"""
iOS Loop long-duration test: inject BG every 5 minutes real-time.
Declares 20g carbs + 2U insulin, feeds BG from 30g reality trajectory.
"""
import sys
import subprocess
import time
import json
import re
from pathlib import Path

SIMULATOR_ID = "4A0939FF-02C0-4000-843E-EEAE8BC727CC"
HEALTHKIT_BUNDLE = "com.test.healthkitinjector"
LOOP_BUNDLE = "com.Exercise.Loop"

def p(msg):
    print(msg, flush=True)

def main():
    # Read saved BG trajectory and fill in early values
    results = json.loads(open('underdeclare_results.json').read())
    bg_values = {r['time_min']: r['bg'] for r in results}
    # Fill in T+0 and T+5 which weren't in the original results
    bg_values[0] = 100.0
    bg_values[5] = 100.0

    # Setup
    p("Setting up...")
    subprocess.run(['xcrun', 'simctl', 'terminate', SIMULATOR_ID, LOOP_BUNDLE], stderr=subprocess.DEVNULL)
    subprocess.run(['xcrun', 'simctl', 'terminate', SIMULATOR_ID, HEALTHKIT_BUNDLE], stderr=subprocess.DEVNULL)
    time.sleep(2)
    subprocess.run(['xcrun', 'simctl', 'launch', SIMULATOR_ID, HEALTHKIT_BUNDLE], check=True)
    time.sleep(3)

    result = subprocess.run(['xcrun', 'simctl', 'get_app_container', SIMULATOR_ID, HEALTHKIT_BUNDLE, 'data'],
        capture_output=True, text=True, check=True)
    container = Path(result.stdout.strip())
    cd = container / "Documents" / "commands"; cd.mkdir(parents=True, exist_ok=True)
    sd = container / "Documents" / "scenarios"; sd.mkdir(parents=True, exist_ok=True)

    # Clear
    for _ in range(2):
        cmd = cd / f'c_{int(time.time()*1000)}.json'
        cmd.write_text(json.dumps({'command': 'clear_all', 'timestamp': time.time()}))
        time.sleep(2)
    p("Cleared HealthKit")

    # Inject initial: 20g carbs + 2U insulin + 3 baseline BGs
    now = time.time()
    initial = {
        'name': 'init',
        'glucoseSamples': [
            {'timestamp': now - 600, 'value': 100.0},
            {'timestamp': now - 300, 'value': 100.0},
            {'timestamp': now, 'value': 100.0},
        ],
        'carbEntries': [{'timestamp': now, 'grams': 20.0, 'absorptionHours': 3.0}],
        'insulinDoses': [{'timestamp': now, 'units': 2.0}]
    }
    (sd / 'init.json').write_text(json.dumps(initial))
    cmd = cd / f'i_{int(time.time()*1000)}.json'
    cmd.write_text(json.dumps({'command': 'inject:init.json', 'timestamp': time.time()}))
    time.sleep(3)
    p("Injected: 20g carbs + 2U insulin + 3 BG at 100")

    # Launch Loop
    subprocess.run(['xcrun', 'simctl', 'launch', SIMULATOR_ID, LOOP_BUNDLE], check=True)
    time.sleep(5)
    p("Loop launched")

    # Real-time BG injection loop
    ios_results = []
    p(f"\n{'Cyc':>3} {'Time':>5} {'BG':>6} {'iOS eBG':>8} {'iOS COB':>8} {'iOS IRC':>8}")
    p("-" * 50)

    for cycle in range(1, 40):
        cycle_time = cycle * 5
        bg = bg_values.get(cycle_time, 200.0)

        # Wait 5 minutes real time
        p(f"  ... waiting 5 min (cycle {cycle}, T+{cycle_time}m, next BG={bg:.1f}) ...")
        time.sleep(300)

        # Inject ONE new BG reading at current real time
        bg_now = time.time()
        bg_scenario = {
            'name': f'bg{cycle}',
            'glucoseSamples': [{'timestamp': bg_now, 'value': bg}],
            'carbEntries': [],
            'insulinDoses': []
        }
        (sd / f'bg{cycle}.json').write_text(json.dumps(bg_scenario))
        cmd = cd / f'i_{int(time.time()*1000)}.json'
        cmd.write_text(json.dumps({'command': f'inject:bg{cycle}.json', 'timestamp': time.time()}))
        time.sleep(3)

        # Restart Loop
        subprocess.run(['xcrun', 'simctl', 'terminate', SIMULATOR_ID, LOOP_BUNDLE], stderr=subprocess.DEVNULL)
        time.sleep(1)
        subprocess.run(['xcrun', 'simctl', 'launch', SIMULATOR_ID, LOOP_BUNDLE], check=True)
        time.sleep(8)

        # Extract
        result = subprocess.run(['xcrun', 'simctl', 'spawn', SIMULATOR_ID, 'log', 'show',
            '--predicate', f'process == "Loop"', '--style', 'compact', '--last', '15s'],
            capture_output=True, text=True, timeout=30)

        ios_ebg = None
        ios_cob = None
        ios_irc = None
        for line in result.stdout.split('\n'):
            if '##LOOP##' not in line:
                continue
            c = line.split('##LOOP##')[1].strip()
            if 'Eventual BG WITH IRC:' in c:
                m = re.search(r'(-?\d+\.?\d*) mg/dL', c)
                if m: ios_ebg = float(m.group(1))
            if 'remainingGrams (COB):' in c:
                m = re.search(r'(\d+\.?\d*)g', c)
                if m: ios_cob = float(m.group(1))
            if 'IRC totalCorrection:' in c:
                m = re.search(r'(-?\d+\.?\d*)', c.split('totalCorrection:')[1])
                if m: ios_irc = float(m.group(1))

        ebg_s = f'{ios_ebg:.1f}' if ios_ebg else '  ---'
        cob_s = f'{ios_cob:.1f}' if ios_cob else '  ---'
        irc_s = f'{ios_irc:+.1f}' if ios_irc else '  ---'
        p(f"{cycle:3d} {cycle_time:4d}m {bg:6.1f} {ebg_s:>8} {cob_s:>8} {irc_s:>8}")

        ios_results.append({
            'cycle': cycle, 'time_min': cycle_time, 'bg': bg,
            'ios_eventual': ios_ebg, 'ios_cob': ios_cob, 'ios_irc': ios_irc
        })

    # Save
    with open('ios_long_test_results.json', 'w') as f:
        json.dump(ios_results, f, indent=2)
    p(f"\nSaved to ios_long_test_results.json")

if __name__ == '__main__':
    main()
