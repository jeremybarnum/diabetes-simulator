#!/usr/bin/env python3
"""
Batch validation harness: Runs scenarios through iOS Loop AND Python Loop, compares results.

Workflow per scenario:
1. Force quit Loop
2. Launch HealthKitInjector (foreground)
3. Clear HealthKit data
4. Inject scenario
5. Launch Loop
6. Wait for prediction
7. Extract iOS Loop results from logs
8. Run Python Loop with same scenario
9. Compare and record results
"""

import sys
import json
import subprocess
import time
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

SIMULATOR_ID = "4A0939FF-02C0-4000-843E-EEAE8BC727CC"
HEALTHKIT_BUNDLE = "com.test.healthkitinjector"
LOOP_BUNDLE = "com.Exercise.Loop"


def get_app_container() -> Path:
    result = subprocess.run(
        ['xcrun', 'simctl', 'get_app_container', SIMULATOR_ID, HEALTHKIT_BUNDLE, 'data'],
        capture_output=True, text=True, check=True
    )
    return Path(result.stdout.strip())


def send_command(container: Path, command: str, wait_seconds: float = 3.0) -> bool:
    commands_dir = container / "Documents" / "commands"
    commands_dir.mkdir(parents=True, exist_ok=True)
    command_file = commands_dir / f"cmd_{int(time.time() * 1000)}.json"
    command_file.write_text(json.dumps({"command": command, "timestamp": time.time()}))

    start = time.time()
    while command_file.exists() and (time.time() - start) < wait_seconds:
        time.sleep(0.1)

    if command_file.exists():
        command_file.unlink()
        return False
    return True


def run_ios_loop(scenario: Dict, scenario_name: str, keep_alive: bool = False) -> Optional[Dict]:
    """
    Run scenario through actual iOS Loop using mechanical workflow.
    Returns parsed results or None on failure.

    Args:
        keep_alive: If True, don't terminate Loop between tests. This preserves
                   Xcode debugger attachment and log streaming. Instead, clears
                   data and injects new scenario while Loop is running.
    """
    if not keep_alive:
        # Original flow: terminate and relaunch
        subprocess.run(['xcrun', 'simctl', 'terminate', SIMULATOR_ID, LOOP_BUNDLE],
                       stderr=subprocess.DEVNULL)
        time.sleep(1)

    # Step 2: Open injector (idempotent if already running)
    subprocess.run(['xcrun', 'simctl', 'launch', SIMULATOR_ID, HEALTHKIT_BUNDLE],
                   check=True, stderr=subprocess.DEVNULL)
    time.sleep(2)

    container = get_app_container()
    scenarios_dir = container / "Documents" / "scenarios"
    scenarios_dir.mkdir(parents=True, exist_ok=True)

    # Step 3: Clear data (send twice with longer wait for reliable clearing)
    if not send_command(container, "clear_all"):
        print(f"  ⚠️  Clear command may not have been processed")
    time.sleep(3)
    # Second clear to ensure all data is gone
    if not send_command(container, "clear_all"):
        pass
    time.sleep(3)

    # Step 4: Write and inject scenario
    scenario_file = scenarios_dir / f"{scenario_name}.json"
    scenario_file.write_text(json.dumps(scenario, indent=2))
    if not send_command(container, f"inject:{scenario_name}.json"):
        print(f"  ⚠️  Inject command may not have been processed")
    time.sleep(2)

    if not keep_alive:
        # Step 5: Launch Loop fresh
        subprocess.run(['xcrun', 'simctl', 'launch', SIMULATOR_ID, LOOP_BUNDLE], check=True)
    else:
        # Loop is already running — ensure it's open (no-op if already foreground)
        subprocess.run(['xcrun', 'simctl', 'launch', SIMULATOR_ID, LOOP_BUNDLE],
                       stderr=subprocess.DEVNULL)

    time.sleep(8 if keep_alive else 6)  # Extra time in keep-alive for Loop to detect new data

    # Step 6: Extract logs
    result = subprocess.run([
        'xcrun', 'simctl', 'spawn', SIMULATOR_ID,
        'log', 'show',
        '--predicate', f'process == "Loop"',
        '--style', 'compact',
        '--last', '20s'
    ], capture_output=True, text=True, timeout=30)

    log_lines = [l for l in result.stdout.split('\n') if '##LOOP##' in l]

    # Save raw logs for post-hoc inspection
    logs_dir = Path(__file__).parent / "ios_logs"
    logs_dir.mkdir(exist_ok=True)
    (logs_dir / f"{scenario_name}.log").write_text('\n'.join(log_lines))

    if not log_lines:
        return None

    # Parse results
    parsed = {
        'eventual_bg': None,
        'momentum_impact': None,
        'irc_impact': None,
        'predictions': [],
        'momentum_effects': [],
        'starting_bg': None,
        'therapy_settings': {},
    }

    for i, line in enumerate(log_lines):
        # Therapy settings (from new logging block)
        if '=== THERAPY SETTINGS ===' in line and 'END' not in line:
            for j in range(i+1, min(i+20, len(log_lines))):
                sl = log_lines[j]
                if 'END THERAPY SETTINGS' in sl:
                    break
                m = re.search(r'ISF:\s*([\d.]+)', sl)
                if m: parsed['therapy_settings']['isf'] = float(m.group(1))
                m = re.search(r'CR:\s*([\d.]+)', sl)
                if m: parsed['therapy_settings']['cr'] = float(m.group(1))
                m = re.search(r'(?<!Max )Basal:\s*([\d.]+)', sl)
                if m: parsed['therapy_settings']['basal'] = float(m.group(1))
                m = re.search(r'Target:\s*([\d.]+)-([\d.]+)', sl)
                if m: parsed['therapy_settings']['target'] = (float(m.group(1)), float(m.group(2)))
                m = re.search(r'Max Basal:\s*([\d.]+)', sl)
                if m: parsed['therapy_settings']['max_basal'] = float(m.group(1))
                m = re.search(r'Max Bolus:\s*([\d.]+)', sl)
                if m: parsed['therapy_settings']['max_bolus'] = float(m.group(1))
                m = re.search(r'Suspend Threshold:\s*([\d.]+)', sl)
                if m: parsed['therapy_settings']['suspend'] = float(m.group(1))
                m = re.search(r'Insulin Model:\s*(\S+)', sl)
                if m: parsed['therapy_settings']['insulin_model'] = m.group(1)
                m = re.search(r'DIA:\s*([\d.]+)', sl)
                if m: parsed['therapy_settings']['dia'] = float(m.group(1))

        # Dosing / loop mode
        if 'Dosing enabled:' in line:
            parsed['dosing_enabled'] = 'true' in line.lower().split('Dosing enabled:')[-1]
        if 'Automatic dosing strategy:' in line:
            m = re.search(r'Automatic dosing strategy:\s*(\S+)', line)
            if m: parsed['dosing_strategy'] = m.group(1)

        # Starting glucose
        if 'Starting glucose:' in line and 'Algorithm settings' not in line:
            match = re.search(r'Starting glucose: (\d+\.?\d*)', line)
            if match:
                parsed['starting_bg'] = float(match.group(1))

        # Momentum impact analysis (use the last one in log - the definitive one)
        if '=== MOMENTUM IMPACT ANALYSIS ===' in line:
            for j in range(i+1, min(i+10, len(log_lines))):
                if 'Momentum net impact on eventual BG:' in log_lines[j]:
                    match = re.search(r'([+-]?\d+\.?\d*)\s*mg/dL', log_lines[j])
                    if match:
                        parsed['momentum_impact'] = float(match.group(1))
                if 'Eventual BG WITH momentum:' in log_lines[j]:
                    match = re.search(r'(-?\d+\.?\d*)\s*mg/dL', log_lines[j])
                    if match:
                        parsed['eventual_bg'] = float(match.group(1))

        # IRC impact analysis
        if '=== IRC IMPACT ANALYSIS ===' in line:
            for j in range(i+1, min(i+10, len(log_lines))):
                if 'IRC net impact on eventual BG:' in log_lines[j]:
                    match = re.search(r'([+-]?\d+\.?\d*)\s*mg/dL', log_lines[j])
                    if match:
                        parsed['irc_impact'] = float(match.group(1))
                if 'Eventual BG WITH IRC:' in log_lines[j]:
                    match = re.search(r'(-?\d+\.?\d*)\s*mg/dL', log_lines[j])
                    if match:
                        parsed['eventual_bg'] = float(match.group(1))

        # Prediction points
        if 'Predictions (count=' in line:
            parsed['predictions'] = []
            for j in range(i+1, min(i+200, len(log_lines))):
                pred_match = re.search(r'\[(\d+)\].*?:\s*(-?\d+\.?\d*)\s*mg/dL', log_lines[j])
                if pred_match:
                    parsed['predictions'].append(float(pred_match.group(2)))
                elif 'Eventual BG:' in log_lines[j] or '===' in log_lines[j]:
                    break

        # Momentum effects
        if 'Momentum effects (total=' in line:
            parsed['momentum_effects'] = []
            for j in range(i+1, min(i+20, len(log_lines))):
                mom_match = re.search(r'\[(\d+)\].*?:\s*(-?\d+\.?\d*)\s*mg/dL', log_lines[j])
                if mom_match:
                    parsed['momentum_effects'].append(float(mom_match.group(2)))
                elif 'Carbs effects' in log_lines[j] or '===' in log_lines[j]:
                    break

    return parsed


def load_settings() -> Dict:
    """Load therapy settings from settings.json (single source of truth)."""
    settings_file = Path(__file__).parent.parent / "settings.json"
    with open(settings_file) as f:
        raw = json.load(f)
    # Convert settings.json format to algorithm format
    target = raw.get('target', 100.0)
    return {
        'insulin_sensitivity_factor': raw['insulin_sensitivity_factor'],
        'carb_ratio': raw['carb_ratio'],
        'basal_rate': raw['basal_rate'],
        'duration_of_insulin_action': raw['duration_of_insulin_action'],
        'target_range': (target, target),
        'suspend_threshold': raw.get('suspend_threshold', 80.0),
        'max_basal_rate': raw.get('max_basal_rate', 2.8),
        'max_bolus': raw.get('max_bolus', 3.0),
        'insulin_type': raw.get('insulin_type', 'fiasp'),
        'enable_momentum': raw.get('enable_momentum', True),
        'enable_irc': raw.get('enable_irc', True),
        'enable_dca': raw.get('enable_dca', True),
    }


def validate_scenario(scenario: Dict, settings: Dict) -> List[str]:
    """Check if a scenario violates settings constraints. Returns list of warnings."""
    warnings = []
    max_bolus = settings.get('max_bolus', 3.0)
    for dose in scenario.get('insulinDoses', []):
        if dose['units'] > max_bolus:
            warnings.append(
                f"Bolus {dose['units']:.1f}U exceeds max_bolus {max_bolus:.1f}U"
            )
    return warnings


def compare_settings(ios_therapy: Dict, python_settings: Dict) -> List[str]:
    """
    Compare iOS therapy settings (from logs) against Python settings.
    Returns list of mismatches. Empty list means settings match.
    """
    mismatches = []
    if not ios_therapy:
        return ["No iOS therapy settings found in logs"]

    checks = [
        ('isf', 'insulin_sensitivity_factor', 'ISF', 0.1),
        ('cr', 'carb_ratio', 'CR', 0.1),
        ('basal', 'basal_rate', 'Basal', 0.01),
        ('max_basal', 'max_basal_rate', 'Max Basal', 0.01),
        ('max_bolus', 'max_bolus', 'Max Bolus', 0.01),
        ('suspend', 'suspend_threshold', 'Suspend', 0.1),
        # DIA removed: iOS reports effectDuration (actionDuration+delay=370min=6.167hr)
        # while Python stores actionDuration (360min=6.0hr). Same insulin model.
    ]

    for ios_key, py_key, label, tol in checks:
        ios_val = ios_therapy.get(ios_key)
        py_val = python_settings.get(py_key)
        if ios_val is not None and py_val is not None:
            if abs(ios_val - py_val) > tol:
                mismatches.append(f"{label}: iOS={ios_val} vs Python={py_val}")

    # Target range
    ios_target = ios_therapy.get('target')
    py_target = python_settings.get('target_range')
    if ios_target and py_target:
        if abs(ios_target[0] - py_target[0]) > 0.1 or abs(ios_target[1] - py_target[1]) > 0.1:
            mismatches.append(f"Target: iOS={ios_target} vs Python={py_target}")

    return mismatches


def run_python_loop(scenario: Dict) -> Optional[Dict]:
    """Run scenario through Python Loop and return results."""
    try:
        from algorithms.loop.loop_algorithm import LoopAlgorithm
        from algorithms.base import AlgorithmInput

        settings = scenario.get('settings', load_settings())

        loop = LoopAlgorithm(settings)

        glucose_samples = scenario['glucoseSamples']
        current_bg = glucose_samples[-1]['value']

        # CRITICAL: Convert unix timestamps (seconds) to minutes for Python Loop
        def to_minutes(ts):
            return ts / 60.0

        current_time = to_minutes(glucose_samples[-1]['timestamp'])

        cgm_history = [(to_minutes(s['timestamp']), s['value']) for s in glucose_samples]
        bolus_history = [(to_minutes(d['timestamp']), d['units']) for d in scenario.get('insulinDoses', [])]
        carb_entries = [(to_minutes(c['timestamp']), c['grams'], c.get('absorptionHours', 3.0))
                       for c in scenario.get('carbEntries', [])]

        def make_input(s):
            return AlgorithmInput(
                cgm_reading=current_bg,
                timestamp=current_time,
                cgm_history=cgm_history,
                current_basal=s['basal_rate'],
                temp_basal=None,
                bolus_history=bolus_history,
                carb_entries=carb_entries,
                settings=s
            )

        output = loop.recommend(make_input(settings))

        predictions = output.glucose_predictions.get('main', [])
        eventual_bg = predictions[-1] if predictions else current_bg

        # Calculate momentum impact: run without momentum and compare
        settings_no_momentum = dict(settings)
        settings_no_momentum['enable_momentum'] = False
        loop_no_mom = LoopAlgorithm(settings_no_momentum)
        out_no_mom = loop_no_mom.recommend(make_input(settings_no_momentum))
        preds_no_mom = out_no_mom.glucose_predictions.get('main', [])
        ebg_no_mom = preds_no_mom[-1] if preds_no_mom else current_bg
        momentum_impact = eventual_bg - ebg_no_mom

        # Calculate IRC impact: run without IRC and compare
        settings_no_irc = dict(settings)
        settings_no_irc['enable_irc'] = False
        loop_no_irc = LoopAlgorithm(settings_no_irc)
        out_no_irc = loop_no_irc.recommend(make_input(settings_no_irc))
        preds_no_irc = out_no_irc.glucose_predictions.get('main', [])
        ebg_no_irc = preds_no_irc[-1] if preds_no_irc else current_bg
        irc_impact = eventual_bg - ebg_no_irc

        return {
            'eventual_bg': eventual_bg,
            'momentum_impact': momentum_impact,
            'irc_impact': irc_impact,
            'predictions': predictions,
            'starting_bg': current_bg,
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None


def create_scenario(name: str, description: str,
                    bg_readings: List[Tuple[float, float]],
                    carb_entries: List[Dict] = None,
                    insulin_doses: List[Dict] = None,
                    reference_time: float = None) -> Dict:
    """
    Create a scenario dict with relative timestamps.
    bg_readings: [(minutes_ago, value), ...] - negative means in the past
    carb_entries: [{'minutes_ago': float, 'grams': float, 'absorptionHours': float}, ...]
    insulin_doses: [{'minutes_ago': float, 'units': float}, ...]
    reference_time: fixed unix timestamp (defaults to time.time() for live runs)
    """
    now = reference_time if reference_time is not None else time.time()

    scenario = {
        'name': name,
        'description': description,
        'glucoseSamples': [
            {'timestamp': now - mins_ago * 60, 'value': value}
            for mins_ago, value in bg_readings
        ],
        'carbEntries': [],
        'insulinDoses': []
    }

    if carb_entries:
        for c in carb_entries:
            scenario['carbEntries'].append({
                'timestamp': now - c['minutes_ago'] * 60,
                'grams': c['grams'],
                'absorptionHours': c.get('absorptionHours', 3.0)
            })

    if insulin_doses:
        for d in insulin_doses:
            scenario['insulinDoses'].append({
                'timestamp': now - d['minutes_ago'] * 60,
                'units': d['units']
            })

    return scenario


def run_test(name: str, scenario: Dict, results_log: List[Dict],
             keep_alive: bool = False) -> Dict:
    """Run a single test scenario through both loops and compare."""
    print(f"\n{'='*80}")
    print(f"TEST: {name}")
    print(f"{'='*80}")

    # Validate scenario against settings constraints
    settings = load_settings()
    warnings = validate_scenario(scenario, settings)
    for w in warnings:
        print(f"  ⚠️  SCENARIO WARNING: {w}")

    # Run iOS Loop
    print("  📱 Running iOS Loop...")
    ios_result = run_ios_loop(scenario, name.replace(' ', '_').lower(),
                              keep_alive=keep_alive)

    # Compare iOS therapy settings against Python settings (first test only to avoid spam)
    if ios_result and ios_result.get('therapy_settings'):
        mismatches = compare_settings(ios_result['therapy_settings'], settings)
        if mismatches:
            print(f"  ❌ SETTINGS MISMATCH between iOS and Python:")
            for m in mismatches:
                print(f"       {m}")
        else:
            print(f"  ✅ iOS therapy settings match Python settings")

    # Check loop mode (open vs closed)
    if ios_result and ios_result.get('dosing_enabled'):
        print(f"  ⚠️  WARNING: iOS Loop is CLOSED LOOP (dosing enabled) - temp basals may contaminate results")

    if ios_result is None or ios_result.get('eventual_bg') is None:
        print("  ❌ iOS Loop failed to produce prediction - skipping")
        result = {
            'name': name,
            'status': 'IOS_FAILED',
            'ios': None,
            'python': None,
            'diffs': {},
        }
        results_log.append(result)
        return result

    print(f"  ✅ iOS Loop: eventual BG = {ios_result['eventual_bg']:.1f} mg/dL")
    if ios_result['momentum_impact'] is not None:
        print(f"     Momentum impact: {ios_result['momentum_impact']:+.1f} mg/dL")
    if ios_result['irc_impact'] is not None:
        print(f"     IRC impact: {ios_result['irc_impact']:+.1f} mg/dL")

    # Run Python Loop
    print("  🐍 Running Python Loop...")
    python_result = run_python_loop(scenario)

    if python_result is None:
        print("  ❌ Python Loop failed")
        result = {
            'name': name,
            'status': 'PYTHON_FAILED',
            'ios': ios_result,
            'python': None,
            'diffs': {},
        }
        results_log.append(result)
        return result

    print(f"  ✅ Python Loop: eventual BG = {python_result['eventual_bg']:.1f} mg/dL")

    # Compare
    diffs = {}
    if ios_result['eventual_bg'] is not None and python_result['eventual_bg'] is not None:
        diffs['eventual_bg'] = python_result['eventual_bg'] - ios_result['eventual_bg']
    if ios_result.get('momentum_impact') is not None:
        diffs['momentum'] = (python_result.get('momentum_impact', 0) or 0) - ios_result['momentum_impact']
    if ios_result.get('irc_impact') is not None:
        diffs['irc'] = (python_result.get('irc_impact', 0) or 0) - ios_result['irc_impact']

    ebg_diff = diffs.get('eventual_bg', 0)
    passed = abs(ebg_diff) <= 2.0

    status = 'PASS' if passed else 'FAIL'
    print(f"\n  📊 RESULT: {status}")
    print(f"     Eventual BG diff: {ebg_diff:+.1f} mg/dL (iOS={ios_result['eventual_bg']:.1f}, Python={python_result['eventual_bg']:.1f})")
    if 'momentum' in diffs:
        print(f"     Momentum diff: {diffs['momentum']:+.1f} mg/dL")
    if 'irc' in diffs:
        print(f"     IRC diff: {diffs['irc']:+.1f} mg/dL")

    result = {
        'name': name,
        'status': status,
        'ios': ios_result,
        'python': python_result,
        'diffs': diffs,
    }
    results_log.append(result)
    return result


def define_test_scenarios(reference_time: float = None) -> List[Tuple[str, Dict]]:
    """Define all test scenarios.

    Args:
        reference_time: Fixed unix timestamp for deterministic results.
                       Defaults to None (uses time.time() for live iOS runs).
    """
    scenarios = []
    rt = reference_time  # shorthand

    # Test 1: Flat BG baseline (3 readings, no momentum expected)
    scenarios.append(("01_flat_bg", create_scenario(
        "Flat BG Baseline",
        "3 flat readings at 120 mg/dL, no carbs, 0.2U trigger at 180m for ICE",
        bg_readings=[(15, 120.0), (10, 120.0), (5, 120.0)],
        insulin_doses=[{'minutes_ago': 180, 'units': 0.2}],
        reference_time=rt,
    )))

    # Test 2: Rising BG 1 mg/dL/min (momentum test)
    scenarios.append(("02_rising_1mg_min", create_scenario(
        "Rising 1 mg/dL/min",
        "4 readings: 100→105→110→115, 1 mg/dL/min, 0.2U trigger at 180m for ICE",
        bg_readings=[(15, 100.0), (10, 105.0), (5, 110.0), (0.5, 115.0)],
        insulin_doses=[{'minutes_ago': 180, 'units': 0.2}],
        reference_time=rt,
    )))

    # Test 3: Rising BG 2 mg/dL/min (momentum scaling test)
    scenarios.append(("03_rising_2mg_min", create_scenario(
        "Rising 2 mg/dL/min",
        "4 readings: 100→110→120→130, 2 mg/dL/min, 0.2U trigger at 180m for ICE",
        bg_readings=[(15, 100.0), (10, 110.0), (5, 120.0), (0.5, 130.0)],
        insulin_doses=[{'minutes_ago': 180, 'units': 0.2}],
        reference_time=rt,
    )))

    # Test 4: Falling BG -1 mg/dL/min
    scenarios.append(("04_falling_1mg_min", create_scenario(
        "Falling 1 mg/dL/min",
        "4 readings: 130→125→120→115, -1 mg/dL/min, 0.2U trigger at 180m for ICE",
        bg_readings=[(15, 130.0), (10, 125.0), (5, 120.0), (0.5, 115.0)],
        insulin_doses=[{'minutes_ago': 180, 'units': 0.2}],
        reference_time=rt,
    )))

    # Test 5: Single insulin bolus (2U, 60 min ago) with 3hr BG history
    # BG has been flat at 150 for 3 hours despite insulin
    bg_3hr_flat_150 = [(i*5, 150.0) for i in range(36, 0, -1)] + [(0.5, 150.0)]
    scenarios.append(("05_insulin_only", create_scenario(
        "Single Insulin Bolus",
        "Flat BG at 150 for 3hrs, 2U bolus 60 min ago",
        bg_readings=bg_3hr_flat_150,
        insulin_doses=[{'minutes_ago': 60, 'units': 2.0}],
        reference_time=rt,
    )))

    # Test 6: Single carb entry (30g, 60 min ago)
    scenarios.append(("06_carbs_only", create_scenario(
        "Single Carb Entry",
        "Flat BG at 100, 30g carbs 60 min ago, 0.2U trigger at 180m for ICE",
        bg_readings=[(15, 100.0), (10, 100.0), (5, 100.0), (0.5, 100.0)],
        carb_entries=[{'minutes_ago': 60, 'grams': 30.0}],
        insulin_doses=[{'minutes_ago': 180, 'units': 0.2}],
        reference_time=rt,
    )))

    # Test 7: Carbs + insulin together
    scenarios.append(("07_carbs_and_insulin", create_scenario(
        "Carbs + Insulin",
        "BG 120, 30g carbs + 3U bolus 30 min ago",
        bg_readings=[(15, 120.0), (10, 120.0), (5, 120.0), (0.5, 120.0)],
        carb_entries=[{'minutes_ago': 30, 'grams': 30.0}],
        insulin_doses=[{'minutes_ago': 30, 'units': 3.0}],
        reference_time=rt,
    )))

    # Test 8: Rising BG + insulin (momentum + insulin interaction)
    scenarios.append(("08_rising_bg_with_insulin", create_scenario(
        "Rising BG + Insulin",
        "Rising BG 1mg/min + 0.1U insulin 15 min ago",
        bg_readings=[(15, 100.0), (10, 105.0), (5, 110.0), (0.5, 115.0)],
        insulin_doses=[{'minutes_ago': 15, 'units': 0.1}],
        reference_time=rt,
    )))

    # Test 9: Large insulin dose with 3hr history
    bg_3hr_flat_200 = [(i*5, 200.0) for i in range(36, 0, -1)] + [(0.5, 200.0)]
    scenarios.append(("09_large_insulin", create_scenario(
        "Large Insulin No Carbs",
        "BG 200 for 3hrs, 3U bolus 30 min ago, no carbs",
        bg_readings=bg_3hr_flat_200,
        insulin_doses=[{'minutes_ago': 30, 'units': 3.0}],
        reference_time=rt,
    )))

    # Test 10: Future carbs (not yet absorbed)
    scenarios.append(("10_future_carbs", create_scenario(
        "Recently Entered Carbs",
        "BG 100, 50g carbs just entered (5 min ago), 0.2U trigger at 180m for ICE",
        bg_readings=[(15, 100.0), (10, 100.0), (5, 100.0), (0.5, 100.0)],
        carb_entries=[{'minutes_ago': 5, 'grams': 50.0}],
        insulin_doses=[{'minutes_ago': 180, 'units': 0.2}],
        reference_time=rt,
    )))

    # Test 11: Multiple carb entries
    scenarios.append(("11_multiple_carbs", create_scenario(
        "Multiple Carb Entries",
        "BG 100, 20g at 90min ago + 30g at 30min ago, 0.2U trigger at 180m for ICE",
        bg_readings=[(15, 100.0), (10, 100.0), (5, 100.0), (0.5, 100.0)],
        carb_entries=[
            {'minutes_ago': 90, 'grams': 20.0},
            {'minutes_ago': 30, 'grams': 30.0},
        ],
        insulin_doses=[{'minutes_ago': 180, 'units': 0.2}],
        reference_time=rt,
    )))

    # Test 12: Multiple insulin doses
    scenarios.append(("12_multiple_insulin", create_scenario(
        "Multiple Insulin Doses",
        "BG 180, 2U at 120min ago + 3U at 30min ago",
        bg_readings=[(15, 180.0), (10, 180.0), (5, 180.0), (0.5, 180.0)],
        insulin_doses=[
            {'minutes_ago': 120, 'units': 2.0},
            {'minutes_ago': 30, 'units': 3.0},
        ],
        reference_time=rt,
    )))

    # Test 13: BG rising despite insulin (IRC should activate)
    scenarios.append(("13_rising_despite_insulin", create_scenario(
        "Rising Despite Insulin (IRC Test)",
        "BG rising from 120→140 despite 2U bolus 60 min ago",
        bg_readings=[
            (60, 120.0), (55, 122.0), (50, 124.0), (45, 126.0),
            (40, 128.0), (35, 130.0), (30, 132.0), (25, 134.0),
            (20, 136.0), (15, 138.0), (10, 140.0), (5, 142.0), (0.5, 144.0)
        ],
        insulin_doses=[{'minutes_ago': 60, 'units': 2.0}],
        reference_time=rt,
    )))

    # Test 14: BG falling faster than expected (negative IRC)
    scenarios.append(("14_falling_fast", create_scenario(
        "Falling Faster Than Expected",
        "BG falling from 200→150 with only 1U bolus 60 min ago",
        bg_readings=[
            (60, 200.0), (55, 196.0), (50, 192.0), (45, 188.0),
            (40, 184.0), (35, 180.0), (30, 176.0), (25, 172.0),
            (20, 168.0), (15, 164.0), (10, 160.0), (5, 156.0), (0.5, 152.0)
        ],
        insulin_doses=[{'minutes_ago': 60, 'units': 1.0}],
        reference_time=rt,
    )))

    # Test 15: Meal scenario - carbs + bolus + rising BG
    scenarios.append(("15_meal_scenario", create_scenario(
        "Typical Meal",
        "30g meal bolused 45 min ago, BG rising",
        bg_readings=[
            (60, 110.0), (55, 110.0), (50, 112.0), (45, 115.0),
            (40, 120.0), (35, 128.0), (30, 135.0), (25, 140.0),
            (20, 143.0), (15, 145.0), (10, 146.0), (5, 146.0), (0.5, 145.0)
        ],
        carb_entries=[{'minutes_ago': 45, 'grams': 30.0}],
        insulin_doses=[{'minutes_ago': 45, 'units': 3.0}],
        reference_time=rt,
    )))

    # Test 16: Complex multi-carb meal
    scenarios.append(("16_multi_carb_meal", create_scenario(
        "Multi-Carb Meal",
        "20g appetizer 90min ago, 30g main 60min ago, 2U+3U boluses",
        bg_readings=[
            (60, 100.0), (55, 105.0), (50, 112.0), (45, 120.0),
            (40, 128.0), (35, 133.0), (30, 136.0), (25, 137.0),
            (20, 136.0), (15, 134.0), (10, 131.0), (5, 128.0), (0.5, 125.0)
        ],
        carb_entries=[
            {'minutes_ago': 90, 'grams': 20.0},
            {'minutes_ago': 60, 'grams': 30.0},
        ],
        insulin_doses=[
            {'minutes_ago': 90, 'units': 2.0},
            {'minutes_ago': 60, 'units': 3.0},
        ],
        reference_time=rt,
    )))

    # Test 17: Low BG near suspend threshold
    scenarios.append(("17_low_bg", create_scenario(
        "Low BG Near Suspend",
        "BG at 75 and falling, 0.2U trigger at 180m for ICE",
        bg_readings=[(15, 85.0), (10, 82.0), (5, 78.0), (0.5, 75.0)],
        insulin_doses=[{'minutes_ago': 180, 'units': 0.2}],
        reference_time=rt,
    )))

    # Test 18: High BG steady state
    scenarios.append(("18_high_bg_steady", create_scenario(
        "High BG Steady",
        "BG flat at 250, 0.2U trigger at 180m for ICE",
        bg_readings=[(15, 250.0), (10, 250.0), (5, 250.0), (0.5, 250.0)],
        insulin_doses=[{'minutes_ago': 180, 'units': 0.2}],
        reference_time=rt,
    )))

    # Test 19: Correction bolus in progress
    scenarios.append(("19_correction_in_progress", create_scenario(
        "Correction Bolus In Progress",
        "BG 220 falling after 3U correction 45 min ago",
        bg_readings=[
            (60, 240.0), (55, 238.0), (50, 235.0), (45, 232.0),
            (40, 228.0), (35, 224.0), (30, 220.0), (25, 216.0),
            (20, 213.0), (15, 210.0), (10, 208.0), (5, 206.0), (0.5, 204.0)
        ],
        insulin_doses=[{'minutes_ago': 45, 'units': 3.0}],
        reference_time=rt,
    )))

    # Test 20: Post-meal with future carbs remaining
    scenarios.append(("20_post_meal_future_carbs", create_scenario(
        "Post-Meal Slow Carbs",
        "30g slow carbs (4hr absorption) eaten 30 min ago, BG starting to rise",
        bg_readings=[
            (45, 100.0), (40, 100.0), (35, 101.0), (30, 103.0),
            (25, 106.0), (20, 110.0), (15, 114.0), (10, 118.0), (5, 121.0), (0.5, 123.0)
        ],
        carb_entries=[{'minutes_ago': 30, 'grams': 30.0, 'absorptionHours': 4.0}],
        insulin_doses=[{'minutes_ago': 30, 'units': 3.0}],
        reference_time=rt,
    )))

    return scenarios


def main():
    keep_alive = '--keep-alive' in sys.argv

    print("="*80)
    print("BATCH VALIDATION: Python Loop vs iOS Loop")
    if keep_alive:
        print("  Mode: KEEP-ALIVE (Loop stays running, preserves Xcode debugger)")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    # Print settings being used
    settings = load_settings()
    print("\nTherapy Settings (from settings.json):")
    print(f"  ISF: {settings['insulin_sensitivity_factor']} mg/dL/U")
    print(f"  CR:  {settings['carb_ratio']} g/U")
    print(f"  Basal: {settings['basal_rate']} U/hr")
    print(f"  DIA: {settings['duration_of_insulin_action']} hr")
    print(f"  Target: {settings['target_range']}")
    print(f"  Suspend: {settings['suspend_threshold']} mg/dL")
    print(f"  Max Basal: {settings['max_basal_rate']} U/hr")
    print(f"  Max Bolus: {settings['max_bolus']} U")
    print(f"  Insulin Type: {settings['insulin_type']}")
    print()

    scenarios = define_test_scenarios()
    results_log = []

    for name, scenario in scenarios:
        run_test(name, scenario, results_log, keep_alive=keep_alive)

    # Print summary
    print("\n\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    passed = sum(1 for r in results_log if r['status'] == 'PASS')
    failed = sum(1 for r in results_log if r['status'] == 'FAIL')
    ios_failed = sum(1 for r in results_log if r['status'] == 'IOS_FAILED')
    python_failed = sum(1 for r in results_log if r['status'] == 'PYTHON_FAILED')

    print(f"\nTotal: {len(results_log)} | PASS: {passed} | FAIL: {failed} | iOS Error: {ios_failed} | Python Error: {python_failed}")
    print()

    for r in results_log:
        ebg_diff = r['diffs'].get('eventual_bg', None)
        diff_str = f"{ebg_diff:+.1f}" if ebg_diff is not None else "N/A"
        icon = "✅" if r['status'] == 'PASS' else "❌" if r['status'] == 'FAIL' else "⚠️"
        print(f"  {icon} {r['name']}: {r['status']} (eventual BG diff: {diff_str} mg/dL)")

    # Save detailed results
    results_file = Path(__file__).parent / "validation_results.json"
    # Convert results to JSON-serializable format
    json_results = []
    for r in results_log:
        jr = {
            'name': r['name'],
            'status': r['status'],
            'diffs': r['diffs'],
        }
        if r['ios']:
            jr['ios_eventual_bg'] = r['ios'].get('eventual_bg')
            jr['ios_momentum_impact'] = r['ios'].get('momentum_impact')
            jr['ios_irc_impact'] = r['ios'].get('irc_impact')
        if r['python']:
            jr['python_eventual_bg'] = r['python'].get('eventual_bg')
            jr['python_momentum_impact'] = r['python'].get('momentum_impact')
            jr['python_irc_impact'] = r['python'].get('irc_impact')
        json_results.append(jr)

    results_file.write_text(json.dumps(json_results, indent=2))
    print(f"\nDetailed results saved to: {results_file}")

    return 0 if failed == 0 and ios_failed == 0 and python_failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
