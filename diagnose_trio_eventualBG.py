#!/usr/bin/env python3
"""
Diagnostic script: Trace intermediate values in the Python Trio pipeline
for all 10 test scenarios, comparing eventualBG against JS baselines.

Goal: understand WHY eventualBG differs between Python and JS.
"""

import sys
import json
from pathlib import Path
from datetime import datetime, timezone

# Setup paths
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "_validation"))
sys.path.insert(0, str(ROOT))

from trio_batch_validate import define_trio_test_scenarios, REFERENCE_TIME, load_settings
from trio_json_exporter import TrioJSONExporter
from algorithms.openaps.iob import generate_iob_array
from algorithms.openaps.cob import recent_carbs
from algorithms.openaps.glucose_stats import get_last_glucose
from algorithms.openaps.predictions import generate_predictions


def main():
    settings = load_settings()
    exporter = TrioJSONExporter(settings)
    scenarios = define_trio_test_scenarios(REFERENCE_TIME)

    # Load JS baselines
    baselines_path = ROOT / "_validation" / "trio_regression_baselines.json"
    with open(baselines_path) as f:
        js_baselines = json.load(f)

    print("=" * 90)
    print("TRIO PIPELINE DIAGNOSTIC: Python intermediate values vs JS baselines")
    print(f"Reference time: {datetime.fromtimestamp(REFERENCE_TIME, tz=timezone.utc).isoformat()}")
    print("=" * 90)

    for name, scenario in scenarios:
        print(f"\n{'=' * 90}")
        print(f"  TEST: {name}")
        print(f"  {scenario.get('description', '')}")
        print(f"{'=' * 90}")

        try:
            # --- Build inputs (same as run_python_trio) ---
            glucose_samples = sorted(scenario['glucoseSamples'], key=lambda x: x['timestamp'])
            most_recent = glucose_samples[-1]
            current_bg = most_recent['value']
            current_time_unix = most_recent['timestamp']
            now_ms = int(current_time_unix * 1000)

            profile = exporter.build_profile(int(current_time_unix))
            if 'max_iob' not in profile or profile['max_iob'] is None:
                profile['max_iob'] = 3.5

            # Pump history
            pump_history = []
            for d in scenario.get('insulinDoses', []):
                iso = datetime.fromtimestamp(d['timestamp'], tz=timezone.utc).strftime(
                    '%Y-%m-%dT%H:%M:%S.000Z')
                pump_history.append({'_type': 'Bolus', 'timestamp': iso, 'amount': d['units']})

            # Glucose data (reverse-chronological)
            glucose_data = []
            for s in glucose_samples:
                glucose_data.append({
                    'glucose': s['value'], 'sgv': s['value'],
                    'date': int(s['timestamp'] * 1000),
                    'dateString': datetime.fromtimestamp(
                        s['timestamp'], tz=timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.000Z'),
                })
            glucose_data.sort(key=lambda x: x['date'], reverse=True)

            # Carb treatments
            carb_treatments = []
            for c in scenario.get('carbEntries', []):
                iso = datetime.fromtimestamp(c['timestamp'], tz=timezone.utc).strftime(
                    '%Y-%m-%dT%H:%M:%S.000Z')
                carb_treatments.append({
                    'carbs': c['grams'], 'nsCarbs': c['grams'],
                    'timestamp': iso, 'created_at': iso,
                })

            bp = exporter.build_basal_profile()
            iob_array = generate_iob_array(pump_history, profile, now_ms)
            iob_data = iob_array[0] if iob_array else {'iob': 0, 'activity': 0}

            meal_data = recent_carbs(
                treatments=carb_treatments, time_ms=now_ms, profile=profile,
                glucose_data=glucose_data, pump_history=pump_history, basalprofile=bp,
            )

            glucose_status = get_last_glucose(glucose_data)
            if not glucose_status:
                glucose_status = {
                    'glucose': current_bg, 'delta': 0,
                    'short_avgdelta': 0, 'long_avgdelta': 0, 'date': now_ms,
                }

            # --- Call generate_predictions ---
            pred_result = generate_predictions(
                bg=glucose_status.get('glucose', current_bg),
                iob_array=iob_array,
                profile=profile,
                glucose_status=glucose_status,
                meal_data=meal_data if meal_data else {'mealCOB': 0, 'carbs': 0, 'lastCarbTime': 0,
                                                        'slopeFromMaxDeviation': 0, 'slopeFromMinDeviation': 999},
                iob_data=iob_data,
                sensitivity_ratio=1.0,
                enable_uam=True,
                clock_ms=now_ms,
            )

            # --- Print intermediate values ---
            bg = glucose_status.get('glucose', current_bg)
            iob = iob_data.get('iob', 0)
            activity = iob_data.get('activity', 0)
            sens = profile.get('sens', 100)

            print(f"\n  [1] CORE INPUTS")
            print(f"      bg              = {bg}")
            print(f"      iob             = {iob}")
            print(f"      activity        = {activity}")
            print(f"      sens (ISF)      = {sens}")
            print(f"      bgi             = {pred_result['bgi']}")

            delta = glucose_status.get('delta', 0)
            short_avgdelta = glucose_status.get('short_avgdelta', 0)
            long_avgdelta = glucose_status.get('long_avgdelta', 0)
            min_delta = min(delta, short_avgdelta)
            min_avg_delta = min(short_avgdelta, long_avgdelta)

            print(f"\n  [2] DELTAS")
            print(f"      delta           = {delta}")
            print(f"      short_avgdelta  = {short_avgdelta}")
            print(f"      long_avgdelta   = {long_avgdelta}")
            print(f"      min_delta       = {min_delta}")
            print(f"      min_avg_delta   = {min_avg_delta}")

            # Recompute deviation logic to show which branch was taken
            bgi = pred_result['bgi']
            dev_from_min_delta = round(30 / 5 * (min_delta - bgi))
            dev_from_min_avg = round(30 / 5 * (min_avg_delta - bgi))
            dev_from_long = round(30 / 5 * (long_avgdelta - bgi))

            if dev_from_min_delta >= 0:
                dev_source = "min_delta (>= 0)"
                dev_val = dev_from_min_delta
            elif dev_from_min_avg >= 0:
                dev_source = "min_avg_delta (min_delta was < 0)"
                dev_val = dev_from_min_avg
            else:
                dev_source = "long_avgdelta (both min_delta and min_avg were < 0)"
                dev_val = dev_from_long

            print(f"\n  [3] DEVIATION")
            print(f"      deviation       = {pred_result['deviation']}")
            print(f"      dev_from_min_delta  = {dev_from_min_delta}")
            print(f"      dev_from_min_avg    = {dev_from_min_avg}")
            print(f"      dev_from_long       = {dev_from_long}")
            print(f"      source selected     = {dev_source}")

            print(f"\n  [4] EVENTUAL BG")
            naive = pred_result['naive_eventualBG']
            eventual = pred_result['eventualBG']
            print(f"      naive_eventualBG = {naive}  (bg - iob*sens = {bg} - {iob}*{sens} = {round(bg - iob*sens)})")
            print(f"      naive + dev      = {naive} + {pred_result['deviation']} = {naive + pred_result['deviation']}")
            print(f"      eventualBG (before COB/UAM max) = {naive + pred_result['deviation']}")

            print(f"\n  [5] CARB IMPACT")
            ci = pred_result.get('ci', 0)
            uci = pred_result.get('uci', 0)
            csf = pred_result.get('csf', 0)
            cid = pred_result.get('cid', 0)
            rci_peak = pred_result.get('remaining_ci_peak', 0)
            rca_time = pred_result.get('remaining_ca_time', 0)
            print(f"      ci              = {ci}  (min_delta - bgi = {min_delta} - {bgi} = {round(min_delta - bgi, 1)})")
            print(f"      uci             = {uci}")
            print(f"      csf             = {csf}  (sens/cr = {sens}/{profile.get('carb_ratio', 10)})")
            print(f"      cid             = {cid}")
            print(f"      remaining_ci_pk = {rci_peak}")
            print(f"      remaining_ca_t  = {rca_time}")

            print(f"\n  [6] MEAL DATA")
            mc = meal_data if meal_data else {}
            print(f"      mealCOB         = {mc.get('mealCOB', 0)}")
            print(f"      carbs           = {mc.get('carbs', 0)}")
            print(f"      lastCarbTime    = {mc.get('lastCarbTime', 0)}")
            print(f"      slopeFromMaxDev = {mc.get('slopeFromMaxDeviation', 0)}")
            print(f"      slopeFromMinDev = {mc.get('slopeFromMinDeviation', 999)}")

            print(f"\n  [7] PREDICTION CURVES")
            pred_bgs = pred_result.get('predBGs', {})
            for curve_name in ['IOB', 'ZT', 'COB', 'UAM']:
                if curve_name in pred_bgs:
                    arr = pred_bgs[curve_name]
                    print(f"      {curve_name:4s}: count={len(arr):3d}, first={arr[0] if arr else '?'}, last={arr[-1] if arr else '?'}")
                else:
                    print(f"      {curve_name:4s}: NOT PRESENT")

            # Show the max refinement
            initial_eventual = naive + pred_result['deviation']
            cob_last = pred_bgs.get('COB', [None])[-1] if 'COB' in pred_bgs else None
            uam_last = pred_bgs.get('UAM', [None])[-1] if 'UAM' in pred_bgs else None

            print(f"\n  [8] EVENTUAL BG REFINEMENT")
            print(f"      initial (naive+dev)       = {initial_eventual}")
            print(f"      COB curve last            = {cob_last}")
            print(f"      UAM curve last            = {uam_last}")
            print(f"      final eventualBG (Python) = {eventual}")
            parts = [initial_eventual]
            if cob_last is not None:
                parts.append(cob_last)
            if uam_last is not None:
                parts.append(uam_last)
            print(f"      max({', '.join(str(p) for p in parts)}) = {max(parts)}")

            # --- Compare with JS baseline ---
            js = js_baselines.get(name, {})
            js_eventual = js.get('eventualBG', '?')
            diff = eventual - js_eventual if isinstance(js_eventual, (int, float)) else '?'

            print(f"\n  [9] JS BASELINE COMPARISON")
            print(f"      JS eventualBG   = {js_eventual}")
            print(f"      Python eventualBG = {eventual}")
            print(f"      DIFF            = {diff}")
            js_pred = js.get('predBGs', {})
            for curve_name in ['IOB', 'ZT', 'COB', 'UAM']:
                jp = js_pred.get(curve_name, {})
                pp = pred_bgs.get(curve_name, None)
                if jp or pp:
                    js_str = f"count={jp.get('count','?')}, first={jp.get('first','?')}, last={jp.get('last','?')}" if jp else "NOT PRESENT"
                    py_str = f"count={len(pp)}, first={pp[0]}, last={pp[-1]}" if pp else "NOT PRESENT"
                    match = ""
                    if jp and pp:
                        c_match = len(pp) == jp.get('count')
                        f_match = pp[0] == jp.get('first')
                        l_match = pp[-1] == jp.get('last')
                        if c_match and f_match and l_match:
                            match = " [MATCH]"
                        else:
                            mismatches = []
                            if not c_match:
                                mismatches.append(f"count: {len(pp)} vs {jp.get('count')}")
                            if not f_match:
                                mismatches.append(f"first: {pp[0]} vs {jp.get('first')}")
                            if not l_match:
                                mismatches.append(f"last: {pp[-1]} vs {jp.get('last')}")
                            match = f" [DIFF: {', '.join(mismatches)}]"
                    print(f"      {curve_name:4s} JS:  {js_str}")
                    print(f"      {curve_name:4s} Py:  {py_str}{match}")

            # Extras from JS reason string
            js_reason = js.get('reason_prefix', '')
            print(f"\n      JS reason: {js_reason}")
            # Parse Dev and BGI from JS reason for comparison
            import re
            dev_match = re.search(r'Dev: (-?\d+)', js_reason)
            bgi_match = re.search(r'BGI: (-?\d+)', js_reason)
            cob_match = re.search(r'COB: (-?\d+)', js_reason)
            if dev_match:
                js_dev = int(dev_match.group(1))
                print(f"      JS Dev={js_dev}, Python Dev={pred_result['deviation']}, diff={pred_result['deviation'] - js_dev}")
            if bgi_match:
                js_bgi = int(bgi_match.group(1))
                print(f"      JS BGI={js_bgi}, Python BGI={pred_result['bgi']}, diff={pred_result['bgi'] - js_bgi}")
            if cob_match:
                js_cob = int(cob_match.group(1))
                py_cob = mc.get('mealCOB', 0)
                print(f"      JS COB={js_cob}, Python COB={py_cob}, diff={py_cob - js_cob}")

        except Exception as e:
            import traceback
            print(f"\n  ERROR: {e}")
            traceback.print_exc()

    print(f"\n{'=' * 90}")
    print("DONE")
    print(f"{'=' * 90}")


if __name__ == "__main__":
    main()
