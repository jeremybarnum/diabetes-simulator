#!/usr/bin/env python3
"""
Algorithm Comparison Framework: Loop vs Trio

Runs two parallel simulations using Loop's prediction model as the "patient."
At each 5-minute step, each algorithm independently:
1. Sees the BG resulting from ITS OWN insulin history
2. Makes a dosing decision (temp basal or SMB)
3. Records the insulin delivered

The BG evolution is deterministic: computed by the Loop algorithm's prediction
model for both patients, but with different insulin histories.

Usage:
    python3 compare_algorithms.py
    python3 compare_algorithms.py --duration 360  # 6 hours
    python3 compare_algorithms.py --carbs 30 --bolus 1.5
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

sys.path.insert(0, str(Path(__file__).parent))

from algorithms.base import AlgorithmInput, AlgorithmOutput
from algorithms.loop.loop_algorithm import LoopAlgorithm
from algorithms.openaps.iob import generate_iob_array
from algorithms.openaps.cob import recent_carbs
from algorithms.openaps.glucose_stats import get_last_glucose
from algorithms.openaps.determine_basal import determine_basal
from trio_json_exporter import TrioJSONExporter


def load_settings():
    with open(Path(__file__).parent / "settings.json") as f:
        return json.load(f)


def create_loop_algorithm(settings, automatic_bolus=False, enable_gbpa=False):
    """Create Loop algorithm from settings.

    Args:
        settings: Therapy settings dict
        automatic_bolus: If True, use automatic bolus dosing instead of temp basal
        enable_gbpa: If True, enable Glucose Based Partial Application
                     (20% at target -> 80% at 200 mg/dL). Only applies when
                     automatic_bolus is True.
    """
    loop_settings = {
        'insulin_sensitivity_factor': settings['insulin_sensitivity_factor'],
        'duration_of_insulin_action': settings['duration_of_insulin_action'],
        'basal_rate': settings['basal_rate'],
        'target_range': (settings['target'], settings['target']),
        'carb_ratio': settings['carb_ratio'],
        'max_basal_rate': settings['max_basal_rate'],
        'max_bolus': settings.get('max_bolus', 5.0),
        'suspend_threshold': settings['suspend_threshold'],
        'enable_irc': True,
        'enable_momentum': True,
        'enable_dca': True,
        'insulin_type': settings.get('insulin_type', 'fiasp'),
        'use_realistic_dosing': False,
    }
    if automatic_bolus:
        loop_settings['dosing_mode'] = 'automatic_bolus'
        loop_settings['enable_gbpa'] = enable_gbpa
    return LoopAlgorithm(loop_settings)


class PatientModel:
    """
    Deterministic patient model using Loop's insulin and carb math.

    Computes BG at any time as:
        BG(t) = starting_BG + cumulative_insulin_effect(t) + cumulative_carb_effect(t)

    This separates the "patient physiology" from the algorithm's dosing decisions.
    The insulin and carb effects are computed independently using Loop's exact
    exponential insulin model and piecewise linear carb model.
    """

    def __init__(self, settings: Dict, starting_bg: float = 120.0):
        from algorithms.loop.insulin_math_exact import create_exact_insulin_math
        from algorithms.loop.carb_math import create_carb_math

        self.starting_bg = starting_bg
        self.isf = settings['insulin_sensitivity_factor']
        self.carb_ratio = settings['carb_ratio']

        self.insulin_math = create_exact_insulin_math(
            insulin_sensitivity_factor=self.isf,
            duration_of_insulin_action=settings['duration_of_insulin_action'],
        )
        self.carb_math = create_carb_math(
            carb_ratio=self.carb_ratio,
            insulin_sensitivity=self.isf,
        )

    def compute_bg(
        self,
        time: int,
        bolus_history: List[Tuple[int, float]],
        carb_entries: List[Tuple[int, float, float]],
    ) -> float:
        """
        Compute BG at given time from all insulin and carb effects.

        Args:
            time: Current time in minutes
            bolus_history: All insulin delivered [(time_min, units), ...]
            carb_entries: All carbs [(time_min, grams, absorption_hrs), ...]

        Returns:
            BG in mg/dL
        """
        # Insulin effect: sum of all insulin doses' effects at this time
        insulin_effect = 0.0
        for dose_time, units in bolus_history:
            time_since = time - dose_time
            if time_since < 0:
                continue
            pct_absorbed = self.insulin_math.insulin_model.percent_absorbed(time_since)
            # Negative effect (insulin lowers BG)
            insulin_effect += -units * self.isf * pct_absorbed

        # Carb effect: compute cumulative effect from each carb entry's start time
        # to the current time. We generate the prediction from the carb entry time
        # and look up the effect at the current time.
        carb_effect = 0.0
        if carb_entries:
            for carb_time, grams, abs_hrs in carb_entries:
                elapsed = time - carb_time
                if elapsed < 0:
                    continue
                # Generate prediction from carb entry time
                preds = self.carb_math.predict_glucose_from_carbs(
                    current_glucose=0.0,
                    carb_entries=[(carb_time, grams, abs_hrs)],
                    current_time=carb_time,
                    prediction_horizon=max(elapsed + 5, 10),
                    time_step=5,
                )
                # Find the prediction closest to current time
                if preds:
                    best_effect = 0.0
                    best_dist = float('inf')
                    for pred_time, effect in preds:
                        dist = abs(pred_time - time)
                        if dist < best_dist:
                            best_dist = dist
                            best_effect = effect
                    carb_effect += best_effect

        bg = self.starting_bg + insulin_effect + carb_effect
        return max(39.0, min(400.0, bg))


def get_loop_recommendation(
    loop_algo: LoopAlgorithm,
    current_bg: float,
    current_time: int,
    cgm_history: List[Tuple[int, float]],
    bolus_history: List[Tuple[int, float]],
    carb_entries: List[Tuple[int, float, float]],
    settings: Dict,
) -> AlgorithmOutput:
    """Get Loop's dosing recommendation for current state."""
    alg_input = AlgorithmInput(
        cgm_reading=current_bg,
        timestamp=current_time,
        cgm_history=cgm_history.copy(),
        current_basal=settings['basal_rate'],
        temp_basal=None,
        bolus_history=bolus_history.copy(),
        carb_entries=carb_entries.copy(),
        settings=settings,
    )
    return loop_algo.recommend(alg_input)


def get_trio_recommendation(
    current_bg: float,
    current_time_min: int,
    cgm_history: List[Tuple[int, float]],
    bolus_history: List[Tuple[int, float]],
    carb_entries: List[Tuple[int, float, float]],
    settings: Dict,
    exporter: TrioJSONExporter,
) -> Dict[str, Any]:
    """
    Get Trio's dosing recommendation for the current state.
    Uses the Python Trio implementation (not JS).
    """
    from datetime import datetime, timezone

    profile = exporter.build_profile(current_time_min * 60)
    # Ensure max_iob is set
    if 'max_iob' not in profile or profile['max_iob'] is None:
        profile['max_iob'] = 3.5

    now_ms = current_time_min * 60 * 1000

    # Build pump history for IOB
    pump_history = []
    for t_min, units in bolus_history:
        iso = datetime.fromtimestamp(t_min * 60, tz=timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.000Z')
        pump_history.append({
            '_type': 'Bolus',
            'timestamp': iso,
            'amount': units,
        })

    # Build glucose data
    glucose_data = []
    for t_min, bg in cgm_history:
        glucose_data.append({
            'glucose': bg,
            'sgv': bg,
            'date': t_min * 60 * 1000,
            'dateString': datetime.fromtimestamp(t_min * 60, tz=timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.000Z'),
        })
    # Sort newest first for Trio
    glucose_data.sort(key=lambda x: x['date'], reverse=True)

    # Build carb treatments
    carb_treatments = []
    for t_min, grams, abs_hrs in carb_entries:
        iso = datetime.fromtimestamp(t_min * 60, tz=timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.000Z')
        carb_treatments.append({
            'carbs': grams,
            'nsCarbs': grams,
            'timestamp': iso,
            'created_at': iso,
        })

    bp = exporter.build_basal_profile()

    # Generate IOB array
    iob_array = generate_iob_array(pump_history, profile, now_ms)
    iob_data = iob_array[0] if iob_array else {'iob': 0, 'activity': 0}

    # Generate meal data
    meal_data = recent_carbs(
        treatments=carb_treatments, time_ms=now_ms, profile=profile,
        glucose_data=glucose_data, pump_history=pump_history, basalprofile=bp,
    )

    # Generate glucose status
    glucose_status = get_last_glucose(glucose_data)
    if not glucose_status:
        glucose_status = {
            'glucose': current_bg,
            'delta': 0, 'short_avgdelta': 0, 'long_avgdelta': 0,
            'date': now_ms,
        }

    # Run determine_basal
    result = determine_basal(
        glucose_status=glucose_status,
        currenttemp={'rate': 0, 'duration': 0},
        iob_data=iob_data,
        profile=profile,
        meal_data=meal_data,
        iob_array=iob_array,
        micro_bolus_allowed=True,
        clock_ms=now_ms,
    )

    return result


def apply_loop_dosing(output: AlgorithmOutput, current_time: int,
                      bolus_history: list, basal_rate: float) -> Tuple[float, float, float]:
    """
    Apply Loop's dosing decision. Records doses into bolus_history.

    Returns: (bolus_amount, temp_rate, net_insulin_this_cycle)
    """
    bolus = output.bolus or 0.0
    rate = output.temp_basal_rate if output.temp_basal_rate is not None else basal_rate
    tb_net = (rate - basal_rate) * 5.0 / 60.0

    if bolus > 0:
        bolus_history.append((current_time, bolus))
    if abs(tb_net) > 0.001:
        bolus_history.append((current_time, tb_net))

    return bolus, rate, bolus + tb_net


def apply_trio_dosing(result: Dict, current_time: int,
                      bolus_history: list, basal_rate: float) -> Tuple[float, float, float]:
    """
    Apply Trio's dosing decision. Records doses into bolus_history.

    Returns: (smb_amount, temp_rate, net_insulin_this_cycle)
    """
    smb = result.get('units') or 0.0
    rate_val = result.get('rate')
    rate = rate_val if rate_val is not None else basal_rate
    tb_net = (rate - basal_rate) * 5.0 / 60.0

    if smb > 0:
        bolus_history.append((current_time, smb))
    if abs(tb_net) > 0.001:
        bolus_history.append((current_time, tb_net))

    return smb, rate, smb + tb_net


def run_comparison(
    initial_bg: float = 120.0,
    carb_grams: float = 20.0,
    carb_time: int = 0,
    initial_bolus: float = 0.5,
    bolus_time: int = 0,
    duration_min: int = 360,
    settings: Dict = None,
    automatic_bolus: bool = False,
    enable_gbpa: bool = False,
):
    """
    Run parallel Loop vs Trio simulation.

    Args:
        initial_bg: Starting BG in mg/dL
        carb_grams: Carbs eaten in grams
        carb_time: When carbs eaten (minutes from start)
        initial_bolus: Initial insulin bolus in units
        bolus_time: When bolus given (minutes from start)
        duration_min: Simulation duration in minutes
        settings: Therapy settings dict
        automatic_bolus: Enable Loop automatic bolus dosing
        enable_gbpa: Enable Glucose Based Partial Application factor
    """
    if settings is None:
        settings = load_settings()

    # Use a fixed epoch offset so times are valid timestamps
    # (needed for Trio's ISO timestamp handling)
    epoch_offset = 1739622600  # 2025-02-15T12:30:00Z in seconds
    t0_min = epoch_offset // 60  # Convert to minutes

    basal_rate = settings['basal_rate']

    # Build initial carb entries (shared between both — same meal)
    carb_entries = [(t0_min + carb_time, carb_grams, 3.0)]

    # Create patient models (same physiology, different insulin histories)
    loop_patient = PatientModel(settings, initial_bg)
    trio_patient = PatientModel(settings, initial_bg)

    # --- Initialize Loop simulation ---
    loop_algo = create_loop_algorithm(settings, automatic_bolus=automatic_bolus,
                                      enable_gbpa=enable_gbpa)
    loop_bg_history = []
    loop_bolus_history = [(t0_min + bolus_time, initial_bolus)]
    loop_current_bg = initial_bg
    loop_insulin_total = initial_bolus
    loop_timeline = []

    # --- Initialize Trio simulation ---
    exporter = TrioJSONExporter(settings)
    trio_bg_history = []
    trio_bolus_history = [(t0_min + bolus_time, initial_bolus)]
    trio_current_bg = initial_bg
    trio_insulin_total = initial_bolus
    trio_timeline = []

    # Seed initial CGM history (need a few points for momentum/deltas)
    for pre_t in range(-15, 1, 5):
        t = t0_min + pre_t
        loop_bg_history.append((t, initial_bg))
        trio_bg_history.append((t, initial_bg))

    steps = duration_min // 5

    # Compute IOB from first principles
    insulin_model = loop_patient.insulin_math.insulin_model
    def compute_iob(doses, t):
        return sum(u * (1.0 - insulin_model.percent_absorbed(t - dt))
                   for dt, u in doses if t >= dt)

    # Describe Loop mode
    if automatic_bolus:
        loop_mode = "Automatic Bolus" + (" + GBPA" if enable_gbpa else " (40% fixed)")
    else:
        loop_mode = "Temp Basal only"

    print(f"\nAlgorithm Comparison: Loop vs Trio")
    print(f"{'='*90}")
    print(f"Scenario: {carb_grams}g carbs + {initial_bolus}U bolus at t=0, "
          f"BG={initial_bg}, Target={settings['target']}")
    print(f"Loop mode: {loop_mode}")
    print(f"Trio mode: SMB + temp basal")
    print(f"Settings: ISF={settings['insulin_sensitivity_factor']}, "
          f"CR={settings['carb_ratio']}, Basal={basal_rate}, "
          f"MaxBasal={settings['max_basal_rate']}")
    print(f"Dose math: cycle_dose = bolus + (TB - {basal_rate}) / 12")
    print(f"{'='*90}")
    print()
    # Column: BG, IOB, then the DECISION breakdown, then the single Cycle Dose number
    print(f"{'':>4} │ {'────────────── LOOP ──────────────':^37} │ {'────────────── TRIO ──────────────':^37} │")
    print(f"{'t':>4} │ {'BG':>6} {'IOB':>6} {'Bolus':>6} {'TB':>5} {'Dose':>7} "
          f"│ {'BG':>6} {'IOB':>6} {'SMB':>6} {'TB':>5} {'Dose':>7} │")
    sep = f"{'─'*4}─┼─{'─'*6}─{'─'*6}─{'─'*6}─{'─'*5}─{'─'*7}─┼─{'─'*6}─{'─'*6}─{'─'*6}─{'─'*5}─{'─'*7}─┤"
    print(sep)

    loop_bgs = []
    trio_bgs = []

    for step in range(steps):
        current_time = t0_min + step * 5
        relative_time = step * 5

        # Compute BG from patient model
        loop_bg = loop_patient.compute_bg(current_time, loop_bolus_history, carb_entries)
        trio_bg = trio_patient.compute_bg(current_time, trio_bolus_history, carb_entries)
        loop_bg_history.append((current_time, loop_bg))
        trio_bg_history.append((current_time, trio_bg))
        loop_bgs.append(loop_bg)
        trio_bgs.append(trio_bg)

        # Compute IOB from first principles
        loop_iob = compute_iob(loop_bolus_history, current_time)
        trio_iob = compute_iob(trio_bolus_history, current_time)

        # --- Loop decision ---
        loop_output = get_loop_recommendation(
            loop_algo, loop_bg, current_time,
            loop_bg_history, loop_bolus_history, carb_entries,
            {**settings, 'dosing_mode': 'automatic_bolus' if automatic_bolus else 'temp_basal',
             'enable_gbpa': enable_gbpa},
        )
        l_bol, l_rate, l_net = apply_loop_dosing(loop_output, current_time,
                                                   loop_bolus_history, basal_rate)
        # Cycle dose = bolus + (TB - scheduled) / 12
        l_tb_dose = (l_rate - basal_rate) / 12.0
        l_cycle = l_bol + l_tb_dose

        # --- Trio decision ---
        trio_result = get_trio_recommendation(
            trio_bg, current_time,
            trio_bg_history, trio_bolus_history, carb_entries,
            settings, exporter,
        )
        t_smb, t_rate, t_net = apply_trio_dosing(trio_result, current_time,
                                                   trio_bolus_history, basal_rate)
        t_tb_dose = (t_rate - basal_rate) / 12.0
        t_cycle = t_smb + t_tb_dose

        # Format: show bolus/SMB, TB rate, and the combined cycle dose
        l_bol_s = f'{l_bol:6.3f}' if l_bol > 0.0005 else '    --'
        l_tb_s = f'{l_rate:5.2f}'
        l_dose_s = f'{l_cycle:+7.4f}' if abs(l_cycle) > 0.0001 else '     --'

        t_smb_s = f'{t_smb:6.3f}' if t_smb > 0.0005 else '    --'
        t_tb_s = f'{t_rate:5.2f}'
        t_dose_s = f'{t_cycle:+7.4f}' if abs(t_cycle) > 0.0001 else '     --'

        print(f'{relative_time:4d} │ {loop_bg:6.1f} {loop_iob:6.3f} {l_bol_s} {l_tb_s} {l_dose_s} '
              f'│ {trio_bg:6.1f} {trio_iob:6.3f} {t_smb_s} {t_tb_s} {t_dose_s} │')

    # Final BG
    ft = t0_min + steps * 5
    lf = loop_patient.compute_bg(ft, loop_bolus_history, carb_entries)
    tf = trio_patient.compute_bg(ft, trio_bolus_history, carb_entries)
    li = compute_iob(loop_bolus_history, ft)
    ti = compute_iob(trio_bolus_history, ft)
    ln = sum(u for _, u in loop_bolus_history)
    tn = sum(u for _, u in trio_bolus_history)

    print(sep)
    print(f' END │ {lf:6.1f} {li:6.3f}                       '
          f'│ {tf:6.1f} {ti:6.3f}                       │')
    print()
    print(f"  {'':20s} {'Loop':>8s}  {'Trio':>8s}")
    print(f"  {'Final BG:':20s} {lf:8.1f}  {tf:8.1f}")
    print(f"  {'Peak BG:':20s} {max(loop_bgs):8.1f}  {max(trio_bgs):8.1f}")
    print(f"  {'Min BG:':20s} {min(loop_bgs):8.1f}  {min(trio_bgs):8.1f}")
    print(f"  {'Net insulin:':20s} {ln:8.3f}  {tn:8.3f}  (ideal: {carb_grams/settings['carb_ratio']:.1f}U)")
    print(f"  {'Final IOB:':20s} {li:8.3f}  {ti:8.3f}")

    return loop_bgs, trio_bgs


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Compare Loop vs Trio algorithms')
    parser.add_argument('--bg', type=float, default=100.0, help='Initial BG (mg/dL)')
    parser.add_argument('--carbs', type=float, default=20.0, help='Carbs (grams)')
    parser.add_argument('--bolus', type=float, default=0.5, help='Initial bolus (units)')
    parser.add_argument('--duration', type=int, default=360, help='Duration (minutes)')
    parser.add_argument('--auto-bolus', action='store_true',
                        help='Enable Loop automatic bolus dosing (default: temp basal only)')
    parser.add_argument('--gbpa', action='store_true',
                        help='Enable Glucose Based Partial Application (requires --auto-bolus)')
    args = parser.parse_args()

    if args.gbpa and not args.auto_bolus:
        print("Warning: --gbpa requires --auto-bolus, enabling both")
        args.auto_bolus = True

    settings = load_settings()
    run_comparison(
        initial_bg=args.bg,
        carb_grams=args.carbs,
        initial_bolus=args.bolus,
        duration_min=args.duration,
        settings=settings,
        automatic_bolus=args.auto_bolus,
        enable_gbpa=args.gbpa,
    )


if __name__ == "__main__":
    main()
