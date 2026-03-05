#!/usr/bin/env python3
"""
Diagnostic single-path simulation tests for debugging sim-vs-reality gaps.

Quick, deterministic tests that print BG traces and key metrics.
No regression baselines — these are visual/diagnostic tools.

Usage:
    python3 debug_sim.py flat_bg
    python3 debug_sim.py single_meal_perfect
    python3 debug_sim.py all
    python3 debug_sim.py --list
"""

import sys
import numpy as np
from pathlib import Path
from typing import List, Optional

from simulation import (
    PatientProfile, MealSpec, ExerciseSpec, SimulationRun, SimulationRunResult,
)


# Day starts at 7am, times are minutes from 7am
def _clock(minutes_from_7am: int) -> str:
    """Convert minutes-from-7am to HH:MM clock string."""
    h = 7 + minutes_from_7am // 60
    m = minutes_from_7am % 60
    return f"{h:02d}:{m:02d}"


def _print_hourly_trace(result: SimulationRunResult, label: str, day: int = 0):
    """Print hourly BG trace for a single day."""
    day_result = result.days[day]
    trace = day_result.bg_trace  # [(t_rel_from_sim_start, bg), ...]

    # Build meal lookup (t_rel -> meal info)
    meals_by_step = {}
    for meal in day_result.meals:
        # meal.time_minutes is absolute; convert to relative
        t0 = result.days[0].bg_trace[0][0] if result.days[0].bg_trace else 0
        t_rel = meal.time_minutes - (result.days[day].bg_trace[0][0] - day * 1440) \
            if day_result.bg_trace else 0
        # Just store by nearest 5-min step relative to day start
        step_in_day = round((meal.time_minutes % 1440 - (result.days[0].bg_trace[0][0] % 1440)) / 5) * 5
        meals_by_step[step_in_day] = meal

    # Sensitivity
    sens = day_result.sensitivity_trace[0][1] if day_result.sensitivity_trace else 1.0

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  Algo: {result.algorithm_name}, Sensitivity: {sens:.3f}")
    print(f"{'='*60}")
    print(f"{'Hour':>5}  {'Clock':>6}  {'BG':>6}  {'Delta':>6}  Notes")
    print(f"{'-'*5}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*20}")

    prev_bg = None
    day_offset = day * 1440
    for t_rel, bg in trace:
        min_in_day = t_rel - day_offset
        # Print every 60 min (hourly)
        if min_in_day % 60 == 0 and min_in_day >= 0:
            hour = min_in_day / 60
            clock = _clock(int(min_in_day))
            delta_str = ""
            if prev_bg is not None:
                d = bg - prev_bg
                delta_str = f"{d:+.1f}"
            # Check for meals near this hour
            notes = ""
            for m in day_result.meals:
                # m.time_minutes is absolute; min_in_day is relative to day start
                # day_result.bg_trace starts at t_rel=day_offset
                m_min = m.time_minutes - (trace[0][0])  # relative to trace start
                if abs(m_min - min_in_day) < 30:
                    if m.undeclared:
                        notes += f"Meal {m.actual_carbs:.0f}g (undecl) "
                    else:
                        notes += f"Meal {m.actual_carbs:.0f}g(decl {m.declared_carbs:.0f}g) "
            print(f"{hour:5.1f}  {clock:>6}  {bg:6.1f}  {delta_str:>6}  {notes}")
            prev_bg = bg

    # Summary
    bgs = [bg for _, bg in trace]
    print(f"\nSummary: Mean={np.mean(bgs):.1f}, Min={min(bgs):.1f}, "
          f"Max={max(bgs):.1f}, Final={bgs[-1]:.1f}")
    rescues = day_result.rescue_carb_events
    if rescues > 0:
        print(f"Rescue carbs: {rescues} events, {day_result.rescue_carb_grams_total:.0f}g total")


def _print_fine_trace(result: SimulationRunResult, label: str,
                      start_min: int, end_min: int, day: int = 0):
    """Print 5-min BG trace for a time window (minutes from 7am)."""
    day_result = result.days[day]
    trace = day_result.bg_trace
    day_offset = day * 1440

    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"  Algo: {result.algorithm_name}")
    print(f"{'='*50}")
    print(f"{'Min':>5}  {'Clock':>6}  {'BG':>6}  {'Delta':>6}")
    print(f"{'-'*5}  {'-'*6}  {'-'*6}  {'-'*6}")

    prev_bg = None
    for t_rel, bg in trace:
        min_in_day = t_rel - day_offset
        if start_min <= min_in_day <= end_min:
            clock = _clock(int(min_in_day))
            delta_str = ""
            if prev_bg is not None:
                d = bg - prev_bg
                delta_str = f"{d:+.1f}"
            print(f"{int(min_in_day):5d}  {clock:>6}  {bg:6.1f}  {delta_str:>6}")
            prev_bg = bg


def _run_sim(profile: PatientProfile, algo: str, n_days: int, seed: int) -> SimulationRunResult:
    """Run a single simulation path."""
    rng = np.random.RandomState(seed)
    sim = SimulationRun(profile=profile, algorithm_name=algo, n_days=n_days, rng=rng)
    return sim.run()


def _default_settings() -> dict:
    """Load default settings.json."""
    import json
    with open(Path(__file__).parent / 'settings.json') as f:
        return json.load(f)


# ─── Scenarios ─────────────────────────────────────────────────────────────────

def test_flat_bg():
    """No meals, no sensitivity error. BG should stay flat at target."""
    settings = _default_settings()
    target = settings.get('target', 100.0)
    profile = PatientProfile(
        meals=[],
        carb_count_sigma=0.0,
        carb_count_bias=0.0,
        sensitivity_sigma=0.0,
        starting_bg=target,  # Start at target so algorithm shouldn't adjust
        rescue_carbs_enabled=False,
        algorithm_settings=settings,
    )
    result = _run_sim(profile, 'trio', n_days=1, seed=42)
    _print_hourly_trace(result, f"flat_bg — No meals, no sensitivity error, start=target={target}")

    bgs = [bg for _, bg in result.days[0].bg_trace]
    drift = max(bgs) - min(bgs)
    print(f"\nMax drift from flat: {drift:.1f} mg/dL")
    if drift < 2.0:
        print("PASS: BG stays flat (drift < 2)")
    else:
        print(f"INVESTIGATE: drift = {drift:.1f} (expected < 2)")


def test_single_meal_perfect():
    """One 50g meal at noon, perfect declarations, no sensitivity error."""
    profile = PatientProfile(
        meals=[MealSpec(time_of_day_minutes=300, carbs_mean=50.0, carbs_sd=0.0,
                        absorption_hrs=3.0)],
        carb_count_sigma=0.0,
        carb_count_bias=0.0,
        absorption_sigma=0.0,
        sensitivity_sigma=0.0,
        starting_bg=100.0,
        rescue_carbs_enabled=False,
        algorithm_settings=_default_settings(),
    )
    result = _run_sim(profile, 'trio', n_days=1, seed=42)
    _print_hourly_trace(result, "single_meal_perfect — 50g at noon, perfect bolus")

    bgs = [bg for _, bg in result.days[0].bg_trace]
    peak = max(bgs)
    final = bgs[-1]
    print(f"\nPeak BG: {peak:.1f}, Final BG: {final:.1f}")
    print(f"Expected: peak ~130-180, final ~90-110")


def test_single_meal_underdeclared():
    """Same 50g meal but declares only ~60% (bias=0.5)."""
    profile = PatientProfile(
        meals=[MealSpec(time_of_day_minutes=300, carbs_mean=50.0, carbs_sd=0.0,
                        absorption_hrs=3.0)],
        carb_count_sigma=0.0,
        carb_count_bias=0.5,
        absorption_sigma=0.0,
        sensitivity_sigma=0.0,
        starting_bg=100.0,
        rescue_carbs_enabled=False,
        algorithm_settings=_default_settings(),
    )
    result = _run_sim(profile, 'trio', n_days=1, seed=42)
    _print_hourly_trace(result, "single_meal_underdeclared — 50g at noon, bias=0.5 (~60% declared)")

    bgs = [bg for _, bg in result.days[0].bg_trace]
    peak = max(bgs)
    final = bgs[-1]
    print(f"\nPeak BG: {peak:.1f}, Final BG: {final:.1f}")
    print(f"Expected: higher peak than perfect, slower return")


def test_sensitivity_only():
    """No meals, sensitivity_sigma=0.28. How much does BG drift across paths?"""
    n_paths = 10
    print(f"\n{'='*70}")
    print(f"  sensitivity_only — No meals, sensitivity_sigma=0.28, {n_paths} paths")
    print(f"{'='*70}")
    print(f"{'Path':>4}  {'Seed':>4}  {'Sens':>6}  {'BG@0h':>6}  "
          f"{'BG@6h':>6}  {'BG@12h':>6}  {'BG@18h':>6}  {'BG@24h':>6}  {'Drift':>6}")
    print("-" * 70)

    settings = _default_settings()
    all_final = []

    for i in range(n_paths):
        seed = i
        profile = PatientProfile(
            meals=[],
            carb_count_sigma=0.0,
            carb_count_bias=0.0,
            sensitivity_sigma=0.28,
            starting_bg=110.0,
            rescue_carbs_enabled=False,
            algorithm_settings=settings,
        )
        result = _run_sim(profile, 'trio', n_days=1, seed=seed)
        trace = result.days[0].bg_trace

        # Extract BG at specific hours (steps: 0, 72, 144, 216, 287)
        def bg_at_hour(h):
            idx = h * 12  # 12 steps per hour
            if idx < len(trace):
                return trace[idx][1]
            return trace[-1][1]

        sens = result.days[0].sensitivity_trace[0][1] if result.days[0].sensitivity_trace else 1.0
        bg0 = bg_at_hour(0)
        bg6 = bg_at_hour(6)
        bg12 = bg_at_hour(12)
        bg18 = bg_at_hour(18)
        bg24 = trace[-1][1]
        drift = bg24 - bg0
        all_final.append(bg24)

        print(f"{i:4d}  {seed:4d}  {sens:6.3f}  {bg0:6.1f}  "
              f"{bg6:6.1f}  {bg12:6.1f}  {bg18:6.1f}  {bg24:6.1f}  {drift:+6.1f}")

    arr = np.array(all_final)
    print(f"\nFinal BG: Mean={np.mean(arr):.1f}, SD={np.std(arr):.1f}, "
          f"Min={np.min(arr):.1f}, Max={np.max(arr):.1f}")
    if np.std(arr) > 5:
        print("PASS: Meaningful BG spread from sensitivity variation")
    else:
        print("INVESTIGATE: Very little BG spread — algorithm may be squashing drift too effectively")


def test_ns_profile_single():
    """Full NS-inferred profile, single path, hour-by-hour."""
    profile_path = Path(__file__).parent / 'patient_profiles' / 'ns_inferred.json'
    if not profile_path.exists():
        print("SKIP: ns_inferred.json not found")
        return

    profile = PatientProfile.from_json(str(profile_path))
    result = _run_sim(profile, 'trio', n_days=1, seed=42)
    _print_hourly_trace(result, "ns_profile_single — Full NS-inferred profile, 1 day")

    # Also print meal details
    print(f"\nMeal details:")
    print(f"{'Time':>6}  {'Actual':>7}  {'Declared':>8}  {'Undecl':>6}")
    # meal.time_minutes is absolute; convert to minutes-from-7am
    t0_min = SimulationRun.EPOCH_OFFSET_SEC // 60
    for m in result.days[0].meals:
        min_from_7am = int((m.time_minutes - t0_min) % 1440)
        clock = _clock(min_from_7am)
        und = "yes" if m.undeclared else ""
        print(f"{clock:>6}  {m.actual_carbs:7.1f}g  {m.declared_carbs:7.1f}g  {und:>6}")

    sens = result.days[0].sensitivity_trace[0][1] if result.days[0].sensitivity_trace else 1.0
    print(f"\nSensitivity scalar: {sens:.3f}")


def test_breakfast_zoom():
    """Isolate the breakfast problem. Fine-grained trace around breakfast."""
    settings = {
        "insulin_sensitivity_factor": 90.0,
        "carb_ratio": 11.3,
        "basal_rate": 0.544,
        "duration_of_insulin_action": 6.0,
        "target": 102.0,
        "suspend_threshold": 80.0,
        "max_basal_rate": 2.8,
        "max_bolus": 3.0,
        "insulin_type": "fiasp",
        "enable_irc": True,
        "enable_momentum": True,
        "enable_dca": True,
    }

    # Case A: Perfect declaration
    profile_a = PatientProfile(
        meals=[MealSpec(time_of_day_minutes=60, carbs_mean=6.8, carbs_sd=0.0,
                        absorption_hrs=3.0)],
        carb_count_sigma=0.0,
        carb_count_bias=0.0,
        absorption_sigma=0.0,
        sensitivity_sigma=0.0,
        starting_bg=110.0,
        rescue_carbs_enabled=False,
        algorithm_settings=settings,
    )

    # Case B: NS-inferred bias
    profile_b = PatientProfile(
        meals=[MealSpec(time_of_day_minutes=60, carbs_mean=6.8, carbs_sd=0.0,
                        absorption_hrs=3.0)],
        carb_count_sigma=0.0,
        carb_count_bias=0.57,
        absorption_sigma=0.0,
        sensitivity_sigma=0.0,
        starting_bg=110.0,
        rescue_carbs_enabled=False,
        algorithm_settings=settings,
    )

    result_a = _run_sim(profile_a, 'trio', n_days=1, seed=42)
    result_b = _run_sim(profile_b, 'trio', n_days=1, seed=42)

    # Print side-by-side fine trace from 7am (0 min) to 11am (240 min)
    trace_a = result_a.days[0].bg_trace
    trace_b = result_b.days[0].bg_trace

    print(f"\n{'='*65}")
    print(f"  breakfast_zoom — 6.8g at 8:00am, ISF=90, CR=11.3")
    print(f"  Case A: Perfect declaration  |  Case B: bias=0.57")
    print(f"{'='*65}")

    # Get meal details
    meal_a = result_a.days[0].meals[0] if result_a.days[0].meals else None
    meal_b = result_b.days[0].meals[0] if result_b.days[0].meals else None
    if meal_a:
        print(f"  A: actual={meal_a.actual_carbs:.1f}g, declared={meal_a.declared_carbs:.1f}g, "
              f"bolus={meal_a.declared_carbs/settings['carb_ratio']:.2f}U")
    if meal_b:
        print(f"  B: actual={meal_b.actual_carbs:.1f}g, declared={meal_b.declared_carbs:.1f}g, "
              f"bolus={meal_b.declared_carbs/settings['carb_ratio']:.2f}U")
    print()

    print(f"{'Min':>5}  {'Clock':>6}  {'BG_A':>6}  {'BG_B':>6}  {'Diff':>6}")
    print(f"{'-'*5}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}")

    for i in range(len(trace_a)):
        t_rel = trace_a[i][0]
        min_in_day = t_rel  # day 0, so t_rel == min_in_day
        if 0 <= min_in_day <= 240:
            clock = _clock(int(min_in_day))
            bg_a = trace_a[i][1]
            bg_b = trace_b[i][1] if i < len(trace_b) else 0
            diff = bg_b - bg_a
            print(f"{int(min_in_day):5d}  {clock:>6}  {bg_a:6.1f}  {bg_b:6.1f}  {diff:+6.1f}")


def test_breakfast_realistic():
    """Realistic breakfast: 5g fast coffee at 7am + 15g yoghurt at 8:30am.

    Both perfectly declared. Uses NS-inferred algorithm settings.
    Expected pattern: rise from coffee, start declining, then second rise from yoghurt.
    """
    settings = {
        "insulin_sensitivity_factor": 90.0,
        "carb_ratio": 11.3,
        "basal_rate": 0.544,
        "duration_of_insulin_action": 6.0,
        "target": 102.0,
        "suspend_threshold": 80.0,
        "max_basal_rate": 2.8,
        "max_bolus": 3.0,
        "insulin_type": "fiasp",
        "enable_irc": True,
        "enable_momentum": True,
        "enable_dca": True,
    }

    profile = PatientProfile(
        meals=[
            MealSpec(time_of_day_minutes=0, carbs_mean=5.0, carbs_sd=0.0,
                     absorption_hrs=1.0),     # Coffee at 7am, fast
            MealSpec(time_of_day_minutes=90, carbs_mean=15.0, carbs_sd=0.0,
                     absorption_hrs=2.5),     # Yoghurt at 8:30am, moderate
        ],
        carb_count_sigma=0.0,
        carb_count_bias=0.0,
        absorption_sigma=0.0,
        sensitivity_sigma=0.0,
        starting_bg=100.0,
        rescue_carbs_enabled=False,
        algorithm_settings=settings,
    )

    result = _run_sim(profile, 'trio', n_days=1, seed=42)
    trace = result.days[0].bg_trace

    # Print fine trace from 7am to 1pm
    print(f"\n{'='*55}")
    print(f"  breakfast_realistic — 5g coffee @7am (1h abs) + 15g yoghurt @8:30am (2.5h abs)")
    print(f"  Perfect declaration, ISF=90, CR=11.3, no sensitivity error")
    csf = settings['insulin_sensitivity_factor'] / settings['carb_ratio']
    print(f"  CSF={csf:.1f} mg/dL per gram")
    print(f"  Coffee: 5g → raw rise ~{5*csf:.0f}, bolus={5/settings['carb_ratio']:.2f}U")
    print(f"  Yoghurt: 15g → raw rise ~{15*csf:.0f}, bolus={15/settings['carb_ratio']:.2f}U")
    print(f"{'='*55}")

    print(f"{'Min':>5}  {'Clock':>6}  {'BG':>6}  {'Delta':>6}  Notes")
    print(f"{'-'*5}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*20}")

    prev_bg = None
    for t_rel, bg in trace:
        if 0 <= t_rel <= 360:  # 7am to 1pm
            clock = _clock(int(t_rel))
            delta_str = ""
            if prev_bg is not None:
                d = bg - prev_bg
                delta_str = f"{d:+.1f}"
            notes = ""
            if t_rel == 0:
                notes = "← Coffee 5g"
            elif t_rel == 90:
                notes = "← Yoghurt 15g"
            print(f"{int(t_rel):5d}  {clock:>6}  {bg:6.1f}  {delta_str:>6}  {notes}")
            prev_bg = bg

    bgs = [bg for t, bg in trace if 0 <= t <= 360]
    print(f"\n7am-1pm: Mean={np.mean(bgs):.1f}, Peak={max(bgs):.1f}, Min={min(bgs):.1f}")


def test_exercise_check():
    """Exercise declared/actual mismatch: spread from stochastic magnitude & duration.

    The algorithm compensates during the declared window, but the actual
    sensitivity effect varies in both magnitude and duration. The mismatch
    window (declared over, actual still active) is where BG variance appears.

    Exercise at 1pm. Declared: 3h (ends 4pm). Actual: mean 6h, σ=0.2 lognormal.
    """
    n_paths = 10
    settings = _default_settings()

    # NS-inferred exercise parameters
    ex_spec = ExerciseSpec(
        time_of_day_minutes=360,  # 1pm
        declared_scalar=0.5,
        declared_duration_hrs=3.0,
        actual_scalar_mean=0.5,
        actual_scalar_sigma=0.15,
        actual_duration_hrs_mean=6.0,
        actual_duration_hrs_sigma=0.2,
    )

    print(f"\n{'='*90}")
    print(f"  exercise_check — Exercise at 1pm, no meals, {n_paths} paths")
    print(f"  Declared: scalar={ex_spec.declared_scalar}, dur={ex_spec.declared_duration_hrs}h (ends 4pm)")
    print(f"  Actual: scalar~LN({ex_spec.actual_scalar_mean}, σ={ex_spec.actual_scalar_sigma}), "
          f"dur~LN({ex_spec.actual_duration_hrs_mean}h, σ={ex_spec.actual_duration_hrs_sigma})")
    print(f"  Mismatch window: after 4pm (declared ends) while actual effect persists")
    print(f"{'='*90}")

    # Column headers: key hours around exercise
    # Exercise at 1pm (h6), declared ends 4pm (h9), actual persists to ~7pm-1am (h12-h18)
    print(f"{'Path':>4}  {'Seed':>4}  {'ActScl':>6}  {'ActDur':>6}  "
          f"{'BG@1p':>6}  {'BG@4p':>6}  {'BG@7p':>6}  {'BG@10p':>6}  {'BG@1a':>6}  "
          f"{'Min':>6}  {'MinAt':>6}")
    print("-" * 90)

    all_bg_4pm = []
    all_bg_7pm = []
    all_bg_10pm = []
    all_min_bg = []

    for i in range(n_paths):
        seed = i
        profile = PatientProfile(
            meals=[],
            carb_count_sigma=0.0,
            carb_count_bias=0.0,
            sensitivity_sigma=0.0,
            starting_bg=100.0,  # Start at target to isolate exercise effect
            exercises_per_week=7.0,
            exercise_spec=ex_spec,
            rescue_carbs_enabled=False,
            algorithm_settings=settings,
        )
        result = _run_sim(profile, 'trio', n_days=1, seed=seed)
        trace = result.days[0].bg_trace

        def bg_at_hour(h):
            idx = h * 12
            if idx < len(trace):
                return trace[idx][1]
            return trace[-1][1]

        # Get actual exercise params from the event
        ex_events = result.days[0].exercises
        if ex_events:
            act_scl = ex_events[0].actual_scalar
            act_dur = ex_events[0].actual_duration_hrs
        else:
            act_scl = act_dur = 0.0

        bg_1pm = bg_at_hour(6)    # h6 from 7am = 1pm
        bg_4pm = bg_at_hour(9)    # declared ends
        bg_7pm = bg_at_hour(12)   # 3h into mismatch window
        bg_10pm = bg_at_hour(15)  # 6h into mismatch window
        bg_1am = bg_at_hour(18)   # 9h post-exercise

        bgs = [bg for _, bg in trace]
        min_bg = min(bgs)
        min_idx = bgs.index(min_bg)
        min_hour = min_idx / 12
        min_clock = _clock(int(min_hour * 60))

        all_bg_4pm.append(bg_4pm)
        all_bg_7pm.append(bg_7pm)
        all_bg_10pm.append(bg_10pm)
        all_min_bg.append(min_bg)

        print(f"{i:4d}  {seed:4d}  {act_scl:6.3f}  {act_dur:5.1f}h  "
              f"{bg_1pm:6.1f}  {bg_4pm:6.1f}  {bg_7pm:6.1f}  {bg_10pm:6.1f}  {bg_1am:6.1f}  "
              f"{min_bg:6.1f}  {min_clock:>6}")

    print()
    for label, arr in [("BG@4pm", all_bg_4pm), ("BG@7pm", all_bg_7pm),
                        ("BG@10pm", all_bg_10pm), ("MinBG", all_min_bg)]:
        a = np.array(arr)
        print(f"  {label:>8}: Mean={np.mean(a):5.1f}, SD={np.std(a):5.1f}, "
              f"Range=[{np.min(a):.1f}, {np.max(a):.1f}]")

    sd_7pm = np.std(all_bg_7pm)
    print(f"\nKey: SD at 7pm (3h into mismatch) = {sd_7pm:.1f} mg/dL")
    if sd_7pm > 3:
        print("PASS: Exercise mismatch creates meaningful BG spread")
    else:
        print("INVESTIGATE: Exercise mismatch not creating enough spread")


# ─── Runner ────────────────────────────────────────────────────────────────────

SCENARIOS = {
    'flat_bg': test_flat_bg,
    'single_meal_perfect': test_single_meal_perfect,
    'single_meal_underdeclared': test_single_meal_underdeclared,
    'sensitivity_only': test_sensitivity_only,
    'ns_profile_single': test_ns_profile_single,
    'breakfast_zoom': test_breakfast_zoom,
    'breakfast_realistic': test_breakfast_realistic,
    'exercise_check': test_exercise_check,
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] == '--list':
        print("Available scenarios:")
        for name in SCENARIOS:
            print(f"  {name}")
        print(f"  all  (run all)")
        return

    names = sys.argv[1:]
    if 'all' in names:
        names = list(SCENARIOS.keys())

    for name in names:
        if name not in SCENARIOS:
            print(f"Unknown scenario: {name}")
            print(f"Available: {', '.join(SCENARIOS.keys())}")
            sys.exit(1)
        SCENARIOS[name]()


if __name__ == '__main__':
    main()
