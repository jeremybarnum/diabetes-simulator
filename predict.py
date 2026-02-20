#!/usr/bin/env python3
"""
Command-line Loop prediction tool.

Usage:
    # First run (no settings.json exists)
    python3 predict.py --isf 100 --cr 9 --basal 0.45 --target 110 \
        --bg -30 120 -20 125 -10 130 0 140

    # Subsequent runs (uses settings.json)
    python3 predict.py --bg -30 120 -20 125 -10 130 0 140

    # With carbs and bolus
    python3 predict.py --bg -30 120 -20 125 -10 130 0 140 \
        --carbs 0 30 3.0 --bolus 0 3.0

    # Override a setting
    python3 predict.py --isf 95 --bg -30 120 -20 125 -10 130 0 140
"""

import sys
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

from algorithms.base import AlgorithmInput
from algorithms.loop.loop_algorithm import LoopAlgorithm


# Settings file location (same directory as script)
SETTINGS_FILE = Path(__file__).parent / "settings.json"

# Default values for optional settings
DEFAULT_SETTINGS = {
    'max_basal_rate': 3.0,
    'max_bolus': 5.0,
    'suspend_threshold': 75.0,
    'enable_irc': True,
    'duration_of_insulin_action': 6.0,  # Fiasp
}


def parse_bg_history(values: List[float]) -> List[Tuple[float, float]]:
    """
    Parse BG history from list of numbers.

    Format: [-30, 120, -25, 122, -20, 125, 0, 140]
    Returns: [(time_minutes, bg_value), ...]
    """
    if not values:
        return []

    if len(values) % 2 != 0:
        raise ValueError(f"BG history must have even number of values (pairs of time,bg). Got {len(values)} values.")

    history = []
    for i in range(0, len(values), 2):
        time_min = values[i]
        bg_value = values[i + 1]
        history.append((time_min, bg_value))

    return history


def interpolate_bg_history(history: List[Tuple[float, float]], interval_minutes: float = 5.0) -> List[Tuple[float, float]]:
    """
    Interpolate BG history to regular intervals (default 5 minutes).

    Takes arbitrary time points and linearly interpolates to create
    the standard 5-minute intervals that Loop expects.

    Args:
        history: List of (time, bg) tuples at arbitrary intervals
        interval_minutes: Target interval (default 5 minutes)

    Returns:
        List of (time, bg) tuples at regular intervals
    """
    if not history:
        return []

    if len(history) == 1:
        return history

    # Sort by time
    history = sorted(history, key=lambda x: x[0])

    # Get time range
    start_time = history[0][0]
    end_time = history[-1][0]

    # Generate regular intervals
    interpolated = []

    # Start from the first time point, rounded down to nearest interval
    current_time = int(start_time / interval_minutes) * interval_minutes

    while current_time <= end_time:
        # Find the two points to interpolate between
        bg_value = None

        # Check if we have an exact match
        for t, bg in history:
            if abs(t - current_time) < 0.01:  # Close enough to exact
                bg_value = bg
                break

        if bg_value is None:
            # Find surrounding points for interpolation
            before = None
            after = None

            for i, (t, bg) in enumerate(history):
                if t <= current_time:
                    before = (t, bg)
                if t >= current_time and after is None:
                    after = (t, bg)
                    break

            if before and after and before[0] != after[0]:
                # Linear interpolation
                t1, bg1 = before
                t2, bg2 = after
                fraction = (current_time - t1) / (t2 - t1)
                bg_value = bg1 + fraction * (bg2 - bg1)
            elif before:
                # At or past the last point
                bg_value = before[1]
            elif after:
                # Before the first point
                bg_value = after[1]

        if bg_value is not None:
            interpolated.append((current_time, bg_value))

        current_time += interval_minutes

    return interpolated


def parse_carbs(values: List[float]) -> List[Tuple[float, float, float]]:
    """
    Parse carb entries from list of numbers.

    Format: [-60, 30, 3.0, -30, 20, 2.5]
    Returns: [(time_minutes, grams, absorption_hours), ...]
    """
    if not values:
        return []

    if len(values) % 3 != 0:
        raise ValueError(f"Carb entries must have values in groups of 3 (time,grams,hours). Got {len(values)} values.")

    carbs = []
    for i in range(0, len(values), 3):
        time_min = values[i]
        grams = values[i + 1]
        hours = values[i + 2]
        carbs.append((time_min, grams, hours))

    return carbs


def parse_bolus(values: List[float]) -> List[Tuple[float, float]]:
    """
    Parse bolus history from list of numbers.

    Format: [-60, 3.0, -30, 1.5]
    Returns: [(time_minutes, units), ...]
    """
    if not values:
        return []

    if len(values) % 2 != 0:
        raise ValueError(f"Bolus history must have even number of values (pairs of time,units). Got {len(values)} values.")

    boluses = []
    for i in range(0, len(values), 2):
        time_min = values[i]
        units = values[i + 1]
        boluses.append((time_min, units))

    return boluses


def load_settings() -> Optional[dict]:
    """Load settings from settings.json if it exists."""
    if not SETTINGS_FILE.exists():
        return None

    try:
        with open(SETTINGS_FILE, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"⚠️  Warning: Could not load {SETTINGS_FILE}: {e}")
        return None


def save_settings(settings: dict):
    """Save settings to settings.json."""
    try:
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(settings, f, indent=2)
    except IOError as e:
        print(f"⚠️  Warning: Could not save {SETTINGS_FILE}: {e}")


def build_settings(args) -> dict:
    """
    Build complete settings dict from file and CLI args.
    CLI args override file values.
    """
    # Load existing settings
    settings = load_settings()

    # Check if this is first run (no file exists)
    if settings is None:
        # First run - require all mandatory settings
        required = ['isf', 'cr', 'basal', 'target']
        missing = [r for r in required if getattr(args, r) is None]

        if missing:
            print(f"❌ Error: No settings.json found.")
            print(f"   Please specify all required settings on first run:")
            print(f"   --isf <value> --cr <value> --basal <value> --target <value>")
            print(f"\n   Missing: {', '.join(missing)}")
            sys.exit(1)

        # Create new settings from CLI args + defaults
        settings = DEFAULT_SETTINGS.copy()
        settings['insulin_sensitivity_factor'] = args.isf
        settings['carb_ratio'] = args.cr
        settings['basal_rate'] = args.basal
        settings['target'] = args.target

        print(f"✅ Created {SETTINGS_FILE} with initial settings")
    else:
        # Subsequent run - use file, override with CLI args
        print(f"ℹ️  Loaded settings from {SETTINGS_FILE}")

        # Override with CLI args if provided
        if args.isf is not None:
            settings['insulin_sensitivity_factor'] = args.isf
            print(f"   Overriding ISF: {args.isf}")
        if args.cr is not None:
            settings['carb_ratio'] = args.cr
            print(f"   Overriding CR: {args.cr}")
        if args.basal is not None:
            settings['basal_rate'] = args.basal
            print(f"   Overriding Basal: {args.basal}")
        if args.target is not None:
            settings['target'] = args.target
            print(f"   Overriding Target: {args.target}")

    # Save updated settings
    save_settings(settings)

    return settings


def print_effect_breakdown_table(
    times: List[float],
    insulin_effects: List[float],
    carb_effects: List[float],
    momentum_effects: List[float],
    irc_effects: List[float],
    glucose_predictions: List[float],
    baseline_bg: float,
    title: str,
    cgm_history: List[Tuple[float, float]] = None,
    momentum_eventual: float = None,
    irc_discrepancy: float = None,
    irc_eventual: float = None,
    irc_discrepancies_timeline: List[Tuple[float, float]] = None,
    irc_proportional: float = None,
    irc_integral: float = None,
    irc_differential: float = None,
    irc_total_correction: float = None,
    irc_discrepancies_count: int = None,
    irc_effect_next: float = None
):
    """Print detailed effect breakdown table with enhanced IRC debugging."""

    print("\n" + "=" * 135)
    print(f"DETAILED EFFECT BREAKDOWN - {title}")
    print("=" * 135)
    print()

    # Print historical CGM data if provided
    if cgm_history:
        current_time = cgm_history[-1][0] if cgm_history else 0
        history_start = current_time - 35.0  # Show last ~35 minutes
        relevant_history = [(t, bg) for t, bg in cgm_history if history_start <= t <= current_time]

        if relevant_history:
            print("HISTORY (CGM Lookback Window)")
            print("-" * 135)
            print(f"{'  Time':>6} | {'    Type':>8} | {'   Insulin':>10} | {'      Carbs':>11} | {'   Momentum':>11} | {'       IRC':>10} | "
                  f"{'   Combined':>11} | {'    Current':>11} | {'   Eventual':>11}")
            print(f"{'(min)':>6} | {'        ':>8} | {'     Effect':>10} | {'     Effect':>11} | {'     Effect':>11} | {'       Adj':>10} | "
                  f"{'     Effect':>11} | {'         BG':>11} | {'         BG':>11}")
            print(f"{'':>6} | {'        ':>8} | {'          ':>10} | {'           ':>11} | {'           ':>11} | {'          ':>10} | "
                  f"{'           ':>11} | {'    (mg/dL)':>11} | {'    (final)':>11}")
            print("-" * 135)

            for t, bg in relevant_history:
                print(f"{t:>6.0f} | {'Actual':>8} | {'---':>10} | {'---':>11} | {'---':>11} | {'---':>10} | "
                      f"{'---':>11} | {bg:>11.1f} | {'---':>11}")

            print("-" * 135)
            print()

            # Calculate velocity from history for summary
            if len(relevant_history) >= 2:
                first_t, first_bg = relevant_history[0]
                last_t, last_bg = relevant_history[-1]
                time_span = last_t - first_t
                bg_change = last_bg - first_bg
                velocity = bg_change / time_span if time_span > 0 else 0.0
            else:
                velocity = 0.0

            # Print summary section
            print("EFFECT SUMMARY (How history influences predictions)")
            print("-" * 135)
            print()
            print(f"Momentum:")
            print(f"  • Velocity from history: {velocity:+.2f} mg/dL/min")
            if momentum_eventual is not None:
                print(f"  • Effect on eventual BG: {momentum_eventual:+.1f} mg/dL")
            else:
                print(f"  • Effect on eventual BG: 0.0 mg/dL (insufficient history)")
            print()

            if irc_discrepancy is not None and irc_eventual is not None:
                print(f"IRC (Integral Retrospective Correction):")
                print(f"  • Most recent discrepancy: {irc_discrepancy:+.1f} mg/dL")
                print(f"  • Effect on eventual BG: {irc_eventual:+.1f} mg/dL")

                # Show discrepancy timeline if available
                if irc_discrepancies_timeline and len(irc_discrepancies_timeline) > 0:
                    print(f"  • Discrepancy history (30-min buckets, last {len(irc_discrepancies_timeline)}):")
                    # Show most recent 6 buckets (last 3 hours)
                    recent_discs = irc_discrepancies_timeline[-6:] if len(irc_discrepancies_timeline) > 6 else irc_discrepancies_timeline
                    for disc in recent_discs:
                        if isinstance(disc, dict):
                            disc_time = disc.get('time', 0)
                            disc_value = disc.get('discrepancy', 0)
                        else:
                            disc_time, disc_value = disc
                        print(f"      t={disc_time:>6.0f} min: {disc_value:>+7.1f} mg/dL")

                if abs(irc_discrepancy) > 10:
                    if irc_discrepancy > 0:
                        print(f"  • Interpretation: BG running higher than expected (under-dosed or settings error)")
                    else:
                        print(f"  • Interpretation: BG running lower than expected (over-dosed or more sensitive)")
                else:
                    print(f"  • Interpretation: Predictions tracking actual BG well")
            else:
                print(f"IRC (Integral Retrospective Correction):")
                print(f"  • Status: Disabled or insufficient history")

            print()
            print("-" * 135)
            print()

    print("PREDICTIONS")
    print("-" * 150)
    print(f"{'  Time':>6} | {'    Type':>8} | {'   Insulin':>10} | {'      Carbs':>11} | {'   Momentum':>11} | {'       IRC':>10} | "
          f"{'   Combined':>11} | {'    Current':>11} | {'       Next':>11} | {'   Eventual':>11}")
    print(f"{'(min)':>6} | {'        ':>8} | {'     Effect':>10} | {'     Effect':>11} | {'     Effect':>11} | {'       Adj':>10} | "
          f"{'     Effect':>11} | {'         BG':>11} | {'         BG':>11} | {'         BG':>11}")
    print(f"{'':>6} | {'        ':>8} | {'          ':>10} | {'           ':>11} | {'           ':>11} | {'          ':>10} | "
          f"{'           ':>11} | {'    (mg/dL)':>11} | {'    (+5min)':>11} | {'    (final)':>11}")
    print("-" * 150)

    eventual_bg = glucose_predictions[-1] if glucose_predictions else baseline_bg

    for i in range(len(times)):
        t = times[i]
        insulin_eff = insulin_effects[i] if i < len(insulin_effects) else 0.0
        carb_eff = carb_effects[i] if i < len(carb_effects) else 0.0
        momentum_eff = momentum_effects[i] if i < len(momentum_effects) else 0.0
        irc_eff = irc_effects[i] if i < len(irc_effects) else 0.0
        combined_eff = insulin_eff + carb_eff + momentum_eff + irc_eff

        current_bg = glucose_predictions[i] if i < len(glucose_predictions) else baseline_bg
        next_bg = glucose_predictions[i + 1] if i + 1 < len(glucose_predictions) else current_bg

        print(f"{t:>6.0f} | {'Predict':>8} | {insulin_eff:>+10.1f} | {carb_eff:>+11.1f} | {momentum_eff:>+11.1f} | {irc_eff:>+10.1f} | "
              f"{combined_eff:>+11.1f} | {current_bg:>11.1f} | {next_bg:>11.1f} | {eventual_bg:>11.1f}")

    print("-" * 150)
    print()


def extract_effects(loop: LoopAlgorithm, input_data: AlgorithmInput, num_points: int = 73):
    """
    Extract effect breakdown from Loop predictions.
    Similar to test_complete_loop.py extract_effects_from_predictions().

    Default num_points=73 covers full 360-minute (6-hour) prediction horizon
    at 5-minute intervals (360/5 = 72 intervals = 73 points including t=0).
    """
    output = loop.recommend(input_data)

    baseline_bg = input_data.cgm_reading

    # Get main predictions
    actual_predictions = output.glucose_predictions.get('main', [])

    # Time points (5-minute intervals)
    times = [float(i * 5) for i in range(num_points)]

    # Get individual effect predictions
    insulin_predictions = output.glucose_predictions.get('insulin_only', [])
    carb_predictions = output.glucose_predictions.get('carbs_only', [])

    # Get momentum timeline from output (before blending)
    momentum_timeline = output.momentum_effect_timeline if hasattr(output, 'momentum_effect_timeline') else []

    # Calculate effects for each time point
    insulin_effects = []
    carb_effects = []
    momentum_effects = []
    irc_effects = []

    for i in range(num_points):
        # Insulin effect - NOTE: insulin_predictions already contains EFFECTS (not absolute BG)
        # The Loop algorithm returns cumulative effects (deltas from baseline), not glucose values
        if i < len(insulin_predictions):
            insulin_eff = insulin_predictions[i]
        else:
            insulin_eff = 0.0
        insulin_effects.append(insulin_eff)

        # Carb effect - NOTE: carb_predictions already contains EFFECTS (not absolute BG)
        # The Loop algorithm returns cumulative effects (deltas from baseline), not glucose values
        if i < len(carb_predictions):
            carb_eff = carb_predictions[i]
        else:
            carb_eff = 0.0
        carb_effects.append(carb_eff)

        # IRC effect (constant across prediction)
        irc_eff = output.irc_effect_eventual if output.irc_effect_eventual is not None else 0.0
        irc_effects.append(irc_eff)

        # Momentum - get from momentum timeline (cumulative effect)
        current_time = times[i]
        momentum_eff = 0.0
        if momentum_timeline:
            # Find the momentum effect for this time point
            for t, effect in momentum_timeline:
                if t == current_time:
                    momentum_eff = effect
                    break
        momentum_effects.append(momentum_eff)

    return {
        'times': times,
        'insulin_effects': insulin_effects,
        'carb_effects': carb_effects,
        'momentum_effects': momentum_effects,
        'irc_effects': irc_effects,
        'glucose_predictions': actual_predictions[:num_points]
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Loop prediction tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Input data
    parser.add_argument('--bg', type=float, nargs='*', required=True,
                        help='BG history: -30 120 -25 122 -20 125 0 140 (pairs of time,bg)')
    parser.add_argument('--carbs', type=float, nargs='*', default=[],
                        help='Carb entries: -60 30 3.0 -30 20 2.5 (triplets of time,grams,hours)')
    parser.add_argument('--bolus', type=float, nargs='*', default=[],
                        help='Bolus history: -60 3.0 -30 1.5 (pairs of time,units)')

    # Settings (override file)
    parser.add_argument('--isf', type=float, default=None,
                        help='Insulin sensitivity factor (mg/dL/U)')
    parser.add_argument('--cr', type=float, default=None,
                        help='Carb ratio (g/U)')
    parser.add_argument('--basal', type=float, default=None,
                        help='Basal rate (U/hr)')
    parser.add_argument('--target', type=float, default=None,
                        help='Target BG (mg/dL)')

    args = parser.parse_args()

    # Build settings
    print()
    settings = build_settings(args)
    print()

    # Parse inputs
    try:
        bg_history_raw = parse_bg_history(args.bg)
        carb_entries = parse_carbs(args.carbs)
        bolus_history = parse_bolus(args.bolus)
    except ValueError as e:
        print(f"❌ Error parsing input: {e}")
        sys.exit(1)

    # Validate we have at least one BG reading
    if not bg_history_raw:
        print("❌ Error: No BG history provided. At least one reading required.")
        sys.exit(1)

    # Interpolate BG history to 5-minute intervals
    bg_history = interpolate_bg_history(bg_history_raw, interval_minutes=5.0)

    # Show interpolation info if input wasn't already at 5-minute intervals
    if len(bg_history) != len(bg_history_raw):
        print(f"ℹ️  Interpolated {len(bg_history_raw)} BG readings to {len(bg_history)} readings at 5-minute intervals")
        print()

    # Check history sufficiency (warn but continue)
    current_time = bg_history[-1][0]
    history_span = current_time - bg_history[0][0] if len(bg_history) > 1 else 0

    if len(bg_history) < 4 or history_span < 15:
        print(f"⚠️  Warning: Only {len(bg_history)} readings spanning {history_span:.0f} minutes.")
        print(f"    Momentum requires 4+ readings spanning 15+ minutes.")

    if len(bg_history) < 7 or history_span < 30:
        print(f"⚠️  Warning: IRC may be limited (requires 7+ readings spanning 30+ minutes).")

    if len(bg_history) == 1:
        # Generate flat history
        single_bg = bg_history[0][1]
        print(f"⚠️  Warning: Only one BG reading provided. Generating flat history at {single_bg:.0f} mg/dL.")
        print(f"    Momentum will be zero. IRC disabled.")
        # Create 2 hours of flat history
        bg_history = [(float(i * -5), single_bg) for i in range(24, -1, -1)]

    print()

    # Display inputs
    print("=" * 80)
    print("PREDICTION INPUT")
    print("=" * 80)
    print()
    print(f"Settings:")
    print(f"  • ISF: {settings['insulin_sensitivity_factor']:.0f} mg/dL/U")
    print(f"  • CR: {settings['carb_ratio']:.1f} g/U")
    print(f"  • Basal: {settings['basal_rate']:.2f} U/hr")
    print(f"  • Target: {settings['target']:.0f} mg/dL")
    print(f"  • IRC: {'Enabled' if settings.get('enable_irc', True) else 'Disabled'}")
    print()

    current_bg = bg_history[-1][1]
    print(f"Current BG: {current_bg:.0f} mg/dL (at t={current_time:.0f} min)")
    print(f"History: {len(bg_history)} readings from t={bg_history[0][0]:.0f} to t={bg_history[-1][0]:.0f} min")

    if carb_entries:
        print(f"\nCarb Entries:")
        for t, g, h in carb_entries:
            print(f"  • t={t:.0f} min: {g:.0f}g ({h:.1f}hr absorption)")

    if bolus_history:
        print(f"\nBolus History:")
        for t, u in bolus_history:
            print(f"  • t={t:.0f} min: {u:.2f}U")

    print()

    # Run Loop
    loop = LoopAlgorithm(settings)

    input_data = AlgorithmInput(
        cgm_reading=current_bg,
        timestamp=current_time,
        cgm_history=bg_history,
        current_basal=settings['basal_rate'],
        bolus_history=bolus_history,
        carb_entries=carb_entries,
        settings=settings
    )

    output = loop.recommend(input_data)

    # Display results
    print("=" * 80)
    print("PREDICTION OUTPUT")
    print("=" * 80)
    print()

    print(f"IOB: {output.iob:.2f} U")
    print(f"COB: {output.cob:.1f} g")

    predictions = output.glucose_predictions.get('main', [])
    if predictions:
        eventual_bg = predictions[-1]
        next_bg = predictions[1] if len(predictions) > 1 else predictions[0]
        print(f"Next BG (5 min): {next_bg:.0f} mg/dL")
        print(f"Eventual BG (6 hr): {eventual_bg:.0f} mg/dL")

    print()
    print(f"Recommendation:")
    print(f"  • Temp Basal: {output.temp_basal_rate:.2f} U/hr for {output.temp_basal_duration:.0f} min")
    print(f"  • Insulin Req: {output.insulin_req:+.2f} U")
    print(f"  • Reason: {output.reason}")

    # Extract and display effect breakdown
    effects = extract_effects(loop, input_data)

    print_effect_breakdown_table(
        times=effects['times'],
        insulin_effects=effects['insulin_effects'],
        carb_effects=effects['carb_effects'],
        momentum_effects=effects['momentum_effects'],
        irc_effects=effects['irc_effects'],
        glucose_predictions=effects['glucose_predictions'],
        baseline_bg=current_bg,
        title="Prediction Analysis",
        cgm_history=bg_history,
        momentum_eventual=output.momentum_effect_eventual,
        irc_discrepancy=output.irc_discrepancy,
        irc_eventual=output.irc_effect_eventual,
        irc_discrepancies_timeline=output.irc_discrepancies_timeline
    )

    print("=" * 80)
    print()


if __name__ == '__main__':
    main()
