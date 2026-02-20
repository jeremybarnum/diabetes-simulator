"""
OpenAPS COB calculation - Full deviation-based model

Ports of:
- trio-oref/lib/determine-basal/cob.js (detectCarbAbsorption)
- trio-oref/lib/meal/total.js (recentCarbs)
- trio-oref/lib/meal/history.js (findMealInputs)

The deviation-based model calculates COB by:
1. Bucketing glucose data into 5-min intervals
2. Computing BGI (BG impact from insulin) at each interval
3. Calculating deviation = actual BG change - BGI
4. Using deviations to estimate carb absorption
"""

from typing import Dict, List, Any, Optional
from algorithms.openaps.insulin_math import iob_calc


def detect_carb_absorption(
    glucose_data: List[Dict[str, Any]],
    iob_inputs: Dict[str, Any],
    basalprofile: List[Dict[str, Any]],
    meal_time_ms: int,
    ci_time_ms: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Detect carb absorption by analyzing glucose dynamics.

    Port of detectCarbAbsorption() from cob.js.

    Buckets glucose data, calculates BGI from IOB at each point,
    computes deviations, and estimates carbs absorbed.

    Args:
        glucose_data: CGM readings (newest first) with 'glucose'/'sgv' and
                     'dateString'/'display_time'/'date'
        iob_inputs: Dict with 'profile' and 'history' for IOB calculation
        basalprofile: Basal schedule array
        meal_time_ms: Meal start time in milliseconds
        ci_time_ms: Optional current time for deviation calculation (CI mode)

    Returns:
        Dict with:
            - carbsAbsorbed: Estimated grams absorbed
            - currentDeviation: Most recent deviation (mg/dL/5min)
            - maxDeviation: Max deviation in window
            - minDeviation: Min deviation in window
            - slopeFromMaxDeviation: Rate of deviation change from max
            - slopeFromMinDeviation: Rate of deviation change from min
            - allDeviations: Array of all computed deviations
    """
    from algorithms.openaps.iob import find_insulin, iob_total

    profile = iob_inputs.get('profile', {})
    history = iob_inputs.get('history', [])

    # Normalize glucose entries
    glucose_data = [_normalize_glucose(g) for g in glucose_data]

    # Get treatments for IOB calculation (once, not per BG point)
    # JS cob.js calls find_insulin(inputs.iob_inputs) which uses the current clock,
    # NOT the meal time. meal_time is only for filtering BG data.
    clock_for_treatments = ci_time_ms if ci_time_ms else meal_time_ms
    # Use the latest BG time if available as a fallback
    if not clock_for_treatments and glucose_data:
        clock_for_treatments = _get_glucose_time_ms(glucose_data[0]) or meal_time_ms
    treatments = find_insulin(history, profile, clock_for_treatments)

    # Bucket glucose data
    bucketed_data = []
    if not glucose_data or not glucose_data[0].get('glucose'):
        return _empty_absorption()

    bucketed_data.append(dict(glucose_data[0]))
    j = 0
    last_bgi = 0
    found_pre_meal = False

    first_bg = glucose_data[0].get('glucose', 0)
    if not first_bg or first_bg < 39:
        last_bgi = -1

    meal_time = meal_time_ms

    for i in range(1, len(glucose_data)):
        bg_time_ms = _get_glucose_time_ms(glucose_data[i])
        if bg_time_ms is None:
            continue

        bg = glucose_data[i].get('glucose', 0)
        if not bg or bg < 39:
            continue

        # Only consider BGs for 6h after meal for calculating COB
        hours_after_meal = (bg_time_ms - meal_time) / (60 * 60 * 1000)
        if hours_after_meal > 6 or found_pre_meal:
            continue
        elif hours_after_meal < 0:
            found_pre_meal = True

        # Only consider last ~45m of data in CI mode
        if ci_time_ms is not None:
            hours_ago = (ci_time_ms - bg_time_ms) / (45 * 60 * 1000)
            if hours_ago > 1 or hours_ago < 0:
                continue

        # Get last bucketed time
        last_bg_time_ms = bucketed_data[-1].get('date', 0)
        if not last_bg_time_ms and last_bgi >= 0:
            last_bg_time_ms = _get_glucose_time_ms(glucose_data[last_bgi])

        if last_bg_time_ms is None:
            continue

        elapsed_minutes = (bg_time_ms - last_bg_time_ms) / (60 * 1000)

        if abs(elapsed_minutes) > 8:
            # Interpolate missing data points
            last_bg = glucose_data[last_bgi].get('glucose', 0) if last_bgi >= 0 else bg
            elapsed_minutes = min(240, abs(elapsed_minutes))

            while elapsed_minutes > 5:
                previous_bg_time_ms = last_bg_time_ms - 5 * 60 * 1000
                j += 1
                gap_delta = bg - last_bg
                previous_bg = last_bg + (5 / elapsed_minutes * gap_delta)
                bucketed_data.append({
                    'date': previous_bg_time_ms,
                    'glucose': round(previous_bg),
                })
                elapsed_minutes -= 5
                last_bg = previous_bg
                last_bg_time_ms = previous_bg_time_ms

        elif abs(elapsed_minutes) > 2:
            j += 1
            entry = dict(glucose_data[i])
            entry['date'] = bg_time_ms
            bucketed_data.append(entry)
        else:
            # Average closely-spaced readings
            bucketed_data[j]['glucose'] = (bucketed_data[j].get('glucose', 0) + bg) / 2

        last_bgi = i

    # Calculate deviations
    current_deviation = None
    slope_from_max = 0
    slope_from_min = 999
    max_deviation = 0
    min_deviation = 999
    all_deviations = []
    carbs_absorbed = 0

    isf_profile = profile.get('isfProfile', {})
    min_5m_ci = profile.get('min_5m_carbimpact', 8)
    carb_ratio = profile.get('carb_ratio', 10)

    for i in range(len(bucketed_data) - 3):
        bg_time_ms = bucketed_data[i].get('date', 0)
        if not bg_time_ms:
            continue

        # ISF lookup
        sens = _isf_lookup(isf_profile, bg_time_ms)

        bg = bucketed_data[i].get('glucose')
        if bg is None or bg < 39:
            continue
        if bucketed_data[i + 3].get('glucose', 0) < 39:
            continue

        avg_delta = (bg - bucketed_data[i + 3].get('glucose', bg)) / 3
        delta = bg - bucketed_data[i + 1].get('glucose', bg)

        avg_delta = round(avg_delta * 100) / 100

        # Calculate IOB at this BG time
        iob = iob_total(treatments, bg_time_ms, profile)
        bgi = round((-iob['activity'] * sens * 5) * 100) / 100

        deviation = round((delta - bgi) * 100) / 100

        # Track current deviation (most recent point)
        if i == 0:
            current_deviation = round((avg_delta - bgi) * 1000) / 1000
            if ci_time_ms and ci_time_ms > bg_time_ms:
                all_deviations.append(round(current_deviation))

        elif ci_time_ms and ci_time_ms > bg_time_ms:
            avg_deviation = round((avg_delta - bgi) * 1000) / 1000
            deviation_slope = (avg_deviation - current_deviation) / (bg_time_ms - ci_time_ms) * 1000 * 60 * 5

            if avg_deviation > max_deviation:
                slope_from_max = min(0, deviation_slope)
                max_deviation = avg_deviation
            if avg_deviation < min_deviation:
                slope_from_min = max(0, deviation_slope)
                min_deviation = avg_deviation

            all_deviations.append(round(avg_deviation))

        # If BG time is after meal, estimate carb absorption
        if bg_time_ms > meal_time:
            ci = max(deviation, (current_deviation or 0) / 2, min_5m_ci)
            absorbed = ci * carb_ratio / sens
            carbs_absorbed += absorbed

    return {
        'carbsAbsorbed': carbs_absorbed,
        'currentDeviation': current_deviation,
        'maxDeviation': max_deviation,
        'minDeviation': min_deviation,
        'slopeFromMaxDeviation': slope_from_max,
        'slopeFromMinDeviation': slope_from_min,
        'allDeviations': all_deviations,
    }


def recent_carbs(
    treatments: List[Dict[str, Any]],
    time_ms: int,
    profile: Dict[str, Any],
    glucose_data: Optional[List[Dict]] = None,
    pump_history: Optional[List[Dict]] = None,
    basalprofile: Optional[List[Dict]] = None,
) -> Dict[str, Any]:
    """
    Calculate recent carbs and COB using deviation-based model.

    Port of recentCarbs() from meal/total.js.

    Iterates carb entries newest-first, calling detectCarbAbsorption for each
    to compute COB. Also calculates deviation slopes for UAM detection.

    Args:
        treatments: Carb treatment entries with 'carbs', 'timestamp'
        time_ms: Current time in milliseconds
        profile: Full profile dict
        glucose_data: CGM readings for deviation calculation
        pump_history: Pump history for IOB calculation
        basalprofile: Basal schedule

    Returns:
        Dict with:
            - carbs: Total carbs in window
            - mealCOB: Carbs on board
            - currentDeviation, maxDeviation, minDeviation
            - slopeFromMaxDeviation, slopeFromMinDeviation
            - allDeviations: Array of deviation values
            - lastCarbTime: Most recent carb entry time
            - bwFound: Bolus wizard usage detected
    """
    bp = basalprofile or profile.get('basalprofile', [])

    carbs_total = 0
    ns_carbs = 0
    bw_carbs = 0
    journal_carbs = 0
    bw_found = False
    meal_cob = 0
    last_carb_time = 0
    carbs_to_remove = 0

    six_hours_ago = time_ms - 6 * 60 * 60 * 1000

    # Build IOB inputs for deviation calculation
    iob_inputs = {
        'profile': profile,
        'history': pump_history or [],
    }

    # Process carb treatments (if any)
    if treatments:
        sorted_treatments = sorted(treatments, key=lambda t: _get_treatment_time(t) or 0, reverse=True)

        for treatment in sorted_treatments:
            treatment_time = _get_treatment_time(treatment)
            if treatment_time is None:
                continue

            if treatment_time <= six_hours_ago or treatment_time > time_ms:
                continue

            carbs = treatment.get('carbs', 0)
            if carbs < 1:
                continue

            # Classify carbs
            if treatment.get('nsCarbs', 0) >= 1:
                ns_carbs += treatment['nsCarbs']
            elif treatment.get('bwCarbs', 0) >= 1:
                bw_carbs += treatment['bwCarbs']
                bw_found = True
            elif treatment.get('journalCarbs', 0) >= 1:
                journal_carbs += treatment['journalCarbs']
            else:
                ns_carbs += carbs  # Default to NS carbs

            carbs_total += carbs
            last_carb_time = max(last_carb_time, treatment_time)

            # Calculate COB for this carb entry
            if glucose_data:
                cob_inputs_meal_time = treatment_time
                absorption = detect_carb_absorption(
                    glucose_data=glucose_data,
                    iob_inputs=iob_inputs,
                    basalprofile=bp,
                    meal_time_ms=cob_inputs_meal_time,
                )
                my_carbs_absorbed = absorption.get('carbsAbsorbed', 0)
                my_meal_cob = max(0, carbs_total - my_carbs_absorbed)

                if isinstance(my_meal_cob, (int, float)) and not _is_nan(my_meal_cob):
                    if my_meal_cob >= meal_cob:
                        meal_cob = my_meal_cob
                        carbs_to_remove = 0
                    else:
                        carbs_to_remove += carbs
                else:
                    carbs_to_remove += carbs
            else:
                # Fallback to simple linear absorption
                meal_cob = _simple_cob(carbs_total, treatment_time, time_ms)

        # Remove carbs from fully-absorbed entries
        carbs_total -= carbs_to_remove
        ns_carbs = max(0, ns_carbs)

    # Always calculate deviations for UAM detection (JS does this even without carbs)
    # meal/total.js lines 98-103: sets ciTime=now, mealTime=6h ago
    c = _empty_deviation_data()
    if glucose_data:
        c = detect_carb_absorption(
            glucose_data=glucose_data,
            iob_inputs=iob_inputs,
            basalprofile=bp,
            meal_time_ms=time_ms - 6 * 60 * 60 * 1000,
            ci_time_ms=time_ms,
        )

    # Apply maxCOB safety limit
    max_cob = profile.get('maxCOB', 120)
    if isinstance(max_cob, (int, float)) and not _is_nan(max_cob):
        meal_cob = min(max_cob, meal_cob)

    # Zombie carb safety: set mealCOB to 0 if deviations are missing
    if c.get('currentDeviation') is None:
        meal_cob = 0
    if c.get('maxDeviation') is None:
        meal_cob = 0

    return {
        'carbs': round(carbs_total * 1000) / 1000,
        'nsCarbs': round(ns_carbs * 1000) / 1000,
        'bwCarbs': round(bw_carbs * 1000) / 1000,
        'journalCarbs': round(journal_carbs * 1000) / 1000,
        'mealCOB': round(meal_cob),
        'currentDeviation': round(c.get('currentDeviation', 0) * 100) / 100 if c.get('currentDeviation') is not None else 0,
        'maxDeviation': round(c.get('maxDeviation', 0) * 100) / 100,
        'minDeviation': round(c.get('minDeviation', 999) * 100) / 100,
        'slopeFromMaxDeviation': round(c.get('slopeFromMaxDeviation', 0) * 1000) / 1000,
        'slopeFromMinDeviation': round(c.get('slopeFromMinDeviation', 999) * 1000) / 1000,
        'allDeviations': c.get('allDeviations', []),
        'lastCarbTime': last_carb_time,
        'bwFound': bw_found,
    }


def create_carb_entry(
    carbs: float,
    time_ms: int,
    absorption_time_hours: float = 3.0
) -> Dict[str, Any]:
    """Create a carb entry dict."""
    from datetime import datetime, timezone
    iso = datetime.fromtimestamp(time_ms / 1000, tz=timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.000Z')
    return {
        'carbs': carbs,
        'nsCarbs': carbs,
        'timestamp': iso,
        'created_at': iso,
        'date': time_ms,
        'absorption_time': absorption_time_hours,
    }


# --- Helper functions ---

def _normalize_glucose(entry: Dict) -> Dict:
    """Normalize glucose entry to have 'glucose' and 'date' fields."""
    result = dict(entry)
    if 'glucose' not in result and 'sgv' in result:
        result['glucose'] = result['sgv']
    if 'date' not in result:
        t = _get_glucose_time_ms(result)
        if t:
            result['date'] = t
    return result


def _get_glucose_time_ms(entry: Dict) -> Optional[int]:
    """Extract time in ms from glucose entry."""
    if 'date' in entry:
        d = entry['date']
        if isinstance(d, (int, float)):
            return int(d) if d > 1e10 else int(d * 1000)
    for field in ('dateString', 'display_time', 'timestamp'):
        if field in entry and entry[field]:
            return _parse_iso_ms(entry[field])
    return None


def _get_treatment_time(treatment: Dict) -> Optional[int]:
    """Extract time in ms from treatment."""
    for field in ('date', 'timestamp', 'created_at'):
        val = treatment.get(field)
        if val is None:
            continue
        if isinstance(val, (int, float)):
            return int(val) if val > 1e10 else int(val * 1000)
        if isinstance(val, str) and val:
            return _parse_iso_ms(val)
    return None


def _parse_iso_ms(iso_str: str) -> Optional[int]:
    """Parse ISO timestamp to milliseconds."""
    from datetime import datetime, timezone
    try:
        ts = iso_str.replace('Z', '+00:00')
        dt = datetime.fromisoformat(ts)
        return int(dt.timestamp() * 1000)
    except (ValueError, TypeError):
        return None


def _isf_lookup(isf_profile: Dict, time_ms: int) -> float:
    """Look up ISF at given time. Port of profile/isf.js isfLookup."""
    sensitivities = isf_profile.get('sensitivities', [])
    if not sensitivities:
        return 100  # default

    from datetime import datetime, timezone
    dt = datetime.fromtimestamp(time_ms / 1000, tz=timezone.utc)
    now_minutes = dt.hour * 60 + dt.minute

    sorted_s = sorted(sensitivities, key=lambda x: x.get('offset', 0))
    isf_val = sorted_s[-1].get('sensitivity', 100)

    for i in range(len(sorted_s) - 1):
        if now_minutes >= sorted_s[i].get('offset', 0) and \
           now_minutes < sorted_s[i + 1].get('offset', 0):
            isf_val = sorted_s[i].get('sensitivity', 100)
            break

    return isf_val


def _is_nan(val):
    """Check if value is NaN."""
    try:
        return val != val
    except TypeError:
        return False


def _simple_cob(total_carbs: float, carb_time_ms: int, now_ms: int) -> float:
    """Simple linear COB fallback when no glucose data available."""
    hours_elapsed = (now_ms - carb_time_ms) / (60 * 60 * 1000)
    absorption_hours = 3.0
    if hours_elapsed >= absorption_hours:
        return 0
    return total_carbs * (1 - hours_elapsed / absorption_hours)


def _empty_meal_data() -> Dict[str, Any]:
    """Return empty meal data structure."""
    return {
        'carbs': 0, 'nsCarbs': 0, 'bwCarbs': 0, 'journalCarbs': 0,
        'mealCOB': 0,
        'currentDeviation': 0, 'maxDeviation': 0, 'minDeviation': 999,
        'slopeFromMaxDeviation': 0, 'slopeFromMinDeviation': 999,
        'allDeviations': [],
        'lastCarbTime': 0, 'bwFound': False,
    }


def _empty_absorption() -> Dict[str, Any]:
    """Return empty absorption result."""
    return {
        'carbsAbsorbed': 0,
        'currentDeviation': None,
        'maxDeviation': 0,
        'minDeviation': 999,
        'slopeFromMaxDeviation': 0,
        'slopeFromMinDeviation': 999,
        'allDeviations': [],
    }


def _empty_deviation_data() -> Dict[str, Any]:
    """Return empty deviation data."""
    return {
        'currentDeviation': 0,
        'maxDeviation': 0,
        'minDeviation': 999,
        'slopeFromMaxDeviation': 0,
        'slopeFromMinDeviation': 999,
        'allDeviations': [],
    }


# Legacy convenience functions (kept for backward compatibility)

def calc_meal_cob_simple(
    carb_entries: List[Dict[str, Any]],
    current_time_ms: int,
    profile: Dict[str, Any]
) -> Dict[str, Any]:
    """Simple linear COB calculation (legacy, used by openaps_algorithm.py)."""
    if not carb_entries:
        return {'mealCOB': 0, 'carbs': 0, 'carbsAbsorbed': 0, 'lastCarbTime': 0}

    total_carbs = 0
    cob = 0
    last_carb_time = 0
    six_hours_ago = current_time_ms - 6 * 60 * 60 * 1000

    for entry in carb_entries:
        carbs = entry.get('carbs', 0)
        if carbs < 1:
            continue
        entry_time = entry.get('timestamp') or entry.get('date', 0)
        if isinstance(entry_time, str):
            entry_time = _parse_iso_ms(entry_time) or 0
        if entry_time <= six_hours_ago or entry_time > current_time_ms:
            continue

        absorption_hours = entry.get('absorption_time', 3.0)
        elapsed_hours = (current_time_ms - entry_time) / (60 * 60 * 1000)

        if elapsed_hours >= absorption_hours:
            pass  # fully absorbed
        else:
            remaining = carbs * (1 - elapsed_hours / absorption_hours)
            cob += remaining

        total_carbs += carbs
        last_carb_time = max(last_carb_time, entry_time)

    max_cob = profile.get('maxCOB', 120)
    cob = min(cob, max_cob)

    return {
        'mealCOB': round(cob),
        'carbs': round(total_carbs, 1),
        'carbsAbsorbed': round(total_carbs - cob, 1),
        'lastCarbTime': last_carb_time,
    }


def calculate_cob_from_history(
    carb_entries: List[tuple],
    current_time_ms: Optional[int] = None,
    profile: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Legacy convenience function."""
    if current_time_ms is None:
        import time
        current_time_ms = int(time.time() * 1000)
    if profile is None:
        profile = {'carb_ratio': 10.0, 'sens': 50.0, 'maxCOB': 120}

    entries = [create_carb_entry(g, t, a) for t, g, a in carb_entries]
    return calc_meal_cob_simple(entries, current_time_ms, profile)
