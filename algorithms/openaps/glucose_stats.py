"""
OpenAPS glucose statistics - Port of trio-oref/lib/glucose-get-last.js

Calculates glucose deltas and trends for prediction algorithms.
Phase 1: Basic delta calculations (delta, short_avgdelta, long_avgdelta).
Phase 2+: Will add autoISF features (parabola fitting, regression slopes).

Reference: ../Trio/trio-oref/lib/glucose-get-last.js (330 lines)
"""

from typing import Dict, List, Any, Optional


def get_last_glucose(
    glucose_data: List[Dict[str, Any]],
    device_filter: Optional[str] = None
) -> Dict[str, Any]:
    """
    Calculate glucose status with deltas from recent BG data.

    Port of getLastGlucose() from glucose-get-last.js (lines 12-327)
    Phase 1: Core delta calculations only (no autoISF extensions)

    Args:
        glucose_data: List of BG readings, sorted newest first, each with:
            - 'glucose' or 'sgv': BG value in mg/dL
            - 'date' or 'dateString' or 'display_time': Time in ms or ISO string
            - 'device': Optional device identifier
            - 'type': Optional, 'cal' for calibration records
        device_filter: Optional device name to filter by

    Returns:
        Dict with:
            - glucose: Current BG (mg/dL)
            - delta: 5-minute delta (mg/dL per 5 min)
            - short_avgdelta: Average delta from ~5-15 min ago (mg/dL per 5 min)
            - long_avgdelta: Average delta from ~20-40 min ago (mg/dL per 5 min)
            - date: Timestamp of current BG (ms)
            - device: Device identifier

    Notes:
        - delta = average of deltas 2.5-7.5 minutes ago
        - short_avgdelta = average of deltas 2.5-17.5 minutes ago
        - long_avgdelta = average of deltas 17.5-42.5 minutes ago
        - All deltas normalized to mg/dL per 5 minutes
        - Phase 2 will add autoISF features (slopes, parabola fit, etc.)
    """
    # Filter and prepare glucose data
    data = []
    for obj in glucose_data:
        # Support both 'glucose' and 'sgv' fields
        bg = obj.get('glucose') or obj.get('sgv')
        if bg is not None:
            obj_copy = obj.copy()
            obj_copy['glucose'] = bg
            data.append(obj_copy)

    if not data:
        return {}

    now = data[0]
    now_date = _get_date_from_entry(now)

    # Get device for filtering
    now_device = now.get('device')
    if device_filter:
        now_device = device_filter

    # Initialize delta accumulators
    last_deltas = []
    short_deltas = []
    long_deltas = []
    last_cal = 0

    # Process historical data
    for i in range(1, len(data)):
        # Stop at calibration records
        if data[i].get('type') == 'cal':
            last_cal = i
            break

        # Filter by device (must match - including both being None)
        then_device = data[i].get('device')
        if then_device != now_device:
            continue

        # Only use valid BG values
        if data[i].get('glucose', 0) <= 38:
            continue

        then = data[i]
        then_date = _get_date_from_entry(then)

        if now_date is None or then_date is None:
            continue

        # Calculate time difference
        minutesago = round((now_date - then_date) / (1000 * 60))

        if minutesago <= 0:
            continue

        # Calculate delta
        # Multiply by 5 to get mg/dL per 5 minutes (standard units)
        change = now['glucose'] - then['glucose']
        avgdelta = change / minutesago * 5

        # Average recent values (within 2.5 minutes) for smoothing
        if -2 < minutesago < 2.5:
            now['glucose'] = (now['glucose'] + then['glucose']) / 2
            now_date = (now_date + then_date) / 2

        # Collect short deltas (~5-15 minutes ago)
        elif 2.5 < minutesago < 17.5:
            short_deltas.append(avgdelta)

            # Collect last deltas (~5 minutes ago)
            if 2.5 < minutesago < 7.5:
                last_deltas.append(avgdelta)

        # Collect long deltas (~20-40 minutes ago)
        elif 17.5 < minutesago < 42.5:
            long_deltas.append(avgdelta)

    # Calculate averages
    last_delta = 0.0
    short_avgdelta = 0.0
    long_avgdelta = 0.0

    if last_deltas:
        last_delta = sum(last_deltas) / len(last_deltas)

    if short_deltas:
        short_avgdelta = sum(short_deltas) / len(short_deltas)

    if long_deltas:
        long_avgdelta = sum(long_deltas) / len(long_deltas)

    # Round to 4 decimal places (matches JavaScript)
    return {
        'delta': round(last_delta, 4),
        'glucose': round(now['glucose'], 4),
        'noise': round(now.get('noise', 1)),  # Default to 1 (Clean)
        'short_avgdelta': round(short_avgdelta, 4),
        'long_avgdelta': round(long_avgdelta, 4),
        'date': now_date,
        'last_cal': last_cal,
        'device': now.get('device')
    }


def _get_date_from_entry(entry: Dict[str, Any]) -> Optional[int]:
    """
    Extract date from entry in various formats.

    Port of getDateFromEntry() from glucose-get-last.js (lines 1-3)

    Args:
        entry: Glucose entry dict

    Returns:
        Timestamp in milliseconds, or None if not found
    """
    # Try 'date' field (milliseconds)
    if 'date' in entry and entry['date'] is not None:
        return int(entry['date'])

    # Try 'dateString' (ISO format)
    if 'dateString' in entry:
        from datetime import datetime
        try:
            dt = datetime.fromisoformat(entry['dateString'].replace('Z', '+00:00'))
            return int(dt.timestamp() * 1000)
        except (ValueError, AttributeError):
            pass

    # Try 'display_time' (ISO format)
    if 'display_time' in entry:
        from datetime import datetime
        try:
            dt = datetime.fromisoformat(entry['display_time'].replace('T', ' ').replace('Z', '+00:00'))
            return int(dt.timestamp() * 1000)
        except (ValueError, AttributeError):
            pass

    return None


def create_glucose_entry(
    glucose: float,
    time_ms: int,
    device: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to create a glucose entry dict.

    Args:
        glucose: BG value in mg/dL
        time_ms: Time in milliseconds
        device: Optional device identifier

    Returns:
        Glucose entry dict compatible with get_last_glucose
    """
    entry = {
        'glucose': glucose,
        'date': time_ms
    }
    if device:
        entry['device'] = device

    return entry


def calculate_glucose_stats(
    bg_history: List[tuple],  # [(time_ms, glucose), ...]
    device: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to calculate glucose stats from BG history.

    Args:
        bg_history: List of (time_ms, glucose) tuples, sorted newest first
        device: Optional device identifier

    Returns:
        Glucose status dict from get_last_glucose
    """
    # Build glucose entry list
    entries = []
    for time_ms, glucose in bg_history:
        entries.append(create_glucose_entry(glucose, time_ms, device))

    return get_last_glucose(entries, device_filter=device)


# Phase 2 additions (to be implemented):
# - autoISF average and duration
# - Linear regression slopes (slope05, slope15, slope40)
# - Parabola fitting for acceleration detection
# - Advanced trend prediction
#
# These will be added when implementing full Trio oref2 features in Phase 2.
