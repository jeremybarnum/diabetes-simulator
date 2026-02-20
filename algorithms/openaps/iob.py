"""
OpenAPS IOB module - Ports of trio-oref/lib/iob/*.js

Includes:
- iob_total: Sum IOB from treatments at a single point (total.js)
- find_insulin: Convert pump history to micro-bolus treatments (history.js)
- generate_iob_array: Generate 48-point IOB array with zero-temp overlay (index.js)

Reference:
- trio-oref/lib/iob/total.js (103 lines)
- trio-oref/lib/iob/history.js (571 lines)
- trio-oref/lib/iob/index.js (85 lines)
"""

from typing import Dict, List, Any, Optional
from algorithms.openaps.insulin_math import iob_calc


def iob_total(
    treatments: List[Dict[str, Any]],
    time_ms: int,
    profile: Dict[str, Any],
    calculate_func: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Calculate total IOB and activity from all treatments.

    Port of iobTotal() from total.js (lines 1-103)

    Args:
        treatments: List of treatment dicts, each with:
            - 'date': Time in milliseconds
            - 'insulin': Insulin amount in units (positive for bolus, can be negative for basal adjustments)
            - Other fields optional
        time_ms: Current time in milliseconds
        profile: Profile dict with:
            - 'dia': Duration of insulin action in hours
            - 'curve': Curve type ('bilinear', 'rapid-acting', 'ultra-rapid')
            - Other insulin curve parameters
        calculate_func: Optional custom IOB calculation function (default: use iob_calc from insulin_math)

    Returns:
        Dict with:
            - iob: Total insulin on board (U)
            - activity: Total insulin activity (U/min)
            - basaliob: IOB from basal adjustments (U)
            - bolusiob: IOB from boluses (U)
            - netbasalinsulin: Net basal insulin delivered (U)
            - bolusinsulin: Total bolus insulin delivered (U)
            - time: Input time (for reference)

    Notes:
        - Treatments < 0.1 U are considered basal adjustments
        - Treatments >= 0.1 U are considered boluses
        - Only includes treatments within DIA window
        - Enforces minimum DIA of 3 hours (5 hours for exponential curves)
        - Values rounded to 3-4 decimal places
    """
    if not treatments:
        return {
            'iob': 0,
            'activity': 0,
            'basaliob': 0,
            'bolusiob': 0,
            'netbasalinsulin': 0,
            'bolusinsulin': 0,
            'time': time_ms,
            'iobArray': []
        }

    # Use provided calculate function or default
    if calculate_func is None:
        calculate_func = iob_calc

    # Get DIA from profile
    dia = profile.get('dia', 6.0)

    # Force minimum DIA of 3 hours
    if dia < 3:
        dia = 3

    # Curve defaults with DIA requirements
    curve_defaults = {
        'bilinear': {
            'requireLongDia': False,
            'peak': 75
        },
        'rapid-acting': {
            'requireLongDia': True,
            'peak': 75,
            'tdMin': 300
        },
        'ultra-rapid': {
            'requireLongDia': True,
            'peak': 55,
            'tdMin': 300
        }
    }

    # Get curve type from profile
    curve = profile.get('curve', 'bilinear')
    if curve:
        curve = curve.lower()

    # Validate curve type
    if curve not in curve_defaults:
        print(f'Warning: Unsupported curve function: "{curve}". Defaulting to "rapid-acting".')
        curve = 'rapid-acting'

    defaults = curve_defaults[curve]

    # Force minimum 5-hour DIA for exponential curves
    if defaults['requireLongDia'] and dia < 5:
        dia = 5

    peak = defaults['peak']

    # Initialize accumulators
    total_iob = 0.0
    total_activity = 0.0
    basal_iob = 0.0
    bolus_iob = 0.0
    net_basal_insulin = 0.0
    bolus_insulin = 0.0

    # Calculate DIA window
    dia_ago = time_ms - (dia * 60 * 60 * 1000)

    # Process each treatment
    for treatment in treatments:
        treatment_date = treatment.get('date', 0)

        # Only process treatments in the past
        if treatment_date <= time_ms:
            # Only process treatments within DIA window
            if treatment_date > dia_ago:
                # Calculate IOB for this treatment
                t_iob = calculate_func(
                    treatment=treatment,
                    time_ms=time_ms,
                    curve=curve,
                    dia=dia,
                    peak=peak,
                    profile=profile
                )

                # Add to totals
                if t_iob and 'iobContrib' in t_iob:
                    total_iob += t_iob['iobContrib']

                if t_iob and 'activityContrib' in t_iob:
                    total_activity += t_iob['activityContrib']

                # Categorize as basal or bolus
                # Basals are typically small adjustments (< 0.1 U)
                # Boluses are larger doses (>= 0.1 U)
                insulin = treatment.get('insulin', 0)
                if insulin and t_iob and 'iobContrib' in t_iob:
                    if insulin < 0.1:
                        # Basal adjustment (including negative for temp basal reductions)
                        basal_iob += t_iob['iobContrib']
                        net_basal_insulin += insulin
                    else:
                        # Bolus
                        bolus_iob += t_iob['iobContrib']
                        bolus_insulin += insulin

    # Round values to appropriate precision
    return {
        'iob': round(total_iob, 3),
        'activity': round(total_activity, 4),
        'basaliob': round(basal_iob, 3),
        'bolusiob': round(bolus_iob, 3),
        'netbasalinsulin': round(net_basal_insulin, 3),
        'bolusinsulin': round(bolus_insulin, 3),
        'time': time_ms
    }


def create_treatment(
    insulin: float,
    time_ms: int,
    treatment_type: str = 'bolus'
) -> Dict[str, Any]:
    """
    Convenience function to create a treatment dict.

    Args:
        insulin: Insulin amount in units
        time_ms: Time in milliseconds
        treatment_type: 'bolus' or 'basal'

    Returns:
        Treatment dict compatible with iob_total
    """
    return {
        'insulin': insulin,
        'date': time_ms,
        'type': treatment_type
    }


def calculate_iob_from_history(
    boluses: List[tuple],  # [(time_ms, units), ...]
    temp_basals: Optional[List[tuple]] = None,  # [(time_ms, rate_u_hr, duration_min), ...]
    current_time_ms: Optional[int] = None,
    profile: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Convenience function to calculate IOB from bolus and temp basal history.

    Args:
        boluses: List of (time_ms, units) tuples
        temp_basals: Optional list of (time_ms, rate_u_hr, duration_min) tuples
        current_time_ms: Current time in milliseconds (default: now)
        profile: Profile dict (default: ultra-rapid, 6h DIA)

    Returns:
        IOB breakdown dict from iob_total
    """
    if current_time_ms is None:
        import time
        current_time_ms = int(time.time() * 1000)

    if profile is None:
        profile = {
            'dia': 6.0,
            'curve': 'ultra-rapid'
        }

    # Build treatment list
    treatments = []

    # Add boluses
    for bolus_time, units in boluses:
        treatments.append({
            'insulin': units,
            'date': bolus_time
        })

    # Add temp basals if provided
    if temp_basals:
        scheduled_basal = profile.get('scheduled_basal', 1.0)  # Default 1 U/hr
        for temp_time, temp_rate, duration_min in temp_basals:
            # Calculate net insulin vs scheduled
            net_rate = temp_rate - scheduled_basal
            net_insulin = net_rate * (duration_min / 60.0)

            # Create treatment for net basal adjustment
            # (can be negative for low temps)
            if abs(net_insulin) > 0.001:  # Ignore very small adjustments
                treatments.append({
                    'insulin': net_insulin,
                    'date': temp_time
                })

    return iob_total(
        treatments=treatments,
        time_ms=current_time_ms,
        profile=profile
    )


def find_insulin(
    history: List[Dict[str, Any]],
    profile: Dict[str, Any],
    clock_ms: int,
    autosens: Optional[Dict[str, Any]] = None,
    zero_temp_duration: int = 0
) -> List[Dict[str, Any]]:
    """
    Convert pump history events into flat micro-bolus treatment list.

    Port of calcTempTreatments() from history.js.

    Processes:
    1. Extract boluses directly from history
    2. Convert temp basals to equivalent micro-boluses (0.05U each)
       by computing net rate (temp rate - scheduled basal)
    3. Optionally add zero-temp overlay for iobWithZeroTemp calculation

    Args:
        history: Pump history events (boluses, temp basals).
                 Boluses: {'_type': 'Bolus', 'timestamp': ISO, 'amount': float}
                 Temp basals: {'_type': 'TempBasal', 'timestamp': ISO, 'rate': float, 'duration': float}
        profile: Profile with 'current_basal', 'basalprofile'
        clock_ms: Current time in milliseconds
        autosens: Optional autosens data with 'ratio'
        zero_temp_duration: If >0, add synthetic zero-temp for this many minutes

    Returns:
        Sorted list of treatment dicts: [{'insulin': float, 'date': int}, ...]
    """
    temp_boluses = []
    temp_history = []
    now_ms = clock_ms

    # Parse pump history events
    for event in history:
        event_type = event.get('_type', '')
        timestamp = event.get('timestamp', event.get('created_at', ''))

        if not timestamp:
            continue

        # Parse timestamp to ms
        event_ms = _parse_timestamp_ms(timestamp)
        if event_ms is None:
            continue

        if event_type == 'Bolus':
            amount = event.get('amount', 0)
            if amount and event_ms <= now_ms:
                temp_boluses.append({
                    'insulin': amount,
                    'date': event_ms,
                })

        elif event_type in ('TempBasal', 'Temp Basal'):
            rate = event.get('rate', event.get('absolute', 0))
            duration = event.get('duration', 0)
            if duration is None:
                duration = 0
            temp_history.append({
                'rate': rate,
                'duration': duration,
                'date': event_ms,
                'timestamp': timestamp,
            })

        elif event.get('eventType') == 'Temp Basal':
            rate = event.get('rate', event.get('absolute', 0))
            duration = event.get('duration', 0)
            if event.get('amount') is not None:
                # Loop-style: amount = total insulin delivered
                rate = event['amount'] / duration * 60 if duration > 0 else 0
            temp_history.append({
                'rate': rate,
                'duration': duration or 0,
                'date': event_ms,
                'timestamp': timestamp,
            })

    # In JS history.js, the zero-temp overlay is added INSIDE the pump history
    # event loop. So when pump history is empty, no zero-temp is created.
    # Only add zero-temp overlay when there are actual pump events.
    has_pump_events = len(history) > 0
    if has_pump_events:
        if zero_temp_duration > 0:
            zero_start_ms = now_ms + 60 * 1000  # 1 min in future
            temp_history.append({
                'rate': 0,
                'duration': zero_temp_duration,
                'date': zero_start_ms,
                'timestamp': '',
            })
        else:
            # Add a zero-duration cancel event at current time
            temp_history.append({
                'rate': 0,
                'duration': 0,
                'date': now_ms + 60 * 1000,
                'timestamp': '',
            })

    # Sort temp history by date
    temp_history.sort(key=lambda x: x['date'])

    # Fix overlapping temp basals (later one truncates earlier)
    for i in range(len(temp_history) - 1):
        end_ms = temp_history[i]['date'] + temp_history[i]['duration'] * 60 * 1000
        if end_ms > temp_history[i + 1]['date']:
            temp_history[i]['duration'] = (
                (temp_history[i + 1]['date'] - temp_history[i]['date']) / 60 / 1000
            )

    # Split temp basals into 30-min chunks (matches JS splitTimespan logic)
    # JS history.js splits any temp basal >30min into 30-min segments
    split_history = []
    for temp in temp_history:
        remaining = temp['duration']
        start_ms = temp['date']
        while remaining > 30:
            split_history.append({
                'rate': temp['rate'],
                'duration': 30,
                'date': start_ms,
            })
            start_ms += 30 * 60 * 1000
            remaining -= 30
        if remaining > 0:
            split_history.append({
                'rate': temp['rate'],
                'duration': remaining,
                'date': start_ms,
            })
        elif temp['duration'] <= 0:
            split_history.append(temp)

    # Convert temp basals to micro-boluses
    current_basal = profile.get('current_basal', 0)
    basalprofile = profile.get('basalprofile', [])

    for temp in split_history:
        if temp['duration'] <= 0:
            continue

        # Look up scheduled basal rate at this time
        scheduled_rate = current_basal
        if basalprofile:
            scheduled_rate = _basal_lookup(basalprofile, temp['date'])

        # Apply autosens if available
        if autosens and autosens.get('ratio'):
            scheduled_rate = scheduled_rate * autosens['ratio']

        net_rate = temp['rate'] - scheduled_rate
        if net_rate == 0:
            continue

        # Convert to micro-boluses
        bolus_size = 0.05 if net_rate > 0 else -0.05
        net_amount = round(net_rate * temp['duration'] * 10 / 6) / 100
        bolus_count = round(net_amount / bolus_size)

        if bolus_count == 0:
            continue

        spacing_ms = temp['duration'] / bolus_count * 60 * 1000

        for j in range(bolus_count):
            temp_boluses.append({
                'insulin': bolus_size,
                'date': temp['date'] + j * spacing_ms,
            })

    # Sort all treatments by date
    temp_boluses.sort(key=lambda x: x['date'])
    return temp_boluses


def generate_iob_array(
    history: List[Dict[str, Any]],
    profile: Dict[str, Any],
    clock_ms: int,
    autosens: Optional[Dict[str, Any]] = None,
    current_iob_only: bool = False
) -> List[Dict[str, Any]]:
    """
    Generate IOB predictions over time (0-240min in 5-min steps).

    Port of generate() from index.js.

    For each 5-minute interval, computes:
    - IOB/activity using actual treatments
    - IOB/activity assuming zero temp from now (iobWithZeroTemp)

    Args:
        history: Pump history events
        profile: Full profile dict
        clock_ms: Current time in milliseconds
        autosens: Optional autosens data
        current_iob_only: If True, only compute index 0 (for COB calculation)

    Returns:
        Array of IOB entries, one per 5-minute interval:
        [
            {
                'iob': float, 'activity': float,
                'basaliob': float, 'bolusiob': float,
                'netbasalinsulin': float, 'bolusinsulin': float,
                'iobWithZeroTemp': {'iob': float, 'activity': float},
                'lastBolusTime': int,  # only on index 0
                'lastTemp': dict,       # only on index 0
            },
            ...
        ]
    """
    # Build treatment lists
    treatments = find_insulin(history, profile, clock_ms, autosens, zero_temp_duration=0)
    treatments_with_zero = find_insulin(history, profile, clock_ms, autosens, zero_temp_duration=240)

    # Determine iteration range
    if current_iob_only:
        i_stop = 1
    else:
        i_stop = 4 * 60  # 240 minutes

    iob_array = []

    for i in range(0, i_stop, 5):
        t_ms = clock_ms + i * 60 * 1000

        # IOB with actual treatments
        iob = iob_total(treatments, t_ms, profile)

        # IOB with zero-temp overlay
        iob_zero = iob_total(treatments_with_zero, t_ms, profile)

        entry = {
            'iob': iob['iob'],
            'activity': iob['activity'],
            'basaliob': iob['basaliob'],
            'bolusiob': iob['bolusiob'],
            'netbasalinsulin': iob['netbasalinsulin'],
            'bolusinsulin': iob['bolusinsulin'],
            'iobWithZeroTemp': {
                'iob': iob_zero['iob'],
                'activity': iob_zero['activity'],
            },
        }
        iob_array.append(entry)

    # Add lastBolusTime and lastTemp to first entry
    if iob_array:
        last_bolus_time = 0
        last_temp = {'date': 0}

        for t in treatments:
            if t.get('insulin') and t.get('insulin') >= 0.1:
                last_bolus_time = max(last_bolus_time, t.get('date', 0))

        iob_array[0]['lastBolusTime'] = last_bolus_time
        iob_array[0]['lastTemp'] = last_temp

    return iob_array


def _parse_timestamp_ms(timestamp) -> Optional[int]:
    """Parse a timestamp (ISO string or ms int) to milliseconds."""
    if isinstance(timestamp, (int, float)):
        # Already milliseconds if > 1e12, seconds if < 1e12
        if timestamp > 1e12:
            return int(timestamp)
        else:
            return int(timestamp * 1000)

    if isinstance(timestamp, str):
        if not timestamp:
            return None
        from datetime import datetime, timezone
        try:
            # Try ISO format
            ts = timestamp.replace('Z', '+00:00')
            dt = datetime.fromisoformat(ts)
            return int(dt.timestamp() * 1000)
        except (ValueError, TypeError):
            return None

    return None


def _basal_lookup(basalprofile: List[Dict], time_ms: int) -> float:
    """
    Look up scheduled basal rate at a given time.

    Port of basalLookup() from profile/basal.js.
    """
    from datetime import datetime, timezone

    if not basalprofile:
        return 0

    dt = datetime.fromtimestamp(time_ms / 1000, tz=timezone.utc)
    now_minutes = dt.hour * 60 + dt.minute

    # Sort by minutes offset
    sorted_profile = sorted(basalprofile, key=lambda x: x.get('minutes', x.get('i', 0)))

    # Default to last entry
    basal_rate = sorted_profile[-1].get('rate', 0)

    for i in range(len(sorted_profile) - 1):
        if now_minutes >= sorted_profile[i].get('minutes', 0) and \
           now_minutes < sorted_profile[i + 1].get('minutes', 0):
            basal_rate = sorted_profile[i].get('rate', 0)
            break

    return round(basal_rate, 3)
