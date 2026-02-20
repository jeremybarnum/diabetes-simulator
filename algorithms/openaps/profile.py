"""
OpenAPS profile and settings - Port of trio-oref/lib/profile/

Handles schedule lookups and profile building.
Phase 1: Basic profile generation and schedule lookups.
Phase 2+: Will add dynamic adjustments (autosens, TDD-based adjustments).

Reference:
- ../Trio/trio-oref/lib/profile/index.js (200 lines) - main profile generation
- ../Trio/trio-oref/lib/profile/basal.js (46 lines) - basal schedule lookups
- ../Trio/trio-oref/lib/profile/isf.js (48 lines) - ISF schedule lookups
- ../Trio/trio-oref/lib/profile/carbs.js (39 lines) - carb ratio lookups
- ../Trio/trio-oref/lib/profile/targets.js (85 lines) - BG target lookups

Total: ~418 lines JavaScript
"""

from typing import Dict, List, Any, Optional
from datetime import datetime


def get_profile_defaults() -> Dict[str, Any]:
    """
    Get default profile settings.

    Port of defaults() from index.js (lines 9-81)

    Returns:
        Dict with default OpenAPS settings
    """
    return {
        'max_iob': 0,  # if max_iob is not provided, will default to zero
        'max_daily_safety_multiplier': 3,
        'current_basal_safety_multiplier': 4,
        'autosens_max': 1.2,
        'autosens_min': 0.7,
        'rewind_resets_autosens': True,
        'high_temptarget_raises_sensitivity': False,
        'low_temptarget_lowers_sensitivity': False,
        'sensitivity_raises_target': False,
        'resistance_lowers_target': False,
        'exercise_mode': False,
        'half_basal_exercise_target': 160,
        'maxCOB': 120,
        'skip_neutral_temps': False,
        'unsuspend_if_no_temp': False,
        'min_5m_carbimpact': 8,  # mg/dL per 5m
        'autotune_isf_adjustmentFraction': 1.0,
        'remainingCarbsFraction': 1.0,
        'remainingCarbsCap': 90,

        # UAM/SMB settings (Phase 2+)
        'enableUAM': False,
        'A52_risk_enable': False,
        'enableSMB_with_COB': False,
        'enableSMB_with_temptarget': False,
        'enableSMB_always': False,
        'enableSMB_after_carbs': False,
        'enableSMB_high_bg': False,
        'enableSMB_high_bg_target': 110,
        'allowSMB_with_high_temptarget': False,
        'maxSMBBasalMinutes': 30,
        'maxUAMSMBBasalMinutes': 30,
        'SMBInterval': 3,
        'bolus_increment': 0.1,
        'maxDelta_bg_threshold': 0.2,

        # Insulin curve settings
        'curve': 'rapid-acting',  # or "ultra-rapid" for Fiasp, "bilinear" for old curve
        'useCustomPeakTime': False,
        'insulinPeakTime': 75,  # defaults to 55m for Fiasp if useCustomPeakTime: false

        'carbsReqThreshold': 1,
        'offline_hotspot': False,
        'noisyCGMTargetMultiplier': 1.3,
        'suspend_zeros_iob': True,
        'enableEnliteBgproxy': False,
        'calc_glucose_noise': False,
        'target_bg': False,  # set to an integer value in mg/dL to override pump min_bg
        'smb_delivery_ratio': 0.5,

        # Trio-specific settings
        'adjustmentFactor': 0.8,
        'adjustmentFactorSigmoid': 0.5,
        'useNewFormula': False,
        'enableDynamicCR': False,
        'sigmoid': False,
        'weightPercentage': 0.65,
        'tddAdjBasal': False,
        'threshold_setting': 60,
    }


def basal_lookup(basals: List[Dict[str, Any]], time_ms: Optional[int] = None) -> float:
    """
    Look up scheduled basal rate for a given time.

    Port of basalLookup() from basal.js (lines 5-28)

    Args:
        basals: List of basal schedule entries with 'minutes' and 'rate' fields
        time_ms: Time in milliseconds (defaults to current time)

    Returns:
        Basal rate in U/hr

    Notes:
        - Basals should be sorted by 'minutes' (or 'i' field)
        - 'minutes' is offset from midnight (0-1439)
        - Returns rate for the time window containing the given time
        - Uses UTC time to avoid timezone issues
    """
    from datetime import timezone

    if time_ms is None:
        now = datetime.now(timezone.utc)
    else:
        now = datetime.fromtimestamp(time_ms / 1000, tz=timezone.utc)

    # Sort by minutes offset (or 'i' field for compatibility)
    sorted_basals = sorted(basals, key=lambda x: x.get('i', x.get('minutes', 0)))

    # Default to last entry
    basal_rate = sorted_basals[-1]['rate']

    if basal_rate == 0:
        raise ValueError(f"ERROR: bad basal schedule - last entry has rate 0")

    # Current time in minutes since midnight
    now_minutes = now.hour * 60 + now.minute

    # Find the appropriate entry
    for i in range(len(sorted_basals) - 1):
        if (now_minutes >= sorted_basals[i]['minutes'] and
            now_minutes < sorted_basals[i + 1]['minutes']):
            basal_rate = sorted_basals[i]['rate']
            break

    return round(basal_rate * 1000) / 1000


def max_daily_basal(basals: List[Dict[str, Any]]) -> float:
    """
    Get maximum basal rate from schedule.

    Port of maxDailyBasal() from basal.js (lines 31-34)

    Args:
        basals: List of basal schedule entries

    Returns:
        Maximum daily basal rate in U/hr
    """
    max_rate = max(basals, key=lambda x: float(x['rate']))
    return round(float(max_rate['rate']) * 1000) / 1000


def max_basal_lookup(settings: Dict[str, Any]) -> float:
    """
    Get maximum basal from pump settings.

    Port of maxBasalLookup() from basal.js (lines 38-40)

    Args:
        settings: Pump settings dict with 'maxBasal' field

    Returns:
        Maximum basal rate in U/hr
    """
    return settings['maxBasal']


def isf_lookup(isf_data: Dict[str, Any], time_ms: Optional[int] = None) -> float:
    """
    Look up insulin sensitivity factor for a given time.

    Port of isfLookup() from isf.js (lines 6-44)

    Args:
        isf_data: ISF schedule data with 'sensitivities' array
        time_ms: Time in milliseconds (defaults to current time)

    Returns:
        ISF in mg/dL per U

    Notes:
        - ISF schedule should start at offset 0
        - Returns sensitivity for the time window containing the given time
        - Uses UTC time to avoid timezone issues
    """
    from datetime import timezone

    if time_ms is None:
        now = datetime.now(timezone.utc)
    else:
        now = datetime.fromtimestamp(time_ms / 1000, tz=timezone.utc)

    now_minutes = now.hour * 60 + now.minute

    # Sort by offset
    sensitivities = sorted(isf_data['sensitivities'], key=lambda x: x['offset'])

    # Validate first entry starts at midnight
    if sensitivities[0]['offset'] != 0:
        return -1

    # Default to last entry
    isf_schedule = sensitivities[-1]

    # Find the appropriate entry
    for i in range(len(sensitivities) - 1):
        current = sensitivities[i]
        next_entry = sensitivities[i + 1]
        if now_minutes >= current['offset'] and now_minutes < next_entry['offset']:
            isf_schedule = sensitivities[i]
            break

    return isf_schedule['sensitivity']


def carb_ratio_lookup(carbratio_data: Dict[str, Any], time_ms: Optional[int] = None) -> Optional[float]:
    """
    Look up carb ratio for a given time.

    Port of carbRatioLookup() from carbs.js (lines 4-38)

    Args:
        carbratio_data: Carb ratio schedule with 'schedule' array and 'units'
        time_ms: Time in milliseconds (defaults to current time)

    Returns:
        Carb ratio in g/U, or None if invalid

    Notes:
        - Supports 'grams' and 'exchanges' units
        - Validates ratio is between 3 and 150
        - Uses UTC time to avoid timezone issues
    """
    from datetime import timezone

    if not carbratio_data or 'schedule' not in carbratio_data:
        return None

    if time_ms is None:
        now = datetime.now(timezone.utc)
    else:
        now = datetime.fromtimestamp(time_ms / 1000, tz=timezone.utc)

    schedule = carbratio_data['schedule']
    units = carbratio_data.get('units', 'grams')

    if units not in ('grams', 'exchanges'):
        print(f"Error: Unsupported carb_ratio units {units}")
        return None

    # Default to last entry
    carb_ratio = schedule[-1]

    # Note: JavaScript version uses getTime() which converts offset string to time
    # For simplicity, we'll assume 'offset' is already in minutes
    now_minutes = now.hour * 60 + now.minute

    # Find the appropriate entry
    for i in range(len(schedule) - 1):
        current_offset = schedule[i].get('offset', 0)
        next_offset = schedule[i + 1].get('offset', 0)

        if now_minutes >= current_offset and now_minutes < next_offset:
            carb_ratio = schedule[i]

            # Validate bounds
            ratio_value = carb_ratio.get('ratio', carb_ratio.get('r', 0))
            if ratio_value < 3 or ratio_value > 150:
                print(f"Error: carbRatio of {ratio_value} out of bounds.")
                return None
            break

    ratio_value = carb_ratio.get('ratio', carb_ratio.get('r', 0))

    # Convert exchanges to grams (12g per exchange)
    if units == 'exchanges':
        ratio_value = 12 / ratio_value

    return ratio_value


def bg_targets_lookup(
    targets_data: Dict[str, Any],
    temptargets_data: List[Dict[str, Any]] = None,
    profile: Optional[Dict[str, Any]] = None,
    time_ms: Optional[int] = None
) -> Dict[str, Any]:
    """
    Look up BG targets for a given time, including temp targets.

    Port of bgTargetsLookup() from targets.js (lines 4-83)

    Args:
        targets_data: BG target schedule with 'targets' array
        temptargets_data: Optional list of temporary target entries
        profile: Optional profile dict (for target_bg override)
        time_ms: Time in milliseconds (defaults to current time)

    Returns:
        Dict with 'low', 'high', 'min_bg', 'max_bg', and optionally 'temptargetSet'

    Notes:
        - Applies temp targets if active
        - Validates and bounds targets to safe ranges (80-200 mg/dL)
        - Converts mmol/L to mg/dL if needed (values < 20)
        - Uses UTC time to avoid timezone issues
    """
    from datetime import timezone

    if time_ms is None:
        now = datetime.now(timezone.utc)
    else:
        now = datetime.fromtimestamp(time_ms / 1000, tz=timezone.utc)

    if temptargets_data is None:
        temptargets_data = []

    if profile is None:
        profile = {}

    # Get scheduled target
    targets = targets_data['targets']

    # Default to last entry
    bg_targets = targets[-1].copy()

    # Note: JavaScript uses getTime() for offset conversion
    # For simplicity, assuming offset is in minutes
    now_minutes = now.hour * 60 + now.minute

    # Find scheduled target for current time
    for i in range(len(targets) - 1):
        current_offset = targets[i].get('offset', 0)
        next_offset = targets[i + 1].get('offset', 0)

        if now_minutes >= current_offset and now_minutes < next_offset:
            bg_targets = targets[i].copy()
            break

    # Apply profile override if set
    if profile.get('target_bg'):
        bg_targets['low'] = profile['target_bg']

    bg_targets['high'] = bg_targets['low']

    # Check for active temp targets
    temp_targets = bg_targets.copy()

    # Sort temp targets by creation date (most recent first)
    sorted_temps = sorted(
        temptargets_data,
        key=lambda x: datetime.fromisoformat(x['created_at'].replace('Z', '+00:00')),
        reverse=True
    )

    for temp in sorted_temps:
        start = datetime.fromisoformat(temp['created_at'].replace('Z', '+00:00'))
        duration_ms = temp['duration'] * 60 * 1000
        expires = datetime.fromtimestamp((start.timestamp() * 1000 + duration_ms) / 1000, tz=timezone.utc)

        # Cancel temp targets (duration 0)
        if now >= start and temp['duration'] == 0:
            temp_targets = bg_targets.copy()
            break

        # Invalid temp target
        if not temp.get('targetBottom') or not temp.get('targetTop'):
            print(f"eventualBG target range invalid: {temp.get('targetBottom')}-{temp.get('targetTop')}")
            break

        # Active temp target
        if now >= start and now < expires:
            temp_targets['high'] = temp['targetTop']
            temp_targets['low'] = temp['targetBottom']
            temp_targets['temptargetSet'] = True
            break

    bg_targets = temp_targets

    # Bound target range
    return _bound_target_range(bg_targets)


def _bound_target_range(target: Dict[str, Any]) -> Dict[str, Any]:
    """
    Bound BG targets to safe ranges.

    Port of bound_target_range() from targets.js (lines 67-78)

    Args:
        target: Target dict with 'low' and 'high'

    Returns:
        Target dict with 'min_bg' and 'max_bg' bounded to 80-200 mg/dL

    Notes:
        - Converts mmol/L to mg/dL if values < 20
        - Hard-codes minimum of 80 mg/dL
        - Hard-codes maximum of 200 mg/dL
    """
    # Convert mmol/L to mg/dL if needed
    if target['high'] < 20:
        target['high'] = target['high'] * 18
    if target['low'] < 20:
        target['low'] = target['low'] * 18

    # Hard-code lower bounds (minimum 80 mg/dL)
    target['max_bg'] = max(80, target['high'])
    target['min_bg'] = max(80, target['low'])

    # Hard-code upper bounds (maximum 200 mg/dL)
    target['min_bg'] = min(200, target['min_bg'])
    target['max_bg'] = min(200, target['max_bg'])

    return target


def generate_profile(inputs: Dict[str, Any], opts: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Generate complete OpenAPS profile from input data.

    Port of generate() from index.js (lines 119-194)

    Args:
        inputs: Input data with:
            - settings: Pump settings (DIA, maxBasal, etc.)
            - basals: Basal schedule
            - isf: ISF schedule
            - targets: BG target schedule
            - carbratio: Carb ratio schedule (optional)
            - temptargets: Temp target entries (optional)
            - model: Pump model (optional)
            - skip_neutral_temps: Whether to skip neutral temps
        opts: Optional settings overrides

    Returns:
        Complete profile dict ready for OpenAPS algorithm

    Notes:
        - Validates all critical parameters
        - Returns -1 if validation fails
        - Merges defaults with user preferences
        - Performs schedule lookups at current time
    """
    # Start with defaults
    profile = get_profile_defaults()

    # Apply opts overrides if provided
    if opts:
        profile.update(opts)

    # Apply input overrides
    for pref in list(profile.keys()):
        if pref in inputs:
            profile[pref] = inputs[pref]

    # Get pump settings
    settings = inputs['settings']

    # DIA (Duration of Insulin Action)
    if settings.get('insulin_action_curve', 0) > 1:
        profile['dia'] = settings['insulin_action_curve']
    else:
        print(f"DIA of {settings.get('insulin_action_curve')} is not supported")
        return -1

    # Pump model
    if 'model' in inputs:
        profile['model'] = inputs['model']

    # Skip neutral temps
    profile['skip_neutral_temps'] = inputs.get('skip_neutral_temps', False)

    # Basal profile
    try:
        profile['current_basal'] = basal_lookup(inputs['basals'])
        profile['basalprofile'] = inputs['basals']

        # Round basal rates to 3 decimal places
        for entry in profile['basalprofile']:
            entry['rate'] = round(entry['rate'], 3)

        profile['max_daily_basal'] = max_daily_basal(inputs['basals'])
        profile['max_basal'] = max_basal_lookup(inputs['settings'])

        # Validate basal rates
        if profile['current_basal'] == 0:
            print(f"current_basal of {profile['current_basal']} is not supported")
            return -1
        if profile['max_daily_basal'] == 0:
            print(f"max_daily_basal of {profile['max_daily_basal']} is not supported")
            return -1
        if profile['max_basal'] < 0.1:
            print(f"max_basal of {profile['max_basal']} is not supported")
            return -1
    except Exception as e:
        print(f"Error processing basal schedule: {e}")
        return -1

    # BG targets
    try:
        temp_targets = inputs.get('temptargets', [])
        range_data = bg_targets_lookup(inputs['targets'], temp_targets, profile)

        profile['out_units'] = inputs['targets'].get('user_preferred_units', 'mg/dL')
        profile['min_bg'] = round(range_data['min_bg'])
        profile['max_bg'] = round(range_data['max_bg'])
        profile['bg_targets'] = inputs['targets']

        # Round BG target entries
        for entry in profile['bg_targets']['targets']:
            entry['high'] = round(entry['high'])
            entry['low'] = round(entry['low'])
            if 'min_bg' in entry:
                entry['min_bg'] = round(entry['min_bg'])
            if 'max_bg' in entry:
                entry['max_bg'] = round(entry['max_bg'])

        # Remove raw data
        if 'raw' in profile['bg_targets']:
            del profile['bg_targets']['raw']

        profile['temptargetSet'] = range_data.get('temptargetSet', False)
    except Exception as e:
        print(f"Error processing BG targets: {e}")
        return -1

    # ISF
    try:
        profile['sens'] = isf_lookup(inputs['isf'])
        profile['isfProfile'] = inputs['isf']

        if profile['sens'] < 5:
            print(f"ISF of {profile['sens']} is not supported")
            return -1
    except Exception as e:
        print(f"Error processing ISF: {e}")
        return -1

    # Carb ratio
    if 'carbratio' in inputs:
        try:
            profile['carb_ratio'] = carb_ratio_lookup(inputs['carbratio'])
            profile['carb_ratios'] = inputs['carbratio']
        except Exception as e:
            print(f"Error processing carb ratio: {e}")
    else:
        print("Profile wasn't given carb ratio data, cannot calculate carb_ratio")

    return profile


# Convenience functions

def create_basal_schedule(entries: List[tuple]) -> List[Dict[str, Any]]:
    """
    Create basal schedule from list of (minutes, rate) tuples.

    Args:
        entries: List of (minutes_from_midnight, rate_U_per_hr) tuples

    Returns:
        Basal schedule compatible with generate_profile
    """
    return [
        {'minutes': minutes, 'rate': rate, 'i': idx}
        for idx, (minutes, rate) in enumerate(entries)
    ]


def create_isf_schedule(entries: List[tuple]) -> Dict[str, Any]:
    """
    Create ISF schedule from list of (offset, sensitivity) tuples.

    Args:
        entries: List of (minutes_from_midnight, sensitivity_mg_dL_per_U) tuples

    Returns:
        ISF schedule compatible with generate_profile
    """
    return {
        'sensitivities': [
            {'offset': offset, 'sensitivity': sens}
            for offset, sens in entries
        ]
    }


def create_target_schedule(entries: List[tuple]) -> Dict[str, Any]:
    """
    Create BG target schedule from list of (offset, low, high) tuples.

    Args:
        entries: List of (minutes_from_midnight, low_mg_dL, high_mg_dL) tuples

    Returns:
        Target schedule compatible with generate_profile
    """
    return {
        'targets': [
            {'offset': offset, 'low': low, 'high': high}
            for offset, low, high in entries
        ],
        'user_preferred_units': 'mg/dL'
    }


def create_carb_ratio_schedule(entries: List[tuple]) -> Dict[str, Any]:
    """
    Create carb ratio schedule from list of (offset, ratio) tuples.

    Args:
        entries: List of (minutes_from_midnight, ratio_g_per_U) tuples

    Returns:
        Carb ratio schedule compatible with generate_profile
    """
    return {
        'schedule': [
            {'offset': offset, 'ratio': ratio}
            for offset, ratio in entries
        ],
        'units': 'grams'
    }
