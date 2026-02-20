"""
OpenAPS determine-basal - Full port of determine-basal.js dosing logic

Computes predictions (IOB/COB/UAM/ZT), then applies the complete decision tree:
- Low glucose suspend
- SMB calculation and delivery
- High temp basal logic
- Safety limits (maxSafeBasal, max_iob)

Reference: trio-oref/lib/determine-basal/determine-basal.js (2051 lines)
"""

from typing import Dict, Any, Optional
import math

from algorithms.openaps.predictions import generate_predictions


def round_val(value, digits=0):
    """Match JS round() behavior."""
    if digits == 0:
        return round(value)
    scale = 10 ** digits
    return round(value * scale) / scale


def round_basal(rate, profile=None):
    """Round basal rate to pump precision. Port of round-basal.js."""
    lowest_rate_scale = 20  # 0.05 increments
    if rate < 1:
        return round(rate * lowest_rate_scale) / lowest_rate_scale
    elif rate < 10:
        return round(rate * 20) / 20
    else:
        return round(rate * 10) / 10


def get_max_safe_basal(profile):
    """Calculate maximum safe basal rate. Port of basal-set-temp.js."""
    max_daily_safety = profile.get('max_daily_safety_multiplier', 3)
    current_basal_safety = profile.get('current_basal_safety_multiplier', 4)
    return min(
        profile.get('max_basal', 2.8),
        max_daily_safety * profile.get('max_daily_basal', profile.get('current_basal', 0.45)),
        current_basal_safety * profile.get('current_basal', 0.45)
    )


def set_temp_basal(rate, duration, profile, rT, currenttemp):
    """Set temp basal with safety checks. Port of basal-set-temp.js."""
    max_safe = get_max_safe_basal(profile)

    if rate < 0:
        rate = 0
    elif rate > max_safe:
        rate = max_safe

    suggested_rate = round_basal(rate, profile)

    # Check if current temp is close enough to suggested
    if (currenttemp.get('duration', 0) > (duration - 10) and
        currenttemp.get('duration', 0) <= 120 and
        suggested_rate <= currenttemp.get('rate', 0) * 1.2 and
        suggested_rate >= currenttemp.get('rate', 0) * 0.8 and
        duration > 0):
        ct_dur = currenttemp.get('duration', 0)
        ct_rate = currenttemp.get('rate', 0)
        rT['reason'] += f" {ct_dur}m left and {ct_rate} ~ req {suggested_rate}U/hr: no temp required"
        return rT

    if suggested_rate == profile.get('current_basal', 0):
        if profile.get('skip_neutral_temps', False):
            if currenttemp.get('duration', 0) > 0:
                rT['reason'] += '. Suggested rate is same as profile rate, a temp basal is active, canceling current temp'
                rT['duration'] = 0
                rT['rate'] = 0
                return rT
            else:
                rT['reason'] += '. Suggested rate is same as profile rate, no temp basal is active, doing nothing'
                return rT
        else:
            rT['reason'] += f'. Setting neutral temp basal of {profile["current_basal"]}U/hr'
            rT['duration'] = duration
            rT['rate'] = suggested_rate
            return rT
    else:
        rT['duration'] = duration
        rT['rate'] = suggested_rate
        return rT


def enable_smb(profile, micro_bolus_allowed, meal_data, bg, target_bg, high_bg):
    """Determine if SMBs should be enabled. Port of enable_smb() from determine-basal.js."""
    if not micro_bolus_allowed:
        return False
    if not profile.get('allowSMB_with_high_temptarget', False) and profile.get('temptargetSet', False) and target_bg > 100:
        return False
    if meal_data.get('bwFound') and not profile.get('A52_risk_enable', False):
        return False
    if bg == 400:
        return False
    if profile.get('enableSMB_always', False):
        return True
    if profile.get('enableSMB_with_COB', False) and meal_data.get('mealCOB', 0):
        return True
    if profile.get('enableSMB_after_carbs', False) and meal_data.get('carbs', 0):
        return True
    if profile.get('enableSMB_with_temptarget', False) and profile.get('temptargetSet', False) and target_bg < 100:
        return True
    if profile.get('enableSMB_high_bg', False) and high_bg is not None and bg >= high_bg:
        return True
    return False


def determine_basal(
    glucose_status: Dict,
    currenttemp: Dict,
    iob_data: Dict,
    profile: Dict,
    meal_data: Dict,
    iob_array: list,
    micro_bolus_allowed: bool = True,
    clock_ms: int = 0,
) -> Dict[str, Any]:
    """
    Full determine_basal decision tree.

    Port of determine_basal() from determine-basal.js.

    Args:
        glucose_status: Dict with glucose, delta, short_avgdelta, long_avgdelta
        currenttemp: Dict with rate, duration
        iob_data: Current IOB data (index 0 of iob_array)
        profile: Full profile dict
        meal_data: Dict with mealCOB, carbs, deviations, slopes
        iob_array: Full IOB array (48 entries)
        micro_bolus_allowed: Whether SMBs are allowed
        clock_ms: Current time in milliseconds

    Returns:
        Dict with rate, duration, units (SMB), reason, eventualBG, predBGs, etc.
    """
    bg = glucose_status.get('glucose', 120)
    sens = profile.get('sens', 100)
    carb_ratio = profile.get('carb_ratio', 10)
    basal = profile.get('current_basal', 0.45)
    target_bg = profile.get('target_bg', profile.get('min_bg', 100))
    min_bg = profile.get('min_bg', target_bg)
    max_bg = profile.get('max_bg', target_bg)
    max_iob = profile.get('max_iob', 3.5)
    threshold = profile.get('threshold_setting', 80)
    high_bg = profile.get('enableSMB_high_bg_target', None) if profile.get('enableSMB_high_bg') else None
    override_factor = 1.0  # No overrides in basic mode

    # Initialize result
    rT = {
        'temp': 'absolute',
        'bg': bg,
        'tick': f"+{round_val(glucose_status.get('delta', 0))}" if glucose_status.get('delta', 0) > -0.5 else str(round_val(glucose_status.get('delta', 0))),
        'reason': '',
        'COB': meal_data.get('mealCOB', 0),
        'IOB': iob_data.get('iob', 0),
        'sensitivityRatio': 1.0,
    }

    # Generate predictions
    pred_result = generate_predictions(
        bg=bg,
        iob_array=iob_array,
        profile=profile,
        glucose_status=glucose_status,
        meal_data=meal_data,
        iob_data=iob_data,
        clock_ms=clock_ms,
    )

    eventual_bg = pred_result['eventualBG']
    naive_eventual_bg = pred_result['naive_eventualBG']
    min_pred_bg = pred_result['minPredBG']
    min_guard_bg = pred_result['minGuardBG']
    bgi = pred_result['bgi']
    deviation = pred_result['deviation']
    ci = pred_result['ci']
    remaining_ci_peak = pred_result.get('remaining_ci_peak', 0)
    cid = pred_result.get('cid', 0)
    csf = pred_result.get('csf', 10)
    min_iob_pred_bg = pred_result.get('minIOBPredBG', 999)
    min_zt_guard_bg = pred_result.get('minZTGuardBG', 999)

    expected_delta = calculate_expected_delta(target_bg, eventual_bg, bgi)
    min_delta = min(glucose_status.get('delta', 0), glucose_status.get('short_avgdelta', 0))

    # Populate rT with prediction results
    rT['eventualBG'] = eventual_bg
    rT['predBGs'] = pred_result['predBGs']
    rT['BGI'] = bgi
    rT['deviation'] = deviation
    rT['ISF'] = sens
    rT['CR'] = round_val(carb_ratio, 1)
    rT['target_bg'] = target_bg
    rT['threshold'] = threshold
    rT['minPredBG'] = min_pred_bg
    rT['minGuardBG'] = min_guard_bg
    rT['insulinReq'] = 0

    # Build reason prefix
    isf_reason = f"Autosens ratio: 1, ISF: {sens}\u2192{sens}"
    last_iob_pred = pred_result['predBGs'].get('IOB', [bg])[-1]
    rT['reason'] = (f"{isf_reason}, COB: {rT['COB']}, Dev: {deviation}, "
                    f"BGI: {bgi}, CR: {round_val(carb_ratio, 1)}, "
                    f"Target: {target_bg}, minPredBG {min_pred_bg}, "
                    f"minGuardBG {min_guard_bg}, IOBpredBG {last_iob_pred}")

    last_cob_pred = pred_result['predBGs'].get('COB', [0])[-1]
    if last_cob_pred > 0:
        rT['reason'] += f", COBpredBG {last_cob_pred}"
    last_uam_pred = pred_result['predBGs'].get('UAM', [0])[-1]
    if last_uam_pred > 0:
        rT['reason'] += f", UAMpredBG {last_uam_pred}"

    rT['reason'] += "; "  # conclusion starts

    # SMB enablement check
    smb_enabled = enable_smb(profile, micro_bolus_allowed, meal_data, bg, target_bg, high_bg)

    # --- Core dosing logic ---

    # carbsReq calculation
    carbs_req_bg = naive_eventual_bg
    if carbs_req_bg < 40:
        carbs_req_bg = min(min_guard_bg, carbs_req_bg)
    bg_undershoot = threshold - carbs_req_bg

    # Low glucose suspend
    if bg < threshold and iob_data.get('iob', 0) < -basal * override_factor * 20 / 60 and min_delta > 0 and min_delta > expected_delta:
        rT['reason'] += f"IOB {iob_data['iob']} < {round_val(-basal*override_factor*20/60, 2)} and minDelta {min_delta} > expectedDelta {expected_delta}; "
    elif bg < threshold or min_guard_bg < threshold:
        rT['reason'] += f"minGuardBG {min_guard_bg}<{threshold}"
        bg_undershoot = target_bg - min_guard_bg
        worst_case_req = bg_undershoot / sens
        duration_req = round_val(60 * worst_case_req / basal * override_factor)
        duration_req = round(duration_req / 30) * 30
        duration_req = min(120, max(30, duration_req))
        return set_temp_basal(0, duration_req, profile, rT, currenttemp)

    # Calculate insulinReq and rate
    insulin_req = 0
    rate = basal

    if eventual_bg < min_bg:
        # Eventual BG below target
        rT['reason'] += f"Eventual BG {eventual_bg} < {min_bg}"

        if min_delta > expected_delta and min_delta > 0 and not (bg_undershoot > 0):
            if naive_eventual_bg < 40:
                rT['reason'] += ", naive_eventualBG < 40. "
                return set_temp_basal(0, 30, profile, rT, currenttemp)
            rT['reason'] += f"; setting current basal of {basal} as temp. "
            return set_temp_basal(basal, 30, profile, rT, currenttemp)

        insulin_req = 2 * min(0, (eventual_bg - target_bg) / sens)
        insulin_req = round_val(insulin_req, 2)

        if min_delta < 0 and min_delta > expected_delta:
            insulin_req = round_val(insulin_req * (min_delta / expected_delta), 2)

        rate = basal + (2 * insulin_req)
        rate = round_basal(rate, profile)

        if rate <= 0:
            bg_undershoot = target_bg - naive_eventual_bg
            worst_case_req = bg_undershoot / sens
            duration_req = round_val(60 * worst_case_req / basal * override_factor)
            if duration_req < 0:
                duration_req = 0
            else:
                duration_req = round(duration_req / 30) * 30
                duration_req = min(120, max(0, duration_req))
            if duration_req > 0:
                rT['reason'] += f", setting {duration_req}m zero temp. "
                return set_temp_basal(rate, duration_req, profile, rT, currenttemp)

        rT['reason'] += f", setting {rate}U/hr. "
        return set_temp_basal(rate, 30, profile, rT, currenttemp)

    # Eventual BG above min_bg but delta falling faster than expected
    if min_delta < expected_delta:
        if not (micro_bolus_allowed and smb_enabled):
            rT['reason'] += f"Eventual BG {eventual_bg} > {min_bg} but Min. Delta {min_delta:.2f} < Exp. Delta {expected_delta}"
            rT['reason'] += f"; setting current basal of {basal} as temp. "
            return set_temp_basal(basal, 30, profile, rT, currenttemp)

    # eventualBG or minPredBG is in range
    if min(eventual_bg, min_pred_bg) < max_bg:
        if not (micro_bolus_allowed and smb_enabled):
            rT['reason'] += f"{eventual_bg}-{min_pred_bg} in range: no temp required"
            rT['reason'] += f"; setting current basal of {basal} as temp. "
            return set_temp_basal(basal, 30, profile, rT, currenttemp)

    # Eventual BG above max_bg
    if eventual_bg >= max_bg:
        rT['reason'] += f"Eventual BG {eventual_bg} >= {max_bg}, "

    # IOB over max_iob
    if iob_data.get('iob', 0) > max_iob:
        rT['reason'] += f"IOB {round_val(iob_data['iob'], 2)} > max_iob {max_iob}"
        rT['reason'] += f"; setting current basal of {basal} as temp. "
        return set_temp_basal(basal, 30, profile, rT, currenttemp)

    # Calculate insulin required to get minPredBG down to target
    insulin_req = round_val((min(min_pred_bg, eventual_bg) - target_bg) / sens, 2)

    # Limit by max_iob
    if insulin_req > max_iob - iob_data.get('iob', 0):
        insulin_req = max_iob - iob_data.get('iob', 0)

    rate = basal + (2 * insulin_req)
    rate = round_basal(rate, profile)
    insulin_req = round_val(insulin_req, 3)
    rT['insulinReq'] = insulin_req

    # SMB calculation
    if micro_bolus_allowed and smb_enabled and bg > threshold:
        smb_minutes = profile.get('maxSMBBasalMinutes', 30)
        uam_minutes = profile.get('maxUAMSMBBasalMinutes', 30)

        meal_insulin_req = round_val(meal_data.get('mealCOB', 0) / carb_ratio, 3)

        if iob_data.get('iob', 0) > meal_insulin_req and iob_data.get('iob', 0) > 0:
            max_bolus = round_val(basal * override_factor * uam_minutes / 60, 1)
        else:
            max_bolus = round_val(basal * override_factor * smb_minutes / 60, 1)

        smb_ratio = min(profile.get('smb_delivery_ratio', 0.5), 1)
        micro_bolus = min(insulin_req * smb_ratio, max_bolus)

        bolus_increment = profile.get('bolus_increment', 0.1)
        round_smb_to = 1 / bolus_increment
        micro_bolus = math.floor(micro_bolus * round_smb_to) / round_smb_to

        # Zero temp duration for SMB
        smb_target = target_bg
        worst_case_req = (smb_target - (naive_eventual_bg + min_iob_pred_bg) / 2) / sens
        duration_req = round_val(60 * worst_case_req / basal * override_factor)

        if insulin_req > 0 and micro_bolus < bolus_increment:
            duration_req = 0

        smb_low_temp = 0
        if duration_req <= 0:
            duration_req = 0
        elif duration_req >= 30:
            duration_req = round(duration_req / 30) * 30
            duration_req = min(60, max(0, duration_req))
        else:
            smb_low_temp = round_val(basal * duration_req / 30, 2)
            duration_req = 30

        rT['reason'] += f" insulinReq {insulin_req}"
        if micro_bolus >= max_bolus:
            rT['reason'] += f"; maxBolus {max_bolus}"
        if duration_req > 0:
            rT['reason'] += f"; setting {duration_req}m low temp of {smb_low_temp}U/h"
        rT['reason'] += ". "

        # Check last bolus timing
        last_bolus_time = iob_data.get('lastBolusTime', 0)
        if clock_ms and last_bolus_time:
            last_bolus_age = round_val((clock_ms - last_bolus_time) / 60000, 1)
        else:
            last_bolus_age = 999  # Allow SMB if no history

        smb_interval = min(10, max(1, profile.get('SMBInterval', 3)))

        if last_bolus_age > smb_interval:
            if micro_bolus > 0:
                rT['units'] = micro_bolus
                rT['reason'] += f"Microbolusing {micro_bolus}U. "
        else:
            next_mins = round_val(smb_interval - last_bolus_age, 0)
            next_secs = round_val((smb_interval - last_bolus_age) * 60, 0) % 60
            rT['reason'] += f"Waiting {next_mins}m {next_secs}s to microbolus again. "

        if duration_req > 0:
            rT['rate'] = smb_low_temp
            rT['duration'] = duration_req
            return rT

    # High temp logic
    max_safe = get_max_safe_basal(profile)
    if rate > max_safe:
        rT['reason'] += f"adj. req. rate: {rate} to maxSafeBasal: {round_val(max_safe, 2)}, "
        rate = round_basal(max_safe, profile)

    if currenttemp.get('duration', 0) == 0:
        rT['reason'] += f"no temp, setting {rate}U/hr. "
        return set_temp_basal(rate, 30, profile, rT, currenttemp)

    if currenttemp.get('duration', 0) > 5 and round_basal(rate) <= round_basal(currenttemp.get('rate', 0)):
        rT['reason'] += f"temp {currenttemp.get('rate', 0)} >~ req {rate}U/hr. "
        return rT

    rT['reason'] += f"temp {currenttemp.get('rate', 0)}<{rate}U/hr. "
    return set_temp_basal(rate, 30, profile, rT, currenttemp)


def calculate_expected_delta(target_bg, eventual_bg, bgi):
    """Port of calculate_expected_delta from determine-basal.js."""
    five_min_blocks = (2 * 60) / 5  # 24
    target_delta = target_bg - eventual_bg
    return round_val(bgi + (target_delta / five_min_blocks), 1)
