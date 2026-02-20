"""
OpenAPS BG predictions - Port of prediction logic from determine-basal.js

Generates all four prediction types (IOB, COB, UAM, ZT) in a single pass
through the IOB array, matching the JS implementation exactly.

Reference: trio-oref/lib/determine-basal/determine-basal.js (lines 1154-1560)
"""

import math
from typing import Dict, List, Any, Optional


def round_val(value, digits=0):
    """Match JS round() behavior."""
    if digits == 0:
        return round(value)
    scale = 10 ** digits
    return round(value * scale) / scale


def generate_predictions(
    bg: float,
    iob_array: List[Dict[str, Any]],
    profile: Dict[str, Any],
    glucose_status: Dict[str, Any],
    meal_data: Dict[str, Any],
    iob_data: Dict[str, Any],
    sensitivity_ratio: float = 1.0,
    enable_uam: bool = True,
    clock_ms: int = 0,
) -> Dict[str, Any]:
    """
    Generate all four prediction curves in a single pass through iobArray.

    Port of the prediction loop from determine-basal.js (lines 1154-1560).

    Args:
        bg: Current BG in mg/dL
        iob_array: Full IOB array (48 entries from generate_iob_array)
        profile: Full profile dict
        glucose_status: Dict with delta, short_avgdelta, long_avgdelta
        meal_data: Dict with mealCOB, carbs, lastCarbTime, slopeFromMaxDeviation, etc.
        iob_data: Current IOB data (index 0 of iob_array)
        sensitivity_ratio: Autosens ratio (default 1.0)
        enable_uam: Whether UAM is enabled

    Returns:
        Dict with:
            - predBGs: {IOB: [...], ZT: [...], COB: [...], UAM: [...]}
            - eventualBG, naive_eventualBG
            - minPredBG, minGuardBG, avgPredBG
            - minIOBPredBG, minCOBPredBG, minUAMPredBG
            - minIOBGuardBG, minCOBGuardBG, minUAMGuardBG, minZTGuardBG
            - bgi, deviation, ci, cid, csf
            - insulinReq, carbsReq
            - threshold, target_bg
    """
    sens = profile.get('sens', 100)
    carb_ratio = profile.get('carb_ratio', 10)
    target_bg = profile.get('target_bg', profile.get('min_bg', 100))
    threshold = profile.get('threshold_setting', 80)

    # Calculate BGI and deviation (lines 1096-1117)
    bgi = round_val(-iob_data.get('activity', 0) * sens * 5, 2)

    min_delta = min(glucose_status.get('delta', 0),
                    glucose_status.get('short_avgdelta', 0))
    min_avg_delta = min(glucose_status.get('short_avgdelta', 0),
                        glucose_status.get('long_avgdelta', 0))

    # Project deviations for 30 minutes (6 x 5-min intervals)
    deviation = round_val(30 / 5 * (min_delta - bgi))
    if deviation < 0:
        deviation = round_val(30 / 5 * (min_avg_delta - bgi))
        if deviation < 0:
            deviation = round_val(30 / 5 * (glucose_status.get('long_avgdelta', 0) - bgi))

    # Naive eventual BG
    naive_eventual_bg = bg
    if iob_data.get('iob', 0) > 0:
        naive_eventual_bg = round_val(bg - (iob_data['iob'] * sens))
    else:
        naive_eventual_bg = round_val(bg - (iob_data.get('iob', 0) * sens))

    eventual_bg = naive_eventual_bg + deviation

    # Carb impact calculations (lines 1188-1270)
    ci = round_val(min_delta - bgi, 1)
    uci = round_val(min_delta - bgi, 1)

    csf = sens / carb_ratio

    max_carb_absorption_rate = 30  # g/h
    max_ci = round_val(max_carb_absorption_rate * csf * 5 / 60, 1)
    if ci > max_ci:
        ci = max_ci

    remaining_ca_time_min = 3.0  # hours minimum
    if sensitivity_ratio:
        remaining_ca_time_min = remaining_ca_time_min / sensitivity_ratio

    assumed_carb_absorption_rate = 20  # g/h
    remaining_ca_time = remaining_ca_time_min
    remaining_carbs_cap = min(90, profile.get('remainingCarbsCap', 90))
    remaining_carbs_fraction = min(1, profile.get('remainingCarbsFraction', 1.0))
    remaining_carbs_ignore = 1 - remaining_carbs_fraction

    meal_cob = meal_data.get('mealCOB', 0)
    meal_carbs = meal_data.get('carbs', 0)
    last_carb_time = meal_data.get('lastCarbTime', 0)

    if meal_carbs:
        remaining_ca_time_min = max(remaining_ca_time_min,
                                     meal_cob / assumed_carb_absorption_rate)
        if not clock_ms:
            import time as time_module
            clock_ms = int(time_module.time() * 1000)
        last_carb_age = round_val((clock_ms - last_carb_time) / 60000) if last_carb_time else 0
        remaining_ca_time = remaining_ca_time_min + 1.5 * last_carb_age / 60
        remaining_ca_time = round_val(remaining_ca_time, 1)

    # Total carb impact and remaining carbs
    total_ci = max(0, ci / 5 * 60 * remaining_ca_time / 2)
    total_ca = total_ci / csf if csf else 0
    remaining_carbs = max(0, meal_cob - total_ca - meal_carbs * remaining_carbs_ignore)
    remaining_carbs = min(remaining_carbs_cap, remaining_carbs)

    # Remaining CI peak (bilinear /\ curve)
    remaining_ci_peak = 0
    if remaining_ca_time > 0:
        remaining_ci_peak = remaining_carbs * csf * 5 / 60 / (remaining_ca_time / 2)

    # Deviation slopes for UAM
    slope_from_max = round_val(meal_data.get('slopeFromMaxDeviation', 0), 2)
    slope_from_min = round_val(meal_data.get('slopeFromMinDeviation', 999), 2)
    slope_from_deviations = min(slope_from_max, -slope_from_min / 3)

    # Carb impact duration
    cid = 0
    if ci != 0:
        cid = min(remaining_ca_time * 60 / 5 / 2,
                  max(0, meal_cob * csf / ci))

    # --- Prediction loop (lines 1296-1386) ---
    iob_pred_bgs = [bg]
    cob_pred_bgs = [bg]
    uam_pred_bgs = [bg]
    zt_pred_bgs = [bg]

    min_iob_pred_bg = 999
    min_cob_pred_bg = 999
    min_uam_pred_bg = 999
    min_cob_guard_bg = 999
    min_uam_guard_bg = 999
    min_iob_guard_bg = 999
    min_zt_guard_bg = 999

    max_iob_pred_bg = bg
    max_cob_pred_bg = bg

    iob_pred_bg_val = eventual_bg
    uam_duration = 0

    insulin_peak_5m = 18  # 90 min / 5 = 18 data points

    for iob_tick in iob_array:
        pred_bgi = round_val(-iob_tick.get('activity', 0) * sens * 5, 2)

        zt_iob = iob_tick.get('iobWithZeroTemp', {})
        pred_zt_bgi = round_val(-zt_iob.get('activity', 0) * sens * 5, 2)

        # IOB prediction: deviation decays linearly over 60min
        pred_dev = ci * (1 - min(1, len(iob_pred_bgs) / (60 / 5)))

        # Default (non-dynamic ISF) path
        iob_pred_bg_val = iob_pred_bgs[-1] + pred_bgi + pred_dev
        zt_pred_bg_val = zt_pred_bgs[-1] + pred_zt_bgi

        # COB prediction
        pred_ci = max(0, max(0, ci) * (1 - len(cob_pred_bgs) / max(cid * 2, 1)))
        intervals = min(len(cob_pred_bgs), (remaining_ca_time * 12) - len(cob_pred_bgs))
        remaining_ci = max(0, intervals / (remaining_ca_time / 2 * 12) * remaining_ci_peak) if remaining_ca_time > 0 else 0
        cob_pred_bg_val = cob_pred_bgs[-1] + pred_bgi + min(0, pred_dev) + pred_ci + remaining_ci

        # UAM prediction: deviation drops at slopeFromDeviations
        pred_uci_slope = max(0, uci + (len(uam_pred_bgs) * slope_from_deviations))
        pred_uci_max = max(0, uci * (1 - len(uam_pred_bgs) / max(3 * 60 / 5, 1)))
        pred_uci = min(pred_uci_slope, pred_uci_max)
        if pred_uci > 0:
            uam_duration = round_val((len(uam_pred_bgs) + 1) * 5 / 60, 1)
        uam_pred_bg_val = uam_pred_bgs[-1] + pred_bgi + min(0, pred_dev) + pred_uci

        # Truncate at 48 points (4 hours)
        if len(iob_pred_bgs) < 48:
            iob_pred_bgs.append(iob_pred_bg_val)
        if len(cob_pred_bgs) < 48:
            cob_pred_bgs.append(cob_pred_bg_val)
        if len(uam_pred_bgs) < 48:
            uam_pred_bgs.append(uam_pred_bg_val)
        if len(zt_pred_bgs) < 48:
            zt_pred_bgs.append(zt_pred_bg_val)

        # Track guard BGs (no wait period)
        if cob_pred_bg_val < min_cob_guard_bg:
            min_cob_guard_bg = round_val(cob_pred_bg_val)
        if uam_pred_bg_val < min_uam_guard_bg:
            min_uam_guard_bg = round_val(uam_pred_bg_val)
        if iob_pred_bg_val < min_iob_guard_bg:
            min_iob_guard_bg = round_val(iob_pred_bg_val)
        if zt_pred_bg_val < min_zt_guard_bg:
            min_zt_guard_bg = round_val(zt_pred_bg_val)

        # Track min pred BGs (with wait periods)
        if len(iob_pred_bgs) > insulin_peak_5m and iob_pred_bg_val < min_iob_pred_bg:
            min_iob_pred_bg = round_val(iob_pred_bg_val)
        if iob_pred_bg_val > max_iob_pred_bg:
            max_iob_pred_bg = iob_pred_bg_val

        if (cid or remaining_ci_peak > 0) and len(cob_pred_bgs) > insulin_peak_5m and cob_pred_bg_val < min_cob_pred_bg:
            min_cob_pred_bg = round_val(cob_pred_bg_val)
        if (cid or remaining_ci_peak > 0) and cob_pred_bg_val > max_cob_pred_bg:
            max_cob_pred_bg = cob_pred_bg_val

        if enable_uam and len(uam_pred_bgs) > 12 and uam_pred_bg_val < min_uam_pred_bg:
            min_uam_pred_bg = round_val(uam_pred_bg_val)

    # --- Post-process prediction arrays (lines 1396-1448) ---
    pred_bgs = {}

    # IOB: bound 39-401, trim trailing flat
    iob_pred_bgs = [round_val(min(401, max(39, p))) for p in iob_pred_bgs]
    i = len(iob_pred_bgs) - 1
    while i > 12:
        if iob_pred_bgs[i - 1] != iob_pred_bgs[i]:
            break
        iob_pred_bgs.pop()
        i -= 1
    pred_bgs['IOB'] = iob_pred_bgs

    # ZT: bound 39-401, trim rising values above target (keep 6+)
    zt_pred_bgs = [round_val(min(401, max(39, p))) for p in zt_pred_bgs]
    i = len(zt_pred_bgs) - 1
    while i > 6:
        if zt_pred_bgs[i - 1] >= zt_pred_bgs[i] or zt_pred_bgs[i] <= target_bg:
            break
        zt_pred_bgs.pop()
        i -= 1
    pred_bgs['ZT'] = zt_pred_bgs

    # COB: only include if mealCOB > 0 and (ci > 0 or remaining carbs)
    if meal_cob > 0 and (ci > 0 or remaining_ci_peak > 0):
        cob_pred_bgs = [round_val(min(1500, max(39, p))) for p in cob_pred_bgs]
        i = len(cob_pred_bgs) - 1
        while i > 12:
            if cob_pred_bgs[i - 1] != cob_pred_bgs[i]:
                break
            cob_pred_bgs.pop()
            i -= 1
        pred_bgs['COB'] = cob_pred_bgs
        eventual_bg = max(eventual_bg, round_val(cob_pred_bgs[-1]))

    # UAM: only include if ci > 0 or remaining_ci_peak > 0, and UAM enabled
    if (ci > 0 or remaining_ci_peak > 0) and enable_uam:
        uam_pred_bgs = [round_val(min(401, max(39, p))) for p in uam_pred_bgs]
        i = len(uam_pred_bgs) - 1
        while i > 12:
            if uam_pred_bgs[i - 1] != uam_pred_bgs[i]:
                break
            uam_pred_bgs.pop()
            i -= 1
        pred_bgs['UAM'] = uam_pred_bgs
        if uam_pred_bgs:
            eventual_bg = max(eventual_bg, round_val(uam_pred_bgs[-1]))

    # --- minPredBG blending (lines 1452-1551) ---
    min_iob_pred_bg = max(39, min_iob_pred_bg)
    min_cob_pred_bg = max(39, min_cob_pred_bg)
    min_uam_pred_bg = max(39, min_uam_pred_bg)
    min_pred_bg = round_val(min_iob_pred_bg)

    fraction_carbs_left = meal_cob / meal_carbs if meal_carbs else 0

    # avgPredBG blending
    if min_uam_pred_bg < 999 and min_cob_pred_bg < 999:
        avg_pred_bg = round_val((1 - fraction_carbs_left) * uam_pred_bg_val + fraction_carbs_left * cob_pred_bg_val)
    elif min_cob_pred_bg < 999:
        avg_pred_bg = round_val((iob_pred_bg_val + cob_pred_bg_val) / 2)
    elif min_uam_pred_bg < 999:
        avg_pred_bg = round_val((iob_pred_bg_val + uam_pred_bg_val) / 2)
    else:
        avg_pred_bg = round_val(iob_pred_bg_val)

    if min_zt_guard_bg > avg_pred_bg:
        avg_pred_bg = min_zt_guard_bg

    # minGuardBG blending
    if cid or remaining_ci_peak > 0:
        if enable_uam:
            min_guard_bg = fraction_carbs_left * min_cob_guard_bg + (1 - fraction_carbs_left) * min_uam_guard_bg
        else:
            min_guard_bg = min_cob_guard_bg
    elif enable_uam:
        min_guard_bg = min_uam_guard_bg
    else:
        min_guard_bg = min_iob_guard_bg
    min_guard_bg = round_val(min_guard_bg)

    # minZTUAMPredBG calculation
    min_zt_uam_pred_bg = min_uam_pred_bg
    if min_zt_guard_bg < threshold:
        min_zt_uam_pred_bg = (min_uam_pred_bg + min_zt_guard_bg) / 2
    elif min_zt_guard_bg < target_bg:
        blend_pct = (min_zt_guard_bg - threshold) / (target_bg - threshold) if target_bg != threshold else 0
        blended = min_uam_pred_bg * blend_pct + min_zt_guard_bg * (1 - blend_pct)
        min_zt_uam_pred_bg = (min_uam_pred_bg + blended) / 2
    elif min_zt_guard_bg > min_uam_pred_bg:
        min_zt_uam_pred_bg = (min_uam_pred_bg + min_zt_guard_bg) / 2
    min_zt_uam_pred_bg = round_val(min_zt_uam_pred_bg)

    # minPredBG selection
    if meal_carbs:
        if not enable_uam and min_cob_pred_bg < 999:
            min_pred_bg = round_val(max(min_iob_pred_bg, min_cob_pred_bg))
        elif min_cob_pred_bg < 999:
            blended_min = fraction_carbs_left * min_cob_pred_bg + (1 - fraction_carbs_left) * min_zt_uam_pred_bg
            min_pred_bg = round_val(max(min_iob_pred_bg, min_cob_pred_bg, blended_min))
        elif enable_uam:
            min_pred_bg = min_zt_uam_pred_bg
        else:
            min_pred_bg = min_guard_bg
    elif enable_uam:
        min_pred_bg = round_val(max(min_iob_pred_bg, min_zt_uam_pred_bg))

    min_pred_bg = min(min_pred_bg, avg_pred_bg)

    if max_cob_pred_bg > bg:
        min_pred_bg = min(min_pred_bg, max_cob_pred_bg)

    return {
        'predBGs': pred_bgs,
        'eventualBG': eventual_bg,
        'naive_eventualBG': naive_eventual_bg,
        'minPredBG': min_pred_bg,
        'minGuardBG': min_guard_bg,
        'avgPredBG': avg_pred_bg,
        'minIOBPredBG': min_iob_pred_bg,
        'minCOBPredBG': min_cob_pred_bg,
        'minUAMPredBG': min_uam_pred_bg,
        'minIOBGuardBG': min_iob_guard_bg,
        'minCOBGuardBG': min_cob_guard_bg,
        'minUAMGuardBG': min_uam_guard_bg,
        'minZTGuardBG': min_zt_guard_bg,
        'bgi': bgi,
        'deviation': deviation,
        'ci': ci,
        'uci': uci,
        'cid': cid,
        'csf': csf,
        'remaining_ci_peak': remaining_ci_peak,
        'remaining_ca_time': remaining_ca_time,
        'UAMduration': uam_duration,
        'threshold': threshold,
        'target_bg': target_bg,
    }


# Keep legacy functions for backward compatibility with openaps_algorithm.py
def calculate_iob_predictions(bg, iob_array, sens, ci=0.0, max_predictions=48):
    """Legacy IOB-only predictions."""
    preds = [bg]
    for iob_tick in iob_array:
        if len(preds) >= max_predictions:
            break
        pred_bgi = round(-iob_tick['activity'] * sens * 5, 2)
        pred_dev = ci * (1 - min(1, len(preds) / (60 / 5)))
        preds.append(preds[-1] + pred_bgi + pred_dev)
    preds = [round(min(401, max(39, p))) for p in preds]
    while len(preds) > 12 and preds[-2] == preds[-1]:
        preds.pop()
    return preds


def calculate_carb_impact(min_delta, bgi, sens, carb_ratio, meal_cob,
                          remaining_ca_time_min=3.0, assumed_carb_absorption_rate=20.0,
                          max_carb_absorption_rate=30.0, sensitivity_ratio=1.0):
    """Legacy carb impact calculation."""
    ci = round(min_delta - bgi, 1)
    csf = sens / carb_ratio
    max_ci = round(max_carb_absorption_rate * csf * 5 / 60, 1)
    if ci > max_ci:
        ci = max_ci
    rcat = remaining_ca_time_min / sensitivity_ratio
    remaining_ca_time = max(rcat, meal_cob / assumed_carb_absorption_rate) if meal_cob > 0 else rcat
    cid = min(remaining_ca_time * 60 / 5 / 2, max(0, meal_cob * csf / ci)) if ci != 0 else 0
    total_ci = max(0, ci / 5 * 60 * remaining_ca_time / 2)
    total_ca = total_ci / csf if csf else 0
    remaining_carbs = max(0, meal_cob - total_ca)
    rci_peak = remaining_carbs * csf * 5 / 60 / (remaining_ca_time / 2) if remaining_ca_time > 0 else 0
    return {'ci': ci, 'cid': cid, 'csf': csf, 'remaining_ci_peak': rci_peak, 'remaining_ca_time': remaining_ca_time}
