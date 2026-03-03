"""
OpenAPS insulin math - Port of trio-oref/lib/iob/calculate.js

Calculates insulin activity and IOB (Insulin On Board) for individual insulin doses.
Supports both bilinear and exponential insulin action curves.

Reference: ../Trio/trio-oref/lib/iob/calculate.js (144 lines)
"""

import math
from typing import Dict, Optional, Any


def iob_calc(
    treatment: Dict[str, Any],
    time_ms: Optional[int] = None,
    curve: str = 'exponential',
    dia: float = 6.0,
    peak: int = 75,
    profile: Optional[Dict[str, Any]] = None
) -> Dict[str, float]:
    """
    Calculate insulin activity and IOB for a single treatment.

    Port of iobCalc() from calculate.js (lines 1-31)

    Args:
        treatment: Dict with 'insulin' (float, units) and 'date' (int, milliseconds)
        time_ms: Current time in milliseconds (default: None for current time)
        curve: 'bilinear' or 'exponential' (default: 'exponential')
        dia: Duration of insulin action in hours (default: 6.0)
        peak: Peak insulin activity time in minutes (default: 75)
        profile: Profile dict with curve settings (optional)

    Returns:
        Dict with:
            - activityContrib: Units of insulin used in previous minute
            - iobContrib: Units of insulin still remaining

    Notes:
        - activityContrib is the instantaneous insulin activity
        - iobContrib is the remaining active insulin
        - Returns empty dict if treatment has no insulin
    """
    if not treatment.get('insulin'):
        return {}

    # Calculate minutes since bolus
    if time_ms is None:
        import time
        time_ms = int(time.time() * 1000)

    bolus_time_ms = treatment.get('date', 0)
    mins_ago = round((time_ms - bolus_time_ms) / 1000 / 60)

    # Choose calculation method
    if curve == 'bilinear':
        return iob_calc_bilinear(treatment, mins_ago, dia)
    else:
        return iob_calc_exponential(treatment, mins_ago, dia, peak, profile)


def iob_calc_bilinear(
    treatment: Dict[str, Any],
    mins_ago: int,
    dia: float
) -> Dict[str, float]:
    """
    Calculate IOB using bilinear (triangular) insulin action curve.

    Port of iobCalcBilinear() from calculate.js (lines 34-78)

    The bilinear curve is a triangle with:
    - Fixed peak at 75 minutes
    - Fixed end at 180 minutes
    - Scaled by user's DIA

    Args:
        treatment: Dict with 'insulin' (float, units)
        mins_ago: Minutes since bolus
        dia: Duration of insulin action in hours

    Returns:
        Dict with activityContrib and iobContrib
    """
    insulin = treatment.get('insulin', 0)

    # Constants for bilinear model
    default_dia = 3.0  # hours
    peak = 75  # minutes
    end = 180  # minutes

    # Scale time by DIA ratio
    time_scalar = default_dia / dia
    scaled_mins_ago = time_scalar * mins_ago

    activity_contrib = 0.0
    iob_contrib = 0.0

    # Calculate activity peak based on triangle area = 1
    # (length * height) / 2 = 1, so height = 2 / length
    activity_peak = 2.0 / (dia * 60)
    slope_up = activity_peak / peak
    slope_down = -1 * (activity_peak / (end - peak))

    if scaled_mins_ago < peak:
        # Before peak: rising slope
        activity_contrib = insulin * (slope_up * scaled_mins_ago)

        # IOB calculation: quadratic fit for pre-peak
        x1 = (scaled_mins_ago / 5) + 1
        iob_contrib = insulin * ((-0.001852 * x1 * x1) + (0.001852 * x1) + 1.000000)

    elif scaled_mins_ago < end:
        # After peak: falling slope
        mins_past_peak = scaled_mins_ago - peak
        activity_contrib = insulin * (activity_peak + (slope_down * mins_past_peak))

        # IOB calculation: quadratic fit for post-peak
        x2 = (scaled_mins_ago - peak) / 5
        iob_contrib = insulin * ((0.001323 * x2 * x2) + (-0.054233 * x2) + 0.555560)

    return {
        'activityContrib': activity_contrib,
        'iobContrib': iob_contrib
    }


def iob_calc_exponential(
    treatment: Dict[str, Any],
    mins_ago: int,
    dia: float,
    peak: int,
    profile: Optional[Dict[str, Any]] = None
) -> Dict[str, float]:
    """
    Calculate IOB using exponential insulin action curve.

    Port of iobCalcExponential() from calculate.js (lines 81-141)

    Uses the same exponential formula as LoopKit:
    https://github.com/LoopKit/Loop/issues/388#issuecomment-317938473

    Args:
        treatment: Dict with 'insulin' (float, units)
        mins_ago: Minutes since bolus
        dia: Duration of insulin action in hours
        peak: Peak insulin activity time in minutes
        profile: Profile dict with curve settings (optional)

    Returns:
        Dict with activityContrib and iobContrib
    """
    insulin = treatment.get('insulin', 0)

    # Determine peak time from profile if provided
    if profile:
        curve_type = profile.get('curve', 'rapid-acting')
        use_custom = profile.get('useCustomPeakTime', False)
        custom_peak = profile.get('insulinPeakTime')

        if curve_type == 'rapid-acting':
            if use_custom and custom_peak is not None:
                # Clamp to valid range for rapid-acting
                if custom_peak > 120:
                    print(f'Warning: Setting maximum Insulin Peak Time of 120m for {curve_type} insulin')
                    peak = 120
                elif custom_peak < 50:
                    print(f'Warning: Setting minimum Insulin Peak Time of 50m for {curve_type} insulin')
                    peak = 50
                else:
                    peak = custom_peak
            else:
                peak = 75  # Default for rapid-acting

        elif curve_type == 'ultra-rapid':
            if use_custom and custom_peak is not None:
                # Clamp to valid range for ultra-rapid
                if custom_peak > 100:
                    print(f'Warning: Setting maximum Insulin Peak Time of 100m for {curve_type} insulin')
                    peak = 100
                elif custom_peak < 35:
                    print(f'Warning: Setting minimum Insulin Peak Time of 35m for {curve_type} insulin')
                    peak = 35
                else:
                    peak = custom_peak
            else:
                peak = 55  # Default for ultra-rapid (Fiasp)
        else:
            print(f'Warning: Curve of {curve_type} is not supported.')

    end = dia * 60  # End of insulin activity in minutes

    activity_contrib = 0.0
    iob_contrib = 0.0

    if mins_ago < end:
        tau = peak * (1 - peak / end) / (1 - 2 * peak / end)
        a = 2 * tau / end
        S = 1 / (1 - a + (1 + a) * math.exp(-end / tau))

        activity_contrib = (
            insulin * (S / (tau ** 2)) * mins_ago *
            (1 - mins_ago / end) * math.exp(-mins_ago / tau)
        )

        iob_contrib = insulin * (
            1 - S * (1 - a) * (
                (mins_ago ** 2 / (tau * end * (1 - a)) - mins_ago / tau - 1) *
                math.exp(-mins_ago / tau) + 1
            )
        )

    return {
        'activityContrib': activity_contrib,
        'iobContrib': iob_contrib
    }


def get_default_peak(curve_type: str = 'ultra-rapid') -> int:
    """
    Get default peak time for insulin curve type.

    Args:
        curve_type: 'rapid-acting' or 'ultra-rapid'

    Returns:
        Peak time in minutes
    """
    if curve_type == 'ultra-rapid':
        return 55  # Fiasp default
    elif curve_type == 'rapid-acting':
        return 75  # Novolog/Humalog default
    else:
        return 75  # Default to rapid-acting


# Convenience function for common use case
def calculate_iob(
    insulin_units: float,
    dose_time_ms: int,
    current_time_ms: int,
    dia_hours: float = 6.0,
    curve_type: str = 'ultra-rapid',
    peak_minutes: Optional[int] = None
) -> Dict[str, float]:
    """
    Convenience function to calculate IOB for a dose.

    Args:
        insulin_units: Units of insulin
        dose_time_ms: Dose time in milliseconds
        current_time_ms: Current time in milliseconds
        dia_hours: Duration of insulin action in hours
        curve_type: 'rapid-acting' or 'ultra-rapid'
        peak_minutes: Optional peak time (uses default if None)

    Returns:
        Dict with activityContrib and iobContrib
    """
    if peak_minutes is None:
        peak_minutes = get_default_peak(curve_type)

    treatment = {
        'insulin': insulin_units,
        'date': dose_time_ms
    }

    profile = {
        'curve': curve_type,
        'useCustomPeakTime': peak_minutes is not None,
        'insulinPeakTime': peak_minutes
    }

    return iob_calc(
        treatment=treatment,
        time_ms=current_time_ms,
        curve='exponential',
        dia=dia_hours,
        peak=peak_minutes,
        profile=profile
    )
