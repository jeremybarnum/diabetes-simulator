"""
Personal therapy settings for realistic simulations.
"""

# Your Core Settings
PERSONAL_SETTINGS = {
    'insulin_sensitivity_factor': 100.0,  # mg/dL per unit
    'carb_ratio': 9.0,                    # grams per unit
    'basal_rate': 0.45,                   # U/hr
    'duration_of_insulin_action': 6.0,    # hours
    'target': 100.0,                      # mg/dL
}

# Loop-specific settings
LOOP_SETTINGS = {
    'insulin_sensitivity_factor': 100.0,
    'duration_of_insulin_action': 6.0,
    'basal_rate': 0.45,
    'target_range': (100, 100),           # Using 100 as both low and high
    'carb_ratio': 9.0,
    'max_basal_rate': 3.0,
    'max_bolus': 2.5
}

# OpenAPS-specific settings
OPENAPS_SETTINGS = {
    'insulin_sensitivity_factor': 100.0,
    'duration_of_insulin_action': 6.0,
    'basal_rate': 0.45,
    'target_bg': 100.0,
    'carb_ratio': 9.0,
    'max_basal': 2.0,                     # Multiplier (2x basal)
    'max_iob': 3.0,
    'max_daily_basal': 0.45
}


def get_loop_settings():
    """Get Loop settings."""
    return LOOP_SETTINGS.copy()


def get_openaps_settings():
    """Get OpenAPS settings."""
    return OPENAPS_SETTINGS.copy()
