#!/usr/bin/env python3
"""
Infer a PatientProfile from Nightscout data.

Pulls real meal/bolus/CGM/profile data and auto-generates a patient profile
compatible with the Monte Carlo simulation framework.

Usage:
    python3 nightscout_profile.py --url https://ccmscout.us.nightscoutpro.com
    python3 nightscout_profile.py --url <url> --days 28 --output patient_profiles/ns_inferred.json
    python3 nightscout_profile.py --url <url> --start-date 2025-10-01 --end-date 2026-01-19
    python3 nightscout_profile.py --url <url> --start-date 2025-10-01  # end defaults to start+days
"""

import json
import math
import argparse
import numpy as np
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from zoneinfo import ZoneInfo

from nightscout_query import (
    fetch_profile,
    fetch_profile_history,
    fetch_treatments,
    fetch_entries,
    fetch_boluses,
    fetch_temp_basals,
    fetch_exercise,
)
from algorithms.openaps.iob import iob_total, find_insulin


# ─── Insulin Type Mapping ─────────────────────────────────────────────────────

# Insulin type → curve mapping for the exponential IOB model.
# The curve name determines peak activity time:
#   rapid-acting (peak=75min): Novolog, Humalog, and similar
#   ultra-rapid  (peak=55min): Fiasp, Lyumjev
INSULIN_TYPES = {
    "novolog":  "rapid-acting",
    "humalog":  "rapid-acting",
    "fiasp":    "ultra-rapid",
    "lyumjev":  "ultra-rapid",
}
DEFAULT_INSULIN_TYPE = "novolog"


# ─── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class InsulinEvent:
    """A single insulin delivery event (basal segment, bolus, or SMB)."""
    time: float           # Unix timestamp
    event_type: str       # 'basal' | 'bolus' | 'smb'
    units: float          # Total units delivered
    rate: float           # U/hr (for basal segments, 0 for boluses)
    duration_min: float   # Duration in minutes (for basal segments, 0 for boluses)


# ─── Step 1: Algorithm Settings from NS Profile ──────────────────────────────

def extract_algorithm_settings(profile_json: dict) -> dict:
    """Extract time-weighted average algorithm settings from NS profile.

    Converts NS time-varying schedules (basal, CR, ISF, target) into
    single scalar values via time-weighted averaging.
    """
    print(f"\n{'='*70}")
    print("Step 1: Algorithm Settings from Nightscout Profile")
    print(f"{'='*70}")

    settings = {}

    # --- Basal schedule ---
    basal_schedule = profile_json.get("basal", [])
    if basal_schedule:
        print(f"\n  Basal schedule ({len(basal_schedule)} entries):")
        basal_avg = _time_weighted_average(basal_schedule, value_key="value")
        for entry in basal_schedule:
            t = _seconds_to_hhmm(entry["timeAsSeconds"])
            print(f"    {t}: {entry['value']:.3f} U/hr")
        print(f"  → Time-weighted average: {basal_avg:.3f} U/hr")
        settings["basal_rate"] = round(basal_avg, 3)

    # --- Carb ratio schedule ---
    cr_schedule = profile_json.get("carbratio", [])
    if cr_schedule:
        print(f"\n  Carb ratio schedule ({len(cr_schedule)} entries):")
        cr_avg = _time_weighted_average(cr_schedule, value_key="value")
        for entry in cr_schedule:
            t = _seconds_to_hhmm(entry["timeAsSeconds"])
            print(f"    {t}: {entry['value']:.1f} g/U")
        print(f"  → Time-weighted average: {cr_avg:.1f} g/U")
        settings["carb_ratio"] = round(cr_avg, 1)

    # --- ISF schedule ---
    isf_schedule = profile_json.get("sens", [])
    if isf_schedule:
        print(f"\n  ISF schedule ({len(isf_schedule)} entries):")
        isf_avg = _time_weighted_average(isf_schedule, value_key="value")
        for entry in isf_schedule:
            t = _seconds_to_hhmm(entry["timeAsSeconds"])
            print(f"    {t}: {entry['value']:.0f} mg/dL/U")
        print(f"  → Time-weighted average: {isf_avg:.0f} mg/dL/U")
        settings["insulin_sensitivity_factor"] = round(isf_avg, 0)

    # --- Target BG schedule ---
    target_low = profile_json.get("target_low", [])
    target_high = profile_json.get("target_high", [])
    if target_low and target_high:
        # Average of (low+high)/2 across schedule
        print(f"\n  Target BG schedule ({len(target_low)} entries):")
        combined = []
        for lo, hi in zip(target_low, target_high):
            mid = (lo["value"] + hi["value"]) / 2.0
            combined.append({"timeAsSeconds": lo["timeAsSeconds"], "value": mid})
            t = _seconds_to_hhmm(lo["timeAsSeconds"])
            print(f"    {t}: {lo['value']:.0f}-{hi['value']:.0f} (mid={mid:.0f})")
        target_avg = _time_weighted_average(combined, value_key="value")
        print(f"  → Time-weighted average: {target_avg:.0f} mg/dL")
        settings["target"] = round(target_avg, 0)
    elif target_low:
        print(f"\n  Target BG schedule ({len(target_low)} entries):")
        target_avg = _time_weighted_average(target_low, value_key="value")
        for entry in target_low:
            t = _seconds_to_hhmm(entry["timeAsSeconds"])
            print(f"    {t}: {entry['value']:.0f} mg/dL")
        print(f"  → Time-weighted average: {target_avg:.0f} mg/dL")
        settings["target"] = round(target_avg, 0)

    # --- DIA: intentionally NOT extracted ---
    dia = profile_json.get("dia")
    if dia:
        print(f"\n  DIA in NS profile: {dia} hrs (NOT used — kept from settings.json)")

    # --- Timezone ---
    tz_str = profile_json.get("timezone", "")
    if tz_str:
        print(f"\n  Timezone: {tz_str}")

    return settings


def _time_weighted_average(schedule: List[dict], value_key: str = "value") -> float:
    """Compute time-weighted average of a NS schedule (24hr cycle).

    Each entry has 'timeAsSeconds' (seconds from midnight) and a value field.
    """
    if len(schedule) == 1:
        return schedule[0][value_key]

    sorted_entries = sorted(schedule, key=lambda e: e["timeAsSeconds"])
    total_weight = 0.0
    total_value = 0.0

    for i, entry in enumerate(sorted_entries):
        start = entry["timeAsSeconds"]
        if i + 1 < len(sorted_entries):
            end = sorted_entries[i + 1]["timeAsSeconds"]
        else:
            end = 86400  # midnight
        duration = end - start
        total_weight += duration
        total_value += duration * entry[value_key]

    return total_value / total_weight if total_weight > 0 else sorted_entries[0][value_key]


def _seconds_to_hhmm(seconds: int) -> str:
    """Convert seconds from midnight to HH:MM string."""
    h = seconds // 3600
    m = (seconds % 3600) // 60
    return f"{h:02d}:{m:02d}"


# ─── Profile Timeline ────────────────────────────────────────────────────────

class ProfileTimeline:
    """Maps any timestamp to the NS profile settings active at that time.

    Built from fetch_profile_history() output (sorted by startDate, deduped).
    Falls back to the most recent profile if no history covers the requested time.
    """

    def __init__(self, profile_entries: List[dict], fallback_profile: dict):
        """
        Args:
            profile_entries: sorted list of profile dicts, each with 'startDate' key
            fallback_profile: profile to use if no history entry covers the time
        """
        self._entries = []
        for entry in profile_entries:
            sd = entry.get("startDate", "")
            try:
                dt = datetime.fromisoformat(sd.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                continue
            self._entries.append((dt, entry))
        self._entries.sort(key=lambda x: x[0])
        self._fallback = fallback_profile

    def get_at(self, dt: datetime) -> dict:
        """Return the profile active at the given datetime."""
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        active = self._fallback
        for entry_dt, entry in self._entries:
            if entry_dt <= dt:
                active = entry
            else:
                break
        return active

    def get_at_ms(self, time_ms: int) -> dict:
        """Return the profile active at the given epoch milliseconds."""
        dt = datetime.fromtimestamp(time_ms / 1000, tz=timezone.utc)
        return self.get_at(dt)

    def isf_at_ms(self, time_ms: int) -> float:
        """Return the time-weighted average ISF from the profile active at time_ms."""
        profile = self.get_at_ms(time_ms)
        isf_schedule = profile.get("sens", [])
        if not isf_schedule:
            return 100.0
        return _time_weighted_average(isf_schedule, value_key="value")

    def cr_at_ms(self, time_ms: int) -> float:
        """Return the time-weighted average CR from the profile active at time_ms."""
        profile = self.get_at_ms(time_ms)
        cr_schedule = profile.get("carbratio", [])
        if not cr_schedule:
            return 10.0
        return _time_weighted_average(cr_schedule, value_key="value")

    def basal_schedule_at(self, dt: datetime) -> List[dict]:
        """Return the basal schedule from the profile active at dt."""
        profile = self.get_at(dt)
        return profile.get("basal", [])

    def average_settings(self, start: datetime, end: datetime) -> dict:
        """Compute time-weighted average ISF, CR, basal across the date range.

        Handles profiles changing mid-range by weighting each profile's contribution
        by the duration it was active within [start, end].
        """
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)

        # Build segments: [(seg_start, seg_end, profile), ...]
        segments = []
        # Find entries relevant to the range
        active_profile = self._fallback
        for entry_dt, entry in self._entries:
            if entry_dt <= start:
                active_profile = entry

        seg_start = start
        for entry_dt, entry in self._entries:
            if entry_dt <= start:
                continue
            if entry_dt >= end:
                break
            segments.append((seg_start, entry_dt, active_profile))
            active_profile = entry
            seg_start = entry_dt
        segments.append((seg_start, end, active_profile))

        total_seconds = (end - start).total_seconds()
        if total_seconds <= 0:
            profile = self.get_at(start)
            return {
                "isf": _time_weighted_average(profile.get("sens", [{"value": 100}]), "value"),
                "cr": _time_weighted_average(profile.get("carbratio", [{"value": 10}]), "value"),
                "basal": _time_weighted_average(profile.get("basal", [{"value": 1}]), "value"),
            }

        weighted_isf = 0.0
        weighted_cr = 0.0
        weighted_basal = 0.0
        for seg_start, seg_end, profile in segments:
            weight = (seg_end - seg_start).total_seconds() / total_seconds
            isf_sched = profile.get("sens", [{"value": 100}])
            cr_sched = profile.get("carbratio", [{"value": 10}])
            basal_sched = profile.get("basal", [{"value": 1}])
            weighted_isf += weight * _time_weighted_average(isf_sched, "value")
            weighted_cr += weight * _time_weighted_average(cr_sched, "value")
            weighted_basal += weight * _time_weighted_average(basal_sched, "value")

        return {
            "isf": round(weighted_isf, 1),
            "cr": round(weighted_cr, 1),
            "basal": round(weighted_basal, 3),
        }

    def summary(self, start: datetime, end: datetime) -> str:
        """Return a human-readable summary of profile changes in the date range."""
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)

        changes = []
        for entry_dt, entry in self._entries:
            if entry_dt < start or entry_dt > end:
                continue
            isf = _time_weighted_average(entry.get("sens", [{"value": "?"}]), "value")
            cr = _time_weighted_average(entry.get("carbratio", [{"value": "?"}]), "value")
            basal = _time_weighted_average(entry.get("basal", [{"value": "?"}]), "value")
            changes.append(f"    {entry_dt.strftime('%Y-%m-%d %H:%M')} ISF={isf:.0f} CR={cr:.1f} Basal={basal:.3f}")
        return f"  {len(changes)} profile changes in range" + ("\n" + "\n".join(changes) if changes else "")


# ─── Step 2: Meal Patterns from Carb Treatments ──────────────────────────────

def extract_meal_pattern(
    treatments: List[dict],
    tz: ZoneInfo,
    n_days: int,
    meal_times: Optional[List[int]] = None,
) -> List[dict]:
    """Extract meal schedule from carb treatments.

    If meal_times is provided (list of clock hours, e.g. [8, 13, 19]),
    carb entries are bucketed to the nearest specified meal time.
    Otherwise, auto-detects meal slots by grouping active hours.

    Returns MealSpec-compatible dicts.
    """
    print(f"\n{'='*70}")
    print("Step 2: Meal Patterns from Carb Treatments")
    print(f"{'='*70}")

    # Collect carb entries with local time and source metadata
    carb_entries = []
    n_loop = 0
    n_trio = 0
    n_other = 0
    for t in treatments:
        carbs = t.get("carbs")
        if not carbs or carbs <= 0:
            continue
        created = t.get("created_at", "")
        if not created:
            continue
        try:
            dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            continue
        local_dt = dt.astimezone(tz)

        entered_by = t.get("enteredBy", "")
        is_loop = "loop" in entered_by.lower()
        is_trio = "trio" in entered_by.lower()
        if is_loop:
            n_loop += 1
        elif is_trio:
            n_trio += 1
        else:
            n_other += 1

        # Loop-era treatments have absorptionTime in minutes
        absorption_min = t.get("absorptionTime")
        absorption_hrs = absorption_min / 60.0 if absorption_min else None

        carb_entries.append({
            "local_dt": local_dt,
            "hour": local_dt.hour,
            "carbs": carbs,
            "absorption_hrs": absorption_hrs,
            "source": "loop" if is_loop else ("trio" if is_trio else "other"),
        })

    print(f"\n  Total carb entries: {len(carb_entries)} over {n_days} days "
          f"({len(carb_entries)/max(1,n_days):.1f}/day)")
    print(f"  Sources: Loop={n_loop}, Trio={n_trio}, Other={n_other}")
    has_loop_absorption = any(e["absorption_hrs"] is not None for e in carb_entries)
    if has_loop_absorption:
        abs_vals = [e["absorption_hrs"] for e in carb_entries if e["absorption_hrs"] is not None]
        print(f"  Loop absorption times available: {len(abs_vals)} entries "
              f"(median={np.median(abs_vals):.1f}h, range={min(abs_vals):.1f}-{max(abs_vals):.1f}h)")

    if not carb_entries:
        return []

    # Hourly distribution (always printed for reference)
    hourly = defaultdict(list)
    for e in carb_entries:
        hourly[e["hour"]].append(e["carbs"])

    print(f"\n  Hourly distribution:")
    print(f"  {'Hour':<8} {'Count':>6} {'Per Day':>8} {'Mean g':>8} {'SD g':>8}")
    print(f"  {'-'*40}")
    for h in range(24):
        vals = hourly[h]
        if vals:
            arr = np.array(vals)
            per_day = len(vals) / n_days
            print(f"  {h:02d}:00    {len(vals):>6} {per_day:>8.2f} "
                  f"{np.mean(arr):>8.1f} {np.std(arr):>8.1f}")

    if meal_times:
        # --- User-specified meal times: bucket each entry to nearest meal ---
        return _bucket_to_meal_times(carb_entries, meal_times, n_days)
    else:
        # --- Auto-detect: merge adjacent active hours into slots ---
        return _auto_detect_meal_slots(carb_entries, hourly, n_days)


def _bucket_to_meal_times(
    carb_entries: List[dict],
    meal_times: List[int],
    n_days: int,
) -> List[dict]:
    """Bucket carb entries to user-specified meal times (clock hours).

    Each carb entry is assigned to the nearest meal time. Daily totals per
    bucket are computed, then mean/sd across days.
    """
    meal_times = sorted(meal_times)
    print(f"\n  User-specified meal times: {', '.join(f'{h:02d}:00' for h in meal_times)}")

    def nearest_meal(clock_hour: int, clock_min: int) -> int:
        """Return the meal_time index nearest to this clock time."""
        entry_min = clock_hour * 60 + clock_min
        best_idx = 0
        best_dist = 1440
        for i, mh in enumerate(meal_times):
            meal_min = mh * 60
            # Handle wrap-around (e.g., 23:00 is close to 00:00 meal)
            dist = min(abs(entry_min - meal_min), 1440 - abs(entry_min - meal_min))
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        return best_idx

    # Bucket entries by (date, meal_index) → sum of carbs for that meal on that day
    daily_meal_carbs = defaultdict(lambda: defaultdict(float))
    daily_meal_absorption = defaultdict(lambda: defaultdict(list))
    bucket_all_carbs = defaultdict(list)  # all individual entries per bucket

    for e in carb_entries:
        idx = nearest_meal(e["hour"], e["local_dt"].minute)
        day = e["local_dt"].date().isoformat()
        daily_meal_carbs[day][idx] += e["carbs"]
        bucket_all_carbs[idx].append(e["carbs"])
        if e["absorption_hrs"] is not None:
            daily_meal_absorption[day][idx].append(e["absorption_hrs"])

    # Compute per-meal stats from daily totals
    meal_specs = []
    print(f"\n  Meal buckets:")
    for i, mh in enumerate(meal_times):
        # Collect daily totals for this meal (including 0 for days with no entry)
        all_days = sorted(set(e["local_dt"].date().isoformat() for e in carb_entries))
        daily_totals = [daily_meal_carbs[day].get(i, 0.0) for day in all_days]
        # Only include days where this meal had carbs (don't count skipped meals as 0g)
        nonzero_totals = [t for t in daily_totals if t > 0]

        if not nonzero_totals:
            print(f"    {mh:02d}:00 → no carb entries")
            continue

        arr = np.array(nonzero_totals)
        n_entries = len(bucket_all_carbs[i])
        skip_rate = 1.0 - len(nonzero_totals) / len(all_days)

        # Absorption: median from Loop metadata if available
        all_abs = []
        for day_abs in daily_meal_absorption.values():
            all_abs.extend(day_abs.get(i, []))
        if all_abs:
            absorption_hrs = round(float(np.median(all_abs)), 1)
            abs_source = f"Loop median of {len(all_abs)}"
        else:
            absorption_hrs = 3.0
            abs_source = "default"

        # Convert clock hour to sim time (minutes from 7am)
        time_of_day_minutes = (mh * 60 - 7 * 60) % 1440

        spec = {
            "time_of_day_minutes": time_of_day_minutes,
            "carbs_mean": round(float(np.mean(arr)), 1),
            "carbs_sd": round(float(np.std(arr)), 1) if len(arr) > 1 else round(float(np.mean(arr)) * 0.3, 1),
            "absorption_hrs": absorption_hrs,
        }
        meal_specs.append(spec)

        print(f"    {mh:02d}:00 → sim t={spec['time_of_day_minutes']}min, "
              f"mean={spec['carbs_mean']}g, sd={spec['carbs_sd']}g, "
              f"abs={absorption_hrs}h ({abs_source}), "
              f"{len(nonzero_totals)}/{len(all_days)} days ({n_entries} entries), "
              f"skip={skip_rate:.0%}")

    return meal_specs


def _auto_detect_meal_slots(
    carb_entries: List[dict],
    hourly: Dict[int, List[float]],
    n_days: int,
) -> List[dict]:
    """Auto-detect meal slots by merging adjacent active hours."""
    min_freq = 0.3
    active_hours = sorted(h for h in range(24) if len(hourly[h]) / n_days >= min_freq)

    if not active_hours:
        min_freq = 0.1
        active_hours = sorted(h for h in range(24) if len(hourly[h]) / n_days >= min_freq)

    max_slot_hours = 2
    slots = []
    if active_hours:
        current_slot = [active_hours[0]]
        for h in active_hours[1:]:
            if h == current_slot[-1] + 1 and len(current_slot) < max_slot_hours:
                current_slot.append(h)
            else:
                slots.append(current_slot)
                current_slot = [h]
        slots.append(current_slot)

    meal_specs = []
    print(f"\n  Auto-detected meal slots:")
    for slot_hours in slots:
        slot_carbs = []
        slot_times_min = []
        slot_absorption_hrs = []
        for e in carb_entries:
            if e["hour"] in slot_hours:
                slot_carbs.append(e["carbs"])
                slot_times_min.append(e["local_dt"].hour * 60 + e["local_dt"].minute)
                if e["absorption_hrs"] is not None:
                    slot_absorption_hrs.append(e["absorption_hrs"])

        if not slot_carbs:
            continue

        arr = np.array(slot_carbs)
        mean_time_min = int(np.mean(slot_times_min))
        time_of_day_minutes = (mean_time_min - 7 * 60) % 1440

        if slot_absorption_hrs:
            absorption_hrs = round(float(np.median(slot_absorption_hrs)), 1)
            abs_source = f"Loop median of {len(slot_absorption_hrs)}"
        else:
            absorption_hrs = 3.0
            abs_source = "default"

        spec = {
            "time_of_day_minutes": time_of_day_minutes,
            "carbs_mean": round(float(np.mean(arr)), 1),
            "carbs_sd": round(float(np.std(arr)), 1) if len(arr) > 1 else round(float(np.mean(arr)) * 0.3, 1),
            "absorption_hrs": absorption_hrs,
        }
        meal_specs.append(spec)

        slot_label = f"{slot_hours[0]:02d}:00-{slot_hours[-1]+1:02d}:00"
        freq = len(slot_carbs) / n_days
        print(f"    {slot_label} → sim t={spec['time_of_day_minutes']}min, "
              f"mean={spec['carbs_mean']}g, sd={spec['carbs_sd']}g, "
              f"abs={absorption_hrs}h ({abs_source}), "
              f"{freq:.1f}/day ({len(slot_carbs)} entries)")

    return meal_specs


# ─── Step 3: Exercise Patterns ───────────────────────────────────────────────

def extract_exercise_pattern(
    treatments: List[dict],
    tz: ZoneInfo,
    n_days: int,
) -> Tuple[float, Optional[dict]]:
    """Extract exercise frequency and typical spec from Exercise treatments.

    Returns (exercises_per_week, exercise_spec_dict_or_None).
    """
    print(f"\n{'='*70}")
    print("Step 3: Exercise Patterns")
    print(f"{'='*70}")

    exercise_entries = [t for t in treatments if t.get("eventType") == "Exercise"]

    if not exercise_entries:
        print("\n  No Exercise events found.")
        return 0.0, None

    # Categorize by notes field
    categories = defaultdict(list)
    for t in exercise_entries:
        notes = t.get("notes", "").strip()
        created = t.get("created_at", "")
        try:
            dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            continue
        local_dt = dt.astimezone(tz)
        duration = t.get("duration", 0)
        categories[notes or "(no notes)"].append({
            "local_dt": local_dt,
            "duration_min": duration,
        })

    print(f"\n  Exercise categories:")
    for notes, entries in sorted(categories.items(), key=lambda x: -len(x[1])):
        freq = len(entries) / n_days * 7
        print(f"    '{notes}': {len(entries)} events ({freq:.1f}/week)")

    # "Cardio" = real exercise; others are Trio override presets
    real_exercise = categories.get("Cardio", [])
    if not real_exercise:
        # Try to find the most frequent category as real exercise
        # but only if it's not clearly an override preset
        override_keywords = {"off", "slight", "override", "preset", "jht"}
        for notes, entries in sorted(categories.items(), key=lambda x: -len(x[1])):
            if not any(kw in notes.lower() for kw in override_keywords):
                real_exercise = entries
                print(f"\n  Using '{notes}' as real exercise category")
                break

    if not real_exercise:
        print("\n  No real exercise events identified (all appear to be override presets).")
        return 0.0, None

    exercises_per_week = len(real_exercise) / n_days * 7
    print(f"\n  Real exercise: {len(real_exercise)} events = {exercises_per_week:.1f}/week")

    # Typical time of day
    times_min = [e["local_dt"].hour * 60 + e["local_dt"].minute for e in real_exercise]
    mean_time = int(np.mean(times_min))
    time_of_day_minutes = (mean_time - 7 * 60) % 1440
    print(f"  Typical time: {mean_time // 60:02d}:{mean_time % 60:02d} "
          f"(sim t={time_of_day_minutes}min)")

    # Duration
    durations = [e["duration_min"] for e in real_exercise if e["duration_min"] > 0]
    if durations:
        mean_dur = np.mean(durations)
        print(f"  Typical duration: {mean_dur:.0f} min")

    exercise_spec = {
        "time_of_day_minutes": time_of_day_minutes,
        "declared_scalar": 0.5,
        "declared_duration_hrs": 3.0,
        "actual_scalar_mean": 0.5,
        "actual_scalar_sigma": 0.15,
        "actual_duration_hrs_mean": 6.0,
        "actual_duration_hrs_sigma": 0.2,
    }

    return round(exercises_per_week, 1), exercise_spec


# ─── Step 4: BG Stats ────────────────────────────────────────────────────────

def extract_bg_stats(entries: List[dict], tz: ZoneInfo) -> dict:
    """Compute BG statistics from CGM entries.

    Returns dict with starting_bg and reference stats.
    """
    print(f"\n{'='*70}")
    print("Step 4: BG Statistics")
    print(f"{'='*70}")

    sgv_values = []
    morning_values = []  # 7-8am local

    for e in entries:
        sgv = e.get("sgv")
        date_ms = e.get("date")
        if sgv is None or date_ms is None or sgv <= 0:
            continue
        sgv_values.append(sgv)

        dt_utc = datetime.fromtimestamp(date_ms / 1000, tz=timezone.utc)
        local_dt = dt_utc.astimezone(tz)
        if 7 <= local_dt.hour < 8:
            morning_values.append(sgv)

    if not sgv_values:
        print("\n  No CGM data available.")
        return {"starting_bg": 120.0}

    arr = np.array(sgv_values)
    n_total = len(arr)

    mean_bg = float(np.mean(arr))
    sd_bg = float(np.std(arr))
    tir = float(np.mean((arr >= 70) & (arr <= 180)) * 100)
    time_below_70 = float(np.mean(arr < 70) * 100)
    time_below_54 = float(np.mean(arr < 54) * 100)
    time_above_180 = float(np.mean(arr > 180) * 100)
    time_above_250 = float(np.mean(arr > 250) * 100)

    starting_bg = float(np.median(morning_values)) if morning_values else mean_bg

    print(f"\n  CGM readings: {n_total}")
    print(f"  Mean BG:      {mean_bg:.0f} mg/dL")
    print(f"  SD:           {sd_bg:.0f} mg/dL")
    print(f"  GMI:          {(3.31 + 0.02392 * mean_bg):.1f}%")
    print(f"  TIR (70-180): {tir:.1f}%")
    print(f"  Below 70:     {time_below_70:.1f}%")
    print(f"  Below 54:     {time_below_54:.1f}%")
    print(f"  Above 180:    {time_above_180:.1f}%")
    print(f"  Above 250:    {time_above_250:.1f}%")
    print(f"  Morning BG:   {starting_bg:.0f} mg/dL (median 7-8am, {len(morning_values)} readings)")

    return {
        "starting_bg": round(starting_bg, 0),
        "mean_bg": round(mean_bg, 0),
        "sd_bg": round(sd_bg, 0),
        "tir": round(tir, 1),
        "time_below_70": round(time_below_70, 1),
        "time_below_54": round(time_below_54, 1),
        "time_above_180": round(time_above_180, 1),
        "time_above_250": round(time_above_250, 1),
    }


# ─── Step 5: Assemble Profile ────────────────────────────────────────────────

def build_profile(
    url: str,
    days: int = 28,
    token: Optional[str] = None,
    output_path: Optional[str] = None,
    layer2: bool = True,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    insulin_type: str = DEFAULT_INSULIN_TYPE,
    meal_times: Optional[List[int]] = None,
) -> dict:
    """Full pipeline: fetch NS data → extract patterns → build PatientProfile dict.

    Date range: if start_date/end_date are provided, they take precedence over days.
    If only one is provided, the other is inferred from days.
    insulin_type: one of INSULIN_TYPES keys (novolog, humalog, fiasp, lyumjev).
    meal_times: optional list of clock hours (e.g. [8, 13, 19]) to bucket carbs into.
    """

    if start_date and end_date:
        days = max(1, (end_date - start_date).days)
    elif start_date:
        end_date = start_date + timedelta(days=days)
    elif end_date:
        start_date = end_date - timedelta(days=days)
    else:
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days)

    # Ensure timezone-aware
    if start_date.tzinfo is None:
        start_date = start_date.replace(tzinfo=timezone.utc)
    if end_date.tzinfo is None:
        end_date = end_date.replace(tzinfo=timezone.utc)

    # Resolve insulin type to curve
    insulin_curve = INSULIN_TYPES.get(insulin_type.lower(), "rapid-acting")

    print(f"Querying {url}")
    print(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"Insulin type: {insulin_type} (curve: {insulin_curve})")

    # --- Fetch NS profile history ---
    print("\nFetching Nightscout profile history...")
    ns_profile = fetch_profile(url, token=token)  # current profile (for timezone, etc.)
    profile_history = fetch_profile_history(url, token=token)
    print(f"  {len(profile_history)} distinct profile entries in history")

    # Build timeline for per-day settings lookup
    profile_timeline = ProfileTimeline(profile_history, fallback_profile=ns_profile)

    # Determine timezone
    tz_str = ns_profile.get("timezone", "America/New_York")
    tz = ZoneInfo(tz_str)
    print(f"  Timezone: {tz_str}")

    # Show profile changes in the date range
    print(profile_timeline.summary(start_date, end_date))

    # --- Step 1: Algorithm settings (time-weighted average for the date range) ---
    avg_settings = profile_timeline.average_settings(start_date, end_date)
    # Also extract full schedule from the midpoint profile for display
    midpoint = start_date + (end_date - start_date) / 2
    midpoint_profile = profile_timeline.get_at(midpoint)
    algo_settings = extract_algorithm_settings(midpoint_profile)
    # Override with the time-weighted averages across the full range
    algo_settings["insulin_sensitivity_factor"] = avg_settings["isf"]
    algo_settings["carb_ratio"] = avg_settings["cr"]
    algo_settings["basal_rate"] = avg_settings["basal"]
    print(f"\n  Time-weighted averages across date range:")
    print(f"    ISF: {avg_settings['isf']:.1f}, CR: {avg_settings['cr']:.1f}, "
          f"Basal: {avg_settings['basal']:.3f}")

    # Merge with defaults from settings.json
    settings_path = Path(__file__).parent / "settings.json"
    with open(settings_path) as f:
        default_settings = json.load(f)
    merged_settings = {**default_settings, **algo_settings}

    # --- Fetch treatments ---
    print("\nFetching treatments...")
    all_treatments = fetch_treatments(url, start_date, end_date, token=token, count=50000)
    print(f"  Got {len(all_treatments)} treatment entries")

    # Detect data source (Loop vs Trio)
    loop_count = sum(1 for t in all_treatments if "loop" in t.get("enteredBy", "").lower())
    trio_count = sum(1 for t in all_treatments if "trio" in t.get("enteredBy", "").lower())
    data_source = "loop" if loop_count > trio_count else ("trio" if trio_count > 0 else "unknown")
    print(f"  Data source: {data_source} (Loop={loop_count}, Trio={trio_count})")

    # --- Step 2: Meal patterns ---
    meal_specs = extract_meal_pattern(all_treatments, tz, days, meal_times=meal_times)

    # --- Step 3: Exercise ---
    exercises_per_week, exercise_spec = extract_exercise_pattern(all_treatments, tz, days)

    # --- Fetch CGM entries ---
    print("\nFetching CGM entries...")
    cgm_entries = fetch_entries(url, start_date, end_date, token=token)
    print(f"  Got {len(cgm_entries)} CGM readings")

    # --- Step 4: BG stats ---
    bg_stats = extract_bg_stats(cgm_entries, tz)

    # --- Step 6: Insulin timeline (built but not included in profile) ---
    print("\nReconstructing insulin timeline...")
    # Use midpoint profile's basal schedule for gap-filling
    basal_schedule = midpoint_profile.get("basal", [])
    insulin_timeline = reconstruct_insulin_timeline(
        all_treatments, basal_schedule, tz, profile_timeline=profile_timeline
    )
    print(f"  Built {len(insulin_timeline)} insulin events")

    # Count by type
    type_counts = defaultdict(int)
    type_units = defaultdict(float)
    for evt in insulin_timeline:
        type_counts[evt.event_type] += 1
        type_units[evt.event_type] += evt.units
    for etype in sorted(type_counts.keys()):
        print(f"    {etype}: {type_counts[etype]} events, {type_units[etype]:.1f}U total")

    # --- Layer 2: Deviation Analysis ---
    l2_sensitivity_sigma = 0.15
    l2_carb_count_sigma = 0.15
    l2_carb_count_bias = 0.0
    l2_undeclared_meal_prob = 0.0
    l2_undeclared_meals = []

    if layer2 and insulin_timeline and cgm_entries:
        # Collect carb timestamps for meal classification
        carb_times_ms = []
        carb_treatment_entries = []
        for t in all_treatments:
            carbs = t.get("carbs")
            if not carbs or carbs <= 0:
                continue
            created = t.get("created_at", "")
            if not created:
                continue
            try:
                dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                continue
            carb_times_ms.append(int(dt.timestamp() * 1000))
            carb_treatment_entries.append(t)

        # Step L2.1: Compute deviations (uses per-point ISF from profile timeline)
        devs = compute_deviations(
            cgm_entries, insulin_timeline, merged_settings, basal_schedule, tz,
            profile_timeline=profile_timeline,
            insulin_curve=insulin_curve,
        )

        if devs:
            # Step L2.2: Sensitivity sigma
            l2_sensitivity_sigma = estimate_sensitivity_sigma(devs, carb_times_ms, tz)

            # Step L2.3: Carb counting error (uses per-day ISF/CR from profile timeline)
            l2_carb_count_sigma, l2_carb_count_bias = estimate_carb_counting_error(
                devs, carb_treatment_entries, carb_times_ms, merged_settings,
                profile_timeline=profile_timeline,
            )

            # Step L2.4: Undeclared meals — skipped (too noisy; picks up
            # dawn phenomenon and BG drift rather than real undeclared food)
    elif layer2:
        print("\n  Skipping Layer 2: insufficient insulin/CGM data")

    # --- Assemble profile ---
    profile = {
        "meals": meal_specs,
        "carb_count_sigma": l2_carb_count_sigma,
        "carb_count_bias": l2_carb_count_bias,
        "absorption_sigma": 0.15,
        "undeclared_meal_prob": 0.0,
        "undeclared_meals": [],
        "sensitivity_sigma": l2_sensitivity_sigma,
        "exercises_per_week": exercises_per_week,
        "starting_bg": bg_stats["starting_bg"],
        "rescue_carbs_enabled": True,
        "rescue_threshold": 65.0,
        "rescue_carbs_grams": 8.0,
        "rescue_absorption_hrs": 1.0,
        "rescue_cooldown_min": 15.0,
        "rescue_carbs_declared_pct": 0.5,
        "algorithm_settings": merged_settings,
        "ns_reference_stats": {
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "days": days,
            "data_source": data_source,
            "mean_bg": bg_stats["mean_bg"],
            "sd_bg": bg_stats["sd_bg"],
            "tir": bg_stats["tir"],
            "time_below_70": bg_stats["time_below_70"],
            "time_below_54": bg_stats["time_below_54"],
            "time_above_180": bg_stats["time_above_180"],
            "time_above_250": bg_stats["time_above_250"],
            "gmi": round(3.31 + 0.02392 * bg_stats["mean_bg"], 1),
        },
    }

    if exercise_spec:
        profile["exercise_spec"] = exercise_spec

    # --- Summary ---
    print(f"\n{'='*70}")
    print("Profile Summary")
    print(f"{'='*70}")
    print(f"\n  Extracted from Nightscout (Layer 1):")
    print(f"    Basal rate:  {merged_settings.get('basal_rate', '?')} U/hr")
    print(f"    Carb ratio:  {merged_settings.get('carb_ratio', '?')} g/U")
    print(f"    ISF:         {merged_settings.get('insulin_sensitivity_factor', '?')} mg/dL/U")
    print(f"    Target:      {merged_settings.get('target', '?')} mg/dL")
    print(f"    Meals:       {len(meal_specs)} slots")
    print(f"    Exercise:    {exercises_per_week}/week")
    print(f"    Starting BG: {bg_stats['starting_bg']} mg/dL")
    if layer2:
        print(f"\n  Estimated from deviation analysis (Layer 2):")
        print(f"    sensitivity_sigma:    {l2_sensitivity_sigma}")
        print(f"    carb_count_sigma:     {l2_carb_count_sigma}")
        print(f"    carb_count_bias:      {l2_carb_count_bias}")
        print(f"    undeclared_meal_prob: 0.0 (detection disabled)")
        print(f"    undeclared_meals:     0 slots (detection disabled)")
    else:
        print(f"\n  Defaulted (Layer 2 skipped):")
        print(f"    carb_count_sigma: 0.15")
        print(f"    carb_count_bias:  0.0")
        print(f"    sensitivity_sigma: 0.15")
        print(f"    undeclared_meal_prob: 0.0")
    print(f"\n  Reference BG stats (for calibration):")
    print(f"    Mean: {bg_stats['mean_bg']}, SD: {bg_stats['sd_bg']}, "
          f"TIR: {bg_stats['tir']}%, <70: {bg_stats['time_below_70']}%")

    # --- Save ---
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(profile, f, indent=4)
        print(f"\n  Saved to {output_path}")

    return profile


# ─── Step 6: Insulin History Reconstruction ──────────────────────────────────

def reconstruct_insulin_timeline(
    treatments: List[dict],
    basal_schedule: List[dict],
    tz: ZoneInfo,
    profile_timeline: Optional['ProfileTimeline'] = None,
) -> List[InsulinEvent]:
    """Reconstruct unified insulin timeline from NS treatments.

    Combines temp basals, boluses, and SMBs into a sorted timeline.
    Fills gaps between temp basals with scheduled basal from the profile.
    If profile_timeline is provided, uses the historically active basal schedule
    for each gap instead of the single basal_schedule.
    """
    events = []

    # --- Parse temp basals ---
    temp_basals = []
    for t in treatments:
        if t.get("eventType") != "Temp Basal":
            continue
        rate = t.get("rate")
        duration = t.get("duration", 0)
        created = t.get("created_at", "")
        if rate is None or not created:
            continue
        try:
            dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            continue
        temp_basals.append({
            "time": dt.timestamp(),
            "rate": float(rate),
            "duration_min": float(duration),
        })

    temp_basals.sort(key=lambda x: x["time"])

    # Clip each temp basal's duration to the start of the next one
    # (Trio issues new temp basals every ~5 min, each declared for 30 min)
    for i in range(len(temp_basals)):
        if i + 1 < len(temp_basals):
            max_dur = (temp_basals[i + 1]["time"] - temp_basals[i]["time"]) / 60.0
            temp_basals[i]["duration_min"] = max(0, min(temp_basals[i]["duration_min"], max_dur))

    # Add temp basal segments as events
    for tb in temp_basals:
        units = tb["rate"] * tb["duration_min"] / 60.0
        events.append(InsulinEvent(
            time=tb["time"],
            event_type="basal",
            units=round(units, 4),
            rate=tb["rate"],
            duration_min=tb["duration_min"],
        ))

    # --- Fill gaps with scheduled basal ---
    if temp_basals and (basal_schedule or profile_timeline):
        fallback_schedule = sorted(basal_schedule, key=lambda e: e["timeAsSeconds"]) if basal_schedule else []
        for i in range(len(temp_basals) - 1):
            gap_start = temp_basals[i]["time"] + temp_basals[i]["duration_min"] * 60
            gap_end = temp_basals[i + 1]["time"]
            gap_duration_min = (gap_end - gap_start) / 60.0

            if gap_duration_min > 1.0:  # Only fill gaps > 1 minute
                gap_dt = datetime.fromtimestamp(gap_start, tz=timezone.utc).astimezone(tz)
                seconds_from_midnight = gap_dt.hour * 3600 + gap_dt.minute * 60 + gap_dt.second

                # Use historically active basal schedule if available
                if profile_timeline:
                    gap_dt_utc = datetime.fromtimestamp(gap_start, tz=timezone.utc)
                    active_basal = profile_timeline.basal_schedule_at(gap_dt_utc)
                    sorted_schedule = sorted(active_basal, key=lambda e: e["timeAsSeconds"]) if active_basal else fallback_schedule
                else:
                    sorted_schedule = fallback_schedule

                scheduled_rate = _get_scheduled_rate(sorted_schedule, seconds_from_midnight)

                units = scheduled_rate * gap_duration_min / 60.0
                events.append(InsulinEvent(
                    time=gap_start,
                    event_type="basal",
                    units=round(units, 4),
                    rate=scheduled_rate,
                    duration_min=round(gap_duration_min, 1),
                ))

    # --- Parse boluses and SMBs ---
    for t in treatments:
        insulin = t.get("insulin")
        if not insulin or insulin <= 0:
            continue
        event_type = t.get("eventType", "")
        if event_type == "Temp Basal":
            continue  # Already handled

        created = t.get("created_at", "")
        if not created:
            continue
        try:
            dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            continue

        is_smb = event_type == "SMB" or (
            t.get("enteredBy", "").lower().startswith("trio") and insulin < 1.0
        )

        events.append(InsulinEvent(
            time=dt.timestamp(),
            event_type="smb" if is_smb else "bolus",
            units=float(insulin),
            rate=0.0,
            duration_min=0.0,
        ))

    events.sort(key=lambda e: e.time)
    return events


def _get_scheduled_rate(sorted_schedule: List[dict], seconds_from_midnight: int) -> float:
    """Get the scheduled basal rate for a given time of day."""
    rate = sorted_schedule[0]["value"]  # Default to first entry
    for entry in sorted_schedule:
        if entry["timeAsSeconds"] <= seconds_from_midnight:
            rate = entry["value"]
        else:
            break
    return rate


# ─── Layer 2: Deviation Analysis ──────────────────────────────────────────────

def _insulin_timeline_to_pump_history(insulin_timeline: List[InsulinEvent]) -> List[dict]:
    """Convert InsulinEvent list to Trio-format pump history for find_insulin()."""
    history = []
    for evt in insulin_timeline:
        ts_iso = datetime.fromtimestamp(evt.time, tz=timezone.utc).isoformat()
        if evt.event_type == "basal":
            history.append({
                "_type": "TempBasal",
                "timestamp": ts_iso,
                "rate": evt.rate,
                "duration": evt.duration_min,
            })
        else:  # bolus or smb
            history.append({
                "_type": "Bolus",
                "timestamp": ts_iso,
                "amount": evt.units,
            })
    return history


def _build_trio_profile(
    profile_settings: dict,
    basal_schedule: List[dict],
    insulin_curve: str = "rapid-acting",
) -> dict:
    """Build a minimal Trio-compatible profile dict for iob_total()."""
    dia = profile_settings.get("duration_of_insulin_action", 6.0)
    current_basal = profile_settings.get("basal_rate", 1.0)

    # Build basalprofile in Trio format: [{minutes, rate}, ...]
    basalprofile = []
    for entry in sorted(basal_schedule, key=lambda e: e["timeAsSeconds"]):
        basalprofile.append({
            "minutes": entry["timeAsSeconds"] // 60,
            "rate": entry["value"],
        })

    return {
        "dia": dia,
        "curve": insulin_curve,
        "current_basal": current_basal,
        "basalprofile": basalprofile,
    }


def compute_deviations(
    cgm_entries: List[dict],
    insulin_timeline: List[InsulinEvent],
    profile_settings: dict,
    basal_schedule: List[dict],
    tz: ZoneInfo,
    profile_timeline: Optional['ProfileTimeline'] = None,
    insulin_curve: str = "rapid-acting",
) -> List[dict]:
    """Compute BG deviations at each 5-min CGM point.

    At each point: deviation = observed_delta - BGI (insulin-predicted delta).
    If profile_timeline is provided, uses per-point ISF from the historically
    active profile. Otherwise uses the single ISF from profile_settings.

    Returns list of dicts: [{time_ms, local_hour, sgv, delta, bgi, deviation}, ...]
    """
    print(f"\n{'='*70}")
    print("Layer 2, Step 1: Computing Deviations")
    print(f"{'='*70}")

    # Sort CGM entries by time
    valid_cgm = []
    for e in cgm_entries:
        sgv = e.get("sgv")
        date_ms = e.get("date")
        if sgv and date_ms and sgv > 0:
            valid_cgm.append({"sgv": sgv, "date_ms": date_ms})
    valid_cgm.sort(key=lambda x: x["date_ms"])

    if len(valid_cgm) < 2:
        print("  Not enough CGM data for deviation analysis.")
        return []

    # Convert insulin timeline to pump history format
    pump_history = _insulin_timeline_to_pump_history(insulin_timeline)

    # Build Trio-compatible profile
    trio_profile = _build_trio_profile(profile_settings, basal_schedule, insulin_curve)
    fallback_isf = profile_settings.get("insulin_sensitivity_factor", 100.0)
    use_timeline_isf = profile_timeline is not None

    if use_timeline_isf:
        print(f"  Using per-point ISF from profile history")
    else:
        print(f"  Using fixed ISF: {fallback_isf}")

    # Pre-compute treatments once (find_insulin is expensive to call per-point)
    # We'll call iob_total at each CGM time using the full treatment list
    # Use a far-future clock so all events are "in the past" for find_insulin
    if pump_history:
        max_time_ms = max(e["date_ms"] for e in valid_cgm) + 60000
        treatments = find_insulin(pump_history, trio_profile, max_time_ms)
    else:
        treatments = []

    print(f"  CGM points: {len(valid_cgm)}")
    print(f"  Insulin treatments (micro-boluses): {len(treatments)}")

    deviations = []
    for i in range(1, len(valid_cgm)):
        prev = valid_cgm[i - 1]
        curr = valid_cgm[i]

        # Only process ~5-min intervals (allow 3-7 min)
        dt_min = (curr["date_ms"] - prev["date_ms"]) / 60000.0
        if dt_min < 3 or dt_min > 7:
            continue

        # Observed BG change
        delta = curr["sgv"] - prev["sgv"]

        # Get insulin activity at this point
        iob_result = iob_total(treatments, curr["date_ms"], trio_profile)
        activity = iob_result.get("activity", 0)

        # ISF: use the historically active profile's ISF at this timestamp
        isf = profile_timeline.isf_at_ms(curr["date_ms"]) if use_timeline_isf else fallback_isf

        # BGI = expected BG change from insulin (activity is U/min)
        bgi = -activity * isf * 5.0

        deviation = delta - bgi

        local_dt = datetime.fromtimestamp(
            curr["date_ms"] / 1000, tz=timezone.utc
        ).astimezone(tz)

        deviations.append({
            "time_ms": curr["date_ms"],
            "local_hour": local_dt.hour,
            "local_date": local_dt.date().isoformat(),
            "sgv": curr["sgv"],
            "delta": round(delta, 2),
            "bgi": round(bgi, 2),
            "deviation": round(deviation, 2),
        })

    print(f"  Valid deviations computed: {len(deviations)}")
    if deviations:
        devs = [d["deviation"] for d in deviations]
        print(f"  Deviation stats: mean={np.mean(devs):.1f}, "
              f"sd={np.std(devs):.1f}, "
              f"median={np.median(devs):.1f}")

    return deviations


def estimate_sensitivity_sigma(
    deviations: List[dict],
    carb_times_ms: List[int],
    tz: ZoneInfo,
) -> float:
    """Estimate daily ISF variation from non-meal deviations.

    Classifies deviations as meal-related (within 3h of declared carb) or non-meal.
    Groups non-meal deviations by day, computes daily median → sensitivity ratio.
    Returns SD of daily ratios.
    """
    print(f"\n{'='*70}")
    print("Layer 2, Step 2: Sensitivity Sigma")
    print(f"{'='*70}")

    MEAL_WINDOW_MS = 3 * 3600 * 1000  # 3 hours

    # Classify deviations
    non_meal_by_day = defaultdict(list)
    n_meal = 0
    n_nonmeal = 0

    for d in deviations:
        # Check if within meal window
        is_meal = any(
            0 <= (d["time_ms"] - ct) <= MEAL_WINDOW_MS
            for ct in carb_times_ms
        )
        if is_meal:
            n_meal += 1
            continue

        n_nonmeal += 1
        non_meal_by_day[d["local_date"]].append(d["deviation"])

    print(f"  Meal-related deviations: {n_meal}")
    print(f"  Non-meal deviations: {n_nonmeal}")
    print(f"  Days with non-meal data: {len(non_meal_by_day)}")

    if len(non_meal_by_day) < 3:
        print("  Not enough days for sensitivity estimate, using default 0.15")
        return 0.15

    # Per-day median deviation → proxy for daily sensitivity shift
    daily_medians = []
    for day, devs in sorted(non_meal_by_day.items()):
        med = float(np.median(devs))
        daily_medians.append(med)

    # Global median (expected ~0 if ISF is correct on average)
    global_median = float(np.median(daily_medians))

    # Convert deviations to ratios: ratio = 1 + (median_dev / typical_bg_change)
    # A typical 5-min BG change from basal insulin is small (~1-2 mg/dL)
    # We use the SD of daily medians relative to the global median
    # as a proxy for sensitivity_sigma
    centered = np.array(daily_medians) - global_median

    # Convert to fractional variation: SD of deviation / typical ISF effect
    # The SD of daily median deviations, divided by a reference BG range,
    # gives the fractional sensitivity variation
    # Use IQR-based estimate for robustness
    q75, q25 = np.percentile(centered, [75, 25])
    iqr = q75 - q25
    robust_sd = iqr / 1.35  # IQR to SD conversion for normal distribution

    # Normalize by typical ISF magnitude to get fractional sigma
    # Median absolute deviation of daily medians / reference scale
    # Reference: a 1-sigma sensitivity shift changes BG by ~ISF * basal_daily_dose
    # Simpler: use the ratio of robust_sd to the absolute global median + offset
    # to avoid division by near-zero
    reference_scale = max(abs(global_median), 2.0) + robust_sd
    sigma = robust_sd / reference_scale

    # Clamp to reasonable range
    sigma = max(0.03, min(sigma, 0.50))

    print(f"  Daily median deviations: mean={np.mean(daily_medians):.2f}, "
          f"sd={np.std(daily_medians):.2f}")
    print(f"  Global median: {global_median:.2f}")
    print(f"  Robust SD (IQR-based): {robust_sd:.2f}")
    print(f"  → sensitivity_sigma: {sigma:.3f}")

    return round(sigma, 3)


def estimate_carb_counting_error(
    deviations: List[dict],
    carb_entries: List[dict],
    carb_times_ms: List[int],
    profile_settings: dict,
    profile_timeline: Optional['ProfileTimeline'] = None,
) -> Tuple[float, float]:
    """Estimate carb counting sigma and bias using daily totals.

    Per-meal analysis fails when carb entries are frequent and their absorption
    windows overlap (92% of entries in typical grazing pattern). Instead, we
    compare daily total declared carbs to daily total implied carbs (from
    baseline-adjusted positive deviations), which avoids double-counting.

    If profile_timeline is provided, uses per-day ISF/CR from the historically
    active profile. Otherwise uses the single values from profile_settings.

    Returns (sigma, bias) for lognormal carb counting error model.
    """
    print(f"\n{'='*70}")
    print("Layer 2, Step 3: Carb Counting Error")
    print(f"{'='*70}")

    fallback_isf = profile_settings.get("insulin_sensitivity_factor", 100.0)
    fallback_cr = profile_settings.get("carb_ratio", 10.0)
    use_timeline = profile_timeline is not None

    if use_timeline:
        print(f"  Using per-day ISF/CR from profile history")
    else:
        isf = fallback_isf
        cr = fallback_cr
        print(f"  Using fixed ISF={isf}, CR={cr}, CSF={isf/cr:.1f}")

    MEAL_EXCLUDE_MS = 3 * 3600 * 1000

    # Compute baseline: median of 1am-6am non-meal deviations (cleanest window)
    night_non_meal_devs = []
    all_non_meal_devs = []
    for d in deviations:
        near_carb = any(
            0 <= (d["time_ms"] - ct) <= MEAL_EXCLUDE_MS
            for ct in carb_times_ms
        )
        if not near_carb:
            all_non_meal_devs.append(d["deviation"])
            if 1 <= d["local_hour"] < 6:
                night_non_meal_devs.append(d["deviation"])

    if night_non_meal_devs:
        baseline = float(np.median(night_non_meal_devs))
        print(f"  Baseline deviation (1am-6am non-meal median): {baseline:.2f} mg/dL/5min "
              f"({len(night_non_meal_devs)} points)")
    else:
        baseline = float(np.median(all_non_meal_devs)) if all_non_meal_devs else 0.0
        print(f"  Baseline deviation (all non-meal median, no overnight data): {baseline:.2f} mg/dL/5min")

    # Group declared carbs by day
    daily_declared = defaultdict(float)
    for t in carb_entries:
        carbs = t.get("carbs", 0)
        if not carbs or carbs <= 0:
            continue
        created = t.get("created_at", "")
        if not created:
            continue
        try:
            dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            continue
        day = dt.date().isoformat()
        daily_declared[day] += carbs

    # Group baseline-adjusted positive deviations by day → implied carbs
    daily_pos_dev_sum = defaultdict(float)
    daily_dev_count = defaultdict(int)
    for d in deviations:
        adjusted = d["deviation"] - baseline
        daily_dev_count[d["local_date"]] += 1
        if adjusted > 0:
            daily_pos_dev_sum[d["local_date"]] += adjusted

    # Compute daily ratios (implied/declared)
    log_ratios = []
    print(f"\n  {'Day':<12} {'Declared':>10} {'Implied':>10} {'CSF':>6} {'Ratio':>8}")
    print(f"  {'-'*50}")
    for day in sorted(daily_declared.keys()):
        declared = daily_declared[day]
        if declared < 10:  # Skip days with very few declared carbs
            continue
        if daily_dev_count.get(day, 0) < 100:  # Need ~8h of CGM data
            continue

        # Get per-day CSF from profile timeline
        if use_timeline:
            # Use noon of that day as representative timestamp
            day_dt = datetime.fromisoformat(day + "T12:00:00+00:00")
            day_ms = int(day_dt.timestamp() * 1000)
            isf = profile_timeline.isf_at_ms(day_ms)
            cr = profile_timeline.cr_at_ms(day_ms)
        csf = isf / cr

        implied = daily_pos_dev_sum.get(day, 0) / csf
        if implied < 1:
            continue

        ratio = implied / declared
        log_ratios.append(math.log(ratio))
        print(f"  {day}   {declared:>8.0f}g   {implied:>8.0f}g  {csf:>5.1f}  {ratio:>7.2f}x")

    print(f"\n  Analyzable days: {len(log_ratios)}")

    if len(log_ratios) < 5:
        print("  Not enough days for reliable estimate, using defaults")
        return 0.15, 0.0

    log_ratios_arr = np.array(log_ratios)

    # sigma = SD of log-ratios (day-to-day variation in counting accuracy)
    sigma = float(np.std(log_ratios_arr))

    # bias = mean(log-ratios)
    # log-ratio = log(actual/declared). Positive mean → actual > declared → under-declaration
    # Sim model: declared = actual * exp(N(-bias, sigma))
    # So positive bias → exp(-bias) < 1 → declared < actual ✓
    bias = float(np.mean(log_ratios_arr))

    # Clamp to reasonable ranges
    sigma = max(0.03, min(sigma, 0.60))
    bias = max(-1.0, min(bias, 1.0))

    implied_ratios = np.exp(log_ratios_arr)
    print(f"  Daily implied/declared ratio: median={np.median(implied_ratios):.2f}x, "
          f"mean={np.mean(implied_ratios):.2f}x")
    print(f"  Log-ratio stats: mean={np.mean(log_ratios_arr):.3f}, "
          f"sd={np.std(log_ratios_arr):.3f}")
    print(f"  → carb_count_sigma: {sigma:.3f}")
    print(f"  → carb_count_bias: {bias:.3f}")

    return round(sigma, 3), round(bias, 3)


def detect_undeclared_meals(
    deviations: List[dict],
    carb_times_ms: List[int],
    tz: ZoneInfo,
    n_days: int,
    profile_settings: dict,
) -> Tuple[float, List[dict]]:
    """Detect undeclared meals from unexplained positive deviation clusters.

    Finds consecutive positive deviations (>3 mg/dL/5min) not within 4h of
    a declared carb. Each cluster → one undeclared meal.

    Returns (undeclared_meal_prob, undeclared_meal_specs).
    """
    print(f"\n{'='*70}")
    print("Layer 2, Step 4: Undeclared Meal Detection")
    print(f"{'='*70}")

    isf = profile_settings.get("insulin_sensitivity_factor", 100.0)
    cr = profile_settings.get("carb_ratio", 10.0)
    csf = isf / cr

    DECLARED_WINDOW_MS = 4 * 3600 * 1000  # Exclude 4h around declared carbs
    MIN_DEVIATION = 3.0  # mg/dL/5min threshold
    MIN_CLUSTER_POINTS = 3  # At least 15 min of positive deviations

    # Find deviations NOT near declared carbs and above threshold
    uam_candidates = []
    for d in deviations:
        near_carb = any(
            -30 * 60000 <= (d["time_ms"] - ct) <= DECLARED_WINDOW_MS
            for ct in carb_times_ms
        )
        if not near_carb and d["deviation"] > MIN_DEVIATION:
            uam_candidates.append(d)

    print(f"  UAM candidate points (dev > {MIN_DEVIATION}, not near carbs): "
          f"{len(uam_candidates)}")

    if not uam_candidates:
        print("  No undeclared meal evidence found.")
        return 0.0, []

    # Cluster consecutive candidates (gap <= 10 min)
    MAX_GAP_MS = 10 * 60000
    clusters = []
    current_cluster = [uam_candidates[0]]

    for i in range(1, len(uam_candidates)):
        if uam_candidates[i]["time_ms"] - current_cluster[-1]["time_ms"] <= MAX_GAP_MS:
            current_cluster.append(uam_candidates[i])
        else:
            if len(current_cluster) >= MIN_CLUSTER_POINTS:
                clusters.append(current_cluster)
            current_cluster = [uam_candidates[i]]

    if len(current_cluster) >= MIN_CLUSTER_POINTS:
        clusters.append(current_cluster)

    print(f"  UAM clusters (≥{MIN_CLUSTER_POINTS} points): {len(clusters)}")

    if not clusters:
        print("  No significant undeclared meal clusters found.")
        return 0.0, []

    # Estimate carbs and timing for each cluster
    n_declared = len(carb_times_ms)
    n_undeclared = len(clusters)

    if n_declared + n_undeclared > 0:
        prob = n_undeclared / (n_declared + n_undeclared)
    else:
        prob = 0.0

    # Group clusters by hour to build meal specs
    hourly_carbs = defaultdict(list)
    for cluster in clusters:
        sum_dev = sum(p["deviation"] for p in cluster)
        implied_carbs = sum_dev / csf
        # Use start time of cluster
        local_dt = datetime.fromtimestamp(
            cluster[0]["time_ms"] / 1000, tz=timezone.utc
        ).astimezone(tz)
        hourly_carbs[local_dt.hour].append(implied_carbs)

    # Build undeclared meal specs from most common hours
    meal_specs = []
    for hour in sorted(hourly_carbs.keys()):
        carbs_list = hourly_carbs[hour]
        if len(carbs_list) < 2:
            continue  # Need at least 2 events to establish a pattern

        mean_carbs = float(np.mean(carbs_list))
        if mean_carbs < 5:
            continue  # Skip tiny clusters

        time_of_day_minutes = (hour * 60 + 30 - 7 * 60) % 1440  # Mid-hour, relative to 7am
        spec = {
            "time_of_day_minutes": time_of_day_minutes,
            "carbs_mean": round(mean_carbs, 1),
            "carbs_sd": round(float(np.std(carbs_list)), 1) if len(carbs_list) > 1 else round(mean_carbs * 0.3, 1),
            "absorption_hrs": 3.0,
        }
        meal_specs.append(spec)
        print(f"    Hour {hour:02d}: {len(carbs_list)} events, "
              f"mean={mean_carbs:.0f}g implied carbs")

    prob = round(max(0.0, min(prob, 0.8)), 2)

    print(f"\n  Declared meals: {n_declared} ({n_declared / max(1,n_days):.1f}/day)")
    print(f"  Undeclared meals: {n_undeclared} ({n_undeclared / max(1,n_days):.1f}/day)")
    print(f"  → undeclared_meal_prob: {prob:.2f}")
    print(f"  → undeclared_meals specs: {len(meal_specs)} slots")

    return prob, meal_specs


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Infer a PatientProfile from Nightscout data"
    )
    parser.add_argument("--url", type=str, required=True,
                        help="Nightscout URL")
    parser.add_argument("--days", type=int, default=28,
                        help="Days of history to analyze (default 28)")
    parser.add_argument("--start-date", type=str, default=None,
                        help="Start date (YYYY-MM-DD). Overrides --days if used with --end-date.")
    parser.add_argument("--end-date", type=str, default=None,
                        help="End date (YYYY-MM-DD). Defaults to today if --start-date is given.")
    parser.add_argument("--token", type=str, default=None,
                        help="API access token")
    parser.add_argument("--output", type=str,
                        default="patient_profiles/ns_inferred.json",
                        help="Output path (default patient_profiles/ns_inferred.json)")
    parser.add_argument("--no-layer2", action="store_true",
                        help="Skip Layer 2 deviation analysis (use defaults)")
    parser.add_argument("--insulin-type", type=str, default=DEFAULT_INSULIN_TYPE,
                        choices=list(INSULIN_TYPES.keys()),
                        help=f"Insulin type for activity curve (default {DEFAULT_INSULIN_TYPE})")
    parser.add_argument("--meal-times", type=str, default=None,
                        help="Comma-separated clock hours for meal times (e.g. '8,13,19'). "
                             "Carb entries are bucketed to nearest meal. Omit for auto-detect.")
    args = parser.parse_args()

    start_dt = None
    end_dt = None
    if args.start_date:
        start_dt = datetime.strptime(args.start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    if args.end_date:
        end_dt = datetime.strptime(args.end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    meal_times_list = None
    if args.meal_times:
        meal_times_list = [int(h.strip()) for h in args.meal_times.split(",")]

    build_profile(
        url=args.url,
        days=args.days,
        token=args.token,
        output_path=args.output,
        layer2=not args.no_layer2,
        start_date=start_dt,
        end_date=end_dt,
        insulin_type=args.insulin_type,
        meal_times=meal_times_list,
    )


if __name__ == "__main__":
    main()
