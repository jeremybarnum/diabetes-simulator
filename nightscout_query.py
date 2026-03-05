#!/usr/bin/env python3
"""
Query Nightscout API to extract real-world patient statistics.

Pulls carb treatments and temp targets (exercise) to calibrate simulation profiles.

Usage:
    python3 nightscout_query.py
    python3 nightscout_query.py --days 30
    python3 nightscout_query.py --token YOUR_TOKEN
"""

import json
import requests
import hashlib
import os
import numpy as np
from datetime import datetime, timedelta, timezone
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Optional, Tuple

NIGHTSCOUT_URL = "https://ccmscout.us.nightscoutpro.com"


def fetch_treatments(
    base_url: str,
    start_date: datetime,
    end_date: datetime,
    token: Optional[str] = None,
    find_filter: Optional[Dict] = None,
    count: int = 10000,
) -> List[Dict]:
    """Fetch treatments from Nightscout API v1."""
    url = f"{base_url}/api/v1/treatments.json"

    params = {"count": count}
    params["find[created_at][$gte]"] = start_date.strftime("%Y-%m-%dT%H:%M:%SZ")
    params["find[created_at][$lte]"] = end_date.strftime("%Y-%m-%dT%H:%M:%SZ")

    if find_filter:
        for key, val in find_filter.items():
            params[key] = val

    headers = {"Accept": "application/json"}
    if token:
        # Try as access token in query string
        params["token"] = token

    resp = requests.get(url, headers=headers, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def fetch_entries(
    base_url: str,
    start_date: datetime,
    end_date: datetime,
    token: Optional[str] = None,
    count: int = 50000,
) -> List[Dict]:
    """Fetch CGM/SGV entries from Nightscout API v1.

    Returns list of dicts with at least 'sgv' and 'date' (epoch ms) fields.
    """
    url = f"{base_url}/api/v1/entries.json"

    params = {"count": count}
    params["find[date][$gte]"] = int(start_date.timestamp() * 1000)
    params["find[date][$lte]"] = int(end_date.timestamp() * 1000)

    headers = {"Accept": "application/json"}
    if token:
        params["token"] = token

    resp = requests.get(url, headers=headers, params=params, timeout=60)
    resp.raise_for_status()
    return resp.json()


def compute_median_daily_trace(
    entries: List[Dict],
    utc_offset_hours: float = -5.0,
    bucket_minutes: int = 5,
) -> List[Tuple[float, float]]:
    """Compute median BG at each time-of-day bucket over many days.

    Day runs 7am to 7am (matching simulation convention).

    Returns:
        List of (hour_from_7am, median_bg) tuples.
    """
    tz_offset = timedelta(hours=utc_offset_hours)
    n_buckets = 24 * 60 // bucket_minutes  # 288 for 5-min

    # Group SGV readings by time-of-day bucket
    buckets = defaultdict(list)

    for entry in entries:
        sgv = entry.get("sgv")
        date_ms = entry.get("date")
        if sgv is None or date_ms is None or sgv <= 0:
            continue

        dt_utc = datetime.fromtimestamp(date_ms / 1000, tz=timezone.utc)
        dt_local = dt_utc + tz_offset

        # Convert to minutes from 7am
        minutes_from_midnight = dt_local.hour * 60 + dt_local.minute
        minutes_from_7am = (minutes_from_midnight - 7 * 60) % 1440

        bucket_idx = minutes_from_7am // bucket_minutes
        if 0 <= bucket_idx < n_buckets:
            buckets[bucket_idx].append(sgv)

    # Compute median at each bucket
    trace = []
    for i in range(n_buckets):
        hour = i * bucket_minutes / 60.0
        if buckets[i]:
            median_bg = float(np.median(buckets[i]))
            trace.append((hour, median_bg))

    return trace


def fetch_and_save_nightscout_trace(
    days: int = 30,
    base_url: str = NIGHTSCOUT_URL,
    token: Optional[str] = None,
    utc_offset_hours: float = -5.0,
    output_dir: Optional[str] = None,
) -> Tuple[List[Tuple[float, float]], str]:
    """Fetch Nightscout CGM data, compute median trace, save to file.

    Returns:
        (median_trace, output_path)
    """
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=days)

    print(f"Fetching CGM entries from {base_url} ({days} days)...")
    entries = fetch_entries(base_url, start_date, end_date, token=token)
    print(f"  Got {len(entries)} CGM readings")

    trace = compute_median_daily_trace(entries, utc_offset_hours=utc_offset_hours)
    print(f"  Computed median trace: {len(trace)} points")

    # Save
    if output_dir is None:
        output_dir = str(Path(__file__).parent / "sim_results")
    Path(output_dir).mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_nightscout_{days}d.json"
    output_path = str(Path(output_dir) / filename)

    data = {
        "metadata": {
            "source": "nightscout",
            "url": base_url,
            "days": days,
            "n_entries": len(entries),
            "utc_offset_hours": utc_offset_hours,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
        },
        "trace": trace,
    }
    with open(output_path, "w") as f:
        json.dump(data, f)
    print(f"  Saved to {output_path}")

    return trace, output_path


def load_nightscout_trace(path: str) -> List[Tuple[float, float]]:
    """Load a previously saved Nightscout median trace."""
    with open(path) as f:
        data = json.load(f)
    return [tuple(p) for p in data["trace"]]


def analyze_carbs(treatments: List[Dict], n_days: int, utc_offset_hours: float = -5.0):
    """Analyze carb entries to get daily patterns."""
    # Group by local date
    daily_carbs = defaultdict(list)
    tz_offset = timedelta(hours=utc_offset_hours)

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

        # Convert to local time
        local_dt = dt + tz_offset
        date_str = local_dt.strftime("%Y-%m-%d")
        hour = local_dt.hour + local_dt.minute / 60.0
        absorption = t.get("absorptionTime")  # in minutes (from Loop, None for Trio)

        daily_carbs[date_str].append({
            "hour": hour,
            "carbs": carbs,
            "absorption_sec": absorption,
            "event_type": t.get("eventType", ""),
            "entered_by": t.get("enteredBy", ""),
        })

    # --- Summary statistics ---
    print(f"\n{'='*70}")
    print(f"Carb Analysis: {len(daily_carbs)} days with carb entries")
    print(f"{'='*70}")

    # Daily totals
    daily_totals = []
    daily_entry_counts = []
    for date_str in sorted(daily_carbs.keys()):
        entries = daily_carbs[date_str]
        total = sum(e["carbs"] for e in entries)
        daily_totals.append(total)
        daily_entry_counts.append(len(entries))

    if daily_totals:
        import numpy as np
        arr = np.array(daily_totals)
        print(f"\nDaily carb totals:")
        print(f"  Mean:   {np.mean(arr):.0f}g")
        print(f"  Median: {np.median(arr):.0f}g")
        print(f"  SD:     {np.std(arr):.0f}g")
        print(f"  Range:  {np.min(arr):.0f}-{np.max(arr):.0f}g")

        cnt = np.array(daily_entry_counts)
        print(f"\nEntries per day:")
        print(f"  Mean:   {np.mean(cnt):.1f}")
        print(f"  Median: {np.median(cnt):.0f}")
        print(f"  Range:  {np.min(cnt)}-{np.max(cnt)}")

    # --- Meal timing and size by time bucket ---
    print(f"\nMeal patterns by time of day:")
    print(f"  {'Time Bucket':<20} {'Count':>6} {'Mean g':>8} {'Median g':>10} {'SD g':>8}")
    print(f"  {'-'*52}")

    buckets = [
        ("Early AM (5-8)", 5, 8),
        ("Breakfast (8-10)", 8, 10),
        ("Morning (10-12)", 10, 12),
        ("Lunch (12-14)", 12, 14),
        ("Afternoon (14-17)", 14, 17),
        ("Dinner (17-20)", 17, 20),
        ("Evening (20-23)", 20, 23),
        ("Night (23-5)", 23, 29),  # wraps
    ]

    all_entries = []
    for entries in daily_carbs.values():
        all_entries.extend(entries)

    import numpy as np
    for label, start_hr, end_hr in buckets:
        if end_hr > 24:
            bucket_entries = [e for e in all_entries
                             if e["hour"] >= start_hr or e["hour"] < end_hr - 24]
        else:
            bucket_entries = [e for e in all_entries
                             if start_hr <= e["hour"] < end_hr]

        if bucket_entries:
            carb_vals = np.array([e["carbs"] for e in bucket_entries])
            print(f"  {label:<20} {len(bucket_entries):>6} {np.mean(carb_vals):>8.1f} "
                  f"{np.median(carb_vals):>10.1f} {np.std(carb_vals):>8.1f}")
        else:
            print(f"  {label:<20} {0:>6}")

    # --- Absorption times ---
    abs_times = [e["absorption_sec"] for e in all_entries if e.get("absorption_sec")]
    if abs_times:
        abs_hrs = np.array(abs_times) / 60  # absorptionTime is in minutes
        print(f"\nDeclared absorption times:")
        print(f"  Mean:   {np.mean(abs_hrs):.1f}h")
        print(f"  Median: {np.median(abs_hrs):.1f}h")
        print(f"  Range:  {np.min(abs_hrs):.1f}-{np.max(abs_hrs):.1f}h")

    # --- Individual days detail ---
    print(f"\nLast 7 days detail:")
    for date_str in sorted(daily_carbs.keys())[-7:]:
        entries = sorted(daily_carbs[date_str], key=lambda e: e["hour"])
        total = sum(e["carbs"] for e in entries)
        meals_str = ", ".join(
            f"{int(e['hour'])}:{int((e['hour']%1)*60):02d}={e['carbs']:.0f}g"
            for e in entries
        )
        print(f"  {date_str}: {total:3.0f}g total [{len(entries)} entries] — {meals_str}")

    return daily_carbs


def analyze_temp_targets(treatments: List[Dict]):
    """Analyze temp target entries to estimate exercise frequency."""
    temp_targets = [t for t in treatments
                    if t.get("eventType") in ("Temporary Target", "Temp Target")]

    if not temp_targets:
        print(f"\nNo temp target entries found.")
        print("  (Exercise may be declared via a different mechanism)")
        return

    print(f"\n{'='*70}")
    print(f"Temp Target Analysis (possible exercise indicators)")
    print(f"{'='*70}")
    print(f"Total temp target events: {len(temp_targets)}")

    # Group by date
    daily_tt = defaultdict(list)
    for t in temp_targets:
        created = t.get("created_at", "")
        try:
            dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            continue

        date_str = dt.strftime("%Y-%m-%d")
        daily_tt[date_str].append({
            "hour": dt.hour + dt.minute / 60.0,
            "duration": t.get("duration", 0),
            "targetTop": t.get("targetTop"),
            "targetBottom": t.get("targetBottom"),
            "reason": t.get("reason", ""),
            "notes": t.get("notes", ""),
        })

    days_with_tt = len(daily_tt)
    total_days = max(1, (max(datetime.fromisoformat(d) for d in daily_tt.keys())
                         - min(datetime.fromisoformat(d) for d in daily_tt.keys())).days + 1) \
        if daily_tt else 1

    print(f"Days with temp targets: {days_with_tt} out of ~{total_days} days")
    print(f"Frequency: ~{days_with_tt/total_days*7:.1f} per week")

    # Show recent entries
    print(f"\nRecent temp targets:")
    for date_str in sorted(daily_tt.keys())[-7:]:
        for tt in daily_tt[date_str]:
            target = f"{tt['targetBottom']}-{tt['targetTop']}" if tt['targetTop'] else "?"
            print(f"  {date_str} {int(tt['hour'])}:{int((tt['hour']%1)*60):02d} "
                  f"target={target} dur={tt['duration']}min "
                  f"reason={tt.get('reason', '')}")


def fetch_profile(
    base_url: str,
    token: Optional[str] = None,
) -> Dict:
    """Fetch the Nightscout profile (basal, CR, ISF, targets, DIA, timezone).

    Returns the default (most recent) profile store entry.
    """
    url = f"{base_url}/api/v1/profile.json"
    params = {}
    headers = {"Accept": "application/json"}
    if token:
        params["token"] = token

    resp = requests.get(url, headers=headers, params=params, timeout=30)
    resp.raise_for_status()
    profiles = resp.json()
    if not profiles:
        raise ValueError("No profiles found in Nightscout")

    # Return the most recent profile store entry
    store = profiles[0].get("store", {})
    # Get the default profile (usually the first/only key)
    default_name = profiles[0].get("defaultProfile", next(iter(store)))
    return store.get(default_name, next(iter(store.values())))


def fetch_boluses(
    base_url: str,
    start_date: datetime,
    end_date: datetime,
    token: Optional[str] = None,
    count: int = 50000,
) -> List[Dict]:
    """Fetch bolus and SMB treatments from Nightscout."""
    bolus_types = [
        "Bolus",
        "Correction Bolus",
        "Meal Bolus",
        "SMB",
    ]
    all_boluses = []
    for event_type in bolus_types:
        treatments = fetch_treatments(
            base_url, start_date, end_date, token=token,
            find_filter={"find[eventType]": event_type},
            count=count,
        )
        all_boluses.extend(treatments)

    # Also grab any treatment with an insulin field but no specific eventType match
    all_treatments = fetch_treatments(
        base_url, start_date, end_date, token=token, count=count,
    )
    seen_ids = {t.get("_id") for t in all_boluses}
    for t in all_treatments:
        if t.get("insulin") and t.get("insulin") > 0 and t.get("_id") not in seen_ids:
            all_boluses.append(t)

    return all_boluses


def fetch_temp_basals(
    base_url: str,
    start_date: datetime,
    end_date: datetime,
    token: Optional[str] = None,
    count: int = 50000,
) -> List[Dict]:
    """Fetch temp basal treatments from Nightscout."""
    return fetch_treatments(
        base_url, start_date, end_date, token=token,
        find_filter={"find[eventType]": "Temp Basal"},
        count=count,
    )


def fetch_exercise(
    base_url: str,
    start_date: datetime,
    end_date: datetime,
    token: Optional[str] = None,
    count: int = 10000,
) -> List[Dict]:
    """Fetch exercise treatments from Nightscout."""
    return fetch_treatments(
        base_url, start_date, end_date, token=token,
        find_filter={"find[eventType]": "Exercise"},
        count=count,
    )


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Query Nightscout for patient stats")
    parser.add_argument("--days", type=int, default=28,
                        help="Number of days to look back (default 28)")
    parser.add_argument("--url", type=str, default=NIGHTSCOUT_URL,
                        help="Nightscout URL")
    parser.add_argument("--token", type=str, default=None,
                        help="API access token (if required)")
    parser.add_argument("--secret", type=str, default=None,
                        help="API secret (will be SHA1 hashed)")
    parser.add_argument("--utc-offset", type=float, default=-5.0,
                        help="UTC offset in hours (default -5 for US Eastern)")
    args = parser.parse_args()

    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=args.days)

    print(f"Querying {args.url}")
    print(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

    # Fetch all treatments (carbs + temp targets)
    try:
        treatments = fetch_treatments(
            args.url, start_date, end_date,
            token=args.token, count=10000,
        )
        print(f"Fetched {len(treatments)} treatment entries")
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            print("Authentication required. Use --token YOUR_TOKEN")
            print("Create a token in Nightscout Admin Tools (hamburger menu -> Admin Tools)")
            return
        raise

    analyze_carbs(treatments, args.days, utc_offset_hours=args.utc_offset)
    analyze_temp_targets(treatments)


if __name__ == "__main__":
    main()
