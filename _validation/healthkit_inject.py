#!/usr/bin/env python3
"""
Command-line wrapper for HealthKit Injector app running in iOS Simulator.
Allows injecting scenarios and clearing data from the command line.
"""

import json
import subprocess
import sys
import os
import time
from pathlib import Path
from datetime import datetime

# Configuration
SIMULATOR_UDID = "4A0939FF-02C0-4000-843E-EEAE8BC727CC"  # iPhone 17 Pro
BUNDLE_ID = "com.test.healthkitinjector"
APP_NAME = "HealthKitInjectorApp"

def run_simctl_command(args):
    """Run xcrun simctl command and return output."""
    cmd = ["xcrun", "simctl"] + args
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running simctl: {e.stderr}")
        sys.exit(1)

def get_app_container():
    """Get the app's data container path."""
    output = run_simctl_command(["get_app_container", SIMULATOR_UDID, BUNDLE_ID, "data"])
    return output.strip()

def launch_app():
    """Launch the HealthKit Injector app."""
    print(f"🚀 Launching {APP_NAME}...")
    output = run_simctl_command(["launch", SIMULATOR_UDID, BUNDLE_ID])
    pid = output.strip().split(":")[-1].strip()
    print(f"✅ App launched (PID: {pid})")
    return pid

def terminate_app():
    """Terminate the HealthKit Injector app."""
    try:
        run_simctl_command(["terminate", SIMULATOR_UDID, BUNDLE_ID])
        print("✅ App terminated")
    except:
        pass  # App might not be running

def write_scenario_file(scenario_data, filename="scenario.json"):
    """Write scenario JSON to app's Documents directory."""
    container = get_app_container()
    scenarios_dir = Path(container) / "Documents" / "scenarios"
    scenarios_dir.mkdir(parents=True, exist_ok=True)

    scenario_path = scenarios_dir / filename
    with open(scenario_path, 'w') as f:
        json.dump(scenario_data, f, indent=2)

    print(f"📝 Wrote scenario to: {scenario_path}")
    return scenario_path

def write_command_file(command):
    """Write a command file for the app to read."""
    container = get_app_container()
    commands_dir = Path(container) / "Documents" / "commands"
    commands_dir.mkdir(parents=True, exist_ok=True)

    command_path = commands_dir / f"command_{int(time.time())}.json"
    with open(command_path, 'w') as f:
        json.dump({"command": command, "timestamp": time.time()}, f)

    return command_path

def clear_healthkit_data():
    """Clear all HealthKit test data."""
    print("🗑️  Clearing all HealthKit data...")

    # Write command file
    write_command_file("clear_all")

    # Launch app to process command
    launch_app()

    print("✅ Clear command sent. Check the app for confirmation.")
    print("   (The app should show 'Successfully cleared all test data')")

def inject_scenario(scenario_file_or_data):
    """Inject a scenario into HealthKit."""
    # Load scenario data
    if isinstance(scenario_file_or_data, str):
        # It's a file path
        with open(scenario_file_or_data, 'r') as f:
            scenario_data = json.load(f)
        print(f"📂 Loaded scenario from: {scenario_file_or_data}")
    else:
        # It's already a dict
        scenario_data = scenario_file_or_data

    # Validate scenario
    if "name" not in scenario_data:
        print("❌ Error: Scenario must have a 'name' field")
        sys.exit(1)

    print(f"📊 Injecting scenario: {scenario_data['name']}")
    print(f"   Glucose samples: {len(scenario_data.get('glucoseSamples', []))}")
    print(f"   Carb entries: {len(scenario_data.get('carbEntries', []))}")
    print(f"   Insulin doses: {len(scenario_data.get('insulinDoses', []))}")

    # Write scenario to app's Documents
    scenario_filename = f"scenario_{int(time.time())}.json"
    write_scenario_file(scenario_data, scenario_filename)

    # Write command to inject it
    write_command_file(f"inject:{scenario_filename}")

    # Launch app to process
    launch_app()

    print("✅ Injection command sent. Check the app for confirmation.")

def create_example_scenario():
    """Create an example scenario for testing."""
    now = time.time()

    return {
        "name": "Example Meal Scenario",
        "glucoseSamples": [
            {"timestamp": now - 600, "value": 110.0},
            {"timestamp": now - 300, "value": 115.0},
            {"timestamp": now, "value": 120.0}
        ],
        "carbEntries": [
            {"timestamp": now - 300, "grams": 40.0, "absorptionHours": 3.0}
        ],
        "insulinDoses": [
            {"timestamp": now - 300, "units": 4.0}
        ]
    }

def inject_single_bg(value, timestamp=None):
    """Inject a single BG reading."""
    if timestamp is None:
        timestamp = time.time()

    print(f"💉 Injecting single BG reading: {value} mg/dL")

    scenario = {
        "name": f"Single BG: {value} mg/dL",
        "glucoseSamples": [
            {"timestamp": timestamp, "value": float(value)}
        ],
        "carbEntries": [],
        "insulinDoses": []
    }

    inject_scenario(scenario)

def inject_single_carb(grams, absorption_hours=3.0, timestamp=None):
    """Inject a single carb entry."""
    if timestamp is None:
        timestamp = time.time()

    print(f"🍎 Injecting single carb entry: {grams}g ({absorption_hours}h absorption)")

    scenario = {
        "name": f"Single Carb: {grams}g",
        "glucoseSamples": [],
        "carbEntries": [
            {"timestamp": timestamp, "grams": float(grams), "absorptionHours": float(absorption_hours)}
        ],
        "insulinDoses": []
    }

    inject_scenario(scenario)

def inject_single_insulin(units, timestamp=None):
    """Inject a single insulin dose."""
    if timestamp is None:
        timestamp = time.time()

    print(f"💊 Injecting single insulin dose: {units}U")

    scenario = {
        "name": f"Single Insulin: {units}U",
        "glucoseSamples": [],
        "carbEntries": [],
        "insulinDoses": [
            {"timestamp": timestamp, "units": float(units)}
        ]
    }

    inject_scenario(scenario)

def inject_multiple_bgs(bg_pairs):
    """Inject multiple BG readings from an array of (offset, value) pairs."""
    if len(bg_pairs) % 2 != 0:
        print("❌ Error: BG readings must be in pairs of (offset, value)")
        sys.exit(1)

    glucose_samples = []
    now = time.time()

    # Parse pairs
    for i in range(0, len(bg_pairs), 2):
        try:
            offset_minutes = float(bg_pairs[i])
            bg_value = float(bg_pairs[i + 1])

            if bg_value < 40 or bg_value > 400:
                print(f"⚠️  Warning: BG value {bg_value} seems unusual (40-400 mg/dL is typical range)")

            timestamp = now + (offset_minutes * 60)
            glucose_samples.append({
                "timestamp": timestamp,
                "value": bg_value
            })

            if offset_minutes < 0:
                time_desc = f"{abs(offset_minutes):.0f} min ago"
            elif offset_minutes > 0:
                time_desc = f"{offset_minutes:.0f} min from now"
            else:
                time_desc = "now"

            print(f"  📍 {bg_value} mg/dL at {time_desc}")

        except (ValueError, IndexError) as e:
            print(f"❌ Error parsing BG pair at position {i}: {e}")
            sys.exit(1)

    print(f"\n💉 Injecting {len(glucose_samples)} BG readings...")

    # Create scenario
    scenario = {
        "name": f"Multiple BG Readings ({len(glucose_samples)} samples)",
        "glucoseSamples": glucose_samples,
        "carbEntries": [],
        "insulinDoses": []
    }

    inject_scenario(scenario)

def print_usage():
    """Print usage information."""
    print("""
HealthKit Injector CLI - Control HealthKit injection from command line

Usage:
    python healthkit_inject.py clear                      Clear all HealthKit data
    python healthkit_inject.py bg <value>                 Add BG reading now (mg/dL)
    python healthkit_inject.py bg <offset> <value>        Add BG reading at offset (minutes from now)
    python healthkit_inject.py bgs <offset1> <value1> <offset2> <value2> ...
                                                          Add multiple BG readings (space-delimited pairs)
    python healthkit_inject.py carb <grams> [hours]       Add single carb entry (default 3h)
    python healthkit_inject.py carb <offset> <grams> [hours]  Add carb at offset
    python healthkit_inject.py insulin <units>            Add single insulin dose
    python healthkit_inject.py insulin <offset> <units>   Add insulin at offset
    python healthkit_inject.py inject <file>              Inject scenario from JSON file
    python healthkit_inject.py example                    Create and inject example scenario
    python healthkit_inject.py launch                     Just launch the app
    python healthkit_inject.py help                       Show this help message

Scenario JSON Format:
{
  "name": "Scenario Name",
  "glucoseSamples": [
    {"timestamp": <unix_timestamp>, "value": <mg/dL>}
  ],
  "carbEntries": [
    {"timestamp": <unix_timestamp>, "grams": <g>, "absorptionHours": <hours>}
  ],
  "insulinDoses": [
    {"timestamp": <unix_timestamp>, "units": <U>}
  ]
}

Examples:
    python healthkit_inject.py clear
    python healthkit_inject.py bg 120                    # BG 120 mg/dL now
    python healthkit_inject.py bg -5 115                 # BG 115 mg/dL 5 minutes ago
    python healthkit_inject.py bg 10 130                 # BG 130 mg/dL 10 minutes from now
    python healthkit_inject.py bgs -10 100 -5 110 0 120  # Multiple BGs: 100@-10min, 110@-5min, 120@now
    python healthkit_inject.py carb 40 3                 # 40g carbs now, 3h absorption
    python healthkit_inject.py carb -10 40 3             # 40g carbs 10 minutes ago
    python healthkit_inject.py insulin 4.5               # 4.5U insulin now
    python healthkit_inject.py insulin -5 4.5            # 4.5U insulin 5 minutes ago
    python healthkit_inject.py inject scenario.json
    python healthkit_inject.py example
""")

def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == "help" or command == "-h" or command == "--help":
        print_usage()

    elif command == "clear":
        clear_healthkit_data()

    elif command == "bgs":
        if len(sys.argv) < 4:
            print("❌ Error: Please provide BG readings as space-delimited pairs")
            print("Usage: python healthkit_inject.py bgs <offset1> <value1> <offset2> <value2> ...")
            print("Example: python healthkit_inject.py bgs -10 100 -5 110 0 120")
            print("         (100 mg/dL 10 min ago, 110 mg/dL 5 min ago, 120 mg/dL now)")
            sys.exit(1)

        bg_pairs = sys.argv[2:]
        inject_multiple_bgs(bg_pairs)

    elif command == "bg":
        if len(sys.argv) < 3:
            print("❌ Error: Please provide a BG value in mg/dL")
            print("Usage: python healthkit_inject.py bg <value>")
            print("       python healthkit_inject.py bg <offset_minutes> <value>")
            print("Example: python healthkit_inject.py bg 120")
            print("         python healthkit_inject.py bg -5 115  (5 minutes ago)")
            sys.exit(1)

        try:
            # Check if we have 2 or 3 arguments (bg, value) or (bg, offset, value)
            if len(sys.argv) == 3:
                # Single argument: bg <value>
                bg_value = float(sys.argv[2])
                timestamp = None
                time_desc = "now"
            else:
                # Two arguments: bg <offset> <value>
                offset_minutes = float(sys.argv[2])
                bg_value = float(sys.argv[3])
                timestamp = time.time() + (offset_minutes * 60)

                if offset_minutes < 0:
                    time_desc = f"{abs(offset_minutes):.0f} minutes ago"
                elif offset_minutes > 0:
                    time_desc = f"{offset_minutes:.0f} minutes from now"
                else:
                    time_desc = "now"

            if bg_value < 40 or bg_value > 400:
                print("⚠️  Warning: BG value seems unusual (40-400 mg/dL is typical range)")

            print(f"📅 Time: {time_desc}")
            inject_single_bg(bg_value, timestamp)
        except ValueError as e:
            print(f"❌ Error: Invalid values: {e}")
            sys.exit(1)

    elif command == "carb":
        if len(sys.argv) < 3:
            print("❌ Error: Please provide carb amount in grams")
            print("Usage: python healthkit_inject.py carb <grams> [hours]")
            print("       python healthkit_inject.py carb <offset_minutes> <grams> [hours]")
            print("Example: python healthkit_inject.py carb 40 3")
            print("         python healthkit_inject.py carb -10 40 3  (10 minutes ago)")
            sys.exit(1)

        try:
            # Parse arguments: carb <grams> [hours] or carb <offset> <grams> [hours]
            # Check if first arg could be an offset (can be negative or a time in the past/future)
            first_val = float(sys.argv[2])

            # If we have 4+ args, definitely <offset> <grams> [hours]
            # If we have 3 args and first val is negative, likely <offset> <grams>
            # Otherwise, <grams> [hours]
            if len(sys.argv) >= 4 and (len(sys.argv) >= 5 or first_val < 0 or first_val > 300):
                # Format: carb <offset> <grams> [hours]
                offset_minutes = first_val
                carb_grams = float(sys.argv[3])
                absorption_hours = float(sys.argv[4]) if len(sys.argv) > 4 else 3.0
                timestamp = time.time() + (offset_minutes * 60)

                if offset_minutes < 0:
                    time_desc = f"{abs(offset_minutes):.0f} minutes ago"
                elif offset_minutes > 0:
                    time_desc = f"{offset_minutes:.0f} minutes from now"
                else:
                    time_desc = "now"
                print(f"📅 Time: {time_desc}")
            else:
                # Format: carb <grams> [hours]
                carb_grams = first_val
                absorption_hours = float(sys.argv[3]) if len(sys.argv) > 3 else 3.0
                timestamp = None

            if carb_grams < 0 or carb_grams > 300:
                print("⚠️  Warning: Carb amount seems unusual (0-300g is typical range)")
            if absorption_hours < 0.5 or absorption_hours > 12:
                print("⚠️  Warning: Absorption time seems unusual (0.5-12h is typical range)")

            inject_single_carb(carb_grams, absorption_hours, timestamp)
        except ValueError as e:
            print(f"❌ Error: Invalid values: {e}")
            sys.exit(1)

    elif command == "insulin":
        if len(sys.argv) < 3:
            print("❌ Error: Please provide insulin amount in units")
            print("Usage: python healthkit_inject.py insulin <units>")
            print("       python healthkit_inject.py insulin <offset_minutes> <units>")
            print("Example: python healthkit_inject.py insulin 4.5")
            print("         python healthkit_inject.py insulin -5 4.5  (5 minutes ago)")
            sys.exit(1)

        try:
            # Check if we have 2 or 3 arguments (insulin, units) or (insulin, offset, units)
            if len(sys.argv) == 3:
                # Single argument: insulin <units>
                insulin_units = float(sys.argv[2])
                timestamp = None
                time_desc = "now"
            else:
                # Two arguments: insulin <offset> <units>
                offset_minutes = float(sys.argv[2])
                insulin_units = float(sys.argv[3])
                timestamp = time.time() + (offset_minutes * 60)

                if offset_minutes < 0:
                    time_desc = f"{abs(offset_minutes):.0f} minutes ago"
                elif offset_minutes > 0:
                    time_desc = f"{offset_minutes:.0f} minutes from now"
                else:
                    time_desc = "now"
                print(f"📅 Time: {time_desc}")

            if insulin_units < 0 or insulin_units > 50:
                print("⚠️  Warning: Insulin amount seems unusual (0-50U is typical range)")
            inject_single_insulin(insulin_units, timestamp)
        except ValueError as e:
            print(f"❌ Error: Invalid values: {e}")
            sys.exit(1)

    elif command == "inject":
        if len(sys.argv) < 3:
            print("❌ Error: Please provide a scenario file")
            print("Usage: python healthkit_inject.py inject <scenario.json>")
            sys.exit(1)

        scenario_file = sys.argv[2]
        if not os.path.exists(scenario_file):
            print(f"❌ Error: Scenario file not found: {scenario_file}")
            sys.exit(1)

        inject_scenario(scenario_file)

    elif command == "example":
        print("📝 Creating example scenario...")
        scenario = create_example_scenario()
        inject_scenario(scenario)

    elif command == "launch":
        launch_app()

    else:
        print(f"❌ Unknown command: {command}")
        print_usage()
        sys.exit(1)

if __name__ == "__main__":
    main()
