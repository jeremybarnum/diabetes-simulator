"""
Generate synthetic BG trajectories for Loop testing.

Creates test scenarios with predictable BG patterns for validating
Loop algorithm implementations.
"""

from typing import List, Tuple, Optional
import math


class TrajectoryGenerator:
    """Generate synthetic glucose trajectories for testing."""

    @staticmethod
    def flat_baseline(
        bg_value: float,
        duration_hours: float,
        time_step_min: float = 5.0
    ) -> List[Tuple[float, float]]:
        """
        Generate flat/stable BG trajectory.

        Args:
            bg_value: BG value in mg/dL
            duration_hours: Duration in hours
            time_step_min: Time step in minutes

        Returns:
            List of (time_minutes, bg_value) tuples
        """
        num_points = int(duration_hours * 60 / time_step_min) + 1
        return [
            (i * time_step_min, bg_value)
            for i in range(num_points)
        ]

    @staticmethod
    def linear_rise(
        start_bg: float,
        end_bg: float,
        duration_hours: float,
        time_step_min: float = 5.0
    ) -> List[Tuple[float, float]]:
        """
        Generate linearly rising BG trajectory.

        Args:
            start_bg: Starting BG in mg/dL
            end_bg: Ending BG in mg/dL
            duration_hours: Duration in hours
            time_step_min: Time step in minutes

        Returns:
            List of (time_minutes, bg_value) tuples
        """
        num_points = int(duration_hours * 60 / time_step_min) + 1
        total_duration = duration_hours * 60

        trajectory = []
        for i in range(num_points):
            time = i * time_step_min
            # Linear interpolation
            fraction = time / total_duration
            bg = start_bg + (end_bg - start_bg) * fraction
            trajectory.append((time, bg))

        return trajectory

    @staticmethod
    def meal_response(
        baseline_bg: float,
        meal_time_min: float,
        meal_carbs: float,
        meal_bolus: float,
        duration_hours: float,
        isf: float = 50.0,
        carb_ratio: float = 10.0,
        time_step_min: float = 5.0
    ) -> List[Tuple[float, float]]:
        """
        Generate realistic meal response trajectory.

        Uses simplified physiological model:
        - Carbs raise BG following absorption curve
        - Insulin lowers BG following action curve

        Args:
            baseline_bg: Starting BG in mg/dL
            meal_time_min: Time of meal in minutes
            meal_carbs: Carb amount in grams
            meal_bolus: Insulin bolus in units
            duration_hours: Total duration in hours
            isf: Insulin sensitivity factor (mg/dL per unit)
            carb_ratio: Carb ratio (g per unit)
            time_step_min: Time step in minutes

        Returns:
            List of (time_minutes, bg_value) tuples
        """
        num_points = int(duration_hours * 60 / time_step_min) + 1

        trajectory = []
        for i in range(num_points):
            time_min = i * time_step_min

            if time_min < meal_time_min:
                # Before meal: baseline
                bg = baseline_bg
            else:
                # After meal: carb and insulin effects
                minutes_since_meal = time_min - meal_time_min

                # Carb effect: rises then falls (simplified absorption)
                carb_effect = TrajectoryGenerator._simple_carb_effect(
                    minutes_since_meal,
                    meal_carbs,
                    carb_ratio,
                    isf
                )

                # Insulin effect: gradually increases (simplified action)
                insulin_effect = TrajectoryGenerator._simple_insulin_effect(
                    minutes_since_meal,
                    meal_bolus,
                    isf
                )

                bg = baseline_bg + carb_effect + insulin_effect

                # Keep in physiological bounds
                bg = max(40, min(400, bg))

            trajectory.append((time_min, bg))

        return trajectory

    @staticmethod
    def _simple_carb_effect(
        minutes: float,
        carbs: float,
        carb_ratio: float,
        isf: float,
        absorption_time_min: float = 180.0
    ) -> float:
        """
        Simplified carb effect calculation.

        Uses piecewise linear absorption:
        - 10 min delay
        - Linear rise to peak at 45 min
        - Linear decay to zero at absorption_time

        Args:
            minutes: Minutes since carb entry
            carbs: Carb amount in grams
            carb_ratio: Carb ratio (g/U)
            isf: Insulin sensitivity factor (mg/dL per U)
            absorption_time_min: Total absorption time

        Returns:
            BG effect in mg/dL (positive = raising)
        """
        if minutes < 10:
            # Delay period
            return 0.0

        # Carb sensitivity factor (mg/dL per gram)
        csf = isf / carb_ratio

        # Maximum effect (if all carbs absorbed instantly)
        max_effect = carbs * csf

        if minutes < 45:
            # Rising phase: 10 min to 45 min
            fraction = (minutes - 10) / 35.0
            return max_effect * 0.4 * fraction  # Peak at 40% of max

        elif minutes < absorption_time_min:
            # Falling phase: 45 min to absorption_time
            remaining_time = absorption_time_min - 45
            elapsed_time = minutes - 45
            fraction = 1.0 - (elapsed_time / remaining_time)
            return max_effect * 0.4 * fraction

        else:
            # Fully absorbed
            return 0.0

    @staticmethod
    def _simple_insulin_effect(
        minutes: float,
        insulin_units: float,
        isf: float,
        dia_minutes: float = 360.0
    ) -> float:
        """
        Simplified insulin effect calculation.

        Uses exponential decay with peak at 55 minutes (Fiasp-like).

        Args:
            minutes: Minutes since insulin dose
            insulin_units: Insulin amount in units
            isf: Insulin sensitivity factor (mg/dL per U)
            dia_minutes: Duration of insulin action

        Returns:
            BG effect in mg/dL (negative = lowering)
        """
        if minutes < 0 or minutes > dia_minutes:
            return 0.0

        # Peak at 55 minutes (Fiasp)
        peak_time = 55.0

        # Use Walsh exponential curve approximation
        # Effect = Total * (1 - fraction_remaining)
        total_effect = -insulin_units * isf  # Negative = lowering

        # Simplified exponential: fraction absorbed
        if minutes <= peak_time:
            # Rising phase
            fraction = (minutes / peak_time) ** 2
        else:
            # Falling phase
            time_after_peak = minutes - peak_time
            remaining_time = dia_minutes - peak_time
            decay_rate = 2.0 / remaining_time  # Exponential decay
            fraction = 1.0 - math.exp(-decay_rate * time_after_peak)

        return total_effect * fraction


# Predefined test scenarios

def scenario_steady_state(duration_hours: float = 3.0) -> dict:
    """Steady state at 110 mg/dL."""
    return {
        "name": "steady_state",
        "trajectory": TrajectoryGenerator.flat_baseline(110.0, duration_hours),
        "meal_time": None,
        "meal_carbs": 0,
        "meal_bolus": 0,
        "description": "Flat baseline at 110 mg/dL, no meals or boluses"
    }


def scenario_meal_perfect_bolus(duration_hours: float = 3.0) -> dict:
    """Meal with perfectly matched bolus."""
    traj = TrajectoryGenerator.meal_response(
        baseline_bg=110.0,
        meal_time_min=30.0,
        meal_carbs=45.0,
        meal_bolus=4.5,  # Perfect: 45g / 10 g/U = 4.5U
        duration_hours=duration_hours
    )

    return {
        "name": "meal_perfect_bolus",
        "trajectory": traj,
        "meal_time": 30.0,
        "meal_carbs": 45.0,
        "meal_bolus": 4.5,
        "description": "45g meal with perfectly matched 4.5U bolus at t=30min"
    }


def scenario_meal_under_bolused(duration_hours: float = 3.0) -> dict:
    """Meal with insufficient bolus (BG will rise)."""
    traj = TrajectoryGenerator.meal_response(
        baseline_bg=110.0,
        meal_time_min=30.0,
        meal_carbs=45.0,
        meal_bolus=3.0,  # Under-bolused: should need 4.5U
        duration_hours=duration_hours
    )

    return {
        "name": "meal_under_bolused",
        "trajectory": traj,
        "meal_time": 30.0,
        "meal_carbs": 45.0,
        "meal_bolus": 3.0,
        "description": "45g meal with insufficient 3.0U bolus (should be 4.5U)"
    }


def scenario_meal_over_bolused(duration_hours: float = 3.0) -> dict:
    """Meal with excessive bolus (BG will drop)."""
    traj = TrajectoryGenerator.meal_response(
        baseline_bg=110.0,
        meal_time_min=30.0,
        meal_carbs=45.0,
        meal_bolus=6.0,  # Over-bolused: only need 4.5U
        duration_hours=duration_hours
    )

    return {
        "name": "meal_over_bolused",
        "trajectory": traj,
        "meal_time": 30.0,
        "meal_carbs": 45.0,
        "meal_bolus": 6.0,
        "description": "45g meal with excessive 6.0U bolus (only need 4.5U)"
    }


def print_trajectory(scenario: dict, show_first_n: int = 20):
    """Print trajectory for debugging."""
    print(f"\n=== {scenario['name']} ===")
    print(f"Description: {scenario['description']}")
    print(f"\nTime (min) | BG (mg/dL)")
    print("-" * 30)

    for time, bg in scenario['trajectory'][:show_first_n]:
        print(f"{time:10.0f} | {bg:10.1f}")

    if len(scenario['trajectory']) > show_first_n:
        print(f"... ({len(scenario['trajectory']) - show_first_n} more points)")


if __name__ == "__main__":
    print("=" * 60)
    print("SYNTHETIC TRAJECTORY GENERATOR")
    print("=" * 60)

    # Generate all test scenarios
    scenarios = [
        scenario_steady_state(),
        scenario_meal_perfect_bolus(),
        scenario_meal_under_bolused(),
        scenario_meal_over_bolused()
    ]

    # Print summaries
    for scenario in scenarios:
        print_trajectory(scenario, show_first_n=15)

    print("\n" + "=" * 60)
    print("✓ Generated 4 test scenarios")
    print("=" * 60)
    print("\nUsage:")
    print("  from loop_trajectory_generator import scenario_meal_perfect_bolus")
    print("  scenario = scenario_meal_perfect_bolus(duration_hours=3)")
    print("  trajectory = scenario['trajectory']")
