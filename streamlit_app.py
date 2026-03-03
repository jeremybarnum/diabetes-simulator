"""
Diabetes Algorithm Simulator — Streamlit Web App

Monte Carlo comparison of Loop and Trio insulin dosing algorithms.
Full UI for editing patient model and algorithm settings.
"""

import json
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from simulation import PatientProfile, MealSpec, ExerciseSpec, SimulationRun, SimulationRunResult
from monte_carlo import compute_metrics, GlycemicMetrics

# ─── Page config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Diabetes Algorithm Simulator",
    page_icon="📊",
    layout="wide",
)

st.title("Diabetes Algorithm Simulator")
st.markdown("Monte Carlo comparison of **Loop** and **Trio** insulin dosing algorithms")

# ─── Profile Loading ─────────────────────────────────────────────────────────

profile_dir = Path(__file__).parent / "patient_profiles"
profile_files = sorted(profile_dir.glob("*.json"))
profile_names = {p.stem.replace("_", " ").title(): str(p) for p in profile_files}


def _minutes_to_clock(minutes_from_7am: int) -> str:
    """Convert minutes from 7am to clock time string."""
    total = 7 * 60 + minutes_from_7am
    h = (total // 60) % 24
    m = total % 60
    return f"{h:02d}:{m:02d}"


def _clock_to_minutes(clock_str: str) -> int:
    """Convert clock time HH:MM to minutes from 7am."""
    parts = clock_str.split(":")
    h, m = int(parts[0]), int(parts[1])
    total = h * 60 + m
    return total - 7 * 60


def save_profile_to_json(file_path: str):
    """Save the current session state to a patient profile JSON file."""
    profile = build_profile_from_state()
    data = {
        "meals": [
            {"time_of_day_minutes": m.time_of_day_minutes,
             "carbs_mean": m.carbs_mean, "carbs_sd": m.carbs_sd,
             "absorption_hrs": m.absorption_hrs}
            for m in profile.meals
        ],
        "undeclared_meals": [
            {"time_of_day_minutes": m.time_of_day_minutes,
             "carbs_mean": m.carbs_mean, "carbs_sd": m.carbs_sd,
             "absorption_hrs": m.absorption_hrs}
            for m in profile.undeclared_meals
        ],
        "carb_count_sigma": profile.carb_count_sigma,
        "carb_count_bias": profile.carb_count_bias,
        "absorption_sigma": profile.absorption_sigma,
        "undeclared_meal_prob": profile.undeclared_meal_prob,
        "sensitivity_sigma": profile.sensitivity_sigma,
        "exercises_per_week": profile.exercises_per_week,
        "starting_bg": profile.starting_bg,
    }
    if profile.exercise_spec:
        ex = profile.exercise_spec
        data["exercise_spec"] = {
            "time_of_day_minutes": ex.time_of_day_minutes,
            "declared_scalar": ex.declared_scalar,
            "declared_duration_hrs": ex.declared_duration_hrs,
            "actual_scalar_mean": ex.actual_scalar_mean,
            "actual_scalar_sigma": ex.actual_scalar_sigma,
            "actual_duration_hrs_mean": ex.actual_duration_hrs_mean,
            "actual_duration_hrs_sigma": ex.actual_duration_hrs_sigma,
        }
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)


def load_profile_to_state(profile_path: str):
    """Load a patient profile JSON and populate session state.

    Sets widget keys directly so widgets pick up values via their key= param.
    """
    profile = PatientProfile.from_json(profile_path)
    settings = profile.get_settings()

    # Meals as DataFrames for data_editor
    st.session_state["meals_df"] = pd.DataFrame([
        {
            "Time": _minutes_to_clock(m.time_of_day_minutes),
            "Avg Carbs (g)": float(m.carbs_mean),
            "SD (g)": float(m.carbs_sd),
            "Absorption (hrs)": float(m.absorption_hrs),
        }
        for m in profile.meals
    ]) if profile.meals else pd.DataFrame(
        columns=["Time", "Avg Carbs (g)", "SD (g)", "Absorption (hrs)"]
    )

    st.session_state["undeclared_meals_df"] = pd.DataFrame([
        {
            "Time": _minutes_to_clock(m.time_of_day_minutes),
            "Avg Carbs (g)": float(m.carbs_mean),
            "SD (g)": float(m.carbs_sd),
            "Absorption (hrs)": float(m.absorption_hrs),
        }
        for m in profile.undeclared_meals
    ]) if profile.undeclared_meals else pd.DataFrame(
        columns=["Time", "Avg Carbs (g)", "SD (g)", "Absorption (hrs)"]
    )

    # Patient model sliders — keys match the widget key= params
    st.session_state["carb_count_sigma_sl"] = float(profile.carb_count_sigma)
    st.session_state["carb_count_bias_sl"] = float(profile.carb_count_bias)
    st.session_state["absorption_sigma_sl"] = float(profile.absorption_sigma)
    st.session_state["undeclared_meal_prob_sl"] = float(profile.undeclared_meal_prob)
    st.session_state["sensitivity_sigma_sl"] = float(profile.sensitivity_sigma)
    st.session_state["starting_bg_sl"] = int(profile.starting_bg)

    # Exercise
    st.session_state["exercises_per_week_sl"] = float(profile.exercises_per_week)
    if profile.exercise_spec:
        ex = profile.exercise_spec
        st.session_state["ex_time_in"] = _minutes_to_clock(ex.time_of_day_minutes)
        st.session_state["ex_declared_scalar_sl"] = float(ex.declared_scalar)
        st.session_state["ex_declared_duration_sl"] = float(ex.declared_duration_hrs)
        st.session_state["ex_actual_scalar_mean_sl"] = float(ex.actual_scalar_mean)
        st.session_state["ex_actual_scalar_sigma_sl"] = float(ex.actual_scalar_sigma)
        st.session_state["ex_actual_duration_mean_sl"] = float(ex.actual_duration_hrs_mean)
        st.session_state["ex_actual_duration_sigma_sl"] = float(ex.actual_duration_hrs_sigma)
    else:
        st.session_state["ex_time_in"] = "17:00"
        st.session_state["ex_declared_scalar_sl"] = 0.5
        st.session_state["ex_declared_duration_sl"] = 4.0
        st.session_state["ex_actual_scalar_mean_sl"] = 0.5
        st.session_state["ex_actual_scalar_sigma_sl"] = 0.1
        st.session_state["ex_actual_duration_mean_sl"] = 7.0
        st.session_state["ex_actual_duration_sigma_sl"] = 0.15

    # Algorithm settings — keys match widget key= params
    st.session_state["isf_in"] = float(settings.get("insulin_sensitivity_factor", 100.0))
    st.session_state["carb_ratio_in"] = float(settings.get("carb_ratio", 10.0))
    st.session_state["basal_rate_in"] = float(settings.get("basal_rate", 0.45))
    st.session_state["dia_in"] = float(settings.get("duration_of_insulin_action", 6.0))
    st.session_state["target_bg_in"] = float(settings.get("target", 100.0))
    st.session_state["suspend_threshold_in"] = float(settings.get("suspend_threshold", 80.0))
    st.session_state["max_basal_rate_in"] = float(settings.get("max_basal_rate", 2.8))
    st.session_state["max_bolus_in"] = float(settings.get("max_bolus", 3.0))
    st.session_state["enable_irc_cb"] = bool(settings.get("enable_irc", True))
    st.session_state["enable_momentum_cb"] = bool(settings.get("enable_momentum", True))
    st.session_state["enable_dca_cb"] = bool(settings.get("enable_dca", True))

    # Trio autosens settings
    st.session_state["autosens_enabled_cb"] = True
    st.session_state["use_new_formula_cb"] = False
    st.session_state["sigmoid_isf_cb"] = False
    st.session_state["adj_factor_sigmoid_sl"] = 0.5

    st.session_state["loaded_profile"] = profile_path


def build_profile_from_state() -> PatientProfile:
    """Construct a PatientProfile from current session state (widget keys)."""
    # Parse meals from data_editor results
    meals = []
    meals_df = st.session_state.get("meals_df")
    if meals_df is not None and len(meals_df) > 0:
        for _, row in meals_df.iterrows():
            try:
                t_min = _clock_to_minutes(str(row["Time"]))
                meals.append(MealSpec(
                    time_of_day_minutes=t_min,
                    carbs_mean=float(row["Avg Carbs (g)"]),
                    carbs_sd=float(row["SD (g)"]),
                    absorption_hrs=float(row["Absorption (hrs)"]),
                ))
            except (ValueError, KeyError):
                continue

    undeclared_meals = []
    und_df = st.session_state.get("undeclared_meals_df")
    if und_df is not None and len(und_df) > 0:
        for _, row in und_df.iterrows():
            try:
                t_min = _clock_to_minutes(str(row["Time"]))
                undeclared_meals.append(MealSpec(
                    time_of_day_minutes=t_min,
                    carbs_mean=float(row["Avg Carbs (g)"]),
                    carbs_sd=float(row["SD (g)"]),
                    absorption_hrs=float(row["Absorption (hrs)"]),
                ))
            except (ValueError, KeyError):
                continue

    # Exercise
    exercise_spec = None
    epw = st.session_state.get("exercises_per_week_sl", 0.0)
    if epw > 0:
        exercise_spec = ExerciseSpec(
            time_of_day_minutes=_clock_to_minutes(
                st.session_state.get("ex_time_in", "17:00")),
            declared_scalar=st.session_state.get("ex_declared_scalar_sl", 0.5),
            declared_duration_hrs=st.session_state.get("ex_declared_duration_sl", 4.0),
            actual_scalar_mean=st.session_state.get("ex_actual_scalar_mean_sl", 0.5),
            actual_scalar_sigma=st.session_state.get("ex_actual_scalar_sigma_sl", 0.1),
            actual_duration_hrs_mean=st.session_state.get("ex_actual_duration_mean_sl", 7.0),
            actual_duration_hrs_sigma=st.session_state.get("ex_actual_duration_sigma_sl", 0.15),
        )

    # Build algorithm settings dict
    algo_settings = {
        "insulin_sensitivity_factor": st.session_state.get("isf_in", 100.0),
        "carb_ratio": st.session_state.get("carb_ratio_in", 10.0),
        "basal_rate": st.session_state.get("basal_rate_in", 0.45),
        "duration_of_insulin_action": st.session_state.get("dia_in", 6.0),
        "target": st.session_state.get("target_bg_in", 100.0),
        "suspend_threshold": st.session_state.get("suspend_threshold_in", 80.0),
        "max_basal_rate": st.session_state.get("max_basal_rate_in", 2.8),
        "max_bolus": st.session_state.get("max_bolus_in", 3.0),
        "insulin_type": "fiasp",
        "enable_irc": st.session_state.get("enable_irc_cb", True),
        "enable_momentum": st.session_state.get("enable_momentum_cb", True),
        "enable_dca": st.session_state.get("enable_dca_cb", True),
    }

    return PatientProfile(
        meals=meals,
        carb_count_sigma=st.session_state.get("carb_count_sigma_sl", 0.15),
        carb_count_bias=st.session_state.get("carb_count_bias_sl", 0.0),
        absorption_sigma=st.session_state.get("absorption_sigma_sl", 0.15),
        undeclared_meal_prob=st.session_state.get("undeclared_meal_prob_sl", 0.0),
        undeclared_meals=undeclared_meals,
        sensitivity_sigma=st.session_state.get("sensitivity_sigma_sl", 0.15),
        exercises_per_week=epw,
        exercise_spec=exercise_spec,
        starting_bg=float(st.session_state.get("starting_bg_sl", 120)),
        algorithm_settings=algo_settings,
    )


# ─── Sidebar controls ───────────────────────────────────────────────────────

st.sidebar.header("Simulation Settings")

# Profile loader
selected_profile_name = st.sidebar.selectbox(
    "Load Profile",
    list(profile_names.keys()),
    index=list(profile_names.keys()).index("Real Patient")
    if "Real Patient" in profile_names else 0,
    help="Loading a profile resets all Patient Model and Algorithm settings.",
)
profile_path = profile_names[selected_profile_name]

# Load profile into session state on first run or profile change
if st.session_state.get("loaded_profile") != profile_path:
    load_profile_to_state(profile_path)

# Algorithms
algo_options = {
    "Loop AB40 (Auto-Bolus)": "loop_ab40",
    "Trio (oref1 SMB)": "trio",
    "Loop Temp Basal": "loop_tb",
}
selected_algos = st.sidebar.multiselect(
    "Algorithms",
    list(algo_options.keys()),
    default=["Loop AB40 (Auto-Bolus)", "Trio (oref1 SMB)"],
)
algorithms = [algo_options[name] for name in selected_algos]

# Simulation parameters
n_paths = st.sidebar.slider("Number of paths", min_value=5, max_value=100, value=20, step=5)
n_days = st.sidebar.slider("Days per path", min_value=1, max_value=14, value=1, step=1)
seed = st.sidebar.number_input("Random seed", value=42, min_value=0, max_value=99999)

run_button = st.sidebar.button("Run Simulation", type="primary", use_container_width=True)


# ─── Helper functions ────────────────────────────────────────────────────────

def run_paths(profile, algorithm_name, n_paths, n_days, seed):
    """Run n_paths simulations, return (traces_by_day, list of metrics).

    traces_by_day: list of lists. Each path produces n_days day-traces.
    Each day-trace is [(hours_in_day, bg), ...].
    """
    all_day_traces = []
    all_metrics = []
    for i in range(n_paths):
        path_seed = seed + i * 1000 + hash(algorithm_name) % (2**31)
        rng = np.random.RandomState(path_seed)
        sim = SimulationRun(profile=profile, algorithm_name=algorithm_name,
                            n_days=n_days, rng=rng)
        result = sim.run()
        metrics = compute_metrics(result)
        all_metrics.append(metrics)

        path_days = []
        for day in result.days:
            day_offset_min = day.day_index * 1440
            trace = [((t_rel - day_offset_min) / 60, bg) for t_rel, bg in day.bg_trace]
            path_days.append(trace)
        all_day_traces.append(path_days)
    return all_day_traces, all_metrics


def plot_spaghetti(traces_by_algo, algo_colors, n_paths, n_days):
    """Create spaghetti plot. Each day of each path overlaid on 0-24h axis."""
    fig, ax = plt.subplots(figsize=(14, 7))

    for algo_name, all_day_traces in traces_by_algo.items():
        color = algo_colors[algo_name]
        flat_traces = []
        for path_days in all_day_traces:
            for day_trace in path_days:
                flat_traces.append(day_trace)
                hours = [h for h, _ in day_trace]
                bgs = [bg for _, bg in day_trace]
                ax.plot(hours, bgs, color=color, alpha=0.08, linewidth=0.6)

        # Median line
        if flat_traces:
            common_hours = np.linspace(0, 24, 289)
            bg_matrix = []
            for trace in flat_traces:
                if len(trace) >= 2:
                    th = np.array([h for h, _ in trace])
                    tb = np.array([bg for _, bg in trace])
                    interp = np.interp(common_hours, th, tb)
                    bg_matrix.append(interp)
            if bg_matrix:
                medians = np.median(np.array(bg_matrix), axis=0)
                label = algo_name.replace("_", " ").upper()
                suffix = f" ({len(flat_traces)} day-traces)"
                ax.plot(common_hours, medians, color=color, linewidth=2.5,
                        label=f"{label} median{suffix}")

    # Target range
    ax.axhspan(70, 180, color="#22c55e", alpha=0.08, zorder=0)
    ax.axhline(70, color="#dc2626", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.axhline(180, color="#f59e0b", linewidth=0.8, linestyle="--", alpha=0.5)

    # X axis
    ax.set_xlim(0, 24)
    hour_ticks = list(range(0, 25, 2))
    hour_labels = []
    for h in hour_ticks:
        clock_hr = (7 + h) % 24
        ampm = "am" if clock_hr < 12 else "pm"
        display = clock_hr if clock_hr <= 12 else clock_hr - 12
        if display == 0:
            display = 12
        hour_labels.append(f"{display}{ampm}")
    ax.set_xticks(hour_ticks)
    ax.set_xticklabels(hour_labels)

    ax.set_xlabel("Time of Day")
    ax.set_ylabel("Blood Glucose (mg/dL)")
    days_label = f" x {n_days} days" if n_days > 1 else ""
    ax.set_title(f"24h BG Traces — {n_paths} paths{days_label}")
    ax.legend(loc="upper left")
    ax.set_ylim(40, 300)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    return fig


def compute_summary_stats(all_metrics):
    """Compute summary statistics from list of GlycemicMetrics."""
    if not all_metrics:
        return {}
    return {
        "Mean BG (mg/dL)": f"{np.mean([m.mean_bg for m in all_metrics]):.1f}",
        "TIR 70-180 (%)": f"{np.mean([m.time_in_range for m in all_metrics]):.1f}",
        "Time <70 (%)": f"{np.mean([m.time_below_70 for m in all_metrics]):.2f}",
        "Time <54 (%)": f"{np.mean([m.time_below_54 for m in all_metrics]):.3f}",
        "Time >180 (%)": f"{np.mean([m.time_above_180 for m in all_metrics]):.1f}",
        "Time >250 (%)": f"{np.mean([m.time_above_250 for m in all_metrics]):.2f}",
        "CV (%)": f"{np.mean([m.cv for m in all_metrics]):.1f}",
        "GMI (%)": f"{np.mean([m.gmi for m in all_metrics]):.2f}",
        "Hypo events/path": f"{np.mean([m.hypo_events for m in all_metrics]):.2f}",
    }


# ─── Main area — Tabs ────────────────────────────────────────────────────────

ALGO_COLORS = {
    "loop_ab40": "#2563eb",
    "loop_tb": "#7c3aed",
    "loop_ab_gbpa": "#0891b2",
    "trio": "#dc2626",
}

tab_results, tab_patient, tab_algo = st.tabs([
    "Results", "Patient Model", "Algorithm Settings",
])

# ─── Tab 2: Patient Model ────────────────────────────────────────────────────

with tab_patient:
    st.caption("These are the active settings used when you click **Run Simulation**. "
               "Use **Load Profile** in the sidebar to reset from a preset.")
    st.subheader("Meal Schedule")
    st.caption("Times are in 24h clock format (HH:MM). Day starts at 07:00.")

    st.markdown("**Declared Meals**")
    edited_meals = st.data_editor(
        st.session_state["meals_df"],
        num_rows="dynamic",
        use_container_width=True,
        key="meals_editor",
        column_config={
            "Time": st.column_config.TextColumn(
                "Time (HH:MM)", help="24h clock time"),
            "Avg Carbs (g)": st.column_config.NumberColumn(
                "Avg Carbs (g)", min_value=0, max_value=200, step=1),
            "SD (g)": st.column_config.NumberColumn(
                "SD (g)", min_value=0, max_value=50, step=1),
            "Absorption (hrs)": st.column_config.NumberColumn(
                "Absorption (hrs)", min_value=0.5, max_value=8.0, step=0.5),
        },
    )
    st.session_state["meals_df"] = edited_meals

    st.markdown("**Undeclared Meals** (always eaten, never bolused)")
    edited_undeclared = st.data_editor(
        st.session_state["undeclared_meals_df"],
        num_rows="dynamic",
        use_container_width=True,
        key="undeclared_meals_editor",
        column_config={
            "Time": st.column_config.TextColumn(
                "Time (HH:MM)", help="24h clock time"),
            "Avg Carbs (g)": st.column_config.NumberColumn(
                "Avg Carbs (g)", min_value=0, max_value=200, step=1),
            "SD (g)": st.column_config.NumberColumn(
                "SD (g)", min_value=0, max_value=50, step=1),
            "Absorption (hrs)": st.column_config.NumberColumn(
                "Absorption (hrs)", min_value=0.5, max_value=8.0, step=0.5),
        },
    )
    st.session_state["undeclared_meals_df"] = edited_undeclared

    st.divider()

    st.subheader("Carb Counting Ability")
    col1, col2 = st.columns(2)
    with col1:
        st.slider(
            "Carb counting error (sigma)",
            min_value=0.0, max_value=0.50, step=0.05,
            help="Lognormal sigma for random carb estimation error",
            key="carb_count_sigma_sl",
        )
        st.slider(
            "Absorption estimate error (sigma)",
            min_value=0.0, max_value=0.30, step=0.05,
            help="Lognormal sigma for absorption time estimation error",
            key="absorption_sigma_sl",
        )
    with col2:
        st.slider(
            "Systematic under-declaration bias",
            min_value=0.0, max_value=0.30, step=0.05,
            help="Positive = patient systematically under-declares carbs",
            key="carb_count_bias_sl",
        )
        st.slider(
            "Probability meal goes undeclared",
            min_value=0.0, max_value=0.50, step=0.05,
            help="Chance that any given meal is eaten but not bolused",
            key="undeclared_meal_prob_sl",
        )

    st.divider()

    st.subheader("Sensitivity Variation")
    st.slider(
        "Daily sensitivity variation (sigma)",
        min_value=0.0, max_value=0.40, step=0.05,
        help="Lognormal sigma for day-to-day insulin sensitivity variation",
        key="sensitivity_sigma_sl",
    )

    st.divider()

    st.subheader("Exercise")
    st.slider(
        "Exercise sessions per week",
        min_value=0.0, max_value=7.0, step=0.5,
        key="exercises_per_week_sl",
    )

    if st.session_state["exercises_per_week_sl"] > 0:
        st.markdown("**Exercise Parameters**")
        ex_col1, ex_col2, ex_col3 = st.columns(3)
        with ex_col1:
            st.text_input(
                "Time of day (HH:MM)",
                key="ex_time_in",
            )
            st.slider(
                "Declared sensitivity scalar",
                min_value=0.1, max_value=1.0, step=0.05,
                help="What patient tells pump (lower = more sensitive)",
                key="ex_declared_scalar_sl",
            )
            st.slider(
                "Declared duration (hrs)",
                min_value=1.0, max_value=8.0, step=0.5,
                key="ex_declared_duration_sl",
            )
        with ex_col2:
            st.slider(
                "Actual scalar mean",
                min_value=0.1, max_value=1.0, step=0.05,
                help="True mean sensitivity multiplier during exercise",
                key="ex_actual_scalar_mean_sl",
            )
            st.slider(
                "Actual scalar sigma",
                min_value=0.0, max_value=0.30, step=0.05,
                key="ex_actual_scalar_sigma_sl",
            )
        with ex_col3:
            st.slider(
                "Actual duration mean (hrs)",
                min_value=1.0, max_value=12.0, step=0.5,
                help="True mean duration of exercise effect",
                key="ex_actual_duration_mean_sl",
            )
            st.slider(
                "Actual duration sigma",
                min_value=0.0, max_value=0.30, step=0.05,
                key="ex_actual_duration_sigma_sl",
            )

    st.divider()

    st.subheader("Starting BG")
    st.slider(
        "Starting blood glucose (mg/dL)",
        min_value=70, max_value=250, step=5,
        key="starting_bg_sl",
    )

    st.divider()

    # ─── Save controls ──────────────────────────────────────────────────────
    st.subheader("Save Profile")
    save_col1, save_col2 = st.columns(2)
    with save_col1:
        if st.button("Save", use_container_width=True,
                      help="Overwrite the currently loaded profile"):
            loaded = st.session_state.get("loaded_profile")
            if loaded:
                save_profile_to_json(loaded)
                profile_name = Path(loaded).stem.replace("_", " ").title()
                st.success(f"Saved to **{profile_name}**.")
            else:
                st.warning("No profile loaded — use Save As instead.")
    with save_col2:
        new_name = st.text_input("Profile name", key="save_as_name",
                                 placeholder="e.g. my_custom_profile")
        if st.button("Save As...", use_container_width=True):
            name = st.session_state.get("save_as_name", "").strip()
            if not name:
                st.warning("Enter a profile name.")
            else:
                filename = name.lower().replace(" ", "_")
                if not filename.endswith(".json"):
                    filename += ".json"
                save_path = str(profile_dir / filename)
                save_profile_to_json(save_path)
                st.session_state["loaded_profile"] = save_path
                st.success(f"Saved as **{filename}**. Reload the page to see it in the dropdown.")


# ─── Tab 3: Algorithm Settings ──────────────────────────────────────────────

with tab_algo:
    st.subheader("Core Pump Settings")

    ac1, ac2, ac3 = st.columns(3)
    with ac1:
        st.number_input(
            "ISF (mg/dL per unit)",
            min_value=10.0, max_value=300.0, step=5.0,
            key="isf_in",
        )
        st.number_input(
            "Carb Ratio (g per unit)",
            min_value=3.0, max_value=30.0, step=0.5,
            key="carb_ratio_in",
        )
    with ac2:
        st.number_input(
            "Basal Rate (U/hr)",
            min_value=0.1, max_value=5.0, step=0.05,
            key="basal_rate_in",
        )
        st.number_input(
            "DIA (hours)",
            min_value=3.0, max_value=8.0, step=0.5,
            key="dia_in",
        )
    with ac3:
        st.number_input(
            "Target BG (mg/dL)",
            min_value=70.0, max_value=150.0, step=5.0,
            key="target_bg_in",
        )
        st.number_input(
            "Suspend Threshold (mg/dL)",
            min_value=50.0, max_value=100.0, step=5.0,
            key="suspend_threshold_in",
        )

    st.divider()

    st.subheader("Limits")
    lc1, lc2 = st.columns(2)
    with lc1:
        st.number_input(
            "Max Basal Rate (U/hr)",
            min_value=0.5, max_value=10.0, step=0.1,
            key="max_basal_rate_in",
        )
    with lc2:
        st.number_input(
            "Max Bolus (U)",
            min_value=0.5, max_value=10.0, step=0.5,
            key="max_bolus_in",
        )

    st.divider()

    st.subheader("Feature Toggles (Loop)")
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        st.checkbox("Enable IRC", key="enable_irc_cb")
    with fc2:
        st.checkbox("Enable Momentum", key="enable_momentum_cb")
    with fc3:
        st.checkbox("Enable DCA", key="enable_dca_cb")

    st.divider()

    st.subheader("Autosens / Dynamic ISF (Trio)")
    tc1, tc2 = st.columns(2)
    with tc1:
        st.checkbox("Autosens Enabled", key="autosens_enabled_cb")
        st.checkbox(
            "Dynamic ISF (Use New Formula)",
            key="use_new_formula_cb",
            help="Logarithmic dynamic ISF: adjusts ISF based on current BG",
        )
    with tc2:
        st.checkbox(
            "Sigmoid ISF",
            key="sigmoid_isf_cb",
            help="Use sigmoid curve instead of logarithmic for dynamic ISF",
        )
        st.slider(
            "Adjustment Factor Sigmoid",
            min_value=0.1, max_value=1.0, step=0.1,
            key="adj_factor_sigmoid_sl",
        )


# ─── Tab 1: Results ──────────────────────────────────────────────────────────

with tab_results:
    if run_button:
        if not algorithms:
            st.error("Select at least one algorithm.")
        else:
            profile = build_profile_from_state()

            traces_by_algo = {}
            metrics_by_algo = {}

            progress = st.progress(0, text="Starting simulation...")
            total_algos = len(algorithms)

            for i, algo_name in enumerate(algorithms):
                progress.progress(
                    i / total_algos,
                    text=f"Running {algo_name} ({n_paths} paths x {n_days} days)..."
                )
                traces, metrics = run_paths(profile, algo_name, n_paths, n_days, seed)
                traces_by_algo[algo_name] = traces
                metrics_by_algo[algo_name] = metrics

            progress.progress(1.0, text="Done!")

            # Spaghetti plot
            st.subheader("BG Traces")
            fig = plot_spaghetti(traces_by_algo, ALGO_COLORS, n_paths, n_days)
            st.pyplot(fig)
            plt.close(fig)

            # Summary stats table
            st.subheader("Summary Statistics")
            cols = st.columns(len(algorithms))
            for col, algo_name in zip(cols, algorithms):
                with col:
                    label = algo_name.replace("_", " ").upper()
                    st.markdown(f"**{label}**")
                    stats = compute_summary_stats(metrics_by_algo[algo_name])
                    for metric, value in stats.items():
                        st.metric(metric, value)

            # Head-to-head (if exactly 2 algorithms)
            if len(algorithms) == 2:
                st.subheader("Head-to-Head")
                a, b = algorithms
                ma = metrics_by_algo[a]
                mb = metrics_by_algo[b]
                n = min(len(ma), len(mb))

                tir_a_wins = sum(1 for x, y in zip(ma[:n], mb[:n])
                                 if x.time_in_range > y.time_in_range)
                tir_b_wins = sum(1 for x, y in zip(ma[:n], mb[:n])
                                 if y.time_in_range > x.time_in_range)
                hypo_a_wins = sum(1 for x, y in zip(ma[:n], mb[:n])
                                  if x.time_below_70 < y.time_below_70)
                hypo_b_wins = sum(1 for x, y in zip(ma[:n], mb[:n])
                                  if y.time_below_70 < x.time_below_70)

                a_label = a.replace("_", " ").upper()
                b_label = b.replace("_", " ").upper()
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Higher TIR**: {a_label} wins {tir_a_wins}, "
                                f"{b_label} wins {tir_b_wins}, "
                                f"tied {n - tir_a_wins - tir_b_wins}")
                with col2:
                    st.markdown(f"**Less hypo**: {a_label} wins {hypo_a_wins}, "
                                f"{b_label} wins {hypo_b_wins}, "
                                f"tied {n - hypo_a_wins - hypo_b_wins}")

    else:
        st.info("Configure settings in the sidebar and the tabs below, "
                "then click **Run Simulation**.")
        st.markdown("""
        ### About

        This simulator runs Monte Carlo comparisons of insulin dosing algorithms
        using a stochastic patient model. Each simulated "path" represents one
        possible day with random variation in:

        - **Meal sizes** and absorption rates
        - **Carb counting accuracy** (over/under-estimation)
        - **Insulin sensitivity** (day-to-day variation)
        - **Exercise** (timing, intensity, duration mismatch)
        - **Undeclared meals** (snacks the pump doesn't know about)

        Use the **Patient Model** tab to edit meals, carb counting skill, exercise,
        etc. Use the **Algorithm Settings** tab to change pump settings like ISF, CR,
        basal, etc.

        Select a patient profile from the sidebar as a starting point, then customize.
        """)
