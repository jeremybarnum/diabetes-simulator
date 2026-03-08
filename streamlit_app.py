"""
Diabetes Algorithm Simulator — Streamlit Web App

Monte Carlo comparison of Loop and Trio insulin dosing algorithms.
Full UI for editing patient model and algorithm settings.
"""

import json
from datetime import datetime, timedelta
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from simulation import PatientProfile, MealSpec, ExerciseSpec, SimulationRun, SimulationRunResult
from monte_carlo import compute_metrics, compute_metrics_by_day_type, GlycemicMetrics, _algo_seed_offset, _profile_to_dict, build_variants
from nightscout_profile import build_profile as ns_build_profile

try:
    import modal
    MODAL_AVAILABLE = True
except ImportError:
    MODAL_AVAILABLE = False

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

ns_ref_dir = Path(__file__).parent / "ns_references"
ns_ref_dir.mkdir(exist_ok=True)
ns_ref_files = sorted(ns_ref_dir.glob("*.json"))
ns_ref_names = {"(none)": None}
ns_ref_names.update({p.stem.replace("_", " ").title(): str(p) for p in ns_ref_files})


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
    def _meal_to_dict(m):
        d = {"time_of_day_minutes": m.time_of_day_minutes,
             "carbs_mean": m.carbs_mean, "carbs_sd": m.carbs_sd,
             "absorption_hrs": m.absorption_hrs,
             "declared": m.declared}
        d["carb_count_sigma"] = m.carb_count_sigma
        return d

    data = {
        "meals_rest": [_meal_to_dict(m) for m in profile.meals_rest],
        "meals_exercise": [_meal_to_dict(m) for m in profile.meals_exercise],
        "undeclared_meals_rest": [_meal_to_dict(m) for m in profile.undeclared_meals_rest],
        "undeclared_meals_exercise": [_meal_to_dict(m) for m in profile.undeclared_meals_exercise],
        "carb_count_bias": profile.carb_count_bias,
        "absorption_sigma": profile.absorption_sigma,
        "undeclared_meal_prob": profile.undeclared_meal_prob,
        "sensitivity_sigma": profile.sensitivity_sigma,
        "exercise_days": profile.exercise_days,
        "starting_bg": profile.starting_bg,
        "rescue_carbs_enabled": profile.rescue_carbs_enabled,
        "rescue_threshold": profile.rescue_threshold,
        "rescue_carbs_grams": profile.rescue_carbs_grams,
        "rescue_absorption_hrs": profile.rescue_absorption_hrs,
        "rescue_cooldown_min": profile.rescue_cooldown_min,
        "rescue_carbs_declared_pct": profile.rescue_carbs_declared_pct,
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
    if profile.algorithm_settings:
        data["algorithm_settings"] = profile.algorithm_settings
    # Preserve ns_reference_stats if present in the existing file
    if Path(file_path).exists():
        try:
            with open(file_path) as f:
                existing = json.load(f)
            if "ns_reference_stats" in existing:
                data["ns_reference_stats"] = existing["ns_reference_stats"]
        except (json.JSONDecodeError, IOError):
            pass
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)


def load_profile_to_state(profile_path: str):
    """Load a patient profile JSON and populate session state.

    Sets widget keys directly so widgets pick up values via their key= param.
    """
    profile = PatientProfile.from_json(profile_path)
    settings = profile.get_settings()

    # Build meal DataFrames for rest and exercise days
    empty_meals_df = pd.DataFrame(
        {"Time": pd.Series(dtype=str), "Avg Carbs (g)": pd.Series(dtype=float),
         "SD (g)": pd.Series(dtype=float), "Absorption (hrs)": pd.Series(dtype=float),
         "Declared": pd.Series(dtype=bool), "Count Error σ": pd.Series(dtype=float)}
    )

    def _build_meals_df(declared_meals, undeclared_meals):
        rows = []
        for m in declared_meals:
            rows.append({
                "Time": _minutes_to_clock(m.time_of_day_minutes),
                "Avg Carbs (g)": float(m.carbs_mean),
                "SD (g)": float(m.carbs_sd),
                "Absorption (hrs)": float(m.absorption_hrs),
                "Declared": m.declared,
                "Count Error σ": float(m.carb_count_sigma),
            })
        for m in undeclared_meals:
            rows.append({
                "Time": _minutes_to_clock(m.time_of_day_minutes),
                "Avg Carbs (g)": float(m.carbs_mean),
                "SD (g)": float(m.carbs_sd),
                "Absorption (hrs)": float(m.absorption_hrs),
                "Declared": False,
                "Count Error σ": float(m.carb_count_sigma),
            })
        rows.sort(key=lambda r: r["Time"])
        return pd.DataFrame(rows) if rows else empty_meals_df.copy()

    st.session_state["meals_rest_df"] = _build_meals_df(profile.meals_rest, profile.undeclared_meals_rest)
    st.session_state["meals_exercise_df"] = _build_meals_df(profile.meals_exercise, profile.undeclared_meals_exercise)

    # Patient model sliders — keys match the widget key= params
    st.session_state["carb_count_bias_sl"] = float(profile.carb_count_bias)
    st.session_state["absorption_sigma_sl"] = float(profile.absorption_sigma)
    st.session_state["undeclared_meal_prob_sl"] = float(profile.undeclared_meal_prob)
    st.session_state["sensitivity_sigma_sl"] = float(profile.sensitivity_sigma)
    st.session_state["starting_bg_sl"] = int(profile.starting_bg)

    # Exercise
    DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    for i, name in enumerate(DAY_NAMES):
        st.session_state[f"ex_day_{i}_cb"] = i in profile.exercise_days
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
    st.session_state["insulin_type_in"] = settings.get("insulin_type", "rapid_acting_adult")
    st.session_state["enable_irc_cb"] = bool(settings.get("enable_irc", True))
    st.session_state["enable_momentum_cb"] = bool(settings.get("enable_momentum", True))
    st.session_state["enable_dca_cb"] = bool(settings.get("enable_dca", True))

    # Trio autosens settings
    st.session_state["autosens_enabled_cb"] = True
    st.session_state["use_new_formula_cb"] = False
    st.session_state["sigmoid_isf_cb"] = False
    st.session_state["adj_factor_sigmoid_sl"] = 0.5

    # Rescue carbs
    st.session_state["rescue_enabled_cb"] = bool(profile.rescue_carbs_enabled)
    st.session_state["rescue_threshold_sl"] = float(profile.rescue_threshold)
    st.session_state["rescue_grams_sl"] = float(profile.rescue_carbs_grams)
    st.session_state["rescue_absorption_sl"] = float(profile.rescue_absorption_hrs)
    st.session_state["rescue_cooldown_sl"] = float(profile.rescue_cooldown_min)
    st.session_state["rescue_declared_pct_sl"] = int(profile.rescue_carbs_declared_pct * 100)

    st.session_state["loaded_profile"] = profile_path


def build_profile_from_state() -> PatientProfile:
    """Construct a PatientProfile from current session state (widget keys)."""
    def _parse_meals_df(df_key):
        declared = []
        undeclared = []
        df = st.session_state.get(df_key)
        if df is not None and len(df) > 0:
            for _, row in df.iterrows():
                try:
                    t_min = _clock_to_minutes(str(row["Time"]))
                    is_declared = bool(row.get("Declared", True))
                    count_sigma_val = row.get("Count Error σ")
                    count_sigma = float(count_sigma_val) if pd.notna(count_sigma_val) else 0.15
                    spec = MealSpec(
                        time_of_day_minutes=t_min,
                        carbs_mean=float(row["Avg Carbs (g)"]),
                        carbs_sd=float(row["SD (g)"]),
                        absorption_hrs=float(row["Absorption (hrs)"]),
                        declared=is_declared,
                        carb_count_sigma=count_sigma,
                    )
                    if is_declared:
                        declared.append(spec)
                    else:
                        undeclared.append(spec)
                except (ValueError, KeyError):
                    continue
        return declared, undeclared

    meals_rest, undeclared_meals_rest = _parse_meals_df("meals_rest_df")
    meals_exercise, undeclared_meals_exercise = _parse_meals_df("meals_exercise_df")

    # Exercise
    exercise_spec = None
    exercise_days = [i for i in range(7) if st.session_state.get(f"ex_day_{i}_cb", False)]
    if exercise_days:
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
        "insulin_type": st.session_state.get("insulin_type_in", "rapid_acting_adult"),
        "enable_irc": st.session_state.get("enable_irc_cb", True),
        "enable_momentum": st.session_state.get("enable_momentum_cb", True),
        "enable_dca": st.session_state.get("enable_dca_cb", True),
    }

    return PatientProfile(
        meals_rest=meals_rest,
        meals_exercise=meals_exercise,
        carb_count_bias=st.session_state.get("carb_count_bias_sl", 0.0),
        absorption_sigma=st.session_state.get("absorption_sigma_sl", 0.15),
        undeclared_meal_prob=st.session_state.get("undeclared_meal_prob_sl", 0.0),
        undeclared_meals_rest=undeclared_meals_rest,
        undeclared_meals_exercise=undeclared_meals_exercise,
        sensitivity_sigma=st.session_state.get("sensitivity_sigma_sl", 0.15),
        exercise_days=exercise_days,
        exercise_spec=exercise_spec,
        starting_bg=float(st.session_state.get("starting_bg_sl", 120)),
        rescue_carbs_enabled=st.session_state.get("rescue_enabled_cb", True),
        rescue_threshold=st.session_state.get("rescue_threshold_sl", 70.0),
        rescue_carbs_grams=st.session_state.get("rescue_grams_sl", 8.0),
        rescue_absorption_hrs=st.session_state.get("rescue_absorption_sl", 1.0),
        rescue_cooldown_min=st.session_state.get("rescue_cooldown_sl", 15.0),
        rescue_carbs_declared_pct=st.session_state.get("rescue_declared_pct_sl", 0) / 100.0,
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

# Delete profile
if st.sidebar.button("Delete Profile", help="Permanently delete the currently selected profile."):
    protected = {"default", "real_patient"}
    stem = Path(profile_path).stem
    if stem in protected:
        st.sidebar.error(f"Cannot delete built-in profile **{selected_profile_name}**.")
    else:
        Path(profile_path).unlink()
        st.sidebar.success(f"Deleted **{selected_profile_name}**. Reloading...")
        st.rerun()

# Additional profiles for multi-profile comparison
additional_profile_names = st.sidebar.multiselect(
    "Additional Profiles to Compare",
    [n for n in profile_names.keys() if n != selected_profile_name],
    help="Select extra profiles to compare alongside the active profile above.",
)

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
    help="Loop AB40 delivers 40% of needed insulin as micro-boluses every 5 min. "
         "Loop Temp Basal adjusts the basal rate only. "
         "Trio uses oref1 SMB logic with 4-path predictions.",
)
algorithms = [algo_options[name] for name in selected_algos]

# Simulation parameters
n_paths = st.sidebar.slider(
    "Number of simulated weeks", min_value=5, max_value=100, value=20, step=5,
    help="Each simulated week runs Mon-Sun with fresh random draws for meal sizes, "
         "carb counting errors, and sensitivity. Exercise occurs on the same fixed days "
         "each week. A larger simulation reduces the chance of spurious conclusions from "
         "unlucky random draws.",
)
seed = st.sidebar.number_input(
    "Random seed", value=42, min_value=0, max_value=99999,
    help="Fixed seed makes results reproducible. Change it to see different random draws.",
)
with st.sidebar.expander("Advanced"):
    n_days = st.slider(
        "Days per path", min_value=1, max_value=14, value=7, step=1,
        help="Default 7 (one full week). Reduce for faster debugging runs.",
    )

run_button = st.sidebar.button("Run Simulation", type="primary", use_container_width=True)

st.sidebar.divider()
st.sidebar.subheader("Display Options")
selected_ns_ref = st.sidebar.selectbox(
    "NS Reference Overlay",
    list(ns_ref_names.keys()),
    index=0,
    help="Overlay the real Nightscout median BG trace on the spaghetti plot for comparison.",
)
ns_ref_path = ns_ref_names[selected_ns_ref]
trace_day_filter = st.sidebar.radio(
    "Day filter",
    ["All days", "Rest days", "Exercise days"],
    index=0,
    horizontal=True,
    help="Filter both sim traces and NS overlay by exercise vs rest days.",
)


# ─── Helper functions ────────────────────────────────────────────────────────

def run_paths(profile, algorithm_name, n_paths, n_days, seed, progress_cb=None, seed_label=None):
    """Run n_paths simulations, return (traces_by_day, metrics_dict, exercise_flags).

    traces_by_day: list of lists. Each path produces n_days day-traces.
    Each day-trace is [(hours_in_day, bg), ...].
    metrics_dict: {"all": [...], "rest": [...], "exercise": [...]} lists of GlycemicMetrics.
    exercise_flags: list of lists of bools, parallel to traces_by_day.
    progress_cb: optional callback(completed_count) called after each path.
    seed_label: label used for seed offset (defaults to algorithm_name).
    """
    if seed_label is None:
        seed_label = algorithm_name
    all_day_traces = []
    all_exercise_flags = []
    all_metrics = []
    rest_metrics = []
    exercise_metrics = []
    for i in range(n_paths):
        path_seed = seed + i * 1000 + _algo_seed_offset(seed_label)
        rng = np.random.RandomState(path_seed)
        sim = SimulationRun(profile=profile, algorithm_name=algorithm_name,
                            n_days=n_days, rng=rng)
        result = sim.run()
        m_all, m_rest, m_exercise = compute_metrics_by_day_type(result)
        all_metrics.append(m_all)
        if m_rest is not None:
            rest_metrics.append(m_rest)
        if m_exercise is not None:
            exercise_metrics.append(m_exercise)

        path_days = []
        path_flags = []
        for day in result.days:
            day_offset_min = day.day_index * 1440
            trace = [((t_rel - day_offset_min) / 60, bg) for t_rel, bg in day.bg_trace]
            path_days.append(trace)
            path_flags.append(day.is_exercise_day)
        all_day_traces.append(path_days)
        all_exercise_flags.append(path_flags)

        if progress_cb:
            progress_cb(i + 1)
    metrics_dict = {"all": all_metrics, "rest": rest_metrics, "exercise": exercise_metrics}
    return all_day_traces, metrics_dict, all_exercise_flags


def run_paths_cloud(variants, n_paths, n_days, seed, progress_cb=None):
    """Run all paths on Modal cloud workers. Returns (traces, metrics, exercise_flags) by variant.

    variants: list of (variant_label, profile_dict, algo_name) tuples from build_variants().
    Each Modal worker runs one path for all variants. Results are collected
    and converted to the same format as run_paths().
    """
    from monte_carlo_cloud import run_path_remote, app as modal_app
    from simulation import SimulationRunResult

    variant_labels = [v[0] for v in variants]

    args_list = []
    for i in range(n_paths):
        path_seed = seed + i * 1000
        args_list.append((path_seed, variants, n_days))

    traces_by_variant = {name: [] for name in variant_labels}
    metrics_by_variant = {name: {"all": [], "rest": [], "exercise": []} for name in variant_labels}
    flags_by_variant = {name: [] for name in variant_labels}

    completed = 0
    with modal_app.run():
        for result in run_path_remote.starmap(args_list, order_outputs=False):
            for variant_label, data in result.items():
                run_result = SimulationRunResult.from_dict(data['result'])
                m_all, m_rest, m_exercise = compute_metrics_by_day_type(run_result)
                metrics_by_variant[variant_label]["all"].append(m_all)
                if m_rest is not None:
                    metrics_by_variant[variant_label]["rest"].append(m_rest)
                if m_exercise is not None:
                    metrics_by_variant[variant_label]["exercise"].append(m_exercise)

                path_days = []
                path_flags = []
                for day in run_result.days:
                    day_offset_min = day.day_index * 1440
                    trace = [((t_rel - day_offset_min) / 60, bg)
                             for t_rel, bg in day.bg_trace]
                    path_days.append(trace)
                    path_flags.append(day.is_exercise_day)
                traces_by_variant[variant_label].append(path_days)
                flags_by_variant[variant_label].append(path_flags)

            completed += 1
            if progress_cb:
                progress_cb(completed / n_paths)

    return traces_by_variant, metrics_by_variant, flags_by_variant


def plot_spaghetti(traces_by_algo, algo_colors, n_paths, n_days,
                   ns_ref_trace=None, ns_ref_label="NS Reference"):
    """Create spaghetti plot. Each day of each path overlaid on 0-24h axis.

    ns_ref_trace: optional list of (hour_from_7am, median_bg) from NS data.
    """
    fig, ax = plt.subplots(figsize=(14, 7))

    # NS reference trace (behind everything)
    if ns_ref_trace:
        ref_hours = [h for h, _ in ns_ref_trace]
        ref_bgs = [bg for _, bg in ns_ref_trace]
        ax.plot(ref_hours, ref_bgs, color="black", linewidth=2.5, linestyle="--",
                alpha=0.7, label=f"{ns_ref_label} (median)", zorder=5)

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
                label = _variant_display_name(algo_name)
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
    weeks_label = f"{n_paths} weeks" if n_days == 7 else f"{n_paths} paths x {n_days} days"
    ax.set_title(f"24h BG Traces — {weeks_label}")
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
        "SD BG (mg/dL)": f"{np.mean([m.sd_bg for m in all_metrics]):.1f}",
        "TIR 70-180 (%)": f"{np.mean([m.time_in_range for m in all_metrics]):.1f}",
        "Time <70 (%)": f"{np.mean([m.time_below_70 for m in all_metrics]):.2f}",
        "Time <54 (%)": f"{np.mean([m.time_below_54 for m in all_metrics]):.3f}",
        "Time >180 (%)": f"{np.mean([m.time_above_180 for m in all_metrics]):.1f}",
        "Time >250 (%)": f"{np.mean([m.time_above_250 for m in all_metrics]):.2f}",
        "CV (%)": f"{np.mean([m.cv for m in all_metrics]):.1f}",
        "GMI (%)": f"{np.mean([m.gmi for m in all_metrics]):.2f}",
        "Hypo events/week": f"{np.mean([m.hypo_events for m in all_metrics]):.2f}",
        "Rescue carb events/week": f"{np.mean([m.rescue_carb_events for m in all_metrics]):.1f}",
        "Rescue carbs (g)/week": f"{np.mean([m.rescue_carb_grams_total for m in all_metrics]):.0f}",
    }


# ─── Main area — Tabs ────────────────────────────────────────────────────────

ALGO_COLORS = {
    "loop_ab40": "#2563eb",
    "loop_tb": "#7c3aed",
    "loop_ab_gbpa": "#0891b2",
    "trio": "#dc2626",
}

EXTENDED_COLORS = [
    "#2563eb", "#dc2626", "#7c3aed", "#0891b2",
    "#d97706", "#059669", "#e11d48", "#4f46e5",
    "#0284c7", "#9333ea", "#c2410c", "#15803d",
]


def _variant_display_name(variant_label):
    """Format variant label for display (e.g., 'loop_ab40/real_patient' -> 'LOOP AB40 / Real Patient')."""
    if "/" in variant_label:
        algo, profile = variant_label.split("/", 1)
        return f"{algo.replace('_', ' ').upper()} / {profile.replace('_', ' ').title()}"
    return variant_label.replace("_", " ").upper()


def get_variant_colors(variant_labels):
    """Assign distinct colors to variant labels."""
    if all(label in ALGO_COLORS for label in variant_labels):
        return {label: ALGO_COLORS[label] for label in variant_labels}
    colors = {}
    for i, label in enumerate(variant_labels):
        colors[label] = EXTENDED_COLORS[i % len(EXTENDED_COLORS)]
    return colors

tab_results, tab_patient, tab_algo, tab_ns, tab_help = st.tabs([
    "Results", "Patient Model", "Algorithm & Therapy Settings",
    "Import from Nightscout", "Help",
])

# ─── Tab 2: Patient Model ────────────────────────────────────────────────────

with tab_patient:
    st.caption("These are the active settings used when you click **Run Simulation**. "
               "Use **Load Profile** in the sidebar to reset from a preset.")

    # Weekly schedule (exercise days) — shown first so user sees context for meal editors
    st.subheader("Weekly Schedule")
    st.caption("Which days are exercise days? Each simulated week runs Mon-Sun.")
    _ex_day_cols = st.columns(7)
    _DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    for i, (_col, _name) in enumerate(zip(_ex_day_cols, _DAY_NAMES)):
        with _col:
            st.checkbox(_name, key=f"ex_day_{i}_cb")

    _MEAL_EDITOR_COLUMNS = {
        "Time": st.column_config.TextColumn(
            "Time (HH:MM)", help="24h clock time"),
        "Avg Carbs (g)": st.column_config.NumberColumn(
            "Avg Carbs (g)", min_value=0, max_value=200, step=1),
        "SD (g)": st.column_config.NumberColumn(
            "SD (g)", min_value=0, max_value=50, step=1),
        "Absorption (hrs)": st.column_config.NumberColumn(
            "Absorption (hrs)", min_value=0.5, max_value=8.0, step=0.5),
        "Declared": st.column_config.CheckboxColumn(
            "Declared", help="If unchecked, meal is eaten but never entered into the pump", default=True),
        "Count Error σ": st.column_config.NumberColumn(
            "Count Error σ", min_value=0.0, max_value=1.0, step=0.05, default=0.15,
            help="Per-meal carb counting error sigma."),
    }

    st.divider()

    st.subheader("Typical Rest Day Meals")
    st.caption("Times are in 24h clock (HH:MM). Day starts at 07:00. "
               "**Declared** meals are entered into the pump (with carb-counting error applied). "
               "Undeclared meals are eaten but never bolused for.")

    @st.fragment
    def rest_meal_editor_fragment():
        edited = st.data_editor(
            st.session_state["meals_rest_df"],
            num_rows="dynamic",
            use_container_width=True,
            key="meals_rest_editor",
            column_config=_MEAL_EDITOR_COLUMNS,
        )
        st.session_state["meals_rest_df"] = edited.sort_values("Time").reset_index(drop=True)

    rest_meal_editor_fragment()

    st.divider()

    st.subheader("Typical Exercise Day Meals")
    st.caption("Leave empty to use rest-day meals on exercise days too.")

    @st.fragment
    def exercise_meal_editor_fragment():
        edited = st.data_editor(
            st.session_state["meals_exercise_df"],
            num_rows="dynamic",
            use_container_width=True,
            key="meals_exercise_editor",
            column_config=_MEAL_EDITOR_COLUMNS,
        )
        st.session_state["meals_exercise_df"] = edited.sort_values("Time").reset_index(drop=True)

    exercise_meal_editor_fragment()

    st.divider()

    st.subheader("Carb Counting Ability")
    col1, col2 = st.columns(2)
    with col1:
        st.slider(
            "Actual absorption variation (sigma)",
            min_value=0.0, max_value=1.0, step=0.05,
            help="Lognormal sigma for day-to-day variation in actual carb absorption speed. "
                 "Declared absorption time stays fixed; actual varies around it.",
            key="absorption_sigma_sl",
        )
    with col2:
        st.slider(
            "Systematic carb-counting bias",
            min_value=-1.0, max_value=1.0, step=0.05,
            help="Positive = patient under-declares carbs, negative = over-declares",
            key="carb_count_bias_sl",
        )
    st.caption("Per-meal carb counting error (σ) is set in the meal table's Count Error σ column.")
    st.slider(
        "Probability meal goes undeclared",
        min_value=0.0, max_value=1.0, step=0.05,
        help="Chance that any given meal is eaten but not bolused",
        key="undeclared_meal_prob_sl",
    )

    st.divider()

    st.subheader("Sensitivity Variation")
    st.slider(
        "Daily sensitivity variation (sigma)",
        min_value=0.0, max_value=1.0, step=0.05,
        help="Lognormal sigma for day-to-day insulin sensitivity variation",
        key="sensitivity_sigma_sl",
    )

    st.divider()

    st.subheader("Exercise Parameters")
    _any_exercise = any(st.session_state.get(f"ex_day_{i}_cb", False) for i in range(7))
    if _any_exercise:
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

    st.subheader("Rescue Carbs")
    st.checkbox(
        "Enable rescue carbs",
        key="rescue_enabled_cb",
        help="When BG drops below threshold, the simulated patient eats fast-acting carbs.",
    )

    if st.session_state.get("rescue_enabled_cb", True):
        rc_col1, rc_col2 = st.columns(2)
        with rc_col1:
            st.slider(
                "BG threshold (mg/dL)",
                min_value=50, max_value=100, step=5,
                key="rescue_threshold_sl",
                help="Patient eats rescue carbs when BG drops below this value.",
            )
            st.slider(
                "Carbs per rescue (g)",
                min_value=4, max_value=30, step=2,
                key="rescue_grams_sl",
                help="Grams of fast-acting carbs consumed per rescue event.",
            )
        with rc_col2:
            st.slider(
                "Absorption time (hrs)",
                min_value=0.5, max_value=3.0, step=0.5,
                key="rescue_absorption_sl",
                help="How quickly rescue carbs are absorbed (shorter = faster-acting).",
            )
            st.slider(
                "Time between rescue carbs (min)",
                min_value=5, max_value=60, step=5,
                key="rescue_cooldown_sl",
                help="Minimum wait before another rescue dose.",
            )
        st.slider(
            "% of rescue carbs declared",
            min_value=0, max_value=100, step=10,
            key="rescue_declared_pct_sl",
            help="What percentage of rescue carbs the patient enters into the pump. "
                 "0% = invisible to algorithm, 100% = fully declared.",
        )

    st.divider()

    st.subheader("Starting BG")
    st.slider(
        "Starting blood glucose (mg/dL)",
        min_value=70, max_value=250, step=5,
        key="starting_bg_sl",
        help="BG at 7:00am when each simulated day begins. "
             "Both algorithms start with no active insulin or carbs.",
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

    st.divider()

    # ─── Export JSON ─────────────────────────────────────────────────────────
    with st.expander("Export Profile JSON"):
        if st.button("Show profile JSON"):
            profile = build_profile_from_state()
            def _meal_export(m):
                d = {"time_of_day_minutes": m.time_of_day_minutes,
                     "carbs_mean": m.carbs_mean, "carbs_sd": m.carbs_sd,
                     "absorption_hrs": m.absorption_hrs, "declared": m.declared}
                d["carb_count_sigma"] = m.carb_count_sigma
                return d
            export_data = {
                "meals_rest": [_meal_export(m) for m in profile.meals_rest],
                "meals_exercise": [_meal_export(m) for m in profile.meals_exercise],
                "undeclared_meals_rest": [_meal_export(m) for m in profile.undeclared_meals_rest],
                "undeclared_meals_exercise": [_meal_export(m) for m in profile.undeclared_meals_exercise],
                "carb_count_bias": profile.carb_count_bias,
                "absorption_sigma": profile.absorption_sigma,
                "undeclared_meal_prob": profile.undeclared_meal_prob,
                "sensitivity_sigma": profile.sensitivity_sigma,
                "exercise_days": profile.exercise_days,
                "starting_bg": profile.starting_bg,
                "rescue_carbs_enabled": profile.rescue_carbs_enabled,
                "rescue_threshold": profile.rescue_threshold,
                "rescue_carbs_grams": profile.rescue_carbs_grams,
                "rescue_absorption_hrs": profile.rescue_absorption_hrs,
                "rescue_cooldown_min": profile.rescue_cooldown_min,
                "rescue_carbs_declared_pct": profile.rescue_carbs_declared_pct,
            }
            if profile.exercise_spec:
                ex = profile.exercise_spec
                export_data["exercise_spec"] = {
                    "time_of_day_minutes": ex.time_of_day_minutes,
                    "declared_scalar": ex.declared_scalar,
                    "declared_duration_hrs": ex.declared_duration_hrs,
                    "actual_scalar_mean": ex.actual_scalar_mean,
                    "actual_scalar_sigma": ex.actual_scalar_sigma,
                    "actual_duration_hrs_mean": ex.actual_duration_hrs_mean,
                    "actual_duration_hrs_sigma": ex.actual_duration_hrs_sigma,
                }
            if profile.algorithm_settings:
                export_data["algorithm_settings"] = profile.algorithm_settings
            st.code(json.dumps(export_data, indent=2), language="json")


# ─── Tab 3: Algorithm & Therapy Settings ──────────────────────────────────────────────

with tab_algo:
    st.subheader("Core Therapy Settings")

    ac1, ac2, ac3 = st.columns(3)
    with ac1:
        st.number_input(
            "ISF (mg/dL per unit)",
            min_value=10.0, max_value=300.0, step=5.0,
            key="isf_in",
            help="Insulin Sensitivity Factor: how much 1 unit of insulin lowers BG. "
                 "Higher = more sensitive.",
        )
        st.number_input(
            "Carb Ratio (g per unit)",
            min_value=3.0, max_value=30.0, step=0.5,
            key="carb_ratio_in",
            help="Carb-to-insulin ratio: grams of carbs covered by 1 unit of insulin.",
        )
    with ac2:
        st.number_input(
            "Basal Rate (U/hr)",
            min_value=0.1, max_value=5.0, step=0.05,
            key="basal_rate_in",
            help="Scheduled background insulin delivery rate. The algorithm adjusts "
                 "this up or down via temp basals.",
        )
        st.number_input(
            "DIA (hours)",
            min_value=3.0, max_value=8.0, step=0.5,
            key="dia_in",
            help="Duration of Insulin Action: how long insulin stays active after delivery. "
                 "Fiasp default is 6 hours.",
        )
        _insulin_type_options = ["rapid_acting_adult", "fiasp", "lyumjev", "rapid_acting_child", "afrezza"]
        _insulin_type_labels = ["Humalog/Novolog (rapid-acting)", "Fiasp (ultra-rapid)", "Lyumjev (ultra-rapid)", "Rapid-Acting (child)", "Afrezza (inhaled)"]
        st.selectbox(
            "Insulin Type",
            options=_insulin_type_options,
            format_func=lambda x: _insulin_type_labels[_insulin_type_options.index(x)],
            key="insulin_type_in",
            help="Insulin curve model used by the algorithm. Affects how quickly insulin "
                 "is predicted to act.",
        )
    with ac3:
        st.number_input(
            "Target BG (mg/dL)",
            min_value=70.0, max_value=150.0, step=5.0,
            key="target_bg_in",
            help="The BG value the algorithm tries to steer toward with corrections.",
        )
        st.number_input(
            "Suspend Threshold (mg/dL)",
            min_value=50.0, max_value=100.0, step=5.0,
            key="suspend_threshold_in",
            help="If predicted BG drops below this, insulin delivery is suspended (zero temp).",
        )

    st.divider()

    st.subheader("Limits")
    lc1, lc2 = st.columns(2)
    with lc1:
        st.number_input(
            "Max Basal Rate (U/hr)",
            min_value=0.5, max_value=10.0, step=0.1,
            key="max_basal_rate_in",
            help="Safety ceiling for temp basal rates. Limits how aggressively "
                 "the algorithm can increase insulin delivery.",
        )
    with lc2:
        st.number_input(
            "Max Bolus (U)",
            min_value=0.5, max_value=10.0, step=0.5,
            key="max_bolus_in",
            help="Maximum single auto-bolus or SMB the algorithm can deliver. "
                 "Also caps meal boluses.",
        )

    st.divider()

    st.subheader("Feature Toggles (Loop)")
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        st.checkbox(
            "Enable IRC", key="enable_irc_cb",
            help="Integral Retrospective Correction: compares predicted vs actual BG "
                 "over the past 3 hours and applies a correction to future predictions.",
        )
    with fc2:
        st.checkbox(
            "Enable Momentum", key="enable_momentum_cb",
            help="Projects the recent BG trend (last ~15 min) forward into predictions, "
                 "helping the algorithm react faster to rapid changes.",
        )
    with fc3:
        st.checkbox(
            "Enable DCA", key="enable_dca_cb",
            help="Dynamic Carb Absorption: adjusts predicted carb effects based on "
                 "observed glucose changes, detecting faster or slower absorption than declared.",
        )

    st.divider()

    st.subheader("Autosens / Dynamic ISF (Trio)")
    tc1, tc2 = st.columns(2)
    with tc1:
        st.checkbox(
            "Autosens Enabled", key="autosens_enabled_cb",
            help="Analyzes the past 24 hours of BG data to detect if the patient is "
                 "more or less insulin sensitive than usual, and adjusts ISF/basal/target.",
        )
        st.checkbox(
            "Dynamic ISF (Use New Formula)",
            key="use_new_formula_cb",
            help="Logarithmic dynamic ISF: adjusts ISF based on current BG. "
                 "Higher BG = lower ISF (more aggressive corrections).",
        )
    with tc2:
        st.checkbox(
            "Sigmoid ISF",
            key="sigmoid_isf_cb",
            help="Uses a sigmoid curve for dynamic ISF instead of logarithmic. "
                 "Provides smoother scaling that levels off at extreme BG values.",
        )
        st.slider(
            "Adjustment Factor Sigmoid",
            min_value=0.1, max_value=1.0, step=0.1,
            key="adj_factor_sigmoid_sl",
            help="Controls the aggressiveness of sigmoid dynamic ISF. "
                 "Lower = gentler adjustments, higher = more aggressive.",
        )

    st.divider()
    if st.button("Save Settings to Profile", key="save_therapy_btn", use_container_width=True):
        loaded = st.session_state.get("loaded_profile")
        if loaded:
            save_profile_to_json(loaded)
            profile_name = Path(loaded).stem.replace("_", " ").title()
            st.success(f"Saved to **{profile_name}**.")
        else:
            st.warning("No profile loaded — save from the Patient Model tab first.")


# ─── Tab 4: Import from Nightscout ───────────────────────────────────────────

_ns_config_path = Path(__file__).parent / ".ns_config.json"

def _load_ns_config():
    if _ns_config_path.exists():
        try:
            with open(_ns_config_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {}

def _save_ns_config(cfg):
    with open(_ns_config_path, "w") as f:
        json.dump(cfg, f)

# Seed session state from saved config on first load
if "ns_url_in" not in st.session_state:
    _cfg = _load_ns_config()
    if _cfg.get("url"):
        st.session_state["ns_url_in"] = _cfg["url"]
    if _cfg.get("token"):
        st.session_state["ns_token_in"] = _cfg["token"]

with tab_ns:
    st.subheader("Import Profile from Nightscout")
    st.caption("Pull real meal/bolus/CGM data and auto-generate a patient profile.")

    ns_url = st.text_input("Nightscout URL", key="ns_url_in",
                           placeholder="https://your-ns-site.nightscoutpro.com")
    col1, col2, col3 = st.columns(3)
    with col1:
        ns_token = st.text_input("API Token (optional)", key="ns_token_in", type="password")
        ns_profile_name = st.text_input("Save as profile name", value="ns_inferred", key="ns_name_in")
    with col2:
        ns_date_mode = st.radio("Date range", ["Last N days", "Custom range"], key="ns_date_mode", horizontal=True)
        if ns_date_mode == "Last N days":
            ns_days = st.number_input("Days of history", value=28, min_value=7, max_value=365, key="ns_days_in")
            ns_start_date = None
            ns_end_date = None
        else:
            from datetime import date as _date
            today = _date.today()
            ns_end_date = st.date_input("End date", value=today, key="ns_end_date")
            ns_start_date = st.date_input("Start date", value=today - timedelta(days=28), key="ns_start_date")
            ns_days = 28  # fallback, overridden by dates
    with col3:
        from nightscout_profile import INSULIN_TYPES, DEFAULT_INSULIN_TYPE
        insulin_options = list(INSULIN_TYPES.keys())
        ns_insulin_type = st.selectbox(
            "Insulin type", options=insulin_options,
            index=insulin_options.index(DEFAULT_INSULIN_TYPE),
            key="ns_insulin_type",
            help="Determines the insulin activity curve used for deviation analysis"
        )
        ns_meal_times_str = st.text_input(
            "Meal times (clock hours)",
            key="ns_meal_times",
            placeholder="e.g. 8, 13, 19",
            help="Comma-separated clock hours. Carb entries are bucketed to the nearest meal. Leave blank for auto-detect."
        )

    import_button = st.button("Import from Nightscout", type="primary")

    if import_button:
        if not ns_url or not ns_url.startswith("http"):
            st.error("Enter a valid Nightscout URL.")
        else:
            url = ns_url.rstrip("/")
            token = ns_token or None
            filename = ns_profile_name.strip().lower().replace(" ", "_")
            if not filename:
                filename = "ns_inferred"
            save_path = str(profile_dir / f"{filename}.json")

            import io, contextlib
            log_buffer = io.StringIO()
            with st.spinner("Fetching data from Nightscout..."):
                try:
                    from datetime import timezone as _tz
                    _start = (datetime.combine(ns_start_date, datetime.min.time()).replace(tzinfo=_tz.utc)
                              if ns_start_date else None)
                    _end = (datetime.combine(ns_end_date, datetime.min.time()).replace(tzinfo=_tz.utc)
                            if ns_end_date else None)
                    _meal_times = None
                    _mt_str = ns_meal_times_str or ""
                    if _mt_str.strip():
                        try:
                            _meal_times = [float(h.strip()) for h in _mt_str.split(",")]
                        except ValueError:
                            st.warning(f"Could not parse meal times: '{_mt_str}'. Using auto-detect.")
                    with contextlib.redirect_stdout(log_buffer):
                        profile_dict = ns_build_profile(
                            url, days=ns_days, token=token, output_path=save_path,
                            start_date=_start, end_date=_end,
                            insulin_type=ns_insulin_type,
                            meal_times=_meal_times,
                        )
                except Exception as e:
                    st.error(f"Import failed: {e}")
                    with st.expander("Full log"):
                        st.code(log_buffer.getvalue())
                    st.stop()

            # Persist NS URL/token for future sessions
            _save_ns_config({"url": url, "token": token or ""})

            st.success(f"Profile saved to **{filename}.json**")

            # Save standalone NS reference
            from nightscout_profile import save_ns_reference
            ref_path = str(ns_ref_dir / f"{filename}.json")
            save_ns_reference(profile_dict, ref_path)
            st.info(f"NS reference trace saved to **ns_references/{filename}.json** — "
                    f"select it from the sidebar to overlay on simulation results.")

            # Algorithm settings
            st.subheader("Extracted Algorithm Settings")
            algo = profile_dict.get("algorithm_settings", {})
            sc1, sc2, sc3, sc4 = st.columns(4)
            sc1.metric("ISF", f"{algo.get('insulin_sensitivity_factor', '?')} mg/dL/U")
            sc2.metric("Carb Ratio", f"{algo.get('carb_ratio', '?')} g/U")
            sc3.metric("Basal Rate", f"{algo.get('basal_rate', '?')} U/hr")
            sc4.metric("Target BG", f"{algo.get('target', '?')} mg/dL")

            # Detected meals
            st.subheader("Detected Meals")
            meals = profile_dict.get("meals_rest", profile_dict.get("meals", []))
            if meals:
                meal_rows = []
                for m in meals:
                    clock = _minutes_to_clock(m["time_of_day_minutes"])
                    meal_rows.append({"Time": clock, "Avg Carbs (g)": m["carbs_mean"],
                                      "SD (g)": m["carbs_sd"], "Absorption (hrs)": m["absorption_hrs"]})
                st.dataframe(pd.DataFrame(meal_rows), use_container_width=True)
            else:
                st.info("No meal patterns detected.")

            # Exercise
            ex_days = profile_dict.get("exercise_days", [])
            if ex_days:
                st.subheader("Exercise")
                _day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
                day_str = ", ".join(_day_labels[d] for d in sorted(ex_days) if d < 7)
                ex = profile_dict.get("exercise_spec", {})
                st.write(f"**{day_str}** at {_minutes_to_clock(ex.get('time_of_day_minutes', 0))}")

            # Starting BG
            st.metric("Starting BG (morning median)", f"{profile_dict.get('starting_bg', '?')} mg/dL")

            # Reference BG stats
            ref = profile_dict.get("ns_reference_stats", {})
            if ref:
                st.subheader("Reference BG Statistics")
                st.caption(f"From Nightscout data: {ref.get('start_date', '?')} to {ref.get('end_date', '?')} "
                           f"({ref.get('days', '?')} days, source: {ref.get('data_source', '?')})")
                rc1, rc2, rc3, rc4 = st.columns(4)
                rc1.metric("Mean BG", f"{ref.get('mean_bg', '?')} mg/dL")
                rc2.metric("SD", f"{ref.get('sd_bg', '?')} mg/dL")
                rc3.metric("GMI", f"{ref.get('gmi', '?')}%")
                rc4.metric("TIR (70-180)", f"{ref.get('tir', '?')}%")
                rc5, rc6, rc7, rc8 = st.columns(4)
                rc5.metric("Time <70", f"{ref.get('time_below_70', '?')}%")
                rc6.metric("Time <54", f"{ref.get('time_below_54', '?')}%")
                rc7.metric("Time >180", f"{ref.get('time_above_180', '?')}%")
                rc8.metric("Time >250", f"{ref.get('time_above_250', '?')}%")
                st.info("Compare these reference stats against simulation results to validate the patient model.")

            # Full log
            with st.expander("Full import log"):
                st.code(log_buffer.getvalue())

            display_name = filename.replace("_", " ").title()
            st.info(f"**Reload the page** (Ctrl-R / Cmd-R), then select **{display_name}** from the **Load Profile** dropdown in the sidebar.")


# ─── Tab 1: Results ──────────────────────────────────────────────────────────

with tab_results:
    if run_button:
        if not algorithms:
            st.error("Select at least one algorithm.")
        else:
            profile = build_profile_from_state()

            # Build profile list for comparison
            primary_label = Path(profile_path).stem
            profile_objs = [(primary_label, profile)]
            for name in additional_profile_names:
                p = PatientProfile.from_json(profile_names[name])
                profile_objs.append((Path(profile_names[name]).stem, p))

            n_prof = len(profile_objs)
            multi_profile = n_prof > 1

            # Build variant tuples: (variant_label, profile_obj, algo_name)
            variants_local = []
            for algo_name in algorithms:
                for plabel, pobj in profile_objs:
                    vlabel = algo_name if not multi_profile else f"{algo_name}/{plabel}"
                    variants_local.append((vlabel, pobj, algo_name))
            variant_labels = [v[0] for v in variants_local]

            traces_by_variant = {}
            metrics_by_variant = {}
            flags_by_variant = {}

            import time as _time
            if len(variant_labels) <= 3:
                variant_summary = " vs ".join(
                    _variant_display_name(vl) for vl in variant_labels)
            else:
                variant_summary = f"{len(variant_labels)} variants"
            total_work = f"{n_paths} weeks"

            progress = st.progress(0, text=f"Starting {variant_summary} — {total_work}...")
            t_start = _time.time()

            def _eta_str(elapsed, frac):
                if frac <= 0:
                    return ""
                remaining = elapsed / frac * (1 - frac)
                if remaining < 60:
                    return f"~{remaining:.0f}s left"
                return f"~{remaining / 60:.1f}m left"

            cloud_success = False
            if MODAL_AVAILABLE and n_paths > 1:
                try:
                    progress.progress(
                        0, text=f"Connecting to Modal Cloud — {variant_summary}, {total_work}...")

                    # Build cloud variants (serialized profile dicts)
                    profile_dicts = [(label, _profile_to_dict(p))
                                     for label, p in profile_objs]
                    cloud_variants = build_variants(algorithms, profile_dicts)

                    def update_progress(frac):
                        done = int(frac * n_paths)
                        elapsed = _time.time() - t_start
                        if done >= n_paths:
                            progress.progress(1.0,
                                text=f"Cloud: all {n_paths} weeks done in {elapsed:.1f}s — rendering...")
                        else:
                            eta = _eta_str(elapsed, frac)
                            progress.progress(frac,
                                text=f"Cloud: {done}/{n_paths} weeks — "
                                     f"{elapsed:.0f}s elapsed, {eta}")

                    traces_by_variant, metrics_by_variant, flags_by_variant = run_paths_cloud(
                        cloud_variants, n_paths, n_days, seed,
                        progress_cb=update_progress)
                    cloud_success = True
                except Exception as e:
                    st.warning(f"Modal cloud failed ({e}), falling back to local...")
                    t_start = _time.time()

            if not cloud_success:
                total_paths = len(variants_local) * n_paths
                paths_done = 0

                for vlabel, pobj, algo_name in variants_local:
                    display = _variant_display_name(vlabel)

                    def local_progress_cb(path_count, _label=display, _offset=paths_done):
                        done = _offset + path_count
                        frac = done / total_paths
                        elapsed = _time.time() - t_start
                        eta = _eta_str(elapsed, frac)
                        progress.progress(frac,
                            text=f"{_label}: week {path_count}/{n_paths} — "
                                 f"{done}/{total_paths} total, "
                                 f"{elapsed:.0f}s elapsed, {eta}")

                    progress.progress(paths_done / total_paths,
                        text=f"{display}: starting ({paths_done}/{total_paths} total)...")
                    traces, metrics, ex_flags = run_paths(
                        pobj, algo_name, n_paths, n_days, seed,
                        progress_cb=local_progress_cb, seed_label=vlabel)
                    traces_by_variant[vlabel] = traces
                    metrics_by_variant[vlabel] = metrics
                    flags_by_variant[vlabel] = ex_flags
                    paths_done += n_paths

            elapsed_total = _time.time() - t_start
            where = "cloud" if cloud_success else "local"
            progress.progress(1.0,
                text=f"Done — {variant_summary}, {total_work} in {elapsed_total:.1f}s ({where})")

            # Assign colors
            variant_colors = get_variant_colors(variant_labels)

            # Cache results in session state so trace toggle can redraw without rerun
            st.session_state["_sim_traces"] = traces_by_variant
            st.session_state["_sim_metrics"] = metrics_by_variant
            st.session_state["_sim_flags"] = flags_by_variant
            st.session_state["_sim_variant_labels"] = variant_labels
            st.session_state["_sim_variant_colors"] = variant_colors
            st.session_state["_sim_n_paths"] = n_paths
            st.session_state["_sim_n_days"] = n_days

    # --- Display results (from session state, updates on trace toggle change) ---
    if "_sim_traces" in st.session_state:
        traces_by_variant = st.session_state["_sim_traces"]
        metrics_by_variant = st.session_state["_sim_metrics"]
        flags_by_variant = st.session_state.get("_sim_flags", {})
        variant_labels = st.session_state["_sim_variant_labels"]
        variant_colors = st.session_state["_sim_variant_colors"]
        n_paths = st.session_state["_sim_n_paths"]
        n_days = st.session_state["_sim_n_days"]

        # Filter traces by exercise/rest day
        filtered_traces = {}
        for vlabel in variant_labels:
            all_traces = traces_by_variant[vlabel]
            all_flags = flags_by_variant.get(vlabel, [])
            if trace_day_filter == "All days" or not all_flags:
                filtered_traces[vlabel] = all_traces
            else:
                want_exercise = (trace_day_filter == "Exercise days")
                filtered = []
                for path_days, path_flags in zip(all_traces, all_flags):
                    filtered_days = [d for d, f in zip(path_days, path_flags)
                                     if f == want_exercise]
                    if filtered_days:
                        filtered.append(filtered_days)
                filtered_traces[vlabel] = filtered

        # Load NS reference trace (re-evaluated on every render, so toggle changes take effect)
        _ns_ref_trace = None
        _ns_ref_stats = None
        if ns_ref_path:
            try:
                with open(ns_ref_path) as _f:
                    _ns_ref_data = json.load(_f)
                _trace_key = {
                    "All days": "median_trace",
                    "Rest days": "median_trace_rest",
                    "Exercise days": "median_trace_exercise",
                }.get(trace_day_filter, "median_trace")
                _raw_trace = _ns_ref_data.get(_trace_key) or _ns_ref_data.get("median_trace", [])
                _ns_ref_trace = [tuple(p) for p in _raw_trace]
                _ns_ref_stats = _ns_ref_data
            except Exception:
                pass

        # Spaghetti plot
        st.subheader("BG Traces")
        _ref_label = f"NS {trace_day_filter}" if ns_ref_path else "NS Reference"
        fig = plot_spaghetti(filtered_traces, variant_colors, n_paths, n_days,
                             ns_ref_trace=_ns_ref_trace, ns_ref_label=_ref_label)
        st.pyplot(fig)
        plt.close(fig)

        # Summary stats table
        st.subheader("Summary Statistics")

        # Use sidebar-selected NS reference, or fall back to profile-embedded stats
        ref_stats = _ns_ref_stats
        if not ref_stats:
            profile_path_loaded = st.session_state.get("loaded_profile", "")
            if profile_path_loaded:
                try:
                    with open(profile_path_loaded) as _f:
                        _pdata = json.load(_f)
                    ref_stats = _pdata.get("ns_reference_stats")
                except Exception:
                    pass

        # Build summary table as DataFrame with rest/exercise/all columns
        metric_rows = [
            "Mean BG (mg/dL)", "SD BG (mg/dL)", "TIR 70-180 (%)",
            "Time <70 (%)", "Time <54 (%)", "Time >180 (%)",
            "Time >250 (%)", "CV (%)", "GMI (%)",
            "Hypo events/wk", "Rescue events/wk", "Rescue carbs (g)/wk",
        ]
        _ns_keys = ["mean_bg", "sd_bg", "tir", "time_below_70", "time_below_54",
                     "time_above_180", "time_above_250", None, "gmi",
                     None, None, None]  # None = not available from NS

        def _ns_col(stats_dict):
            """Build a column of formatted values from an NS stats dict."""
            if not stats_dict:
                return ["—"] * len(metric_rows)
            vals = []
            for key in _ns_keys:
                if key is None:
                    vals.append("—")
                else:
                    v = stats_dict.get(key, "?")
                    vals.append(f"{v}")
            return vals

        def _sim_col(metrics_list):
            """Build a column of formatted values from a list of GlycemicMetrics."""
            if not metrics_list:
                return ["—"] * len(metric_rows)
            return [
                f"{np.mean([m.mean_bg for m in metrics_list]):.1f}",
                f"{np.mean([m.sd_bg for m in metrics_list]):.1f}",
                f"{np.mean([m.time_in_range for m in metrics_list]):.1f}",
                f"{np.mean([m.time_below_70 for m in metrics_list]):.2f}",
                f"{np.mean([m.time_below_54 for m in metrics_list]):.3f}",
                f"{np.mean([m.time_above_180 for m in metrics_list]):.1f}",
                f"{np.mean([m.time_above_250 for m in metrics_list]):.2f}",
                f"{np.mean([m.cv for m in metrics_list]):.1f}",
                f"{np.mean([m.gmi for m in metrics_list]):.2f}",
                f"{np.mean([m.hypo_events for m in metrics_list]):.2f}",
                f"{np.mean([m.rescue_carb_events for m in metrics_list]):.1f}",
                f"{np.mean([m.rescue_carb_grams_total for m in metrics_list]):.0f}",
            ]

        # Build columns dict: {column_header: [values]}
        columns = {}

        if ref_stats:
            columns["NS All"] = _ns_col(ref_stats)
            columns["NS Rest"] = _ns_col(ref_stats.get("stats_rest"))
            columns["NS Exercise"] = _ns_col(ref_stats.get("stats_exercise"))

        for vlabel in variant_labels:
            display = _variant_display_name(vlabel)
            mdict = metrics_by_variant[vlabel]
            columns[f"{display} All"] = _sim_col(mdict["all"])
            columns[f"{display} Rest"] = _sim_col(mdict["rest"])
            columns[f"{display} Exercise"] = _sim_col(mdict["exercise"])

        df = pd.DataFrame(columns, index=metric_rows)
        if ref_stats:
            _ns_days = ref_stats.get('days', '?')
            _ns_ex = ref_stats.get('exercise_days', '?')
            _ns_ex_per_wk = f"{_ns_ex / _ns_days * 7:.1f}" if isinstance(_ns_ex, (int, float)) and isinstance(_ns_days, (int, float)) and _ns_days > 0 else "?"
            st.caption(f"NS Reference: {ref_stats.get('start_date', '?')} to {ref_stats.get('end_date', '?')} ({_ns_days} days, {_ns_ex_per_wk} exercise days/wk)")
        st.dataframe(df, use_container_width=True)

        # Head-to-head (if exactly 2 variants)
        if len(variant_labels) == 2:
            st.subheader("Head-to-Head")
            a, b = variant_labels
            ma = metrics_by_variant[a]["all"]
            mb = metrics_by_variant[b]["all"]
            n = min(len(ma), len(mb))

            tir_a_wins = sum(1 for x, y in zip(ma[:n], mb[:n])
                             if x.time_in_range > y.time_in_range)
            tir_b_wins = sum(1 for x, y in zip(ma[:n], mb[:n])
                             if y.time_in_range > x.time_in_range)
            hypo_a_wins = sum(1 for x, y in zip(ma[:n], mb[:n])
                              if x.time_below_70 < y.time_below_70)
            hypo_b_wins = sum(1 for x, y in zip(ma[:n], mb[:n])
                              if y.time_below_70 < x.time_below_70)

            a_label = _variant_display_name(a)
            b_label = _variant_display_name(b)
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
        etc. Use the **Algorithm & Therapy Settings** tab to change therapy settings like ISF, CR,
        basal, etc.

        Select a patient profile from the sidebar as a starting point, then customize.
        """)


# ─── Tab 4: Help ─────────────────────────────────────────────────────────────

with tab_help:
    st.markdown("""
## Quick Start

1. **Pick a profile** — select a patient profile from the sidebar (e.g., "Real Patient"),
   or import one from Nightscout
2. **Tweak settings** — adjust meals, carb counting ability, and algorithm settings in the tabs
3. **Run** — click **Run Simulation** to generate Monte Carlo results

To compare the same algorithm across different patient profiles, select additional profiles
in the sidebar under **Additional Profiles to Compare**.

---

## How the Simulator Works

The simulator runs many independent weekly "paths" — each one simulates a full Mon-Sun week
of a Type 1 diabetic wearing an insulin pump controlled by a closed-loop algorithm.
Exercise occurs on the same fixed days each week; all other variation is random.

Each 5-minute step:
1. **Patient physiology** computes BG change from active insulin, absorbing carbs, and
   basal deficit (if sensitivity differs from the algorithm's assumption)
2. **The algorithm** sees the CGM reading and its own insulin/carb history, then decides
   whether to adjust the temp basal, deliver a micro-bolus, or suspend delivery
3. If BG drops below the rescue threshold (default 70), the simulated patient eats rescue
   carbs. Rescue carb behavior is configurable — amount, absorption speed, cooldown, and
   what percentage the patient declares to the pump

Every week draws fresh random values for meal sizes, carb counting errors, and insulin
sensitivity — so the collection of weeks shows the range of outcomes you'd see over
many real weeks. Exercise days are deterministic (fixed to the days you select).

When Modal (a serverless compute platform) is available, weeks are automatically
dispatched to cloud workers in parallel for faster results. Otherwise, weeks run locally.

---

## Understanding the Algorithms

### Loop (Auto-Bolus 40%)
Loop is a prediction-based system that builds a single 6-hour forecast from several
components:

- **Insulin effect** — where current insulin-on-board will take BG
- **Carb effect** — expected BG rise from declared carbs (adjusted by DCA if absorption
  differs from declared)
- **Momentum** — short-term CGM trend projected forward (~15 min)
- **IRC** — correction based on how well predictions matched reality over the past 3 hours

In Auto-Bolus mode, Loop delivers 40% of the calculated insulin requirement as a
micro-bolus every 5 minutes, plus a reduced temp basal.

### Loop (Temp Basal)
Same prediction pipeline, but corrections are delivered entirely via temporary basal rate
adjustments — no micro-boluses. Generally more conservative.

### Trio (oref1 SMB)
Trio uses the OpenAPS oref1 algorithm. Instead of a single prediction, it computes four
parallel forecasts:

- **IOB** — assumes no more carb absorption
- **ZT (Zero Temp)** — what happens if delivery stops now
- **COB** — carbs absorb at declared rate
- **UAM** — unannounced-meal detection using observed glucose rise

It picks the most conservative relevant prediction and delivers Super Micro-Boluses (SMBs)
for aggressive corrections. Trio also supports **Autosens** (24h sensitivity detection)
and **Dynamic ISF** (BG-dependent sensitivity adjustments).

---

## Patient Model Explained

The patient model introduces realistic sources of randomness that challenge the algorithm:

- **Meal variation** — actual carbs consumed are drawn from a normal distribution around
  the average, so a "50g lunch" might be 40g or 60g on any given day
- **Carb counting error** — the patient's declaration to the pump differs from reality.
  Sigma controls random error; bias controls systematic under-counting
- **Absorption error** — declared absorption time may not match the actual rate
- **Undeclared meals** — some meals are eaten but never entered, forcing the algorithm
  to detect and react to unexpected BG rises
- **Sensitivity variation** — daily insulin sensitivity varies lognormally. The algorithm
  doesn't know today's true sensitivity, creating a "basal deficit" that drifts BG
- **Exercise mismatch** — the patient tells the pump about exercise, but the actual
  sensitivity change and duration differ from what was declared
- **Rescue carbs** — when BG drops below a threshold, the patient eats fast-acting carbs.
  You can configure how much, how fast they absorb, cooldown between doses, and what
  fraction the patient declares to the pump (0% = fully invisible to the algorithm)

---

## Importing from Nightscout

The **Import from Nightscout** tab pulls real data from a Nightscout instance and
auto-generates a patient profile:

- Detects recurring meal patterns (timing, average carbs, variability)
- Extracts therapy settings (ISF, CR, basal rate, target)
- Estimates carb counting accuracy and sensitivity variation from historical data
- Detects exercise patterns
- **Records reference BG statistics** (TIR, mean BG, time below 70, etc.) from the
  selected period — these appear alongside simulation results for easy comparison

You can choose a specific date range instead of "last N days". This is useful for:
- Pulling data from a **Loop-only** or **Trio-only** period (the importer detects
  which app uploaded each entry via the `enteredBy` field)
- For **Loop-era data**, per-meal absorption times are extracted from Nightscout
  metadata and used in the profile (instead of a blanket 3h default)
- Analyzing a specific period of interest (e.g., before/after a settings change)

This lets you simulate how different algorithms would perform for a real patient based
on their actual eating and dosing patterns, and directly compare the simulator's output
against the real-world statistics from that period.

---

## Multi-Profile Comparison

You can compare the same algorithm(s) across different patient profiles by selecting
**Additional Profiles to Compare** in the sidebar. Each combination of algorithm and
profile becomes a "variant" — for example, running Loop and Trio against both a
"Real Patient" and "High Carb" profile produces four variants, each plotted and scored
independently.

---

## Reading the Results

### Spaghetti Plot
Each thin line is one simulated day (from all weeks). The thick line is the **median**
across all day-traces. Use the **Day filter** to show only exercise or rest days.
The green band marks the target range (70-180 mg/dL). Tight bundles = consistent
performance; wide spread = high variability.

### Summary Statistics
- **Mean BG** — average blood glucose across all weeks
- **TIR 70-180** — Time In Range: percentage of readings between 70-180 mg/dL (higher is better; clinical target >70%)
- **Time <70 / <54** — time in hypoglycemia / severe hypoglycemia (lower is better; target <4% / <1%)
- **Time >180 / >250** — time in hyperglycemia / severe hyperglycemia (lower is better; target <25%)
- **CV** — Coefficient of Variation: glucose variability as a percentage (lower is better; target <36%)
- **GMI** — Glucose Management Indicator: estimated A1C from mean glucose
- **Hypo events/week** — average distinct hypoglycemia episodes per week (3+ consecutive readings below 70)

### Head-to-Head
When exactly 2 variants are selected, each paired path is compared: which had higher TIR,
which had less time below 70. This controls for randomness — both variants face the
exact same patient on each path.

---

## Glossary

| Term | Meaning |
|------|---------|
| **ISF** | Insulin Sensitivity Factor — how much 1U of insulin lowers BG (mg/dL per unit) |
| **CR** | Carb Ratio — grams of carbs covered by 1U of insulin |
| **DIA** | Duration of Insulin Action — how long insulin stays active (hours) |
| **IOB** | Insulin on Board — active insulin remaining from recent deliveries |
| **COB** | Carbs on Board — carbs still being absorbed |
| **TIR** | Time in Range — % of readings between 70-180 mg/dL |
| **GMI** | Glucose Management Indicator — estimated A1C from average glucose |
| **CV** | Coefficient of Variation — measure of glucose variability |
| **IRC** | Integral Retrospective Correction — Loop's error correction feedback loop |
| **DCA** | Dynamic Carb Absorption — Loop's real-time carb absorption tracking |
| **ICE** | Insulin Counteraction Effects — observed BG change not explained by insulin |
| **SMB** | Super Micro-Bolus — Trio's small correction boluses delivered every few minutes |
| **UAM** | Unannounced Meal — Trio's detection of unexplained glucose rise |
| **Autosens** | Automatic Sensitivity — Trio's 24h rolling sensitivity detection |

---

*For a detailed technical description of the algorithms and patient model, see the
[Technical Guide](https://github.com/jeremybarnum/diabetes-simulator/blob/main/docs/technical_guide.md) on GitHub.*
""")
