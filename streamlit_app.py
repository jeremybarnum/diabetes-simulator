"""
Diabetes Algorithm Simulator — Streamlit Web App

Monte Carlo comparison of Loop and Trio insulin dosing algorithms.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from simulation import PatientProfile, SimulationRun, SimulationRunResult
from monte_carlo import compute_metrics, GlycemicMetrics

# ─── Page config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Diabetes Algorithm Simulator",
    page_icon="📊",
    layout="wide",
)

st.title("Diabetes Algorithm Simulator")
st.markdown("Monte Carlo comparison of **Loop** and **Trio** insulin dosing algorithms")

# ─── Sidebar controls ───────────────────────────────────────────────────────

st.sidebar.header("Simulation Settings")

# Patient profile
profile_dir = Path(__file__).parent / "patient_profiles"
profile_files = sorted(profile_dir.glob("*.json"))
profile_names = {p.stem.replace("_", " ").title(): str(p) for p in profile_files}
selected_profile_name = st.sidebar.selectbox(
    "Patient Profile",
    list(profile_names.keys()),
    index=list(profile_names.keys()).index("Real Patient")
    if "Real Patient" in profile_names else 0,
)
profile_path = profile_names[selected_profile_name]

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
seed = st.sidebar.number_input("Random seed", value=42, min_value=0, max_value=99999)

run_button = st.sidebar.button("Run Simulation", type="primary", use_container_width=True)


# ─── Helper functions ────────────────────────────────────────────────────────

def run_paths(profile, algorithm_name, n_paths, seed):
    """Run n_paths 1-day simulations, return (traces, list of metrics)."""
    traces = []
    all_metrics = []
    for i in range(n_paths):
        path_seed = seed + i * 1000 + hash(algorithm_name) % (2**31)
        rng = np.random.RandomState(path_seed)
        sim = SimulationRun(profile=profile, algorithm_name=algorithm_name,
                            n_days=1, rng=rng)
        result = sim.run()
        metrics = compute_metrics(result)
        all_metrics.append(metrics)

        day = result.days[0]
        trace = [(t_rel / 60, bg) for t_rel, bg in day.bg_trace]
        traces.append(trace)
    return traces, all_metrics


def plot_spaghetti(traces_by_algo, algo_colors):
    """Create spaghetti plot with median lines."""
    fig, ax = plt.subplots(figsize=(14, 7))

    for algo_name, traces in traces_by_algo.items():
        color = algo_colors[algo_name]
        for trace in traces:
            hours = [h for h, _ in trace]
            bgs = [bg for _, bg in trace]
            ax.plot(hours, bgs, color=color, alpha=0.12, linewidth=0.7)

        # Median line
        hours = [h for h, _ in traces[0]]
        bg_matrix = np.array([[bg for _, bg in t] for t in traces])
        medians = np.median(bg_matrix, axis=0)
        label = algo_name.replace("_", " ").upper()
        ax.plot(hours, medians, color=color, linewidth=2.5, label=f"{label} (median)")

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
    ax.set_title(f"24h BG Traces — {n_paths} paths")
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


# ─── Main area ───────────────────────────────────────────────────────────────

ALGO_COLORS = {
    "loop_ab40": "#2563eb",
    "loop_tb": "#7c3aed",
    "loop_ab_gbpa": "#0891b2",
    "trio": "#dc2626",
}

if run_button:
    if not algorithms:
        st.error("Select at least one algorithm.")
    else:
        profile = PatientProfile.from_json(profile_path)

        traces_by_algo = {}
        metrics_by_algo = {}

        progress = st.progress(0, text="Starting simulation...")
        total_algos = len(algorithms)

        for i, algo_name in enumerate(algorithms):
            progress.progress(
                i / total_algos,
                text=f"Running {algo_name} ({n_paths} paths)..."
            )
            traces, metrics = run_paths(profile, algo_name, n_paths, seed)
            traces_by_algo[algo_name] = traces
            metrics_by_algo[algo_name] = metrics

        progress.progress(1.0, text="Done!")

        # Spaghetti plot
        st.subheader("BG Traces")
        fig = plot_spaghetti(traces_by_algo, ALGO_COLORS)
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
    st.info("Configure settings in the sidebar and click **Run Simulation** to start.")
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

    Select a patient profile, choose algorithms, and click Run to see how they compare.
    """)
