# Diabetes Algorithm Simulator

Monte Carlo comparison of **Loop** and **Trio** (OpenAPS/oref1) insulin dosing algorithms using a stochastic patient model. Built for the T1D community to explore how different algorithms handle real-world variability: imperfect carb counting, exercise, daily sensitivity swings, and undeclared meals.

## What it does

- Runs hundreds of simulated days through validated Python ports of Loop and Trio
- Stochastic patient model with configurable meals, exercise, sensitivity variation, and carb counting skill
- Produces spaghetti plots (24h BG traces), summary statistics (TIR, hypo time, mean BG), and month-long comparisons
- Optional Nightscout overlay to compare simulated vs real data
- Web UI via Streamlit or CLI scripts

## Quick Start

```bash
pip install -r requirements.txt

# Web app
streamlit run streamlit_app.py

# CLI: 50-path spaghetti plot
python3 plot_traces.py --paths 50

# CLI: Monte Carlo comparison (10 paths x 1 day)
python3 monte_carlo.py --paths 10 --days 1

# CLI: specific patient profile
python3 plot_traces.py --paths 30 --profile patient_profiles/poor_carb_counter.json
```

## CLI Tools

| Script | Purpose |
|---|---|
| `plot_traces.py` | Spaghetti chart: Loop vs Trio BG traces over 24h |
| `monte_carlo.py` | Full Monte Carlo with summary stats and head-to-head comparison |
| `plot_month.py` | 30-day Nightscout vs simulated Trio side-by-side |
| `compare_algorithms.py` | Deterministic single-scenario algorithm comparison |
| `predict.py` | Single-point prediction from a scenario |

## Patient Profiles

Seven profiles in `patient_profiles/`:

| Profile | Description |
|---|---|
| `default.json` | 3 meals/day, moderate variability |
| `real_patient.json` | Calibrated from Nightscout data (~70g/day, exercise 4x/week) |
| `real_patient_no_carbs.json` | Same patient, no carbs declared to pump |
| `poor_carb_counter.json` | High carb counting error (sigma=0.35) |
| `well_controlled.json` | Low variability, accurate carb counting |
| `active_exerciser.json` | Frequent exercise with sensitivity mismatch |
| `variable_sensitivity.json` | High day-to-day insulin sensitivity variation |

## Algorithms

**Loop AB40** — Python port of [Loop](https://github.com/LoopKit/Loop) 3.x auto-bolus mode:
- Exponential insulin model (Fiasp), momentum effects, insulin counteraction effects
- Dynamic carb absorption (DCA), retrospective correction
- Validated against iOS Loop simulator (6 regression tests, exact match)

**Trio** — Python port of [Trio](https://github.com/nightscout/Trio) / [oref1](https://github.com/openaps/oref0):
- IOB array, deviation-based COB, UAM detection
- SMB dosing with configurable safety limits
- 4 prediction curves (ZT, IOB, COB, UAM)
- Validated against actual oref1 JavaScript (10 regression tests, exact match)

## Project Structure

```
diabetes_simulator/
├── algorithms/              # Core algorithm implementations
│   ├── base.py              # Shared interfaces
│   ├── loop/                # Loop: prediction, dosing, insulin models, DCA
│   └── openaps/             # Trio/oref1: IOB, COB, predictions, determine_basal
├── patient_profiles/        # 7 JSON patient configurations
├── simulation.py            # Multi-day simulation engine
├── stochastic_patient.py    # Stochastic patient model (sensitivity, exercise, meals)
├── monte_carlo.py           # Monte Carlo framework with parallel execution
├── plot_traces.py           # Spaghetti chart generation
├── plot_month.py            # Month-long comparison plots
├── compare_algorithms.py    # Deterministic algorithm comparison
├── predict.py               # Single prediction entry point
├── nightscout_query.py      # Nightscout CGM data fetching
├── trio_json_exporter.py    # Trio JSON format converter
├── streamlit_app.py         # Web application
├── settings.json            # Default pump settings (ISF, CR, basal, DIA)
├── my_settings.py           # Settings wrapper
├── _validation/             # iOS and Trio validation tooling (not needed to run)
└── docs/                    # Internal implementation notes
```

## Validation

The Python implementations are validated against their reference implementations:

- **Loop**: 6 iOS-validated regression tests against Loop 3.10.0 in Xcode simulator. LoopTestRunner matches all 78 predictions exactly (0.000 mg/dL) against a live capture fixture.
- **Trio**: 10 JS-validated regression tests against actual oref1 JavaScript. IOB matches exactly, predictions within 1-3 mg/dL, determine_basal decisions match 9/10 (1 rounding diff).

Validation tooling is in `_validation/` (requires iOS simulator / Node.js).

## Safety Notice

**This is a simulation tool for educational and research purposes only.**

- NOT for medical decision-making
- NOT validated for clinical use
- NOT a substitute for healthcare professional advice
- Algorithm implementations are faithful ports but may have edge-case differences
- Simulated results should not influence real diabetes management decisions

## References

- [Loop](https://loopkit.github.io/loopdocs/) — iOS automated insulin delivery
- [Trio](https://github.com/nightscout/Trio) — iOS port of OpenAPS
- [OpenAPS oref0](https://openaps.readthedocs.io/) — reference algorithm
- [LoopKit](https://github.com/LoopKit/LoopKit) — Loop algorithm source

## License

MIT
