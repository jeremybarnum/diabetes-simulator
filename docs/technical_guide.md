# Diabetes Algorithm Simulator — Technical Guide

## Introduction

This simulator performs Monte Carlo comparisons of closed-loop insulin dosing algorithms for Type 1 diabetes. It answers the question: *given a realistic patient with day-to-day variability, how do different algorithms perform over many possible days?*

The simulator implements two algorithms:
- **Loop** — a port of the iOS Loop 3.10.0 prediction and dosing pipeline
- **Trio** — a port of the Trio/oref1 (OpenAPS) algorithm with SMB support

Both algorithms run against the same stochastic patient model, using the same therapy settings (ISF, CR, basal rate, etc.), so differences in outcomes reflect genuine algorithmic differences.

---

## The Loop Algorithm

Loop is a prediction-based closed-loop system originally developed as an iOS app. Every 5 minutes, it builds a single 6-hour BG forecast and calculates how much insulin is needed to bring the predicted eventual BG to target.

### Prediction Pipeline

Loop's prediction is built by combining several effects:

#### 1. Insulin Effect
Computes the expected BG impact of all active insulin (IOB). Uses an exponential insulin model parameterized by peak time and duration of action. For Fiasp: peak at 55 minutes, total action duration of 360 minutes, with a 10-minute delay.

The insulin effect curve shows where BG would go if no carbs were absorbing — typically a downward curve proportional to remaining IOB.

#### 2. Insulin Counteraction Effects (ICE)
ICE compares the *observed* glucose change over recent history against what the insulin model *predicted* should have happened. The difference represents everything the insulin model doesn't account for — primarily carb absorption, but also sensitivity changes, exercise, and sensor noise.

ICE is computed as: `observed_BG_change - predicted_insulin_effect` for each 5-minute interval. This signal feeds into both DCA and IRC.

#### 3. Dynamic Carb Absorption (DCA)
DCA uses ICE to determine whether declared carbs are absorbing faster or slower than the declared absorption rate. It compares the counteraction effects against the expected carb absorption curve (piecewise linear model) and adjusts the remaining carb effect prediction accordingly.

Key behaviors:
- If ICE shows more glucose rise than expected → carbs are absorbing faster → DCA accelerates the predicted remaining absorption
- If ICE shows less rise → absorption is slower → DCA extends the predicted absorption timeline
- The `initialAbsorptionTimeOverrun` parameter (1.5x by default in iOS) controls how much slower-than-declared absorption is tolerated before carbs are considered "missing"

#### 4. Momentum Effect
Linear regression on the last ~15 minutes of CGM readings, projected forward. This gives the algorithm fast-reacting awareness of rapid BG changes (e.g., a post-meal spike or a fast drop). The momentum contribution tapers to zero by ~15 minutes into the prediction, since the other effects dominate the longer-term forecast.

#### 5. Integral Retrospective Correction (IRC)
IRC compares what the model predicted over the past 30-60 minutes against what actually happened, and applies a correction to future predictions. It works by:
1. Computing "discrepancies" — the difference between ICE and the carb effect model — over the retrospective window
2. Integrating these discrepancies into a cumulative correction
3. Projecting the correction forward into the prediction

IRC helps the algorithm adapt to situations the carb and insulin models don't capture, such as unannounced meals, sensitivity shifts, or site absorption issues.

#### 6. Combining Effects
All effects are converted to BG deltas and summed sequentially to build the final 6-hour prediction curve. The "eventual BG" is the endpoint of this curve, and the "minimum predicted BG" is the lowest point.

### Dosing Modes

#### Temp Basal (`loop_tb`)
The traditional Loop dosing mode. Corrections are delivered entirely through temporary basal rate adjustments:
- If predicted BG is above target → increase temp basal (up to max_basal_rate)
- If predicted BG is below target → decrease or zero temp basal
- If minimum predicted BG is below suspend_threshold → zero temp (delivery suspended)

Temp basals are set for 30-minute durations and re-evaluated every 5 minutes.

#### Automatic Bolus 40% (`loop_ab40`)
The recommended Loop dosing mode. Instead of relying solely on temp basals:
- Calculates the total insulin requirement to correct predicted BG to target
- Delivers 40% of this requirement as an immediate micro-bolus
- Sets a reduced temp basal to avoid double-counting
- The 40% fraction means the algorithm self-corrects over multiple cycles — aggressive enough to respond quickly but conservative enough to avoid stacking

### Key Settings
| Setting | Description |
|---------|-------------|
| ISF | Insulin Sensitivity Factor — mg/dL drop per unit |
| CR | Carb Ratio — grams covered per unit |
| Basal Rate | Scheduled background delivery (U/hr) |
| DIA | Duration of Insulin Action (hours) |
| Target | BG target for corrections (mg/dL) |
| Suspend Threshold | Zero-temp if predicted BG drops below this |
| Max Basal Rate | Safety ceiling for temp basals |
| Max Bolus | Maximum single auto-bolus |

---

## The Trio/oref1 Algorithm

Trio uses the OpenAPS oref1 algorithm, originally developed for the DIY open-source artificial pancreas community. Its approach differs fundamentally from Loop: instead of a single combined prediction, it computes four parallel forecasts and uses the most conservative one.

### IOB and COB Pipeline

#### Insulin on Board (IOB)
Trio computes IOB by iterating over recent insulin deliveries (boluses and temp basals), applying an exponential decay model. Each delivery's remaining activity is summed to get total IOB and the current "activity" (rate of insulin effect on BG).

#### Carbs on Board (COB)
COB tracking uses the "deviation" approach:
1. Compute BG deviations — the difference between observed BG change and expected insulin effect (BGI) at each 5-minute interval
2. Positive deviations indicate carb absorption (or other BG-raising effects)
3. COB is decremented as deviations account for the expected carb absorption

This is conceptually similar to Loop's ICE but implemented differently — deviations are classified into categories (carb-related, UAM, non-meal) rather than being continuously integrated.

### Four-Path Predictions

Trio generates four 6-hour BG prediction curves:

#### IOB Prediction
Assumes no more carbs will absorb. BG is driven purely by remaining insulin. This is the most pessimistic forecast for a post-meal scenario (assumes all carbs are done).

#### ZT (Zero Temp) Prediction
Projects what happens if insulin delivery is stopped immediately. Shows the "floor" — the lowest BG could go if the algorithm suspends right now.

#### COB Prediction
Assumes remaining COB will absorb at the declared rate while insulin continues. This is the standard prediction when carbs are active.

#### UAM (Unannounced Meal) Prediction
Detects unexplained BG rises (positive deviations not accounted for by declared carbs) and projects them forward. This allows Trio to respond to meals the user didn't enter, site changes, or other glucose-raising events.

### Decision Logic

Trio's `determine_basal` function evaluates the predictions and decides on an action:

1. **Safety first**: if BG or any minimum predicted BG is below the suspend threshold → zero temp
2. **Falling BG**: if eventual BG is below target and BG is dropping → reduce or zero temp
3. **In range**: if eventual BG is near target → set a neutral temp or cancel the current temp
4. **Rising/high BG**: if eventual BG is above target → calculate insulin required and deliver via high temp basal and/or SMB

### Super Micro-Boluses (SMBs)

SMBs are Trio's aggressive correction mechanism. When the algorithm determines more insulin is needed:
- It calculates the insulin requirement based on the gap between eventual BG and target
- Delivers a fraction (`smb_delivery_ratio`, default 0.5) as an immediate micro-bolus
- SMBs are capped by `maxSMBBasalMinutes` (default 30) — the maximum SMB is `basal_rate × 30/60`
- A minimum interval between SMBs is enforced (default 3 minutes)
- SMBs are rounded down to pump increment (0.1U)

### Autosens

Autosens analyzes the past 24 hours of BG data to detect systematic sensitivity changes:
1. For each historical BG reading, compute the expected insulin effect (BGI) at that time
2. Calculate the deviation: `observed change - expected insulin effect`
3. Classify each interval as carb-related (near meals), UAM (unexplained rise), or non-meal
4. Take the median of non-meal deviations and convert to a sensitivity ratio
5. Apply the ratio to adjust ISF, basal, and target (within configurable bounds, typically 0.8–1.2)

Autosens is recalculated every 30 minutes in the simulation (cached to avoid running on every 5-minute step).

### Dynamic ISF

When enabled, Dynamic ISF makes the sensitivity factor depend on current BG:
- **Logarithmic mode**: ISF scales logarithmically with BG — higher BG means lower ISF (more aggressive corrections)
- **Sigmoid mode**: ISF follows a sigmoid curve, providing smoother scaling that levels off at extreme BG values. The `adjustmentFactorSigmoid` parameter controls aggressiveness

Dynamic ISF is calibrated so that the ratio equals 1.0 at the target BG — the algorithm behaves identically to static ISF when BG is at target.

---

## The Patient Model

The patient model simulates a Type 1 diabetic with realistic day-to-day variability. The algorithm runs against this simulated physiology, creating a realistic test of how well it adapts to uncertainty.

### BG Dynamics

Each 5-minute step, the patient's BG changes based on three components:

1. **Insulin effect**: each unit of active insulin lowers BG by `ISF / S(t)` mg/dL (where `S(t)` is the current sensitivity scalar). The insulin model uses the same exponential curve as the algorithm.

2. **Carb effect**: absorbing carbs raise BG by `grams × CSF × absorption_fraction`, where `CSF = ISF / CR`. A 10-minute absorption delay is applied. The carb model uses a piecewise linear absorption curve.

3. **Basal deficit**: when the patient's true sensitivity `S(t) ≠ 1`, the scheduled basal rate is wrong for the patient's actual needs. If `S(t) > 1` (more resistant), the pump under-delivers relative to need, causing BG to drift up. If `S(t) < 1` (more sensitive), the pump over-delivers. This is modeled as a "virtual insulin" injection the algorithm cannot see.

BG is clamped to [39, 400] mg/dL. When BG drops below the rescue threshold (default 70 mg/dL), the patient automatically eats rescue carbs. Rescue behavior is fully configurable: the threshold, carb amount (default 8g), absorption time (default 1 hour), cooldown between doses (default 15 minutes), and what percentage of rescue carbs the patient declares to the pump (default 0% — fully invisible to the algorithm). Setting the declared percentage above zero models a patient who enters their rescue carbs, giving the algorithm better information but also introducing potential insulin stacking.

### Sources of Randomness

#### Meal Size Variation
Each declared meal's actual carb content is drawn from `Normal(mean, sd)` (clamped to ≥2g). A meal with mean 50g and SD 10g might be anywhere from ~30g to ~70g on a given day.

#### Carb Counting Error
The patient's carb declaration to the pump differs from reality:
- **Sigma** (random error): declared carbs = `actual × exp(Normal(-bias, sigma))`. At sigma=0.15, declarations are typically 85-115% of actual.
- **Bias** (systematic under-counting): a bias of 0.1 means the median declaration is ~90% of actual. This simulates the common tendency to undercount.

#### Absorption Time Error
The declared absorption time is randomly perturbed: `declared_absorption = actual × exp(Normal(0, sigma))`. This models uncertainty in how fast carbs will actually absorb.

#### Undeclared Meals
Two mechanisms:
- **Probability**: each regular meal has a configurable chance of going completely undeclared (eaten but not entered in the pump)
- **Undeclared meal list**: a separate list of meals that are always eaten but never declared (e.g., a regular afternoon snack)

#### Insulin Sensitivity Variation
Each simulated day draws a sensitivity scalar: `S ~ lognormal(0, sigma)`. During waking hours (7am–9pm), `S` is constant. From 9pm to 7am, `S` exponentially decays back toward 1.0 (half-life ~3 hours), modeling the overnight normalization of sensitivity.

When `S > 1`, the patient is more insulin resistant than the algorithm assumes — the algorithm under-delivers, and BG drifts up. When `S < 1`, the patient is more sensitive — the algorithm over-delivers, and BG drifts down. The algorithm must detect and adapt to these shifts using its prediction and correction mechanisms.

#### Exercise
On exercise days (probability = `exercises_per_week / 7`):
- The algorithm is told about exercise with a declared sensitivity scalar and duration
- The patient's actual sensitivity change is drawn from a lognormal distribution around `actual_scalar_mean`
- The actual duration is drawn from a lognormal distribution around `actual_duration_hrs_mean`
- The mismatch between declared and actual creates a realistic challenge — the algorithm knows exercise happened but doesn't know exactly how much sensitivity changed or how long it will last

---

## Monte Carlo Framework

### How Paths Work

Each "path" is an independent simulation of one or more days:
1. A random seed is computed from the base seed + path index + algorithm hash
2. Random draws are made for: meal sizes, carb counting errors, absorption times, sensitivity scalar, undeclared meal decisions, exercise occurrence
3. The simulation runs minute-by-minute (with 5-minute algorithm steps) from 7am through the configured number of days
4. BG trace and all insulin/carb history are recorded

### Metrics Collection

After each path completes, standard glycemic metrics are computed from all 5-minute BG readings:
- Time in Range (70-180 mg/dL)
- Time below 70 and below 54 (hypoglycemia)
- Time above 180 and above 250 (hyperglycemia)
- Mean BG, SD, CV
- GMI (estimated A1C)
- Hypo events (≥3 consecutive readings below 70)

Summary statistics (mean, median, percentiles) are computed across all paths.

### Fair Comparison Design

When comparing two algorithms, fairness is ensured by:
- **Same patient, same day**: both algorithms face the exact same random draws for each path (same meal sizes, same sensitivity, same carb counting errors). The seed includes the algorithm name so each gets different randomness between algorithms, but the patient-level randomness is controlled.
- **Same settings**: both use the same ISF, CR, basal rate, DIA, and target
- **Head-to-head pairing**: each path is compared directly — "did Loop or Trio achieve higher TIR on this particular day?" — rather than just comparing averages

---

## Validation

### Loop Validation
The Loop algorithm implementation has been validated against iOS Loop 3.10.0:
- **LoopTestRunner**: a Swift tool that injects scenarios into the iOS Loop simulator and captures prediction outputs
- **78 predictions match exactly** (0.000 mg/dL difference) against a live_capture fixture from Loop 3.10.0
- A regression suite of 6 iOS-validated scenarios runs on every change
- An additional 14 scenarios are in various stages of iOS validation

### Trio Validation
The Trio/oref1 implementation has been validated against the actual JavaScript source:
- **Ground truth runner**: a Node.js wrapper that calls the real trio-oref `determine-basal.js`
- **Phase-by-phase validation**: IOB (0.000 max diff), COB (0 diff), predictions (1-3 mg/dL max for COB/UAM, 0 for ZT), determine_basal (9/10 exact match, 1 test with 0.05 rounding diff)
- Autosens and Dynamic ISF validated separately against JS outputs
- 10 regression baselines locked against JS ground truth

---

## Development Process

This project was built iteratively over about six weeks (late January to early March 2026), with each algorithm ported from its original source code and validated against the real system before being integrated into the simulator. The entire codebase — algorithm ports, patient model, simulation engine, validation infrastructure, and UI — was developed with Claude Code as a pair-programming partner.

### Porting the Loop Algorithm

The Loop algorithm was ported from Swift (LoopKit) to Python, working module by module through the prediction pipeline:

1. **Insulin and carb math** came first — the exponential insulin model and piecewise linear carb absorption model, validated against LoopKit's own test fixtures
2. **DCA (Dynamic Carb Absorption)** was the most complex component, requiring deep study of LoopKit's `CarbStatus.swift` and `DynamicCarbAbsorption.swift` to understand the three-mode COB calculation and ICE splitting logic
3. **Momentum, IRC, and dosing logic** completed the prediction pipeline
4. **The `subtracting()` rewrite** (February 15) was a critical bug fix — Python's IRC discrepancy calculation didn't match iOS's `LoopMath.swift:279-334` due to two subtle issues: mutual trim logic on misaligned grids, and a fixed 5-minute effect interval that iOS uses regardless of actual ICE interval duration

The port went through several iterations where behavior appeared correct on simple tests but diverged from iOS on complex scenarios, requiring progressively deeper study of the Swift source.

### Validating Loop Against iOS

The hardest engineering challenge was building infrastructure to validate the Python port against the real iOS Loop app running in the Xcode simulator. The problem: iOS Loop reads all its data exclusively from HealthKit. There is no test API or dependency injection point.

**The solution was a custom iOS "injector" app** (`HealthKitInjectorApp`) that runs in the same simulator alongside Loop:

1. **HealthKit injection**: The injector app writes glucose samples, carb entries, and insulin doses directly into HealthKit with the correct metadata — CGM-style device tags (not "manually entered," which Loop ignores), LoopKit absorption time keys, Fiasp insulin type metadata, and proper origin flags
2. **Command-file protocol**: Python drives the injector via the simulator's shared filesystem. It writes scenario JSON files into the app's sandbox, then drops command files (`clear_all`, `inject:<filename>`) that the injector polls for every second. Python confirms execution by watching for the command file to disappear
3. **Modified Loop build**: A custom build of iOS Loop (Loop 3.10.0, bundle ID `com.Exercise.Loop`) with added `NSLog()` statements tagged with `##LOOP##` markers that emit structured data — therapy settings, intermediate effects (ICE, momentum, IRC), and the full prediction array
4. **Log capture**: After injecting data and waiting for Loop to compute (~6-8 seconds), `batch_validate.py` extracts the simulator's log stream via `xcrun simctl spawn ... log show`, parses the structured `##LOOP##` lines, and compares iOS's predictions against the Python port

This yielded two levels of validation:
- **LoopTestRunner** (a Swift CLI tool calling LoopKit directly) validated algorithm math against offline fixtures: 78 predictions match at 0.000 mg/dL
- **batch_validate.py** (the full iOS pipeline) validated against the running app including HealthKit data ingestion, with 6 scenarios passing at ≤1.0 mg/dL tolerance

A key workaround: iOS Loop won't compute ICE (and thus IRC) without insulin history in the dose store. Tests that don't naturally include insulin use a 0.2U "trigger bolus" at t-180 minutes — small enough not to affect predictions, but sufficient to activate the ICE pipeline.

### Porting the Trio/oref1 Algorithm

Trio was ported from JavaScript (OpenAPS's `oref0` library) to Python. The approach was phase-by-phase with a ground truth runner:

1. **Ground truth runner**: A Node.js wrapper (`trio_runner.js`) that calls the actual `determine-basal.js` from Trio's source. Python converts scenarios to Trio's JSON format, shells out to Node.js, and compares results. This made validation much faster than the iOS approach — no simulator, no HealthKit, just function-call-level comparison
2. **IOB pipeline**: Ported the insulin-on-board calculation and validated to 0.000 max difference
3. **Deviation COB**: Ported the deviation-based carb tracking (conceptually similar to Loop's ICE but classified into carb/UAM/non-meal categories)
4. **Predictions**: Ported all four prediction paths (IOB, ZT, COB, UAM). ZT matched exactly; COB and UAM had 1-3 mg/dL differences due to floating-point ordering
5. **determine_basal**: Ported the full decision tree — 9 of 10 test scenarios produced exact rate/duration/SMB matches, with 1 test showing a 0.05 U/hr rounding difference
6. **Autosens and Dynamic ISF**: Ported the 24-hour sensitivity analysis and both logarithmic and sigmoid ISF scaling modes

Key gotchas discovered during the Trio port:
- `console.log` redirect: `determine-basal.js` uses `console.log` (not `console.error`) for debug output, so the Node.js wrapper had to redirect stdout to stderr to keep it out of the JSON result
- **CGM unchanged guard**: oref1 exits early if all deltas are zero — test scenarios need non-zero BG changes
- **Top-of-hour guard**: oref1 cancels temps if minutes ≥ 55 (an OpenAPS safety feature) — test scenarios use :30 past the hour

### Building the Patient Model

The patient model evolved through several iterations:

1. **Phase 1** attempted to use `simglucose` (the FDA-approved UVA/Padova simulator) but it proved too complex and opaque for the comparison use case
2. **A custom patient model** was built instead, using the same insulin and carb math as the algorithms. This is intentional — the patient's physiology uses identical insulin curves, so any prediction errors come from the *information gap* (carb counting, sensitivity variation) rather than model mismatch
3. **Stochastic infrastructure** added lognormal variation to carb amounts, absorption times, and insulin sensitivity, with proper dual tracking of declared vs. actual parameters
4. **The delta-based approach** (the most recent commit before documentation) was a critical fix for multi-day simulations: instead of computing absolute BG from cumulative insulin/carb effects (which drifted over days due to floating-point accumulation), each step computes only the incremental BG change

### Streamlit UI

Streamlit was chosen for the web UI because it provides a fast path from Python simulation code to an interactive app with minimal frontend work. Key implementation choices:

- **Tabbed layout** separates results, patient model editing, algorithm settings, Nightscout import, and help, keeping the interface manageable despite many parameters
- **`st.data_editor`** for meal tables allows users to add, remove, and edit meals directly in the browser
- **Session state management** bridges the gap between Streamlit's rerun model and the need to persist editable state — profile loading writes directly to `st.session_state` keys that widgets read via their `key=` parameters
- **Patient profiles** are stored as JSON files and can be saved/loaded, making it easy to share and reproduce scenarios
- **Nightscout import** pulls real meal/bolus/CGM data from a Nightscout instance and auto-generates a patient profile with detected meal patterns, therapy settings, and estimated variability parameters
- **Multi-profile comparison** allows selecting additional profiles in the sidebar to compare the same algorithm(s) across different patients. Each algorithm × profile combination becomes a "variant" with its own trace and metrics
- **Multiple insulin types** are supported (Humalog/Novolog, Fiasp, Lyumjev, rapid-acting child, Afrezza), each with different action curve parameters

The app is deployed on Streamlit Community Cloud, which auto-deploys from the GitHub repository.

### Modal for Cloud Parallelization

The Monte Carlo simulation is embarrassingly parallel — each path is independent — but running hundreds of paths locally is slow. Modal (a serverless compute platform) was integrated to fan out paths across cloud workers:

- **Architecture**: `monte_carlo_cloud.py` wraps the same `_run_single_path()` function from `monte_carlo.py`. Each cloud worker receives one `(seed, profile, algorithms, n_days)` tuple, runs the simulation, and returns metrics. No algorithm code is duplicated
- **Fan-out**: Modal's `starmap` dispatches all N paths simultaneously. Results stream back as workers complete, enabling real-time progress reporting
- **Container setup**: The entire project directory is synced into a Debian slim container with numpy and scipy. Each worker is allocated 1 CPU and 512MB — no GPU needed since the simulation is pure Python math
- **Reproducibility**: Seeding is identical to the local runner, so the same seed produces the same results whether run locally or in the cloud

Modal is available through two execution paths:
- **Streamlit app**: When Modal is installed, the app automatically dispatches paths to cloud workers instead of running locally. If Modal is unavailable or fails, it falls back to local execution transparently. A progress bar shows cloud completion in real time
- **CLI**: `modal run monte_carlo_cloud.py` provides the same cloud parallelism for batch runs outside the UI

### Role of Claude Code

The entire project was developed using Claude Code (Anthropic's AI coding assistant) as a pair-programming partner. Claude Code wrote the majority of the code across all components — algorithm ports, validation infrastructure, patient model, simulation engine, Streamlit UI, and Modal integration — working from the original Swift and JavaScript source code, iOS documentation, and iterative debugging against the validation harnesses. The human role was primarily architectural direction, clinical domain knowledge, and reviewing/testing the outputs.

---

## Design Choices

### Why Fiasp as Default?
Fiasp (faster-acting insulin aspart) was the original default insulin type because its faster onset and shorter peak make closed-loop performance differences more apparent. The simulator now supports multiple insulin types (Humalog/Novolog, Fiasp, Lyumjev, rapid-acting child, Afrezza), each with different exponential model parameters. The selected insulin type is used consistently across both algorithms.

### Why Piecewise Linear Carbs?
The carb absorption model uses a piecewise linear curve (matching iOS Loop's "nonlinear" mode, which is actually piecewise linear). This was chosen because it matches the validated iOS implementation and provides a reasonable approximation of real absorption dynamics with a simple, predictable shape.

### Why Same Settings for Both Algorithms?
Using identical therapy settings (ISF, CR, basal, DIA, target) for both Loop and Trio ensures that performance differences reflect the algorithms themselves, not tuning. In practice, users might optimize settings differently for each system, but a fair baseline comparison requires a level playing field.

### Why Start Each Day Fresh?
Single-day simulations start with no active insulin or carbs at 7am. This eliminates carry-over effects that could confound comparisons. Multi-day mode is available for scenarios where carry-over matters (e.g., autosens needs 24h of history to be meaningful).

### Why Rescue Carbs?
The simulated patient automatically eats rescue carbs when BG drops below a configurable threshold (default 70 mg/dL). This prevents unrealistically severe hypoglycemia and models real patient behavior — nobody sits still while their BG crashes. By default, rescue carbs are undeclared, so the algorithm must detect and respond to the unexpected glucose rise. The rescue system is fully configurable: threshold, carb amount, absorption speed, cooldown between doses, and what fraction the patient declares to the pump.
