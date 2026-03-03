"""
Multi-day simulation engine for Monte Carlo algorithm comparison.

Orchestrates:
- PatientProfile: configuration for a simulated patient (meal schedule, uncertainty, etc.)
- SimulationDay: runs one day with random meal/exercise events
- SimulationRun: chains days together, carrying forward insulin history
"""

import json
import math
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timezone

from stochastic_patient import (
    EnhancedPatientModel,
    SensitivityModel,
    ExerciseEvent,
    MealEvent,
)
from algorithms.base import AlgorithmInput, AlgorithmOutput
from algorithms.loop.loop_algorithm import LoopAlgorithm
from algorithms.loop.insulin_models_exact import create_insulin_model, InsulinType
from algorithms.openaps.iob import generate_iob_array
from algorithms.openaps.cob import recent_carbs
from algorithms.openaps.glucose_stats import get_last_glucose
from algorithms.openaps.determine_basal import determine_basal
from trio_json_exporter import TrioJSONExporter


# ─── Patient Profile ──────────────────────────────────────────────────────────

@dataclass
class MealSpec:
    """Specification for a recurring daily meal."""
    time_of_day_minutes: int   # Minutes from 7am (day start)
    carbs_mean: float          # Mean actual carbs (grams)
    carbs_sd: float            # SD of actual carbs (grams)
    absorption_hrs: float = 3.0  # True mean absorption time (hours)


@dataclass
class ExerciseSpec:
    """Specification for exercise events."""
    time_of_day_minutes: int = 600  # Default 5pm (600 min from 7am)
    declared_scalar: float = 0.5
    declared_duration_hrs: float = 4.0
    actual_scalar_mean: float = 0.5
    actual_scalar_sigma: float = 0.1  # Lognormal sigma for actual scalar
    actual_duration_hrs_mean: float = 7.0
    actual_duration_hrs_sigma: float = 0.15  # Lognormal sigma for duration


@dataclass
class PatientProfile:
    """Complete configuration for a simulated patient."""

    # Meal schedule
    meals: List[MealSpec] = field(default_factory=list)

    # Carb estimation skill (lognormal sigma)
    carb_count_sigma: float = 0.15    # Random error in carb estimation
    carb_count_bias: float = 0.0      # Systematic bias: >0 means under-declares
                                       # declared = actual * exp(N(-bias, sigma))
                                       # e.g., bias=0.2 -> median declaration is ~82% of actual
    absorption_sigma: float = 0.15    # How well patient estimates absorption time

    # Undeclared meals
    undeclared_meal_prob: float = 0.0  # Probability a meal goes completely undeclared
    undeclared_meals: List[MealSpec] = field(default_factory=list)  # Extra undeclared snacks

    # Daily sensitivity variation
    sensitivity_sigma: float = 0.15

    # Exercise
    exercises_per_week: float = 0.0
    exercise_spec: Optional[ExerciseSpec] = None

    # Starting BG
    starting_bg: float = 120.0

    # Algorithm settings (pump programming) — loaded from settings.json if None
    algorithm_settings: Optional[Dict] = None

    @classmethod
    def from_json(cls, path: str) -> 'PatientProfile':
        """Load profile from JSON file."""
        with open(path) as f:
            data = json.load(f)

        meals = [MealSpec(**m) for m in data.get('meals', [])]
        exercise_spec = None
        if 'exercise_spec' in data:
            exercise_spec = ExerciseSpec(**data['exercise_spec'])

        settings = data.get('algorithm_settings')
        if settings is None:
            settings_path = Path(__file__).parent / 'settings.json'
            with open(settings_path) as f:
                settings = json.load(f)

        undeclared_meals = [MealSpec(**m) for m in data.get('undeclared_meals', [])]

        return cls(
            meals=meals,
            carb_count_sigma=data.get('carb_count_sigma', 0.15),
            carb_count_bias=data.get('carb_count_bias', 0.0),
            absorption_sigma=data.get('absorption_sigma', 0.15),
            undeclared_meal_prob=data.get('undeclared_meal_prob', 0.0),
            undeclared_meals=undeclared_meals,
            sensitivity_sigma=data.get('sensitivity_sigma', 0.15),
            exercises_per_week=data.get('exercises_per_week', 0.0),
            exercise_spec=exercise_spec,
            starting_bg=data.get('starting_bg', 100.0),
            algorithm_settings=settings,
        )

    def get_settings(self) -> Dict:
        """Get algorithm settings, loading defaults if needed."""
        if self.algorithm_settings is not None:
            return self.algorithm_settings
        with open(Path(__file__).parent / 'settings.json') as f:
            return json.load(f)


# ─── Simulation Result ────────────────────────────────────────────────────────

@dataclass
class DayResult:
    """Results from one simulated day."""
    day_index: int
    bg_trace: List[Tuple[float, float]]  # [(time_min_from_sim_start, bg), ...]
    bolus_history: List[Tuple[float, float]]
    meals: List[MealEvent]
    exercises: List[ExerciseEvent]
    sensitivity_trace: List[Tuple[float, float]]  # [(time, scalar), ...]

    def to_dict(self) -> Dict:
        return {
            'day_index': self.day_index,
            'bg_trace': self.bg_trace,
            'bolus_history': self.bolus_history,
            'meals': [
                {'time_minutes': m.time_minutes, 'actual_carbs': m.actual_carbs,
                 'actual_absorption_hrs': m.actual_absorption_hrs,
                 'declared_carbs': m.declared_carbs,
                 'declared_absorption_hrs': m.declared_absorption_hrs,
                 'undeclared': m.undeclared}
                for m in self.meals
            ],
            'exercises': [
                {'start_time_minutes': e.start_time_minutes,
                 'declared_scalar': e.declared_scalar,
                 'declared_duration_hrs': e.declared_duration_hrs,
                 'actual_scalar': e.actual_scalar,
                 'actual_duration_hrs': e.actual_duration_hrs}
                for e in self.exercises
            ],
            'sensitivity_trace': self.sensitivity_trace,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> 'DayResult':
        meals = [MealEvent(**m) for m in d.get('meals', [])]
        exercises = [ExerciseEvent(**e) for e in d.get('exercises', [])]
        return cls(
            day_index=d['day_index'],
            bg_trace=[tuple(p) for p in d['bg_trace']],
            bolus_history=[tuple(p) for p in d.get('bolus_history', [])],
            meals=meals,
            exercises=exercises,
            sensitivity_trace=[tuple(p) for p in d.get('sensitivity_trace', [])],
        )


@dataclass
class SimulationRunResult:
    """Results from a multi-day simulation run."""
    algorithm_name: str
    days: List[DayResult]
    settings: Dict

    def all_bg_values(self) -> List[float]:
        """Flat list of all BG readings."""
        bgs = []
        for day in self.days:
            bgs.extend(bg for _, bg in day.bg_trace)
        return bgs

    def bg_trace(self) -> List[Tuple[float, float]]:
        """Full (time, bg) trace across all days."""
        trace = []
        for day in self.days:
            trace.extend(day.bg_trace)
        return trace

    def to_dict(self) -> Dict:
        return {
            'algorithm_name': self.algorithm_name,
            'days': [d.to_dict() for d in self.days],
            'settings': self.settings,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> 'SimulationRunResult':
        return cls(
            algorithm_name=d['algorithm_name'],
            days=[DayResult.from_dict(day) for day in d['days']],
            settings=d.get('settings', {}),
        )

    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path: str) -> 'SimulationRunResult':
        with open(path) as f:
            return cls.from_dict(json.load(f))


# ─── Result Storage ──────────────────────────────────────────────────────────

SIM_RESULTS_DIR = Path(__file__).parent / 'sim_results'


def save_batch_results(
    results: List['SimulationRunResult'],
    label: str,
    metadata: Optional[Dict] = None,
) -> str:
    """Save a batch of simulation results to sim_results/.

    Args:
        results: List of SimulationRunResult (one per path).
        label: Descriptive label for the filename (e.g. 'loop_ab40_50paths').
        metadata: Optional dict of run parameters to store alongside.

    Returns:
        Path to the saved file.
    """
    SIM_RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'{timestamp}_{label}.json'
    path = SIM_RESULTS_DIR / filename

    data = {
        'metadata': metadata or {},
        'results': [r.to_dict() for r in results],
    }
    with open(path, 'w') as f:
        json.dump(data, f)

    print(f'Saved {len(results)} results to {path}')
    return str(path)


def load_batch_results(path: str) -> Tuple[List['SimulationRunResult'], Dict]:
    """Load a batch of simulation results.

    Returns:
        (list of SimulationRunResult, metadata dict)
    """
    with open(path) as f:
        data = json.load(f)
    results = [SimulationRunResult.from_dict(r) for r in data['results']]
    return results, data.get('metadata', {})


# ─── Algorithm Wrappers ──────────────────────────────────────────────────────

def create_loop_algorithm(settings: Dict, mode: str = 'temp_basal') -> LoopAlgorithm:
    """Create Loop algorithm instance.

    Args:
        settings: Therapy settings
        mode: 'temp_basal', 'auto_bolus', or 'auto_bolus_gbpa'
    """
    loop_settings = {
        'insulin_sensitivity_factor': settings['insulin_sensitivity_factor'],
        'duration_of_insulin_action': settings['duration_of_insulin_action'],
        'basal_rate': settings['basal_rate'],
        'target_range': (settings['target'], settings['target']),
        'carb_ratio': settings['carb_ratio'],
        'max_basal_rate': settings['max_basal_rate'],
        'max_bolus': settings.get('max_bolus', 5.0),
        'suspend_threshold': settings['suspend_threshold'],
        'enable_irc': True,
        'enable_momentum': True,
        'enable_dca': True,
        'insulin_type': settings.get('insulin_type', 'fiasp'),
        'use_realistic_dosing': False,
    }
    if mode in ('auto_bolus', 'auto_bolus_gbpa'):
        loop_settings['dosing_mode'] = 'automatic_bolus'
        loop_settings['enable_gbpa'] = (mode == 'auto_bolus_gbpa')
    return LoopAlgorithm(loop_settings)


def get_loop_recommendation(
    loop_algo: LoopAlgorithm,
    current_bg: float,
    current_time: float,
    cgm_history: List[Tuple[float, float]],
    bolus_history: List[Tuple[float, float]],
    carb_entries: List[Tuple[float, float, float]],
    settings: Dict,
) -> AlgorithmOutput:
    """Get Loop dosing recommendation."""
    alg_input = AlgorithmInput(
        cgm_reading=current_bg,
        timestamp=int(current_time),
        cgm_history=[(int(t), bg) for t, bg in cgm_history],
        current_basal=settings['basal_rate'],
        temp_basal=None,
        bolus_history=[(int(t), u) for t, u in bolus_history],
        carb_entries=[(int(t), g, a) for t, g, a in carb_entries],
        settings=settings,
    )
    return loop_algo.recommend(alg_input)


def get_trio_recommendation(
    current_bg: float,
    current_time_min: float,
    cgm_history: List[Tuple[float, float]],
    bolus_history: List[Tuple[float, float]],
    carb_entries: List[Tuple[float, float, float]],
    settings: Dict,
    exporter: TrioJSONExporter,
) -> Dict[str, Any]:
    """Get Trio dosing recommendation."""
    profile = exporter.build_profile(int(current_time_min) * 60)
    if 'max_iob' not in profile or profile['max_iob'] is None:
        profile['max_iob'] = 3.5

    now_ms = int(current_time_min) * 60 * 1000

    pump_history = []
    for t_min, units in bolus_history:
        iso = datetime.fromtimestamp(int(t_min) * 60, tz=timezone.utc).strftime(
            '%Y-%m-%dT%H:%M:%S.000Z')
        pump_history.append({'_type': 'Bolus', 'timestamp': iso, 'amount': units})

    glucose_data = []
    for t_min, bg in cgm_history:
        glucose_data.append({
            'glucose': bg, 'sgv': bg,
            'date': int(t_min) * 60 * 1000,
            'dateString': datetime.fromtimestamp(
                int(t_min) * 60, tz=timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.000Z'),
        })
    glucose_data.sort(key=lambda x: x['date'], reverse=True)

    carb_treatments = []
    for t_min, grams, abs_hrs in carb_entries:
        iso = datetime.fromtimestamp(int(t_min) * 60, tz=timezone.utc).strftime(
            '%Y-%m-%dT%H:%M:%S.000Z')
        carb_treatments.append({
            'carbs': grams, 'nsCarbs': grams,
            'timestamp': iso, 'created_at': iso,
        })

    bp = exporter.build_basal_profile()
    iob_array = generate_iob_array(pump_history, profile, now_ms)
    iob_data = iob_array[0] if iob_array else {'iob': 0, 'activity': 0}

    meal_data = recent_carbs(
        treatments=carb_treatments, time_ms=now_ms, profile=profile,
        glucose_data=glucose_data, pump_history=pump_history, basalprofile=bp,
    )

    glucose_status = get_last_glucose(glucose_data)
    if not glucose_status:
        glucose_status = {
            'glucose': current_bg, 'delta': 0,
            'short_avgdelta': 0, 'long_avgdelta': 0, 'date': now_ms,
        }

    return determine_basal(
        glucose_status=glucose_status,
        currenttemp={'rate': 0, 'duration': 0},
        iob_data=iob_data,
        profile=profile,
        meal_data=meal_data,
        iob_array=iob_array,
        micro_bolus_allowed=True,
        clock_ms=now_ms,
    )


# ─── Simulation Engine ───────────────────────────────────────────────────────

class SimulationRun:
    """
    Run a multi-day simulation for one algorithm.

    Carries forward insulin history across days. Each 5-min step:
    1. Compute true BG from patient model
    2. Feed BG to algorithm (with declared carbs/settings)
    3. Record algorithm's dosing decision
    4. Advance
    """

    # Epoch offset so times are valid ISO timestamps for Trio
    EPOCH_OFFSET_SEC = 1739622600  # 2025-02-15T12:30:00Z

    def __init__(
        self,
        profile: PatientProfile,
        algorithm_name: str,
        n_days: int = 7,
        rng: np.random.RandomState = None,
    ):
        self.profile = profile
        self.algorithm_name = algorithm_name
        self.n_days = n_days
        self.rng = rng or np.random.RandomState()
        self.settings = profile.get_settings()

        # Time base: t0 in minutes
        self.t0_min = self.EPOCH_OFFSET_SEC // 60

    def run(self) -> SimulationRunResult:
        """Execute the full multi-day simulation."""
        settings = self.settings
        basal_rate = settings['basal_rate']

        # Build sensitivity model
        sensitivity = SensitivityModel(
            sigma_day=self.profile.sensitivity_sigma,
            rng=self.rng,
        )

        # Generate exercise events for the full run
        exercise_events = self._generate_exercise_events()

        # Build patient model
        patient = EnhancedPatientModel(
            settings=settings,
            starting_bg=self.profile.starting_bg,
            sensitivity_model=sensitivity,
            exercise_events=exercise_events,
        )

        # Generate all meals for all days
        all_meals = self._generate_meals()

        # Build algorithm
        algo, exporter = self._create_algorithm()

        # Insulin model for IOB computation
        insulin_model = create_insulin_model(InsulinType.FIASP,
                                             settings['duration_of_insulin_action'])

        # State carried across days
        bolus_history: List[Tuple[float, float]] = []
        # Declared carb entries (what algorithm sees)
        declared_carbs: List[Tuple[float, float, float]] = []
        # Actual carb entries (what patient absorbs)
        actual_carbs: List[Tuple[float, float, float]] = []
        # Virtual insulin entries from sensitivity mismatch (patient-only)
        basal_deficit_entries: List[Tuple[float, float]] = []

        # Delta-based BG: accumulated step by step
        running_bg = self.profile.starting_bg

        # Pre-seed CGM history (algorithm needs a few points for momentum/deltas)
        cgm_history: List[Tuple[float, float]] = []
        for pre_t in range(-15, 1, 5):
            cgm_history.append((self.t0_min + pre_t, self.profile.starting_bg))

        day_results: List[DayResult] = []
        total_steps = self.n_days * 288  # 288 five-min steps per day

        # Rescue carbs state
        last_rescue_time = -1e9  # last time rescue carbs were eaten
        RESCUE_THRESHOLD = 70    # BG below this triggers rescue
        RESCUE_CARBS = 8.0       # grams
        RESCUE_ABSORPTION = 1.0  # hours
        RESCUE_COOLDOWN = 15     # minutes between rescue doses

        # Pruning cutoff: at 400 min, both insulin (DIA=370min) and carbs
        # (max ~4h absorption) are fully absorbed → delta is 0 → safe to prune.
        PRUNE_AGE_MIN = 400

        sim_start = self.t0_min

        for step in range(total_steps):
            current_time = self.t0_min + step * 5
            day_index = step // 288
            step_in_day = step % 288

            # --- Check for meals at this step ---
            for meal in all_meals:
                if abs(current_time - meal.time_minutes) < 0.1:
                    # Patient always absorbs actual carbs
                    actual_carbs.append((
                        meal.time_minutes,
                        meal.actual_carbs,
                        meal.actual_absorption_hrs,
                    ))

                    if not meal.undeclared:
                        # Patient tells pump about the meal and boluses
                        declared_carbs.append((
                            meal.time_minutes,
                            meal.declared_carbs,
                            meal.declared_absorption_hrs,
                        ))
                        bolus_units = meal.declared_carbs / settings['carb_ratio']
                        if bolus_units > 0:
                            bolus_history.append((current_time, bolus_units))
                    # If undeclared: patient eats but doesn't tell pump, no bolus

            # --- Basal deficit from sensitivity mismatch ---
            # Pump delivers scheduled_basal, but patient needs scheduled_basal * S(t).
            # Deficit = basal * (1 - S(t)) * dt/60 units of "virtual insulin"
            # that the patient experiences but the algorithm doesn't see.
            t_rel = current_time - sim_start
            s_now = patient.get_sensitivity_scalar(t_rel)
            if abs(s_now - 1.0) > 1e-6:
                deficit_units = basal_rate * (1.0 - s_now) * 5.0 / 60.0
                basal_deficit_entries.append((current_time, deficit_units))

            # --- Determine effective basal (exercise reduces delivery) ---
            effective_basal = basal_rate
            for ex in exercise_events:
                if ex.is_declared_active(current_time):
                    effective_basal = basal_rate * ex.declared_scalar
                    break

            # Record basal reduction as negative bolus so both patient model
            # and algorithm see less insulin delivered during exercise
            if effective_basal != basal_rate:
                basal_reduction = (effective_basal - basal_rate) * 5.0 / 60.0
                bolus_history.append((current_time, basal_reduction))

            # --- Compute BG delta for this step ---
            delta = patient.compute_bg_delta(
                current_time, 5.0, bolus_history, basal_deficit_entries,
                actual_carbs, sim_start_time=sim_start,
            )
            running_bg = max(39.0, min(400.0, running_bg + delta))
            bg = running_bg

            # --- Rescue carbs: patient eats if BG < 70 (undeclared) ---
            # Effect starts next step (more realistic — carbs take time)
            if bg < RESCUE_THRESHOLD and (current_time - last_rescue_time) >= RESCUE_COOLDOWN:
                actual_carbs.append((current_time, RESCUE_CARBS, RESCUE_ABSORPTION))
                last_rescue_time = current_time

            cgm_history.append((current_time, bg))

            # --- Get algorithm recommendation ---
            # Algorithm sees the effective (exercise-adjusted) basal rate
            algo_settings = settings
            if effective_basal != basal_rate:
                algo_settings = {**settings, 'basal_rate': effective_basal}

            l_bol, l_rate = 0.0, effective_basal
            try:
                if self.algorithm_name == 'trio':
                    result = get_trio_recommendation(
                        bg, current_time, cgm_history, bolus_history,
                        declared_carbs, algo_settings, exporter,
                    )
                    smb = result.get('units') or 0.0
                    rate_val = result.get('rate')
                    l_rate = rate_val if rate_val is not None else effective_basal
                    l_bol = smb
                else:
                    output = get_loop_recommendation(
                        algo, bg, current_time, cgm_history,
                        bolus_history, declared_carbs, algo_settings,
                    )
                    l_bol = output.bolus or 0.0
                    l_rate = (output.temp_basal_rate
                              if output.temp_basal_rate is not None else effective_basal)
            except Exception:
                # Algorithm failure — maintain effective basal
                pass

            # --- Apply dosing ---
            # Algorithm's temp adjustment is relative to effective_basal
            tb_net = (l_rate - effective_basal) * 5.0 / 60.0
            if l_bol > 0:
                bolus_history.append((current_time, l_bol))
            if abs(tb_net) > 0.001:
                bolus_history.append((current_time, tb_net))

            # --- Record to day result ---
            if step_in_day == 0:
                # Start new day
                day_results.append(DayResult(
                    day_index=day_index,
                    bg_trace=[],
                    bolus_history=[],
                    meals=[m for m in all_meals
                           if day_index * 1440 <= (m.time_minutes - self.t0_min) < (day_index + 1) * 1440],
                    exercises=[e for e in exercise_events
                               if day_index * 1440 <= (e.start_time_minutes - self.t0_min) < (day_index + 1) * 1440],
                    sensitivity_trace=[],
                ))

            day_results[-1].bg_trace.append((t_rel, bg))
            day_results[-1].sensitivity_trace.append(
                (t_rel, patient.get_sensitivity_scalar(t_rel)))

            # --- Prune old entries ---
            # With delta-based BG, fully absorbed entries contribute 0 delta
            # and can be safely removed — their effect is baked into running_bg.
            prune_cutoff = current_time - PRUNE_AGE_MIN
            cgm_cutoff = current_time - 480
            if len(cgm_history) > 200:
                cgm_history = [(t, v) for t, v in cgm_history if t >= cgm_cutoff]
            bolus_history = [(t, u) for t, u in bolus_history if t >= prune_cutoff]
            basal_deficit_entries = [(t, u) for t, u in basal_deficit_entries if t >= prune_cutoff]
            actual_carbs = [(t, g, a) for t, g, a in actual_carbs if t >= prune_cutoff]
            declared_carbs = [(t, g, a) for t, g, a in declared_carbs if t >= prune_cutoff]

        return SimulationRunResult(
            algorithm_name=self.algorithm_name,
            days=day_results,
            settings=settings,
        )

    def _generate_meals(self) -> List[MealEvent]:
        """Generate all meal events for the full simulation.

        Handles:
        - Normal declared meals (with optional bias and random error)
        - Undeclared probability: any meal may randomly go undeclared
        - Undeclared-only meals: always eaten, never declared (e.g., morning coffee)
        """
        meals = []
        bias = self.profile.carb_count_bias
        sigma = self.profile.carb_count_sigma
        abs_sigma = self.profile.absorption_sigma

        for day in range(self.n_days):
            day_offset = self.t0_min + day * 1440

            # --- Regular meals (may be declared or randomly undeclared) ---
            for spec in self.profile.meals:
                meal_time = day_offset + spec.time_of_day_minutes

                # Draw actual carbs (normal distribution, clamp > 0)
                actual_g = max(2.0, self.rng.normal(spec.carbs_mean, spec.carbs_sd))
                actual_abs = spec.absorption_hrs

                # Check if this meal goes undeclared
                undeclared = (self.profile.undeclared_meal_prob > 0 and
                              self.rng.random() < self.profile.undeclared_meal_prob)

                if undeclared:
                    meals.append(MealEvent(
                        time_minutes=meal_time,
                        actual_carbs=actual_g,
                        actual_absorption_hrs=actual_abs,
                        declared_carbs=0.0,
                        declared_absorption_hrs=actual_abs,
                        undeclared=True,
                    ))
                else:
                    # Patient estimates carb count with bias + random error
                    # declared = actual * exp(N(-bias, sigma))
                    # bias > 0 means systematic under-declaration
                    if sigma > 0 or bias > 0:
                        count_error = float(np.exp(
                            self.rng.normal(-bias, max(sigma, 0.001))))
                        declared_g = actual_g * count_error
                    else:
                        declared_g = actual_g

                    # Patient estimates absorption with error
                    if abs_sigma > 0:
                        abs_error = float(np.exp(
                            self.rng.normal(0, abs_sigma)))
                        declared_abs = actual_abs * abs_error
                    else:
                        declared_abs = actual_abs

                    meals.append(MealEvent(
                        time_minutes=meal_time,
                        actual_carbs=actual_g,
                        actual_absorption_hrs=actual_abs,
                        declared_carbs=declared_g,
                        declared_absorption_hrs=declared_abs,
                        undeclared=False,
                    ))

            # --- Undeclared-only meals (always eaten, never declared) ---
            for spec in self.profile.undeclared_meals:
                meal_time = day_offset + spec.time_of_day_minutes
                actual_g = max(2.0, self.rng.normal(spec.carbs_mean, spec.carbs_sd))
                actual_abs = spec.absorption_hrs

                meals.append(MealEvent(
                    time_minutes=meal_time,
                    actual_carbs=actual_g,
                    actual_absorption_hrs=actual_abs,
                    declared_carbs=0.0,
                    declared_absorption_hrs=actual_abs,
                    undeclared=True,
                ))

        return meals

    def _generate_exercise_events(self) -> List[ExerciseEvent]:
        """Generate exercise events across the full simulation."""
        if self.profile.exercises_per_week <= 0 or self.profile.exercise_spec is None:
            return []

        events = []
        spec = self.profile.exercise_spec
        exercise_prob_per_day = self.profile.exercises_per_week / 7.0

        for day in range(self.n_days):
            if self.rng.random() < exercise_prob_per_day:
                day_offset = self.t0_min + day * 1440
                start = day_offset + spec.time_of_day_minutes

                # Draw actual scalar
                if spec.actual_scalar_sigma > 0:
                    actual_scalar = spec.actual_scalar_mean * float(np.exp(
                        self.rng.normal(0, spec.actual_scalar_sigma)))
                else:
                    actual_scalar = spec.actual_scalar_mean

                # Draw actual duration
                if spec.actual_duration_hrs_sigma > 0:
                    actual_dur = spec.actual_duration_hrs_mean * float(np.exp(
                        self.rng.normal(0, spec.actual_duration_hrs_sigma)))
                else:
                    actual_dur = spec.actual_duration_hrs_mean

                events.append(ExerciseEvent(
                    start_time_minutes=start,
                    declared_scalar=spec.declared_scalar,
                    declared_duration_hrs=spec.declared_duration_hrs,
                    actual_scalar=actual_scalar,
                    actual_duration_hrs=actual_dur,
                ))
        return events

    def _create_algorithm(self):
        """Create algorithm instance and optional Trio exporter."""
        settings = self.settings
        exporter = None

        if self.algorithm_name == 'trio':
            exporter = TrioJSONExporter(settings)
            algo = None
        elif self.algorithm_name == 'loop_tb':
            algo = create_loop_algorithm(settings, mode='temp_basal')
        elif self.algorithm_name == 'loop_ab40':
            algo = create_loop_algorithm(settings, mode='auto_bolus')
        elif self.algorithm_name == 'loop_ab_gbpa':
            algo = create_loop_algorithm(settings, mode='auto_bolus_gbpa')
        else:
            # Default to Loop temp basal
            algo = create_loop_algorithm(settings, mode='temp_basal')

        return algo, exporter
