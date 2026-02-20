"""Base classes and interfaces for diabetes algorithm implementations."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple


@dataclass
class AlgorithmInput:
    """Input data provided to an algorithm for decision-making."""

    cgm_reading: float  # mg/dL
    timestamp: int  # minutes since start
    cgm_history: List[Tuple[int, float]]  # [(time, glucose), ...]
    current_basal: float  # U/hr
    temp_basal: Optional[Tuple[float, int]] = None  # (rate, duration) if active
    bolus_history: List[Tuple[int, float]] = field(default_factory=list)  # [(time, units), ...]
    carb_entries: List[Tuple[int, float, float]] = field(default_factory=list)  # [(time, grams, absorption_hrs), ...]
    settings: Dict = field(default_factory=dict)


@dataclass
class AlgorithmOutput:
    """Output from an algorithm decision."""

    timestamp: int
    temp_basal_rate: Optional[float] = None  # U/hr (None means cancel temp)
    temp_basal_duration: int = 30  # minutes
    bolus: Optional[float] = None  # units (SMB for OpenAPS, recommendation for Loop)
    iob: float = 0.0
    cob: float = 0.0
    glucose_predictions: Dict[str, List[float]] = field(default_factory=dict)  # {'main': [...], 'iob': [...], ...}
    reason: str = ""

    # Raw insulin recommendation (can be negative)
    insulin_req: Optional[float] = None  # units (positive = need more, negative = need less)
    insulin_req_explanation: str = ""  # e.g., "(eventual 180 - target 110) / ISF 50 = +1.4U"

    # IRC effect tracking
    irc_effect_next: Optional[float] = None  # IRC contribution to next BG (5 min) in mg/dL
    irc_effect_eventual: Optional[float] = None  # IRC contribution to eventual BG (6 hr) in mg/dL
    irc_discrepancy: Optional[float] = None  # Most recent discrepancy (actual - predicted) in mg/dL
    irc_proportional: Optional[float] = None  # IRC P term in mg/dL
    irc_integral: Optional[float] = None  # IRC I term in mg/dL
    irc_differential: Optional[float] = None  # IRC D term in mg/dL
    irc_total_correction: Optional[float] = None  # IRC total PID correction in mg/dL
    irc_discrepancies_count: Optional[int] = None  # Number of discrepancies used for IRC
    irc_discrepancies_timeline: Optional[List[Tuple[float, float]]] = None  # Timeline of (time, discrepancy) tuples

    # Momentum effect tracking
    momentum_effect_eventual: Optional[float] = None  # Momentum contribution to eventual BG (6 hr) in mg/dL
    momentum_effect_timeline: Optional[List[Tuple[float, float]]] = None  # Timeline of (time, cumulative_effect) tuples

    # IRC effect tracking (timeline)
    irc_effect_timeline: Optional[List[Tuple[float, float]]] = None  # Timeline of (time, cumulative_effect) tuples

    # NEW: RealisticDoseMath fields
    insulin_per_5min: Optional[float] = None  # Actual insulin deliverable in 5-min period
    ideal_temp_basal_rate: Optional[float] = None  # Ideal rate before constraints
    temp_basal_was_capped: bool = False  # True if rate hit max_basal limit
    temp_basal_was_floored: bool = False  # True if rate hit zero floor


class Algorithm(ABC):
    """Abstract base class for diabetes algorithms."""

    def __init__(self, settings: Dict):
        """
        Initialize the algorithm with settings.

        Args:
            settings: Algorithm-specific configuration
        """
        self.settings = settings

    @abstractmethod
    def recommend(self, input_data: AlgorithmInput) -> AlgorithmOutput:
        """
        Generate a dosing recommendation based on current state.

        Args:
            input_data: Current glucose, history, and insulin/carb data

        Returns:
            AlgorithmOutput with dosing recommendation and predictions
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return the name of this algorithm."""
        pass
