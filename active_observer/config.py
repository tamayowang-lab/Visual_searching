"""Explicit spatial heuristics and independently selectable timing parameters."""
from dataclasses import dataclass, field, asdict
from pathlib import Path
import yaml

@dataclass
class Config:
    seed: int = 42
    max_fixations: int = 16
    time_budget_ms: float = 6000
    pixels_per_degree: float | None = None
    central_degrees: float = 2.5
    central_fraction: float = .075
    memory_decay_ms: float = 12000
    grid_columns: int = 8
    grid_rows: int = 5
    temperature: float = .25
    min_duration_ms: float = 100
    max_duration_ms: float = 800
    stop_on_target: bool = False
    target_threshold: float = .8
    depth_enabled: bool = True
    memory_enabled: bool = True
    uncertainty_enabled: bool = True
    novelty_enabled: bool = True
    history_enabled: bool = True
    exploration_enabled: bool = True
    semantic_enabled: bool = True
    information_enabled: bool = True
    information_model: str = "expected_reduction"
    evidence_model: str = "bounded"
    evidence_time_ms: float = 250.
    stay_enabled: bool = True
    max_consecutive_stays: int = 2
    perception: dict = field(default_factory=lambda: dict(objects={'provider':'classical'},depth={'provider':'none'}))
    timing_model: str = 'phase2'
    timing: dict = field(default_factory=dict)
    policy: str = "full"
    weights: dict = field(default_factory=lambda: dict(salience=.8, task=1.3, context=.35, novelty=.45, info=.65, move=.5, history=.6))
    active: dict = field(default_factory=dict)
    experiment: dict = field(default_factory=dict)
    stopping: dict = field(default_factory=dict)
    context_priors: dict = field(default_factory=dict)

    def validate(self):
        if self.max_fixations < 1 or self.time_budget_ms <= 0: raise ValueError("Positive fixation and time budgets required")
        if self.grid_columns < 1 or self.grid_rows < 1 or self.temperature <= 0: raise ValueError("Invalid grid/temperature")
        if self.memory_decay_ms <= 0 or not 0 < self.central_fraction <= .5 or self.central_degrees <= 0: raise ValueError("Invalid memory/foveation settings")
        if self.pixels_per_degree is not None and self.pixels_per_degree <= 0: raise ValueError("pixels_per_degree must be positive")
        if not 0 < self.min_duration_ms <= self.max_duration_ms: raise ValueError("Invalid duration bounds")
        if self.policy not in {"full", "center", "random_center", "salience", "salience_ior", "random_walk"}: raise ValueError("Unknown policy")
        if self.evidence_model not in {'bounded','legacy'} or self.information_model not in {'expected_reduction','uncertainty_novelty'}: raise ValueError('Unknown evidence/information model')
        if self.evidence_time_ms<=0 or self.max_consecutive_stays<0: raise ValueError('Invalid evidence/stay parameters')
        if self.timing_model not in {'phase2','heuristic_lognormal','accumulator','accumulator_hold'}:raise ValueError('Unknown timing model')
        from .attention.timing import TimingParameters
        TimingParameters(**self.timing).validate()
        from .experiment.options import ExperimentOptions,StoppingOptions
        experiment=ExperimentOptions(**self.experiment).validate();StoppingOptions(**self.stopping).validate()
        if experiment.enabled and self.timing_model=='phase2':raise ValueError('Phase 5 uses one of the three preserved Phase 3 timing models')
        from .active.options import ActiveOptions
        options=ActiveOptions(**self.active).validate()
        if options.enabled and self.timing_model=='phase2':raise ValueError('Phase 4B requires one of the three Phase 3 timing models; archived Phase 2 stays available with active disabled')
        if set(self.weights) != {"salience","task","context","novelty","info","move","history"}: raise ValueError("Seven explicit utility weights required")
        if any(not 0 <= float(v) < 100 for v in self.weights.values()): raise ValueError("Weights must be finite and nonnegative")
        return self

    @classmethod
    def load(cls, path=None):
        return cls(**(yaml.safe_load(Path(path).read_text(encoding='utf-8')) or {})).validate() if path else cls().validate()

    def to_dict(self): return asdict(self)
