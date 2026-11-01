from dataclasses import dataclass,asdict
import math

@dataclass
class ExperimentOptions:
    enabled: bool=False
    behavior_mode: str='explore'
    observation_condition: str='fixed_time'
    fixed_time_ms: float=5000.
    administrative_timeout_ms: float=60000.
    administrative_max_steps: int | None=None
    def validate(self):
        if self.behavior_mode not in {'explore','search'}:raise ValueError('Phase 5 has exactly two behavioral modes: explore and search')
        if self.observation_condition not in {'fixed_time','until_finished'}:raise ValueError('Unknown observation condition')
        if not all(math.isfinite(x) and x>0 for x in [self.fixed_time_ms,self.administrative_timeout_ms]):raise ValueError('Positive finite observation and administrative times required')
        if self.fixed_time_ms>self.administrative_timeout_ms:raise ValueError('Requested view exceeds administrative timeout; increase the safeguard explicitly')
        if self.administrative_max_steps is not None and (not isinstance(self.administrative_max_steps,int) or self.administrative_max_steps<1):raise ValueError('Optional step limit is an administrative safeguard only')
        return self

@dataclass
class StoppingOptions:
    observation_cost: float=.035
    movement_cost: float=.02
    expected_value_threshold: float=.01
    recent_gain_rate_threshold: float=.04
    recent_window_ms: float=1500.
    minimum_elapsed_ms: float=1000.
    persistence: int=3
    exploration_min_coverage: float=.1
    absence_min_coverage: float=.85
    absence_max_unresolved_fraction: float=.15
    allow_absence: bool=True
    possible_target_score: float=.25
    verification_score: float=.65
    verification_observations: int=2
    verification_exposure_ms: float=200.
    def validate(self):
        if not all(math.isfinite(v) and v>=0 for k,v in asdict(self).items() if k!='allow_absence'):raise ValueError('Stopping parameters must be finite and nonnegative')
        if self.persistence<1 or self.verification_observations<1 or self.recent_window_ms<=0:raise ValueError('Positive persistence, evidence counts and rate window required')
        if not all(0<=v<=1 for v in [self.exploration_min_coverage,self.absence_min_coverage,self.absence_max_unresolved_fraction,self.possible_target_score,self.verification_score]):raise ValueError('Stopping support thresholds must be in [0,1]')
        return self
