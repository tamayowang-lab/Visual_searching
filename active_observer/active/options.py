from dataclasses import dataclass
import math

@dataclass
class ActiveOptions:
    enabled: bool=False
    variant: str='A'
    valid_stimulus: bool=False
    behavior_mode: str='legacy'
    peripheral_guidance: bool=False
    selection: str='legacy'
    adaptive_candidates: bool=False
    dynamic_perception: bool=False
    coarse_width: int=160
    peripheral_cap: float=.35
    task_weight: float=3.
    movement_scale: float=.20
    trace_maps: bool=True
    peripheral_sensor: str='observed'
    proto_persistence: bool=True

    def validate(self):
        if not all(math.isfinite(x) for x in [self.coarse_width,self.peripheral_cap,self.task_weight,self.movement_scale]):raise ValueError('Active parameters must be finite')
        if self.variant not in list('ABCDEFSN'):raise ValueError('Unknown Phase 4B variant')
        if self.behavior_mode not in {'legacy','explore','search'} or self.selection not in {'legacy','balanced'}:raise ValueError('Invalid active policy')
        if not 32<=self.coarse_width<=160 or not 0<self.peripheral_cap<=1 or self.task_weight<0 or self.movement_scale<=0:raise ValueError('Invalid active options')
        if self.peripheral_sensor not in {'observed','bounded_rgb'}:raise ValueError('Invalid peripheral sensor')
        return self
