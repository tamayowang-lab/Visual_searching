"""Configuration for one reconstruction-oriented execution path."""
from dataclasses import dataclass, field, asdict
from pathlib import Path
import math
import yaml

@dataclass
class Config:
    seed: int = 42
    central_fraction: float = .075
    peripheral_width: int = 32
    memory_decay_ms: float = 12000.
    evidence_time_ms: float = 250.
    priority_width: int = 64
    spatial_temperature: float = .08
    timing_mean_ms: float = 250.
    timing_sigma: float = .18
    administrative_timeout_ms: float = 60000.
    minimum_observation_ms: float = 1500.
    minimum_coverage: float = .7
    expected_value_threshold: float = .01
    low_value_patience: int = 3
    observation_cost: float = .035
    cache_directory: str = '.cache/observer-reconstruction'
    objects: dict = field(default_factory=lambda: dict(provider='torchvision_ssdlite',checkpoint='.cache/models/ssdlite320_mobilenet_v3_large_coco-a79551df.pth',threshold=.3,context_scale=3.))
    depth: dict = field(default_factory=lambda: dict(provider='none'))

    def validate(self):
        for name in ['memory_decay_ms','evidence_time_ms','spatial_temperature','timing_mean_ms','timing_sigma','administrative_timeout_ms']:
            if not math.isfinite(getattr(self,name)) or getattr(self,name)<=0:raise ValueError(f'{name} must be positive and finite')
        if not 0<self.central_fraction<=.5:raise ValueError('Invalid central fraction')
        if not isinstance(self.seed,int) or self.seed<0:raise ValueError('Nonnegative integer seed required')
        for name in ['peripheral_width','priority_width','low_value_patience']:
            if not isinstance(getattr(self,name),int) or getattr(self,name)<1:raise ValueError(f'{name} must be a positive integer')
        if not 0<=self.minimum_coverage<=1:raise ValueError('Invalid coverage threshold')
        for name in ['minimum_observation_ms','expected_value_threshold','observation_cost']:
            if not math.isfinite(getattr(self,name)) or getattr(self,name)<0:raise ValueError(f'Invalid {name}')
        return self

    @classmethod
    def load(cls,path=None):
        return cls(**(yaml.safe_load(Path(path).read_text(encoding='utf-8')) or {})).validate() if path else cls().validate()

    def to_dict(self):return asdict(self)
