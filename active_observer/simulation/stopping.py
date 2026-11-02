"""Inspect estimated benefit without treating it as correct reconstruction."""
import numpy as np

class StopController:
    def __init__(self,config):self.config=config;self.low_value_streak=0

    def assess(self,priority,valid,coverage,elapsed,depth_available):
        c=self.config;gain=float(np.max(priority['gain'][valid]));value=gain-c.observation_cost
        low=value<=c.expected_value_threshold and coverage>=c.minimum_coverage and elapsed>=c.minimum_observation_ms
        self.low_value_streak=self.low_value_streak+1 if low else 0
        stop=self.low_value_streak>=c.low_value_patience
        return dict(decision='stop' if stop else 'continue',
                    outcome=('low_estimated_benefit' if depth_available else 'reconstruction_unresolved') if stop else None,
                    reason=('Estimated benefit remained below threshold; geometric correctness is unvalidated' if depth_available else 'Low acquisition proxy, but no gaze-dependent depth provider is available') if stop else 'Estimated information needs still support another observation',
                    estimated_information_gain=gain,expected_observation_value=value,
                    observed_reconstruction_improvement=None,reconstruction_validated=False,
                    coverage=coverage,threshold=c.expected_value_threshold,observation_cost=c.observation_cost,
                    minimum_coverage=c.minimum_coverage,minimum_observation_ms=c.minimum_observation_ms,
                    low_value_streak=self.low_value_streak,persistence_required=c.low_value_patience,
                    depth_available=depth_available,administrative_timeout=False,
                    caveat='Uncalibrated prediction proxy, not measured 3D improvement or evidence of scene correctness')
