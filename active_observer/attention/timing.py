"""One heuristic lognormal dwell model; timing never chooses spatial targets."""
import numpy as np

def fixation_duration(config,rng,initial=False):
    if initial:return float(config.timing_mean_ms)
    return float(np.clip(rng.lognormal(np.log(config.timing_mean_ms)-.5*config.timing_sigma**2,config.timing_sigma),100,800))

def saccade_duration(distance):
    return 0. if distance<1 else float(np.clip(20+.04*distance,10,90))
