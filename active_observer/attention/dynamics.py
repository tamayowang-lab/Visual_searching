"""Bounded lognormal dwell and soft history; explicit heuristic, not physiology."""
import numpy as np

def duration_ms(candidate: dict,config, rng: np.random.Generator) -> float:
    mean=180+160*candidate['terms']['info']+100*candidate['terms']['task']+80*candidate['terms']['novelty']
    return float(np.clip(rng.lognormal(np.log(mean)-.5*.18**2,.18),config.min_duration_ms,config.max_duration_ms))
