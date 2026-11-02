"""Inspection is a causal view of one trajectory, never a second policy."""
from copy import deepcopy
import math
import numpy as np

def inspect_at(result,time_ms):
    if not math.isfinite(time_ms) or time_ms<0:raise ValueError('Invalid inspection time')
    t=min(time_ms,result['duration_ms']);last=None;memory=None;completed=[]
    for fixation in result['fixations']:
        if fixation['onset_ms']>t:break
        last=fixation
        if fixation['offset_ms']<=t:
            memory=deepcopy(fixation['memory_after']);completed.append(fixation['step'])
            dt=min(max(0,t-fixation['offset_ms']),fixation.get('saccade_duration_ms',0))
            memory['time_ms']+=dt
            for obj in memory['objects']:
                obj['confidence']*=float(np.exp(-dt/result['config']['memory_decay_ms']))
                uncertainty=obj['perceptual_uncertainty']
                uncertainty['identity']=min(1,uncertainty['identity']+.02*dt/result['config']['memory_decay_ms'])
        else:
            memory=deepcopy(fixation['memory_before']);break
    return dict(elapsed_ms=t,gaze=last['gaze'] if last else None,memory=memory,
                completed_fixations=completed,
                stopping=deepcopy(last['stopping']) if last and last['offset_ms']<=t else None,
                next_choice=deepcopy(last['selected_next']) if last and last['offset_ms']<=t else None,
                termination=deepcopy(result['termination']) if t>=result['duration_ms'] else None,
                note='Memory commits at fixation offset; no future depth, cuboid, or stopping state is exposed')
