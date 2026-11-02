"""One causal clock for sensory input, committed memory, saccades and decisions."""
from copy import deepcopy
import math
import numpy as np


def inspect_at(result,time_ms):
    if not math.isfinite(time_ms) or time_ms<0:raise ValueError('Invalid inspection time')
    t=min(float(time_ms),result['duration_ms']);index=0;committed=-1;completed=[]
    for i,f in enumerate(result['fixations']):
        if f['onset_ms']<=t:index=i
        if f['offset_ms']<=t:committed=i;completed.append(f['step'])
    f=result['fixations'][index];done=f['offset_ms']<=t
    memory=deepcopy(f['memory_after'] if done else f['memory_before']);gaze=list(f['gaze'])
    phase='fixation';sensor_available=True
    if done:
        phase='between_fixations';dt=min(max(0,t-f['offset_ms']),f.get('saccade_duration_ms',0))
        memory['time_ms']+=dt
        for obj in memory['objects']:
            obj['confidence']*=float(np.exp(-dt/result['config']['memory_decay_ms']))
            u=obj['perceptual_uncertainty'];u['identity']=min(1,u['identity']+.02*dt/result['config']['memory_decay_ms'])
        if f.get('saccade_duration_ms',0)>0 and t<f['offset_ms']+f['saccade_duration_ms']:
            phase='saccade';sensor_available=False
            destination=f.get('saccade_to') or ([f['selected_next']['x'],f['selected_next']['y']] if f.get('selected_next') else gaze)
            fraction=dt/max(1e-12,f.get('saccade_planned_ms',f['saccade_duration_ms']))
            gaze=[float(a+(b-a)*fraction) for a,b in zip(gaze,destination)]
    terminal=t>=result['duration_ms']
    if terminal and f.get('termination_after_saccade'):
        sensor_available=False;phase='saccade'
        destination=f.get('saccade_to',gaze);fraction=f.get('saccade_duration_ms',0)/max(1e-12,f.get('saccade_planned_ms',1))
        gaze=[float(a+(b-a)*fraction) for a,b in zip(f['gaze'],destination)]
    last=result['fixations'][committed] if committed>=0 else None
    return dict(elapsed_ms=t,fixation_index=index,committed_index=committed,phase='terminal' if terminal else phase,
                sensory_available=sensor_available,gaze=gaze,memory=memory,completed_fixations=completed,
                stopping=deepcopy(last['stopping']) if last else None,
                next_choice=deepcopy(last['selected_next']) if last else None,
                depth=deepcopy(last.get('depth')) if last else None,
                termination=deepcopy(result['termination']) if terminal else None,
                note='End-of-fixation updates only; during saccades the last sensory frame is retained, not new input')
