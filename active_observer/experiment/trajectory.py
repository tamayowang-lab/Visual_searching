"""Causal fixed-time views of one recorded autonomous history; no random draws."""
from copy import deepcopy
import math
import numpy as np

def decay_snapshot(snapshot,dt,decay_ms):
    s=deepcopy(snapshot);s['time_ms']+=dt
    for m in s['objects']:
        m['confidence']*=float(np.exp(-dt/decay_ms));m['ior']*=float(np.exp(-dt/1200));m['visibility']=0.;m['state']='remembered' if m['confidence']>.2 else 'uncertain';m['identity_uncertainty']=min(1,m['identity_uncertainty']+.02*dt/decay_ms);m['position_uncertainty']=min(1,m['position_uncertainty']+.04*dt/decay_ms)
    return s

def inspect_at(result,time_ms):
    if not math.isfinite(time_ms) or time_ms<0:raise ValueError('Snapshot time must be finite and nonnegative')
    terminal_time=result['duration_ms'];t=min(float(time_ms),terminal_time);completed=[];partial=None;memory=None;phase='not_started';gaze=None
    for f in result['fixations']:
        if f['fixation_onset_ms']>t:break
        gaze=[f['x_px'],f['y_px']];tr=f['decision_trace']
        if f['fixation_offset_ms']<=t:
            ff=deepcopy(f);ff['decision_trace']['selection']['arrived']=bool(f.get('next_fixation_onset_ms') is not None and f['next_fixation_onset_ms']<=t and f['step']+1<len(result['fixations']));completed.append(ff);memory=deepcopy(tr['memory_after']);phase='between_fixations'
            gap=min(t-f['fixation_offset_ms'],f.get('saccade_duration_ms',0.));memory=decay_snapshot(memory,max(0,gap),result['config']['memory_decay_ms'])
            if gap>0:phase='saccade'
        else:
            partial={k:deepcopy(f[k]) for k in ['step','x_px','y_px','fixation_onset_ms']};partial.update(duration_ms=t-f['fixation_onset_ms'],fixation_offset_ms=t,right_censored=True)
            state=deepcopy(tr.get('perceptual_state',{}));state['proto_objects']=deepcopy(tr.get('perceptual_memory_before',[]));state['proto_events']=[]
            partial['decision_trace']=dict(step=f['step'],observation=deepcopy(tr['observation']),available_candidates=deepcopy(tr['onset_candidates']),memory_before=deepcopy(tr['memory_before']),memory_after=None,memory_updates=[],perceptual_state=state,selection=dict(action='not_yet_decided',reason='Current fixation is still in progress'),stopping=None,experimental_control=dict(behavior_mode=result['experimental_control']['behavior_mode'],observation_condition='fixed_time',elapsed_ms=t))
            memory=deepcopy(tr['memory_before']);phase='fixation';break
    finished=terminal_time<=time_ms
    return dict(source_model=result['model'],source_seed=result['config']['seed'],requested_time_ms=float(time_ms),elapsed_ms=t,phase='terminal' if finished else phase,gaze=gaze,completed_fixations=completed,current_partial_fixation=partial,memory=memory,termination=deepcopy(result['termination']) if finished else None,actual_completion_time_ms=terminal_time if finished else None,administrative_timeout=bool(result['termination']['administrative_timeout']) if finished else False,note='End-of-fixation memory updates are unavailable until committed. Partial snapshots do not invent continuous perceptual updates.')
