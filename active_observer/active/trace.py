"""Actual observation/decision logs, not inferred post-hoc rationales."""
from dataclasses import asdict
from copy import deepcopy
import hashlib
import numpy as np
import cv2

def memory_changes(before,after):
    old={m['id']:m for m in before['objects']};new={m['id']:m for m in after['objects']};events=[]
    for key,m in new.items():
        b=old.get(key);events.append(dict(id=key,event='created' if b is None else 'observed_update' if m['observation_count']>b['observation_count'] else 'decay',before=b,after=m,confidence_delta=m['confidence']-(b['confidence'] if b else 0)))
    for key,m in old.items():
        if key not in new:events.append(dict(id=key,event='removed_by_world_filter',before=m,after=None))
    return events

def priority_raster(candidates,mask,width=80,enforce_validity=True):
    h,w=mask.mask.shape;rh=max(1,round(h*width/w));a=np.zeros((rh,width),np.float32)
    for c in candidates:
        if c['probability']<=0:continue
        x=min(width-1,int(c['x']/w*width));y=min(rh-1,int(c['y']/h*rh));a[y,x]+=c['probability']
    a=cv2.GaussianBlur(a,(0,0),1.2);small=cv2.resize(mask.mask.astype(np.uint8),(width,rh),interpolation=cv2.INTER_NEAREST).astype(bool);
    if enforce_validity:a[~small]=0
    if a.sum()>0:a/=a.sum()
    return dict(width=width,height=rh,values=np.round(a,7).tolist(),meaning='display raster of actual candidate probabilities; exact validity bitmap must also be applied when drawing')

def decision_trace(step,obs,detections,before,after,candidates,rejected,mask,maps=None,enforce_validity=True):
    return dict(schema_version=1,step=step,observation=dict(gaze=list(obs.gaze),input_rgb_sha256=hashlib.sha256(obs.rgb.tobytes()).hexdigest(),input_shape=list(obs.rgb.shape),central_radius_px=obs.radius_px,source='actual gaze-dependent RGB; no future human information',detections=[asdict(d) for d in detections]),memory_before=before,memory_after=after,memory_updates=memory_changes(before,after),available_candidates=deepcopy(candidates),rejected_candidates=deepcopy(rejected),priority_map=priority_raster(candidates,mask,enforce_validity=enforce_validity),perceptual_state=maps or {},selection=dict(action='stop',reason='not selected yet'),incoming_choice=None)
