"""Evaluation-only spatial diagnostics. Never imported by simulation/inference."""
import numpy as np

TOLERANCES=(8,16,24,32,48,64)

def lattice(width,height,columns=8,rows=5):
    return np.array([[(x+.5)*width/columns,(y+.5)*height/rows] for y in range(rows) for x in range(columns)],float)

def multiscale(width,height,gaze):
    """Forty proposals: 16 global, current gaze, 8 inner and 15 outer ring points."""
    points=lattice(width,height,4,4).tolist()+[list(gaze)]
    for count,radius in [(8,.08*min(width,height)),(15,.2*min(width,height))]:
        for a in np.arange(count)*2*np.pi/count:points.append([np.clip(gaze[0]+radius*np.cos(a),0,width-1),np.clip(gaze[1]+radius*np.sin(a),0,height-1)])
    return np.unique(np.asarray(points),axis=0)

def amplitudes(xy,exclude_first=False):
    xy=np.asarray(xy,float).reshape(-1,2)
    a=np.linalg.norm(np.diff(xy,axis=0),axis=1)
    return a[1:] if exclude_first else a

def inside(xy,boxes):
    xy=np.asarray(xy,float).reshape(-1,2);result=np.zeros(len(xy),bool)
    for x1,y1,x2,y2 in boxes:result|=(xy[:,0]>=x1)&(xy[:,0]<=x2)&(xy[:,1]>=y1)&(xy[:,1]<=y2)
    return result

def selection_diagnostics(candidates,next_xy):
    """Future point is a scoring label only. Candidates must already be frozen."""
    xy=np.array([[c['x'],c['y']] for c in candidates]);u=np.array([c['utility'] for c in candidates]);p=np.array([c['probability'] for c in candidates]);d=np.linalg.norm(xy-np.asarray(next_xy),axis=1);j=int(np.argmin(d));entropy=float(-np.sum(p*np.log(np.maximum(p,1e-300))))
    expected=float(p@d);nearest=float(d[j]);top=int(np.argmax(u));q=float(p[j]/(1+p[j]))
    result=dict(candidate_count=len(candidates),nearest_error_px=nearest,nearest_rank_min=1+int(np.sum(u>u[j]+1e-12)),nearest_rank_max=int(np.sum(u>=u[j]-1e-12)),nearest_utility=float(u[j]),nearest_probability=float(p[j]),nearest_kind=candidates[j]['kind'],greedy_error_px=float(d[top]),expected_error_px=expected,selection_excess_px=expected-nearest,entropy_fraction=entropy/np.log(len(p)) if len(p)>1 else 0.,effective_candidates=float(1/np.sum(p*p)),max_probability=float(p.max()),object_probability=float(sum(c['probability'] for c in candidates if c['kind']=='object')),unexplored_region_probability=float(sum(c['probability'] for c in candidates if c['kind']=='region' and c['confidence']<.1)),target_term_any=float(any(c['terms']['task']>0 for c in candidates)),context_term_any=float(any(c['terms']['context']>0 for c in candidates)))
    # Oracle: add the FUTURE coordinate with nearest candidate's EXISTING utility.
    # No new semantics or reweighting. This is not a normal prediction.
    result.update(oracle_added_probability=q,oracle_added_expected_error_px=expected/(1+p[j]),oracle_added_nearest_error_px=0.)
    for tolerance in TOLERANCES:result.update({f'coverage_{tolerance}':float(nearest<=tolerance),f'probability_within_{tolerance}':float(p[d<=tolerance].sum())})
    angular=angular_distance(xy,[next_xy]);result['nearest_error_deg']=float(angular.min());result['expected_error_deg']=float(p@angular)
    for tolerance in [1,2,3,4]:result.update({f'coverage_deg_{tolerance}':float(angular.min()<=tolerance),f'probability_within_deg_{tolerance}':float(p[angular<=tolerance].sum())})
    return result

def aligned_model_fixations(fixations,budget_ms):
    """Merge stay observations while carrying the final segment's censor flag."""
    merged=[]
    for f in fixations:
        f=dict(f);onset=f.get('fixation_onset_ms',0.)
        if onset>=budget_ms:break
        original_duration=f['duration_ms'];f['duration_ms']=min(original_duration,budget_ms-onset)
        f['right_censored']=f.get('right_censored',False) or f['duration_ms']<original_duration
        if f['duration_ms']<=0:continue
        if merged and f.get('mode')=='stay' and np.hypot(f['x_px']-merged[-1]['x_px'],f['y_px']-merged[-1]['y_px'])<1:
            merged[-1]['duration_ms']+=f['duration_ms'];merged[-1]['right_censored']=f.get('right_censored',False)
        else:merged.append(f)
    return merged


def display_rays(xy,width=640,height=400):
    """Approximate flat-display geometry from the published 54° × 35° field.

    This is evaluation geometry, not the observer's pseudo-camera or per-person
    calibration. Coordinates retain the continuous display origin convention.
    """
    xy=np.asarray(xy,float).reshape(-1,2);v=np.column_stack([(xy[:,0]/width-.5)*2*np.tan(np.deg2rad(27)),(xy[:,1]/height-.5)*2*np.tan(np.deg2rad(17.5)),np.ones(len(xy))]);return v/np.linalg.norm(v,axis=1)[:,None]

def angular_distance(a,b,width=640,height=400):
    x=display_rays(a,width,height);y=display_rays(b,width,height)
    # atan2 gives numerical stability even for coincident or very close points.
    return np.rad2deg(np.arctan2(np.linalg.norm(np.cross(x,y),axis=1),np.sum(x*y,axis=1)))

def angular_amplitudes(xy,width=640,height=400):
    a=np.asarray(xy,float).reshape(-1,2);return angular_distance(a[:-1],a[1:],width,height)


def content_box(geometry,width=640,height=400):
    """Known photograph extent, excluding the adapter's artificial black padding."""
    w,h=geometry['source_size'];px,py=geometry['padding_xy'];return [px*width/(w+2*px),py*height/(h+2*py),(px+w)*width/(w+2*px),(py+h)*height/(h+2*py)]
