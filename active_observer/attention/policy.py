"""Score deterministically, then sample separately. No human data enters policy."""
import numpy as np
from scipy.special import softmax


def feature_at(obs,x: float,y: float,global_color=None) -> float:
    h,w=obs.rgb.shape[:2];x=int(np.clip(x,0,w-1));y=int(np.clip(y,0,h-1));r=max(2,int(w*.025))
    crop=obs.rgb[max(0,y-r):min(h,y+r+1),max(0,x-r):min(w,x+r+1)].astype(float)/255
    contrast=float(np.std(crop,axis=(0,1)).mean());global_color=obs.rgb.mean(axis=(0,1))/255 if global_color is None else global_color
    return float(np.clip(contrast*3+np.linalg.norm(crop.mean(axis=(0,1))-global_color)/np.sqrt(3),0,1))


def score(world,obs,target: str,config) -> list[dict]:
    candidates=[];global_color=obs.rgb.mean(axis=(0,1))/255;diag=np.hypot(world.width,world.height)
    for m in (world.objects if config.policy not in {'center','random_center','random_walk'} else []):
        candidates.append(dict(id=m.id,kind="object",x=m.centroid[0],y=m.centroid[1],confidence=m.confidence,probabilities=m.probabilities,salience=m.salience,novelty=m.novelty,history=m.ior,relative_depth=m.relative_depth,depth_uncertainty=m.depth_uncertainty))
    if config.exploration_enabled or config.policy in {'center','random_center','random_walk'}:
        for g in world.grid:
            history=0 if g['last_seen'] is None else np.exp(-(world.time-g['last_seen'])/1200)
            candidates.append(dict(id=g['id'],kind="region",x=g['x'],y=g['y'],confidence=g['coverage'],probabilities={},salience=feature_at(obs,g['x'],g['y'],global_color),novelty=1-g['coverage'],history=float(history),relative_depth=None,depth_uncertainty=1.))
    if not candidates:
        candidates=[dict(id="standby",kind="region",x=obs.gaze[0],y=obs.gaze[1],confidence=0.,probabilities={},salience=0.,novelty=1.,history=0.,relative_depth=None,depth_uncertainty=1.)]
    if config.stay_enabled and config.policy=='full' and world.consecutive_stays<config.max_consecutive_stays:
        current=[m for m in world.objects if np.hypot(m.centroid[0]-obs.gaze[0],m.centroid[1]-obs.gaze[1])<=obs.radius_px]
        m=max(current,key=lambda m:m.visibility,default=None)
        if m is not None and m.sensory_ceiling-m.confidence>.025:
            candidates.append(dict(id='stay-current',kind='stay',x=obs.gaze[0],y=obs.gaze[1],confidence=m.confidence,probabilities=m.probabilities,salience=0.,novelty=0.,history=m.ior,relative_depth=m.relative_depth,depth_uncertainty=m.depth_uncertainty,additional_evidence_gap=m.sensory_ceiling-m.confidence))
    by_id={m.id:m for m in world.objects}
    for c in candidates:
        memory=by_id.get(c['id'])
        uncertainty=(.5*(1-c['confidence'])+.3*memory.identity_uncertainty+.2*memory.position_uncertainty) if memory else 1-c['confidence']
        uncertainty=float(np.clip(uncertainty,0,1)) if config.uncertainty_enabled else .5
        expected_reliability=float(np.clip(memory.sensory_ceiling if memory else .75,.1,.95))
        expected_reduction=uncertainty*(1-np.exp(-250*expected_reliability/config.evidence_time_ms))
        movement=float(np.hypot(c['x']-obs.gaze[0],c['y']-obs.gaze[1])/diag)
        observation_cost=.15+.25*movement
        information=expected_reduction/(1+observation_cost) if config.information_model=='expected_reduction' else uncertainty*c['novelty']
        if c['kind']=='stay':information=c['additional_evidence_gap']*(1-np.exp(-250/config.evidence_time_ms))/(1+observation_cost)
        if not config.information_enabled:information=0.
        context=0.
        for m in world.objects:
            known=max(m.probabilities,key=m.probabilities.get)
            relation=config.context_priors.get(known,{}).get(target,0) if target else 0
            context=max(context,relation*m.confidence*np.exp(-np.hypot(c['x']-m.centroid[0],c['y']-m.centroid[1])/(.2*diag)))
        terms=dict(salience=c['salience'],task=c['probabilities'].get(target,0)*c['confidence'] if target else 0.,context=float(context),novelty=c['novelty'] if config.novelty_enabled else 0.,info=float(information),move=float(np.hypot(c['x']-obs.gaze[0],c['y']-obs.gaze[1])/diag),history=c['history'] if config.history_enabled else 0.)
        if config.depth_enabled and config.uncertainty_enabled and config.information_enabled and c['relative_depth'] is not None:
            # Depth dispersion is an information-seeking hypothesis, not physical truth.
            terms['info']=float(np.clip(.8*terms['info']+.2*c['depth_uncertainty'],0,1))
        if config.policy=='random_walk':utility=-movement**2/(2*.12**2)
        elif config.policy in {'center','random_center'}:
            center=np.hypot(c['x']-world.width/2,c['y']-world.height/2)/diag
            utility=-center**2/(2*.22**2)
            if config.policy=='random_center': utility*=.35
        elif config.policy in {'salience','salience_ior'}: utility=terms['salience']-(.6*terms['history'] if config.policy=='salience_ior' else 0)
        else: utility=sum(config.weights[k]*v*(-1 if k in {'move','history'} else 1) for k,v in terms.items())
        # Bounded hysteresis: a small near-gaze preference, never a hard lock.
        persistence=.08*float(np.hypot(c['x']-obs.gaze[0],c['y']-obs.gaze[1])<obs.radius_px/2) if config.policy=='full' else 0.
        contributions={k:float(config.weights[k]*v*(-1 if k in {'move','history'} else 1)) for k,v in terms.items()}
        if config.policy in {'center','random_center','random_walk'}: contributions={'random_walk' if config.policy=='random_walk' else 'center_prior':float(utility)}
        elif config.policy in {'salience','salience_ior'}: contributions={'salience':terms['salience'],'history':-.6*terms['history'] if config.policy=='salience_ior' else 0.}
        contributions['persistence']=persistence
        c.update(expected_reliability=expected_reliability,expected_uncertainty_reduction=float(expected_reduction),observation_cost=observation_cost,information_model=config.information_model,terms=terms,utility=float(utility+persistence),utility_components=contributions,persistence=persistence,uncertainty=uncertainty)
    probabilities=softmax(np.array([c['utility'] for c in candidates])/config.temperature)
    for c,p in zip(candidates,probabilities): c['probability']=float(p)
    return candidates


def choose(candidates: list[dict],rng: np.random.Generator) -> dict:
    return candidates[int(rng.choice(len(candidates),p=[c['probability'] for c in candidates]))]
