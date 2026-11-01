"""Separately switchable Phase 4B attention components."""
from copy import deepcopy
import numpy as np
from scipy.special import softmax
from active_observer.attention.policy import score,feature_at
from .candidates import refine_candidates

def decision_candidates(world,obs,target,cfg,options,mask,perception=None):
    effective_target='' if options.behavior_mode=='explore' else target
    candidates=score(world,obs,effective_target,cfg);rejected=[]
    for c in candidates:c['probability_before_validity']=c['probability'];c['proposal_area']=1.;c['valid_stimulus']=mask.valid(c['x'],c['y'])
    if options.dynamic_perception and perception:
        global_color=obs.rgb.mean(axis=(0,1))/255
        for hypothesis in perception.hypotheses:
            if hypothesis.get('linked_memory_id') in {c['id'] for c in candidates}:continue
            x,y=hypothesis['point'];c=deepcopy(candidates[0]);confidence=hypothesis['confidence'];probs=hypothesis['probabilities'];novelty=float(np.exp(-hypothesis['observations']/5));movement=float(np.hypot(x-obs.gaze[0],y-obs.gaze[1])/np.hypot(world.width,world.height));uncertainty=float(1-confidence);expected_reduction=float(uncertainty*(1-np.exp(-250*.75/cfg.evidence_time_ms)));observation_cost=.15+.25*movement;salience=feature_at(obs,x,y,global_color)
            terms=dict(salience=salience,task=float(probs.get(effective_target,0)*confidence) if effective_target else 0.,context=0.,novelty=novelty if cfg.novelty_enabled else 0.,info=expected_reduction/(1+observation_cost) if cfg.information_enabled else 0.,move=movement,history=0.)
            components={k:float(cfg.weights[k]*value*(-1 if k in {'move','history'} else 1)) for k,value in terms.items()};components['persistence']=0.
            c.update(id=hypothesis['id'],kind='proto',x=x,y=y,confidence=confidence,probabilities=probs,salience=salience,novelty=novelty,history=0.,relative_depth=None,depth_uncertainty=1.,uncertainty=uncertainty,expected_reliability=.75,observation_cost=observation_cost,proposal_area=1.,valid_stimulus=mask.valid(x,y),expected_uncertainty_reduction=expected_reduction,persistence=0.,proto_status=hypothesis['status'],terms=terms,utility_components=components,utility=float(sum(components.values())),probability=0.,probability_before_validity=0.)
            candidates.append(c)
    if options.adaptive_candidates:
        candidates=refine_candidates(candidates,obs,world,cfg,mask,perception)
        for c in candidates:c['valid_stimulus']=mask.valid(c['x'],c['y'])
    if options.valid_stimulus:
        rejected=[dict(c,probability=0.,rejection_reason='outside valid stimulus') for c in candidates if not c['valid_stimulus']];candidates=[c for c in candidates if c['valid_stimulus']]
        if not candidates:
            c=deepcopy(rejected[0]);c.update(id='valid-standby',kind='region',x=float(obs.gaze[0]),y=float(obs.gaze[1]),utility=0.,probability=1.,valid_stimulus=True,probability_before_validity=0.);candidates=[c]
        p=softmax(np.array([c['utility'] for c in candidates])/cfg.temperature)
        for c,v in zip(candidates,p):c['probability']=float(v)
    if options.adaptive_candidates or options.dynamic_perception:
        p=softmax(np.array([c['utility'] for c in candidates])/cfg.temperature)
        for c,v in zip(candidates,p):c['probability']=float(v)
    if perception:
        for c in candidates:c['perceptual_support']=perception.support_at(c['x'],c['y'])
    if options.peripheral_guidance and perception and cfg.policy=='full':
        for c in candidates:
            old=c['terms']['task'];hint=perception.target_at(c['x'],c['y'],effective_target);value=max(old,hint);c['terms']['task']=value;c['peripheral_target_evidence']=hint;c['perceptual_support']=perception.support_at(c['x'],c['y']);c['utility']+=cfg.weights['task']*(value-old);c['utility_components']['task']=cfg.weights['task']*value
        p=softmax(np.array([c['utility'] for c in candidates])/cfg.temperature)
        for c,v in zip(candidates,p):c['probability']=float(v)
    if options.selection=='balanced' and cfg.policy=='full':candidates=balanced_competition(candidates,options,effective_target)
    for c in candidates:c['rank']=int(1+sum(a['utility']>c['utility']+1e-12 for a in candidates))
    return candidates,rejected


def balanced_competition(candidates,options,target):
    """Area-weighted mixture: proposal multiplicity is not evidence for a group.

    Dimensionless log-priority units, temperature 1. These are declared design
    priors, not fitted human probabilities or a claim of neural implementation.
    """
    groups={}
    for c in candidates:
        role='task' if target and c['terms']['task']>=.05 else 'stay' if c['kind']=='stay' else 'object' if c['kind'] in {'object','proto'} else 'region'
        c['selection_role']=role;groups.setdefault(role,[]).append(c)
    priors={'task':.55,'object':.15,'region':.25,'stay':.05} if target else {'object':.45,'region':.50,'stay':.05}
    active_prior=sum(priors[k] for k in groups)
    for role,items in groups.items():
        area=sum(c.get('proposal_area',1.) for c in items)
        for c in items:
            old_components=dict(c['utility_components']);raw=dict(c['terms']);support=c.get('perceptual_support',.0);useful=float(c['expected_uncertainty_reduction']*support*np.clip(raw['novelty'],0,1));distance=raw['move'];task=raw['task']
            components=dict(salience=.8*raw['salience'],useful_information=useful,task=options.task_weight*task,movement=-.5*(distance/options.movement_scale)**2,inspection=-.35*raw['history']*(1-np.clip(task,0,1)),persistence=c.get('persistence',0.),group_prior=float(np.log(priors[role]/active_prior)),proposal_density=float(np.log(c.get('proposal_area',1.)/area)))
            c.update(original_utility_components=old_components,utility_components=components,utility=float(sum(components.values())),useful_information=useful,scoring_model='area-normalized candidate-group log priority; temperature 1')
    p=softmax(np.array([c['utility'] for c in candidates]))
    masses={k:0. for k in groups}
    for c,v in zip(candidates,p):c['probability']=float(v);masses[c['selection_role']]+=float(v)
    for c in candidates:c['group_probability']=masses[c['selection_role']]
    return candidates
