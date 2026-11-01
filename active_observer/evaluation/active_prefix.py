"""Evaluation-only observed-prefix replay. Future coordinates are labels only."""
from copy import deepcopy
from dataclasses import asdict
import hashlib,json
import numpy as np
from active_observer.io.image import load_image,Camera
from active_observer.world.model import World
from active_observer.perception.foveation import observe
from active_observer.perception.depth import update_observed_depth
from active_observer.attention.processing import processing_context
from active_observer.attention.policy import choose
from active_observer.attention.timing import TimingParameters,simulate_timing
from active_observer.active.options import ActiveOptions
from active_observer.active.masking import StimulusMask
from active_observer.active.policy import decision_candidates
from active_observer.active.perception import ActivePerception,bounded_peripheral_sensor,update_workspace
from active_observer.active.trace import decision_trace
from .diagnostics import selection_diagnostics,lattice,inside


def replay(record,cfg,factory,target_box=None):
    image=load_image(record['image_path']);h,w=image.shape[:2]
    if (w,h)!=(640,400):raise ValueError('This empirical replay uses the audited 640x400 display transform')
    mask=StimulusMask(w,h,bounds=record['valid_stimulus_bounds']);opt=ActiveOptions(**cfg.active).validate();world=World(w,h,cfg);camera=Camera(w,h);objects,depth=factory.get(cfg,record);d=depth.predict(image) if depth else None;perception=ActivePerception(objects,opt,mask,cfg.perception.get('cache_directory','.cache/observer')+'/peripheral') if (opt.peripheral_guidance or opt.adaptive_candidates or opt.dynamic_perception or opt.selection=='balanced') else None;rng=np.random.default_rng(cfg.seed);fix=[];metrics=[];labels=[];contexts=[];previous=None;clock=0.
    for step,f in enumerate(record['fixations'][:-1]):
        gaze=(f['x'],f['y']);before=deepcopy(world.snapshot());proto_before=deepcopy(perception.hypotheses) if perception else [];observation=observe(image,gaze,cfg);detections=objects.detect(observation)
        if perception:perception.observe(observation,detections,bounded_peripheral_sensor(image,gaze,observation.radius_px,opt.coarse_width) if opt.peripheral_sensor=='bounded_rgb' else None)
        onset,_=decision_candidates(world,observation,record['task'],cfg,opt,mask,perception);process=processing_context(world,observation,detections,250.)
        if step>0:
            nearest=min(previous,key=lambda c:np.hypot(c['x']-gaze[0],c['y']-gaze[1]));contexts.append(dict(image_id=record['image_id'],participant=record['participant'],task=record['task'],step=step,duration_ms=f['duration_ms'],utilities=[c['utility'] for c in onset],heuristic_terms=nearest['terms'],processing=asdict(process)))
        world.update(detections,observation,f['duration_ms'])
        if d is not None:update_observed_depth(world,observation,d,camera)
        if opt.dynamic_perception:update_workspace(perception,observation,detections,f['duration_ms'],world)
        candidates,rejected=decision_candidates(world,observation,record['task'],cfg,opt,mask,perception);uniform=float(deepcopy(rng).random());chosen=choose(candidates,rng)
        trace=decision_trace(step,observation,detections,before,deepcopy(world.snapshot()),candidates,rejected,mask,maps=perception.snapshot(record['task']) if perception else None,enforce_validity=opt.valid_stimulus);trace['onset_candidates']=deepcopy(onset);trace['perceptual_memory_before']=proto_before;trace['perceptual_memory_after']=deepcopy(perception.hypotheses) if perception else [];trace['incoming_choice']=dict(action='observed human location; not sampled by model',location=list(gaze));trace['selection']=dict(action='conditional prediction',chosen_candidate=chosen['id'],location=[chosen['x'],chosen['y']],probability=chosen['probability'],uniform_draw=uniform,method='softmax',selection_source='after_observation',arrived=None,reason='evaluation-only one-step prediction; next gaze remains human supplied')
        fix.append(dict(step=step,x_px=gaze[0],y_px=gaze[1],duration_ms=f['duration_ms'],fixation_onset_ms=clock,fixation_offset_ms=clock+f['duration_ms'],memory=world.snapshot(),next_candidates=candidates,decision_trace=trace,saccade_duration_ms=0.));clock+=f['duration_ms']
        # Only now read the NEXT HUMAN LABEL. It is never an input to the policy.
        nxt=record['fixations'][step+1];point=[nxt['x'],nxt['y']];row=selection_diagnostics(candidates,point);row.update(image_id=record['image_id'],participant=record['participant'],task=record['task'],step=step,current_valid_stimulus=float(mask.valid(*gaze)),target_guidance_before_choice=float(any(c['terms']['task']>0 for c in candidates)),first_transition=float(step==0))
        if target_box is not None:
            xy=np.array([[c['x'],c['y']] for c in candidates]);p=np.array([c['probability'] for c in candidates]);row['target_probability']=float(p@inside(xy,[target_box]));row['human_next_on_target']=float(inside([point],[target_box])[0]);probes=lattice(w,h);probes=probes[[mask.valid(*a) for a in probes]];prediction=np.array([perception.target_at(x,y,record['task']) if perception else 0. for x,y in probes]);truth=inside(probes,[target_box]).astype(float);row['target_location_brier_proxy']=float(np.mean((prediction-truth)**2))
        metrics.append(row);labels.append(point);previous=candidates
        world.consecutive_stays=world.consecutive_stays+1 if step and np.hypot(gaze[0]-record['fixations'][step-1]['x'],gaze[1]-record['fixations'][step-1]['y'])<1 else 0
    result=dict(schema_version=3,model='observed-human-prefix conditional replay, not an autonomous scanpath',variant=opt.variant,width=w,height=h,config=cfg.to_dict(),target=record['task'],fixations=fix,validity=mask.metadata(),duration_ms=clock,assumptions=['Observed human prefixes condition perception and memory; no saccade-gap times available','Next human points used only for scoring frozen candidates and choices','No target ground-truth box passed to perception or selection'])
    return result,metrics,labels,contexts
