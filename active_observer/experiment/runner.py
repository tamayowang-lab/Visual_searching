"""Three timing variants with unchanged Phase-2 policy and world-update functions."""
from dataclasses import asdict,replace
from .options import ExperimentOptions,StoppingOptions
from .stopping import StopController
from copy import deepcopy
from active_observer.active.options import ActiveOptions
from active_observer.active.masking import StimulusMask
from active_observer.active.policy import decision_candidates
from active_observer.active.trace import decision_trace
from active_observer.active.perception import ActivePerception,bounded_peripheral_sensor,update_workspace
import hashlib
import numpy as np
from active_observer.io.image import Camera
from active_observer.world.model import World
from active_observer.perception.foveation import observe
from active_observer.perception.objects import ClassicalObjects
from active_observer.perception.depth import update_observed_depth
from active_observer.perception.factory import ProviderFactory
from active_observer.attention.policy import score,choose
from active_observer.attention.processing import processing_context
from active_observer.attention.timing import TimingParameters,simulate_timing,saccade_duration_ms


def decay_without_observation(world,elapsed_ms):
    world.time+=elapsed_ms
    for m in world.objects:
        m.confidence*=np.exp(-elapsed_ms/world.config.memory_decay_ms);m.ior*=np.exp(-elapsed_ms/1200);m.visibility=0.;m.state='remembered' if m.confidence>.2 else 'uncertain'
        m.identity_uncertainty=min(1,m.identity_uncertainty+.02*elapsed_ms/world.config.memory_decay_ms);m.position_uncertainty=min(1,m.position_uncertainty+.04*elapsed_ms/world.config.memory_decay_ms)


def run_experiment(image,cfg,*,image_id='image',target='',start=None,object_provider=None,depth_provider=None,camera=None,provider_factory=None,include_depth_reference=True,valid_mask=None,stimulus_bounds=None):
    experiment=ExperimentOptions(**cfg.experiment).validate();target=target if experiment.behavior_mode=='search' else ''
    cfg=replace(cfg,active={**cfg.active,'enabled':True,'behavior_mode':experiment.behavior_mode})
    controller=StopController(experiment.behavior_mode,target,StoppingOptions(**cfg.stopping))
    h,w=image.shape[:2]
    if image.ndim!=3 or image.shape[2]!=3 or image.dtype!=np.uint8:raise ValueError('Expected uint8 RGB image')
    camera=camera or Camera(w,h)
    if (camera.width,camera.height)!=(w,h):raise ValueError('Camera/image dimensions differ')
    options=ActiveOptions(**cfg.active).validate();mask=StimulusMask(w,h,valid_mask,stimulus_bounds)
    gaze=tuple(start or ((w-1)/2,(h-1)/2));camera.bearing(*gaze);world=World(w,h,cfg)
    if options.valid_stimulus and not mask.valid(*gaze):raise ValueError('Starting fixation lies outside the valid stimulus')
    if object_provider is None:
        objects,selected_depth=(provider_factory or ProviderFactory()).get(cfg);object_provider=objects
        if depth_provider is None:depth_provider=selected_depth
    if options.behavior_mode=='search' and not target:raise ValueError('Search mode requires a target instruction')
    objects=object_provider or ClassicalObjects();perception=ActivePerception(objects,options,mask,cfg.perception.get('cache_directory','.cache/observer')+'/peripheral') if (options.peripheral_guidance or options.adaptive_candidates or options.dynamic_perception or options.selection=='balanced') else None
    depth=depth_provider.predict(image) if depth_provider is not None and cfg.depth_enabled else None
    selection=np.random.default_rng(cfg.seed);timing_rng=np.random.default_rng(np.random.SeedSequence([cfg.seed,1]));params=TimingParameters(**cfg.timing).validate();fixations=[];saccades=[];elapsed=0.;pending=None;reason='administrative timeout';outcome='unresolved';administrative=True
    step=0
    while elapsed<experiment.administrative_timeout_ms:
        if experiment.administrative_max_steps is not None and step>=experiment.administrative_max_steps:reason='administrative step safeguard';break
        onset=elapsed;before=deepcopy(world.snapshot());proto_before=deepcopy(perception.hypotheses) if perception else [];observation=observe(image,gaze,cfg);detections=objects.detect(observation)
        if perception:
            sensor=bounded_peripheral_sensor(image,gaze,observation.radius_px,options.coarse_width) if options.peripheral_sensor=='bounded_rgb' else None
            perception.observe(observation,detections,sensor)
        plans,_=decision_candidates(world,observation,target,cfg,options,mask,perception);processing=processing_context(world,observation,detections,params.sensory_tau_ms)
        timing_rng_before=deepcopy(timing_rng.bit_generator.state)
        timing=simulate_timing([p['utility'] for p in plans],processing,cfg.timing_model,params,timing_rng,remaining_ms=experiment.administrative_timeout_ms-elapsed,heuristic_terms=pending['terms'] if pending else None,initial=pending is None)
        dwell=timing['duration_ms']
        if dwell<=0:reason='time budget';break
        # Commit once at termination: same evidence/association/habituation functions as V2.
        world.update(detections,observation,dwell)
        if depth is not None:update_observed_depth(world,observation,depth,camera)
        if options.dynamic_perception:update_workspace(perception,observation,detections,dwell,world)
        scores,rejected=decision_candidates(world,observation,target,cfg,options,mask,perception);current=pending or dict(id='initial',kind='initial',terms={},uncertainty=1.,relative_depth=None,depth_uncertainty=1.)
        fixation=dict(image_id=image_id,step=step,x_px=float(gaze[0]),y_px=float(gaze[1]),x_norm=float(gaze[0]/w),y_norm=float(gaze[1]/h),duration_ms=dwell,fixation_onset_ms=onset,fixation_offset_ms=onset+dwell,chosen_candidate=current['id'],candidate_kind=current['kind'],mode='initial' if pending is None else 'stay' if pending.get('distance',1)<1 else 'shift',utility_terms=current['terms'],candidate_details=current,random_seed=cfg.seed,foveation_geometry=observation.geometry,memory=world.snapshot(),next_candidates=scores,timing=timing,right_censored=timing['right_censored'],saccade_onset_ms=None,saccade_duration_ms=0.,next_fixation_onset_ms=None)
        fixation['timing']['candidate_ids']=[p['id'] for p in plans];fixation['timing']['plan_winner_id']=plans[timing['winner_index']]['id'] if timing['winner_index'] is not None else None
        trace=decision_trace(step,observation,detections,before,deepcopy(world.snapshot()),scores,rejected,mask,maps=perception.snapshot('' if options.behavior_mode=='explore' else target) if perception else None,enforce_validity=options.valid_stimulus)
        trace['onset_candidates']=deepcopy(plans);trace['perceptual_memory_before']=proto_before;trace['perceptual_memory_after']=deepcopy(perception.hypotheses) if perception else [];trace['timing_rng_state_before_json']=__import__('json').dumps(timing_rng_before)
        trace['incoming_choice']=fixations[-1]['decision_trace']['selection'] if fixations else dict(action='supplied starting location',location=list(gaze))
        fixation['decision_trace']=trace
        fixations.append(fixation);elapsed+=dwell
        stop=controller.assess(world,observation,detections,scores,perception,mask,elapsed,dwell)
        trace['experimental_control']=dict(behavior_mode=experiment.behavior_mode,observation_condition='autonomous_source_trajectory',requested_condition=experiment.observation_condition,elapsed_ms=elapsed)
        trace['stopping']=stop
        if not timing['completed']:
            reason='administrative fixation-timing safeguard' if timing['safety_timeout'] else 'administrative timeout';break
        if elapsed>=experiment.administrative_timeout_ms-1e-9:reason='administrative timeout';break
        if stop['decision']=='stop':reason=stop['reason'];outcome=stop['terminal_outcome'];administrative=False;break
        if experiment.administrative_max_steps is not None and step+1>=experiment.administrative_max_steps:reason='administrative step safeguard';break
        rng_before=deepcopy(selection.bit_generator.state);uniform=float(deepcopy(selection).random())
        if params.selection_mode=='race' and timing['winner_index'] is not None:
            planned=plans[timing['winner_index']];pending=dict(next((p for p in scores if p['id']==planned['id']),planned))
        else:pending=dict(choose(scores,selection))
        trace['selection']=dict(action='fixate',chosen_candidate=pending['id'],location=[pending['x'],pending['y']],probability=pending['probability'] if params.selection_mode=='softmax' else None,candidate_softmax_probability=pending['probability'],selection_source='after_observation' if params.selection_mode=='softmax' else 'onset_readiness_race',uniform_draw=uniform if params.selection_mode=='softmax' else None,method=params.selection_mode,rng_state_before_json=__import__('json').dumps(rng_before) if params.selection_mode=='softmax' else None,reason='sampled actual candidate distribution')
        distance=float(np.hypot(pending['x']-gaze[0],pending['y']-gaze[1]));pending['distance']=distance;world.consecutive_stays=world.consecutive_stays+1 if pending['kind']=='stay' else 0
        execution=saccade_duration_ms(distance,params,cfg.pixels_per_degree);observed_execution=min(execution,max(0,experiment.administrative_timeout_ms-elapsed));fixation['saccade_onset_ms']=elapsed if distance>=1 else None;fixation['saccade_duration_ms']=execution;fixation['next_fixation_onset_ms']=elapsed+execution;fixation['next_chosen_candidate']=pending['id'];fixation['next_fixation_xy']=[pending['x'],pending['y']]
        if distance>=1:saccades.append(dict(onset_ms=elapsed,duration_ms=execution,observed_duration_ms=observed_execution,amplitude_px=distance,amplitude_degrees=distance/cfg.pixels_per_degree if cfg.pixels_per_degree else None,right_censored=observed_execution<execution,from_xy=list(gaze),to_xy=[pending['x'],pending['y']]))
        elapsed+=observed_execution;decay_without_observation(world,observed_execution)
        if observed_execution<execution or elapsed>=experiment.administrative_timeout_ms-1e-9:reason='administrative timeout during saccade';break
        gaze=(pending['x'],pending['y']);step+=1
    if fixations and fixations[-1]['decision_trace']['selection']['action']=='stop':fixations[-1]['decision_trace']['selection']['reason']=reason
    for f in fixations:
        f['decision_trace']['selection']['arrived']=f.get('next_fixation_onset_ms') is not None and f['next_fixation_onset_ms']<elapsed+1e-9 and f['step']+1<len(fixations)
    if fixations:
        fixations[-1]['decision_trace']['stopping'].update(applied_decision='administrative_stop' if administrative else 'cognitive_stop',terminal_outcome=outcome,administrative_timeout=administrative,termination_reason=reason)
    return dict(schema_version=5,model='rgb-active-observer-phase5',experimental_control=asdict(experiment),variant=options.variant,validity=mask.metadata(),image_id=image_id,image_sha256=hashlib.sha256(image.tobytes()).hexdigest(),width=w,height=h,config=cfg.to_dict(),camera=camera.metadata(),target=target,fixations=fixations,saccades=saccades,duration_ms=elapsed,stop_reason=reason,termination=dict(action='stop',reason=reason,elapsed_ms=elapsed,outcome=outcome,administrative_timeout=administrative,cognitive_completion=not administrative),depth=dict(available=depth is not None,reference=depth.reference() if depth is not None and include_depth_reference else None,metadata=depth.metadata if depth is not None else None),perception=objects.metadata() if hasattr(objects,'metadata') else dict(provider=type(objects).__name__),timing_parameters=asdict(params),assumptions=['Opt-in Phase 4B path; consult active flags for changes; archived runners retained','Online saturating processing proxy drives hold; world state committed once per fixation to avoid tick-count-dependent familiarity','Primary softmax mode preserves spatial selector; competing plans control readiness, optional race selection is separate','Motor preparation is part of fixation; saccade execution is excluded from fixation duration','No future human fixation information; timing parameters require train/validation calibration'])
