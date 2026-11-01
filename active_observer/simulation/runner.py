"""Sequential RGB observer; no novel viewpoints, head/eye controller or test labels."""
from active_observer.perception.depth import update_observed_depth
import hashlib
import numpy as np
from active_observer.config import Config
from active_observer.io.image import Camera
from active_observer.perception.foveation import observe
from active_observer.perception.objects import ClassicalObjects
from active_observer.world.model import World
from active_observer.attention.policy import score,choose
from active_observer.attention.dynamics import duration_ms


def run_phase2(image: np.ndarray,config: Config | None=None,*,image_id="image",target="",start=None,object_provider=None,depth_provider=None,camera=None,provider_factory=None,include_depth_reference=True):
    cfg=(config or Config()).validate();h,w=image.shape[:2]
    if image.ndim!=3 or image.shape[2]!=3 or image.dtype!=np.uint8: raise ValueError("Expected uint8 RGB image")
    camera=camera or Camera(w,h)
    if (camera.width,camera.height)!=(w,h): raise ValueError("Camera/image dimensions differ")
    gaze=tuple(start or ((w-1)/2,(h-1)/2));camera.bearing(*gaze)
    world=World(w,h,cfg)
    if object_provider is None and (provider_factory is not None or cfg.perception.get('objects',{}).get('provider','classical')!='classical' or cfg.perception.get('depth',{}).get('provider','none')!='none'):
        from active_observer.perception.factory import ProviderFactory
        selected_objects,selected_depth=(provider_factory or ProviderFactory()).get(cfg)
        object_provider=selected_objects
        if depth_provider is None:depth_provider=selected_depth
    objects=object_provider or ClassicalObjects()
    selection=np.random.default_rng(cfg.seed);timing=np.random.default_rng(np.random.SeedSequence([cfg.seed,1]))
    depth_frame=depth_provider.predict(image) if depth_provider is not None and cfg.depth_enabled else None
    fixations=[];reason="fixation budget";elapsed=0.;pending=None
    for step in range(cfg.max_fixations):
        observation=observe(image,gaze,cfg)
        dwell=duration_ms(pending,cfg,timing) if pending else float(np.clip(250,cfg.min_duration_ms,cfg.max_duration_ms))
        dwell=min(dwell,cfg.time_budget_ms-elapsed)
        if dwell<=0: reason="time budget";break
        detections=objects.detect(observation);world.update(detections,observation,dwell)
        if depth_frame is not None: update_observed_depth(world,observation,depth_frame,camera)
        scores=score(world,observation,target,cfg)
        current=pending or dict(id="initial",kind="initial",terms={},uncertainty=1.,relative_depth=None,depth_uncertainty=1.)
        fixations.append(dict(image_id=image_id,step=step,x_px=float(gaze[0]),y_px=float(gaze[1]),x_norm=float(gaze[0]/w),y_norm=float(gaze[1]/h),duration_ms=float(dwell),chosen_candidate=current['id'],candidate_kind=current['kind'],mode="initial" if pending is None else "stay" if current['kind']=='stay' else "standby" if current["id"]=="standby" else "stay" if current.get('distance',1)<observation.radius_px/2 else "shift",utility_terms=current['terms'],candidate_details=current,random_seed=cfg.seed,foveation_geometry=observation.geometry,memory=world.snapshot(),next_candidates=scores))
        elapsed+=dwell
        if cfg.stop_on_target and target and any(m.state=='observed' and observation.central[min(h-1,int(m.centroid[1])),min(w-1,int(m.centroid[0]))] and m.probabilities.get(target,0)*m.confidence>=cfg.target_threshold for m in world.objects): reason="explicit target detection";break
        if elapsed>=cfg.time_budget_ms: reason="time budget";break
        pending=dict(choose(scores,selection));world.consecutive_stays=world.consecutive_stays+1 if pending['kind']=='stay' else 0;pending['distance']=float(np.hypot(pending['x']-gaze[0],pending['y']-gaze[1]));gaze=(pending['x'],pending['y'])
    return dict(schema_version=1,model="rgb-active-observer-v2",image_id=image_id,image_sha256=hashlib.sha256(image.tobytes()).hexdigest(),width=w,height=h,config=cfg.to_dict(),camera=camera.metadata(),target=target,fixations=fixations,duration_ms=elapsed,stop_reason=reason,termination=dict(action="stop",reason=reason,elapsed_ms=elapsed),depth=dict(available=depth_frame is not None,reference=depth_frame.reference() if depth_frame is not None and include_depth_reference else None,metadata=depth_frame.metadata if depth_frame is not None else None,reason="provider supplied" if depth_frame is not None else "No depth provider or depth ablated"),perception=objects.metadata() if hasattr(objects,'metadata') else dict(provider=type(objects).__name__,semantic_input='foveated RGB; private manual test boxes gated centrally'),assumptions=["RGB static scene; no fabricated viewpoints","Heuristic utility and uncalibrated belief confidence","Classical foveated proto-objects unless a replaceable provider is supplied","No fitted human-likeness claim"])


def run(image,config=None,**kwargs):
    """Dispatch without changing the archived Phase-2 numerical path by default."""
    cfg=(config or Config()).validate()
    if cfg.experiment.get('enabled',False):
        from active_observer.experiment.runner import run_experiment
        return run_experiment(image,cfg,**kwargs)
    if cfg.active.get('enabled',False):
        from active_observer.active.runner import run_active
        return run_active(image,cfg,**kwargs)
    if cfg.timing_model=='phase2':return run_phase2(image,cfg,**kwargs)
    from .timed_runner import run_timed
    return run_timed(image,cfg,**kwargs)
