"""Single reconstruction-oriented runner; every sensor receives current input only."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import numpy as np
from active_observer.config import Config
from active_observer.io.image import Camera
from active_observer.perception.foveation import observe
from active_observer.perception.factory import ProviderFactory
from active_observer.perception.depth import update_observed_depth,record_depth
from active_observer.world.model import World
from active_observer.world.ensemble import summarize
from active_observer.attention.priority import PriorityField
from active_observer.attention.timing import fixation_duration,saccade_duration
from .stopping import StopController


def central_detections(detections,observation):
    h,w=observation.rgb.shape[:2];kept=[]
    for d in detections:
        if not np.isfinite(d.confidence) or not 0<=d.confidence<=1:raise ValueError('Invalid detector confidence')
        if not d.probabilities or not all(np.isfinite(v) and 0<=v<=1 for v in d.probabilities.values()):raise ValueError('Invalid class support')
        x1,y1,x2,y2=d.bbox
        if not np.isfinite(d.bbox).all() or not (0<=x1<x2<=w and 0<=y1<y2<=h):raise ValueError('Invalid detector box')
        x,y=(x1+x2)/2,(y1+y2)/2
        if d.source!='semantic' or np.hypot(x-observation.gaze[0],y-observation.gaze[1])>observation.radius_px:continue
        gx,gy=observation.gaze;rad=observation.radius_px
        clipped=[max(0,x1,gx-rad),max(0,y1,gy-rad),min(w,x2,gx+rad),min(h,y2,gy+rad)]
        if clipped[2]<=clipped[0] or clipped[3]<=clipped[1]:continue
        partial=d.partial or any(abs(a-b)>1e-9 for a,b in zip(d.bbox,clipped))
        partial=partial or any(np.hypot(px-gx,py-gy)>rad for px,py in [(clipped[0],clipped[1]),(clipped[2],clipped[3])])
        kept.append(replace(d,bbox=clipped,partial=bool(partial)))
    return kept


def run(image,config=None,*,image_id='image',start=None,object_provider=None,depth_provider=None,camera=None,frame_callback=None):
    cfg=(config or Config()).validate()
    if image.ndim!=3 or image.shape[2]!=3 or image.dtype!=np.uint8:raise ValueError('Expected uint8 RGB photograph')
    h,w=image.shape[:2];camera=camera or Camera(w,h)
    if (camera.width,camera.height)!=(w,h):raise ValueError('Camera/image dimensions differ')
    gaze=tuple(start) if start is not None else ((w-1)/2,(h-1)/2)
    if not (0<=gaze[0]<w and 0<=gaze[1]<h):raise ValueError('Off-screen starting gaze')
    factory=ProviderFactory()
    if object_provider is None:object_provider=factory.get_objects(cfg)
    if depth_provider is None:depth_provider=factory.get_depth(cfg)
    world=World(w,h,cfg);field=PriorityField(w,h,np.ones((h,w),bool),cfg.priority_width,cfg.spatial_temperature)
    controller=StopController(cfg);selection=np.random.default_rng(cfg.seed)
    timing=np.random.default_rng(np.random.SeedSequence([cfg.seed,1]))
    fixations=[];elapsed=0.;termination=None
    while elapsed<cfg.administrative_timeout_ms:
        before=deepcopy(world.snapshot());onset=elapsed
        observation=observe(image,gaze,cfg)
        sensory_hash=hashlib.sha256(observation.rgb.tobytes()).hexdigest()
        observation.rgb.setflags(write=False)
        detections=central_detections(object_provider.detect(observation),observation)
        # There is no depth call on `image`, including at initialization or cache fill.
        depth=depth_provider.predict(observation.rgb.copy()) if depth_provider is not None else None
        if depth is not None:depth.validate((h,w))
        depth_record=record_depth(depth,sensory_hash,len(fixations)) if depth is not None else dict(available=False,reason='Depth provider disabled; no depth or geometry is fabricated')
        summary=summarize(observation,cfg.peripheral_width)
        planned=fixation_duration(cfg,timing,initial=not fixations)
        dwell=min(planned,cfg.administrative_timeout_ms-elapsed)
        measured=world.update(detections,observation,dwell,summary)
        if depth is not None:update_observed_depth(world,observation,depth,camera,depth_record)
        field.update(observation,dwell);elapsed+=dwell
        priority=field.score(observation,world,summary);selected=field.choose(priority,selection)
        coverage=float((1-np.exp(-field.exposure)).mean())
        decision=controller.assess(priority,field.valid,coverage,elapsed,depth is not None)
        timeout=elapsed>=cfg.administrative_timeout_ms
        if timeout:decision.update(decision='stop',outcome='administrative_timeout',reason='Elapsed-time safety limit; reconstruction completion is unknown',administrative_timeout=True)
        if decision['decision']=='stop':termination=dict(decision)
        f=dict(step=len(fixations),gaze=list(gaze),onset_ms=onset,offset_ms=elapsed,duration_ms=dwell,
               sensory_sha256=sensory_hash,object_input_sha256=sensory_hash,radius_px=observation.radius_px,
               depth_input_sha256=hashlib.sha256(observation.rgb.tobytes()).hexdigest() if depth is not None else None,
               depth_metadata=deepcopy(depth.metadata) if depth is not None else None,depth=depth_record,
               memory_before=before,memory_after=deepcopy(world.snapshot()),ensemble=summary.to_dict(),
               priority=field.snapshot(priority),selected_next=None if termination else selected,
               predicted_information_gain=selected['estimated_information_gain'],measured_proxies=measured,
               stopping=decision,right_censored=dwell<planned,saccade_duration_ms=0.)
        fixations.append(f)
        if frame_callback:frame_callback(f,observation)
        if termination:break
        distance=float(np.hypot(selected['x']-gaze[0],selected['y']-gaze[1]))
        planned_execution=saccade_duration(distance)
        execution=min(planned_execution,cfg.administrative_timeout_ms-elapsed)
        f['saccade_planned_ms']=planned_execution
        f['saccade_to']=list([selected['x'],selected['y']])
        f['saccade_duration_ms']=execution;elapsed+=execution;world.elapse(execution)
        field.ior*=np.exp(-execution/1800)
        if elapsed>=cfg.administrative_timeout_ms:
            termination=dict(decision='stop',outcome='administrative_timeout',reason='Safety limit during saccade',administrative_timeout=True,reconstruction_validated=False)
            f['termination_after_saccade']=dict(termination);break
        gaze=(selected['x'],selected['y'])
    return dict(schema_version=2,model='gaze-dependent-reconstruction-foundation',image_id=image_id,width=w,height=h,
                image_sha256=hashlib.sha256(image.tobytes()).hexdigest(),config=cfg.to_dict(),camera=camera.metadata(),
                fixations=fixations,duration_ms=elapsed,termination=termination,
                capabilities=dict(gaze_dependent_depth=depth_provider is not None,ensemble_object_counts=False,
                                  complete_cuboid_extent=False,cross_fixation_depth_scale_alignment=False,
                                  measured_3d_reconstruction_improvement=False),
                providers=dict(objects=object_provider.metadata() if hasattr(object_provider,'metadata') else {'provider':type(object_provider).__name__},depth={'configured':cfg.depth,'runtime':type(depth_provider).__name__,'input':'current foveated RGB only'} if depth_provider is not None else {'provider':'none'}),
                status='Diagnostic reconstruction hypotheses; independent depth scales, unknown extent and unvalidated improvement')
