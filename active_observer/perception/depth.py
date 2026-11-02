"""Relative-only depth contract; aligned RGB-D can later implement this protocol."""
from typing import Protocol
from dataclasses import dataclass
from copy import deepcopy
from pathlib import Path
import hashlib,json
import numpy as np

@dataclass
class DepthFrame:
    inverse_depth: np.ndarray
    valid: np.ndarray
    quality: np.ndarray
    metadata: dict

    def validate(self,shape=None):
        arrays=[np.asarray(self.inverse_depth),np.asarray(self.valid),np.asarray(self.quality)]
        if arrays[0].ndim!=2 or any(a.shape!=arrays[0].shape for a in arrays):raise ValueError('Depth arrays must share HxW shape')
        if shape is not None and arrays[0].shape!=shape:raise ValueError('Depth must align with gaze-dependent input')
        if arrays[1].dtype!=bool:raise ValueError('Depth validity must be boolean')
        if not np.isfinite(arrays[0]).all() or np.any((arrays[0]<0)|(arrays[0]>1)):raise ValueError('Stored inverse depth must be finite and normalized to [0,1]')
        if not np.isfinite(arrays[2]).all() or np.any((arrays[2]<0)|(arrays[2]>1)):raise ValueError('Depth quality must be finite in [0,1]')
        return self

    def reference(self):
        return dict(values=np.where(self.valid,self.inverse_depth,None).tolist(),valid=self.valid.tolist(),quality=self.quality.tolist(),metadata=self.metadata,visibility="Estimator reference only; unobserved pixels are not in world memory")

class DepthProvider(Protocol):
    def predict(self,image: np.ndarray) -> DepthFrame: ...


def normalize(raw: np.ndarray,*,valid=None,quality=None,estimator="precomputed",kind="inverse_depth") -> DepthFrame:
    raw=np.asarray(raw,float)
    if raw.ndim!=2: raise ValueError("Depth must be HxW")
    if kind not in {"inverse_depth","relative_range"}: raise ValueError("Depth kind must be explicit")
    if valid is not None and np.asarray(valid).shape!=raw.shape: raise ValueError('Depth mask shape mismatch')
    mask=np.isfinite(raw) if valid is None else np.isfinite(raw)&np.asarray(valid,bool)
    if mask.shape!=raw.shape: raise ValueError("Depth mask shape mismatch")
    if kind=='relative_range': mask &= raw>0;raw=np.divide(1.,raw,out=np.full_like(raw,np.nan),where=mask)
    q=np.full_like(raw,.5) if quality is None else np.asarray(quality,float)
    if q.shape!=raw.shape or not np.isfinite(q).all() or np.any((q<0)|(q>1)): raise ValueError("Quality must be finite HxW in [0,1]")
    out=np.zeros_like(raw);flags=[];lo=hi=None
    if mask.any():
        lo,hi=np.quantile(raw[mask],[.02,.98]).tolist()
        if hi-lo<1e-9: out[mask]=.5;q=np.minimum(q,.05);flags.append("constant_or_degenerate")
        else: out=np.clip((raw-lo)/(hi-lo),0,1)
    else: flags.append("no_valid_depth")
    out[~mask]=0;q=np.where(mask,q,0)
    return DepthFrame(out,mask,q,dict(estimator=estimator,normalization="per-image 2nd/98th percentiles of inverse depth",normalization_bounds=[lo,hi],source_kind=kind,units="normalized inverse depth; relative range = 1/(0.1+D), not meters",calibrated_uncertainty=False,quality_flags=flags))

class TransformersDepth:
    """Optional local-checkpoint CPU adapter; never downloads without explicit opt-in."""
    def __init__(self,model_id: str,*,revision: str="main",allow_download: bool=False,output_kind: str='inverse_depth'):
        if output_kind not in {'inverse_depth','relative_range'}: raise ValueError('Declare model output kind')
        self.output_kind=output_kind;self.model_id=model_id;self.revision=revision;self.allow_download=allow_download;self.processor=None
    def predict(self,image):
        try:
            import torch
            from PIL import Image
            from transformers import AutoImageProcessor,AutoModelForDepthEstimation
        except ImportError as exc: raise RuntimeError("Install the pretrained extra and supply a compatible checkpoint") from exc
        if self.processor is None:
            kw=dict(revision=self.revision,local_files_only=not self.allow_download)
            self.processor=AutoImageProcessor.from_pretrained(self.model_id,**kw)
            self.model=AutoModelForDepthEstimation.from_pretrained(self.model_id,**kw).eval().to('cpu')
        with torch.inference_mode():
            raw=self.model(**self.processor(images=Image.fromarray(image),return_tensors='pt')).predicted_depth
            depth=torch.nn.functional.interpolate(raw[:,None],size=image.shape[:2],mode='bicubic',align_corners=False)[0,0].cpu().numpy()
        f=normalize(depth,estimator=self.model_id,kind=self.output_kind)
        f.metadata.update(revision=self.revision,preprocessing="model AutoImageProcessor; bicubic output resize",device="cpu",quality_flags=["uncalibrated quality proxy","perturbation consistency not measured"])
        return f

class CachedDepth:
    """Cache key includes pixels, shape, and caller's explicit model identity."""
    def __init__(self,provider: DepthProvider,directory,identity: str):
        self.provider=provider;self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True);self.identity=identity
    def predict(self,image):
        key=hashlib.sha256(image.tobytes()+str(image.shape).encode()+self.identity.encode()).hexdigest();p=self.directory/(key+'.npz');meta=self.directory/(key+'.json')
        if p.exists() and meta.exists():
            with np.load(p,allow_pickle=False) as a:return DepthFrame(a['depth'],a['valid'],a['quality'],json.loads(meta.read_text(encoding='utf-8')))
        f=self.provider.predict(image);np.savez_compressed(p,depth=f.inverse_depth,valid=f.valid,quality=f.quality);meta.write_text(json.dumps(f.metadata,allow_nan=False), encoding='utf-8');return f


def array_hash(array):
    a=np.ascontiguousarray(array)
    return hashlib.sha256(a.dtype.str.encode()+str(a.shape).encode()+a.tobytes()).hexdigest()


def record_depth(frame,input_sha256,step):
    """Exact arrays plus an explicit independently normalized coordinate contract."""
    frame.validate()
    arrays={name:np.asarray(getattr(frame,name)).copy() for name in ['inverse_depth','valid','quality']}
    hashes={name:array_hash(a) for name,a in arrays.items()}
    frame_id=f'depth-{step:04d}-'+hashes['inverse_depth'][:12]
    return dict(available=True,frame_id=frame_id,input_sha256=input_sha256,array_hashes=hashes,
                metadata=deepcopy(frame.metadata),
                coordinate_contract=dict(axes='x right, y down, z forward',depth='normalized inverse depth',
                    z_mapping='1/(0.1+D); display/geometry proxy, not calibrated physical depth',
                    normalization_scope='independent per observation',local_scale_id=frame_id,
                    shared_scale_id=None,provider_scale_claim=frame.metadata.get('scale_id'),
                    cross_fixation_alignment='unavailable; do not fuse',units='nonmetric relative proxy units'),
                arrays={name:dict(dtype=a.dtype.str,values=a.tolist()) for name,a in arrays.items()})


def update_observed_depth(world,observation,frame: DepthFrame,camera,record=None):
    """Keep every central surface estimate as a separate scale hypothesis."""
    from active_observer.world.model import ReconstructionEstimate
    h,w=observation.rgb.shape[:2];frame.validate((h,w))
    record=record or record_depth(frame,hashlib.sha256(observation.rgb.tobytes()).hexdigest(),len(world.ensembles)-1)
    for m in world.objects:
        if m.last_seen!=world.time:continue
        x,y,x2,y2=m.bbox;mask=np.zeros((h,w),bool)
        mask[max(0,int(np.ceil(y))):min(h,int(np.ceil(y2))),max(0,int(np.ceil(x))):min(w,int(np.ceil(x2)))]=True
        mask &= observation.central & frame.valid
        if not mask.any():continue
        values=frame.inverse_depth[mask];median=float(np.median(values));spread=float(np.quantile(values,.75)-np.quantile(values,.25))
        quality=float(np.median(frame.quality[mask]));uncertainty=float(np.clip(1-quality+spread,0,1))
        z=1/(.1+median);center=camera.unproject_z(*m.centroid,z)
        width=(x2-x)*z/camera.fx;height=(y2-y)*z/camera.fy
        corners=[camera.unproject_z(px,py,z).tolist() for px,py in [(x,y),(x2,y),(x2,y2),(x,y2)]]
        hypothesis=dict(time_ms=world.time,frame_id=record['frame_id'],scale_id=None,
                        input_sha256=record['input_sha256'],depth_array_sha256=record['array_hashes']['inverse_depth'],
                        relative_depth=z,center=center.tolist(),dimensions=[float(width),float(height),None],
                        front_corners=corners,orientation=None,bbox=list(m.bbox),extent_is_partial=m.extent_is_partial,
                        depth_uncertainty=uncertainty,dimension_uncertainty=[max(uncertainty,m.localization_uncertainty)]*2+[1.],
                        sample_count=int(mask.sum()),sample_median_inverse_depth=median,mean_quality=quality,
                        alignment='unaligned independent observation; rectangle is image-plane support, not object orientation')
        history=m.reconstruction.observation_history+[hypothesis]
        m.reconstruction=ReconstructionEstimate(relative_depth=z,center=center.tolist(),
                    dimensions=hypothesis['dimensions'],depth_uncertainty=uncertainty,
                    dimension_uncertainty=hypothesis['dimension_uncertainty'],
                    frame_id=record['frame_id'],observation_history=history)
