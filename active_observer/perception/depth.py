"""Relative-only depth contract; aligned RGB-D can later implement this protocol."""
from typing import Protocol
from dataclasses import dataclass
from pathlib import Path
import hashlib,json
import numpy as np

@dataclass
class DepthFrame:
    inverse_depth: np.ndarray
    valid: np.ndarray
    quality: np.ndarray
    metadata: dict

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

class ConstantDepth:
    def predict(self,image):
        return normalize(np.ones(image.shape[:2]),quality=np.full(image.shape[:2],.05),estimator="deterministic constant dummy (not inferred depth)")

class ArrayDepth:
    def __init__(self,raw,*,valid=None,quality=None,kind="inverse_depth"):
        self.frame=normalize(raw,valid=valid,quality=quality,kind=kind)
    def predict(self,image):
        if self.frame.inverse_depth.shape!=image.shape[:2]: raise ValueError("Depth/image dimensions differ; align explicitly")
        return self.frame

class PrecomputedDepth(ArrayDepth):
    def __init__(self,path,kind="inverse_depth"):
        data=np.load(path,allow_pickle=False)
        if isinstance(data,np.lib.npyio.NpzFile):
            try: super().__init__(data['depth'],valid=data['valid'] if 'valid' in data else None,quality=data['quality'] if 'quality' in data else None,kind=kind)
            finally:data.close()
        else: super().__init__(data,kind=kind)
        self.frame.metadata['source_file']=Path(path).name

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


def update_observed_depth(world,observation,frame: DepthFrame,camera):
    """Only central observed surface samples can enter uncertain object memory."""
    h,w=observation.rgb.shape[:2]
    if frame.inverse_depth.shape!=(h,w): raise ValueError("Unaligned depth frame")
    for m in world.objects:
        if m.state!='observed': continue
        x,y,x2,y2=m.bbox;mask=np.zeros((h,w),bool);mask[max(0,int(y)):min(h,int(np.ceil(y2))),max(0,int(x)):min(w,int(np.ceil(x2)))]=True
        mask &= observation.central & frame.valid
        if not mask.any(): continue
        values=frame.inverse_depth[mask];quality=float(np.median(frame.quality[mask]));median=float(np.median(values));dispersion=float(np.quantile(values,.75)-np.quantile(values,.25))
        m.depth_dispersion=dispersion
        m.depth_uncertainty=float(np.clip(1-quality+dispersion,0,1));m.relative_depth=1/(.1+median)
        m.xyz=(camera.bearing(*m.centroid)*m.relative_depth).tolist();image_var=m.image_covariance_px or [[1.,0.],[0.,1.]]
        m.covariance=np.diag([image_var[0][0]*(m.relative_depth/camera.fx)**2,image_var[1][1]*(m.relative_depth/camera.fy)**2,m.depth_uncertainty*m.relative_depth**2]).tolist()
        # Bounding volume spans only observed image extents and the dispersion proxy.
        corners=[]
        for px in [m.bbox[0],min(camera.width-1,m.bbox[2])]:
            for py in [m.bbox[1],min(camera.height-1,m.bbox[3])]:
                for scale in [max(.1,1-.5*m.depth_uncertainty),1+.5*m.depth_uncertainty]:corners.append(camera.bearing(px,py)*m.relative_depth*scale)
        corners=np.array(corners);m.volume=dict(min=corners.min(axis=0).tolist(),max=corners.max(axis=0).tolist(),units='relative image-anchored units',extent_is_partial=m.partial,interpretation='uncertainty envelope of observed surfaces, not complete physical shape')
