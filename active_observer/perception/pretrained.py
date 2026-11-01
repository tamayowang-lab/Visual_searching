"""Verified-checkpoint CPU adapters with explicit central semantic gating.

No downloads occur here. Use scripts/setup_perception.py to obtain pinned assets.
"""
from pathlib import Path
from dataclasses import asdict
import hashlib,json,time,sys
import numpy as np
import cv2
from .objects import Detection,ClassicalObjects,descriptor
from .depth import normalize

SSD_SHA='a79551df90c79834bcd3bb3845ef9d966b5449a3a9b2833ae8404778ca5d65d2'
MIDAS_SHA='70d6b9c891758c67f974a6097fb0c608c7ee67fb81ac3e5588847d5596d56fca'
MIDAS_REVISION='454597711a62eabcbf7d1e89f3fb9f569051ac9b'


def verify(path,expected):
    path=Path(path)
    if not path.is_file():raise RuntimeError(f'Missing pretrained checkpoint: {path}; run scripts/setup_perception.py. No classical fallback was substituted.')
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if digest!=expected:raise ValueError(f'Checkpoint SHA256 mismatch: {path.name}')
    return digest


def iou(a,b):
    inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/max(union,1e-9)

class TorchvisionFoveal:
    """SSDLite sees a foveated context crop; only central labels/local boxes leave it."""
    def __init__(self,checkpoint,threshold=.35,context_scale=3.,threads=2):
        self.sha=verify(checkpoint,SSD_SHA)
        try:
            import torch,torchvision
            from torchvision.models.detection import ssdlite320_mobilenet_v3_large,SSDLite320_MobileNet_V3_Large_Weights
        except ImportError as e:raise RuntimeError('Install the vision extra: torch, torchvision, timm. No fallback.') from e
        self.torch=torch;torch.set_num_threads(threads);self.version=torchvision.__version__;self.threshold=threshold;self.context_scale=context_scale
        self.model=ssdlite320_mobilenet_v3_large(weights=None,weights_backbone=None).eval().to('cpu')
        self.model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True));self.labels=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT.meta['categories'];self.calls=0;self.seconds=0.
    def metadata(self):return dict(provider='torchvision_ssdlite_foveal',checkpoint_sha256=self.sha,architecture='ssdlite320_mobilenet_v3_large',torchvision=self.version,preprocessing='RGB /255; torchvision detection resize/normalization',semantic_gate='center inside central circle; box clipped to central square',context_scale=self.context_scale,threshold=self.threshold,device='cpu',full_image_semantics=False)
    def detect(self,observation):
        torch=self.torch;rgb=observation.rgb;h,w=rgb.shape[:2];x,y=observation.gaze;rad=observation.radius_px;context=rad*self.context_scale
        x1,y1=max(0,int(x-context)),max(0,int(y-context));x2,y2=min(w,int(np.ceil(x+context))),min(h,int(np.ceil(y+context)))
        tensor=torch.from_numpy(rgb[y1:y2,x1:x2].transpose(2,0,1).copy()).float()/255
        t=time.perf_counter()
        with torch.inference_mode():found=self.model([tensor])[0]
        self.calls+=1;self.seconds+=time.perf_counter()-t;semantic=[]
        for box,label,prob in zip(found['boxes'].tolist(),found['labels'].tolist(),found['scores'].tolist()):
            if prob<self.threshold:continue
            full=[box[0]+x1,box[1]+y1,box[2]+x1,box[3]+y1];cx,cy=(full[0]+full[2])/2,(full[1]+full[3])/2
            if np.hypot(cx-x,cy-y)>rad:continue
            local=[max(0,full[0],x-rad),max(0,full[1],y-rad),min(w,full[2],x+rad),min(h,full[3],y+rad)]
            if local[2]<=local[0] or local[3]<=local[1]:continue
            if any(iou(local,d.bbox)>.5 for d in semantic):continue
            name=self.labels[label];partial=any(abs(a-b)>.5 for a,b in zip(local,full))
            semantic.append(Detection(local,{name:float(prob),'unknown':1-float(prob)},descriptor(rgb,local),float(prob),.6,partial=partial,source='semantic'))
        coarse=[d for d in ClassicalObjects().detect(observation) if not any(iou(d.bbox,s.bbox)>.2 for s in semantic)]
        return semantic+coarse

class MidasSmall:
    """Pinned MiDaS 2.1 small decoder/backbone. Its map remains a private sensor output."""
    def __init__(self,checkpoint,source,revision=MIDAS_REVISION,threads=2):
        self.sha=verify(checkpoint,MIDAS_SHA)
        if revision!=MIDAS_REVISION:raise ValueError('Unsupported unpinned MiDaS revision')
        source=Path(source).resolve()
        if not (source/'midas/midas_net_custom.py').exists():raise RuntimeError('Missing pinned MiDaS source; run setup_perception.py')
        import torch,timm
        torch.set_num_threads(threads);self.torch=torch;self.source_hashes={name:hashlib.sha256((source/name).read_bytes()).hexdigest() for name in ['midas/midas_net_custom.py','midas/blocks.py']}
        sys.path.insert(0,str(source))
        import midas.blocks as blocks
        def backbone(use_pretrained,exportable=False):
            network=timm.create_model('tf_efficientnet_lite3',pretrained=False);network.act1=torch.nn.Identity()
            return blocks._make_efficientnet_backbone(network)
        blocks._make_pretrained_efficientnet_lite3=backbone
        from midas.midas_net_custom import MidasNet_small
        self.model=MidasNet_small(None,features=64,backbone='efficientnet_lite3',exportable=True,non_negative=True,blocks={'expand':True})
        self.model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True));self.model.eval();self.revision=revision;self.calls=0;self.seconds=0.
    def predict(self,image):
        torch=self.torch;h,w=image.shape[:2];factor=256/max(h,w);size=(max(32,int(round(w*factor/32))*32),max(32,int(round(h*factor/32))*32))
        inp=cv2.resize(image.astype(np.float32)/255,size,interpolation=cv2.INTER_CUBIC);inp=(inp-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
        start=time.perf_counter()
        with torch.inference_mode():raw=self.model(torch.from_numpy(inp.transpose(2,0,1).copy()).unsqueeze(0))[0].cpu().numpy()
        self.calls+=1;self.seconds+=time.perf_counter()-start
        raw=cv2.resize(raw,(w,h),interpolation=cv2.INTER_CUBIC);frame=normalize(raw,quality=np.full((h,w),.45),estimator='MiDaS v2.1 small')
        frame.metadata.update(checkpoint_sha256=self.sha,source_revision=self.revision,source_sha256=self.source_hashes,device='cpu',preprocessing='upper-bound aspect resize to multiples of 32; ImageNet RGB normalization; bicubic upsample',quality_flags=frame.metadata['quality_flags']+['heuristic_quality_not_calibrated','no_perturbation_consistency'])
        return frame

class CachedObjects:
    """Cache only actual foveated observations, never a hidden whole-image detector bank."""
    def __init__(self,provider,directory):
        self.provider=provider;self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True);self.hits=0;self.misses=0
        self.identity=json.dumps(provider.metadata(),sort_keys=True)
    def metadata(self):return dict(**self.provider.metadata(),cache='foveated RGB+gaze+radius+provider fingerprint; no cached full-image semantics')
    def detect(self,observation):
        key=hashlib.sha256(observation.rgb.tobytes()+json.dumps([observation.gaze,observation.radius_px,self.identity],sort_keys=True).encode()).hexdigest();p=self.directory/(key+'.json')
        if p.exists():self.hits+=1;return [Detection(**d) for d in json.loads(p.read_text(encoding='utf-8'))]
        self.misses+=1;detections=self.provider.detect(observation);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps([asdict(d) for d in detections],allow_nan=False),encoding='utf-8');tmp.replace(p);return detections
