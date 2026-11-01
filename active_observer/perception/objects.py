"""Replaceable sensor providers. Manual labels are revealed only centrally."""
from typing import Protocol
from dataclasses import dataclass
import cv2
import numpy as np
from .foveation import Observation

@dataclass
class Detection:
    bbox: list[float]
    probabilities: dict[str,float]
    appearance: list[float]
    confidence: float
    salience: float
    partial: bool=False
    source: str="proto"

class ObjectProvider(Protocol):
    def detect(self, observation: Observation) -> list[Detection]: ...


def descriptor(rgb, box):
    h,w=rgb.shape[:2];x1,y1,x2,y2=map(int,box)
    crop=rgb[max(0,y1):min(h,y2),max(0,x1):min(w,x2)]
    return (crop.mean(axis=(0,1))/255).tolist() if crop.size else [0.,0.,0.]

class ClassicalObjects:
    """Color components from the foveated image; names are colors, not semantics."""
    def detect(self, observation):
        rgb=observation.rgb;h,w=rgb.shape[:2];out=[]
        # Independent color channels give simple deterministic proto-objects.
        for i,name in enumerate(["red-region","green-region","blue-region"]):
            other=np.max(np.delete(rgb,i,axis=2),axis=2).astype(float)
            mask=((rgb[:,:,i].astype(float)-other>35)&(rgb[:,:,i]>100)).astype(np.uint8)
            n,labels,stats,centers=cv2.connectedComponentsWithStats(mask)
            for k in range(1,n):
                x,y,bw,bh,area=stats[k]
                if area < max(12,h*w*.002): continue
                cx,cy=centers[k];reliability=float(observation.reliability[min(h-1,int(cy)),min(w-1,int(cx))])
                conf=.85*reliability;box=[float(x),float(y),float(x+bw),float(y+bh)]
                out.append(Detection(box,{name:conf,"unknown":1-conf},descriptor(rgb,box),conf,float(min(1,area/(h*w*.08)))))
        return sorted(out,key=lambda d:-d.salience)[:32]

class ManualObjects:
    """Private fixture/proposal bank. Remote boxes and labels never leave adapter."""
    def __init__(self, records: list[dict]): self.records=records
    def detect(self, observation):
        h,w=observation.rgb.shape[:2];out=[]
        for rec in self.records:
            box=rec["bbox"];x1,y1,x2,y2=box
            if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h): raise ValueError("Manual box outside image")
            cx,cy=(x1+x2)/2,(y1+y2)/2
            if np.hypot(cx-observation.gaze[0],cy-observation.gaze[1]) > observation.radius_px: continue
            # A local observation bounds geometry, too: do not reveal full remote extents.
            rad=observation.radius_px;clipped=[max(x1,observation.gaze[0]-rad),max(y1,observation.gaze[1]-rad),min(x2,observation.gaze[0]+rad),min(y2,observation.gaze[1]+rad)]
            conf=.9;out.append(Detection(clipped,{rec['label']:conf,'unknown':1-conf},descriptor(observation.rgb,clipped),conf,.5))
        return out

class FovealTransformersObjects:
    """Optional CPU detector: only a foveal crop receives semantic inference.

    Peripheral proto-objects are classical color components. Downloads require
    explicit opt-in; model/checkpoint and revision are recorded in run metadata.
    """
    def __init__(self,model_id: str,*,revision: str='main',allow_download: bool=False,threshold: float=.4):
        self.model_id=model_id;self.revision=revision;self.allow_download=allow_download;self.threshold=threshold;self.processor=None
    def metadata(self):
        return dict(provider='foveal HuggingFace object detector + classical peripheral cues',model_id=self.model_id,revision=self.revision,device='cpu',semantic_input='central crop only',threshold=self.threshold,preprocessing='model AutoImageProcessor')
    def detect(self,observation):
        try:
            import torch
            from PIL import Image
            from transformers import AutoImageProcessor,AutoModelForObjectDetection
        except ImportError as exc:raise RuntimeError('Install pretrained extra and provide a compatible object-detection checkpoint') from exc
        if self.processor is None:
            kw=dict(revision=self.revision,local_files_only=not self.allow_download)
            self.processor=AutoImageProcessor.from_pretrained(self.model_id,**kw);self.model=AutoModelForObjectDetection.from_pretrained(self.model_id,**kw).eval().to('cpu')
        h,w=observation.rgb.shape[:2];x,y=observation.gaze;rad=observation.radius_px;x1,y1=max(0,int(x-rad)),max(0,int(y-rad));x2,y2=min(w,int(np.ceil(x+rad))),min(h,int(np.ceil(y+rad)));crop=observation.rgb[y1:y2,x1:x2]
        with torch.inference_mode():
            output=self.model(**self.processor(images=Image.fromarray(crop),return_tensors='pt'))
            found=self.processor.post_process_object_detection(output,target_sizes=torch.tensor([crop.shape[:2]]),threshold=self.threshold)[0]
        proposals=ClassicalObjects().detect(observation)
        for box,label,confidence in zip(found['boxes'].tolist(),found['labels'].tolist(),found['scores'].tolist()):
            b=[float(np.clip(box[0]+x1,x1,x2)),float(np.clip(box[1]+y1,y1,y2)),float(np.clip(box[2]+x1,x1,x2)),float(np.clip(box[3]+y1,y1,y2))]
            if b[2]<=b[0] or b[3]<=b[1] or np.hypot((b[0]+b[2])/2-x,(b[1]+b[3])/2-y)>rad:continue
            # Suppress overlapping color proto-objects rather than allocating a second ID.
            def overlap(d):
                a=d.bbox;inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]));union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
                return inter/max(union,1e-9)
            proposals=[d for d in proposals if overlap(d)<.35]
            name=self.model.config.id2label[label];proposals.append(Detection(b,{name:float(confidence),'unknown':1-float(confidence)},descriptor(observation.rgb,b),float(confidence),.6))
        return proposals
