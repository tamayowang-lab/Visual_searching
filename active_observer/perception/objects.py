"""Centrally gated object evidence from the current gaze-dependent input."""
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

class FovealTransformersObjects:
    """Optional CPU detector: only a foveal crop receives semantic inference.

    Only centrally gated detections are returned. Downloads require
    explicit opt-in; model/checkpoint and revision are recorded in run metadata.
    """
    def __init__(self,model_id: str,*,revision: str='main',allow_download: bool=False,threshold: float=.4):
        self.model_id=model_id;self.revision=revision;self.allow_download=allow_download;self.threshold=threshold;self.processor=None
    def metadata(self):
        return dict(provider='foveal HuggingFace object detector',model_id=self.model_id,revision=self.revision,device='cpu',semantic_input='central crop only',threshold=self.threshold,preprocessing='model AutoImageProcessor')
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
        proposals=[]
        for box,label,confidence in zip(found['boxes'].tolist(),found['labels'].tolist(),found['scores'].tolist()):
            b=[float(np.clip(box[0]+x1,x1,x2)),float(np.clip(box[1]+y1,y1,y2)),float(np.clip(box[2]+x1,x1,x2)),float(np.clip(box[3]+y1,y1,y2))]
            if b[2]<=b[0] or b[3]<=b[1] or np.hypot((b[0]+b[2])/2-x,(b[1]+b[3])/2-y)>rad:continue
            # Suppress duplicate overlapping central detections.
            def overlap(d):
                a=d.bbox;inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]));union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
                return inter/max(union,1e-9)
            proposals=[d for d in proposals if overlap(d)<.35]
            name=self.model.config.id2label[label];proposals.append(Detection(b,{name:float(confidence),'unknown':1-float(confidence)},descriptor(observation.rgb,b),float(confidence),.6,source='semantic'))
        return proposals
