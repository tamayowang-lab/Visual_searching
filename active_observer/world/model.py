"""Image-anchored, uncertain 2.5D object memory; no physical-world claim."""
from dataclasses import dataclass,asdict
import numpy as np
from .association import associate

def entropy(probabilities):
    values=np.array(list(probabilities.values()),float);values=values/values.sum() if values.sum() else np.ones(len(values))/max(1,len(values))
    return float(-np.sum(values*np.log(np.maximum(values,1e-12)))/np.log(max(2,len(values))))

@dataclass
class ObjectMemory:
    id: str
    bbox: list
    probabilities: dict
    appearance: list
    centroid: list
    confidence: float
    salience: float
    last_seen: float
    visits: int=0
    familiarity: float=.1
    novelty: float=.9
    visibility: float=1.
    state: str="observed"
    relative_depth: float | None=None
    depth_uncertainty: float=1.
    depth_dispersion: float | None=None
    xyz: list | None=None
    covariance: list | None=None
    image_covariance_px: list | None=None
    ior: float=0.
    extent: list | None=None
    observation_count: int=0
    identity_uncertainty: float=1.
    position_uncertainty: float=1.
    sensory_ceiling: float=0.
    exposure_ms: float=0.
    partial: bool=False
    source: str='proto'
    volume: dict | None=None
    coordinate_frame: str='image-anchored relative camera frame; not physical world'
    def to_dict(self):return asdict(self)

class World:
    def __init__(self,width,height,config):
        self.width=width;self.height=height;self.config=config;self.objects=[];self.counter=0;self.time=0.;self.consecutive_stays=0
        self.grid=[dict(id=f"region-{y}-{x}",x=(x+.5)*width/config.grid_columns,y=(y+.5)*height/config.grid_rows,coverage=0.,last_seen=None,visits=0) for y in range(config.grid_rows) for x in range(config.grid_columns)]
    def update(self,detections,observation,elapsed_ms: float):
        cfg=self.config;self.time+=elapsed_ms
        if not cfg.memory_enabled:self.objects=[]
        for m in self.objects:
            m.confidence*=np.exp(-elapsed_ms/cfg.memory_decay_ms);m.ior*=np.exp(-elapsed_ms/1200);m.visibility=0.;m.state='remembered' if m.confidence>.2 else 'uncertain'
            m.identity_uncertainty=min(1,m.identity_uncertainty+.02*elapsed_ms/cfg.memory_decay_ms);m.position_uncertainty=min(1,m.position_uncertainty+.04*elapsed_ms/cfg.memory_decay_ms)
        pairs=associate(self.objects,detections,np.hypot(self.width,self.height));matched={j:i for i,j in pairs}
        for j,d in enumerate(detections):
            center=[(d.bbox[0]+d.bbox[2])/2,(d.bbox[1]+d.bbox[3])/2];central=bool(observation.central[min(self.height-1,int(center[1])),min(self.width-1,int(center[0]))]);alpha=float(1-np.exp(-elapsed_ms/cfg.evidence_time_ms))
            if j in matched:
                m=self.objects[matched[j]];gain=float(alpha*d.confidence/(m.confidence+d.confidence+1e-9));keys=set(m.probabilities)|set(d.probabilities)
                m.probabilities={k:(1-gain)*m.probabilities.get(k,0)+gain*d.probabilities.get(k,0) for k in keys}
                if d.partial:
                    # Union only surfaces already observed; no inferred whole-object extent.
                    m.bbox=[min(m.bbox[0],d.bbox[0]),min(m.bbox[1],d.bbox[1]),max(m.bbox[2],d.bbox[2]),max(m.bbox[3],d.bbox[3])]
                else:m.bbox=list(d.bbox)
                m.centroid=[(m.bbox[0]+m.bbox[2])/2,(m.bbox[1]+m.bbox[3])/2];m.appearance=((1-gain)*np.array(m.appearance)+gain*np.array(d.appearance)).tolist()
                if cfg.evidence_model=='legacy':m.confidence=min(.99,1-(1-m.confidence)*(1-d.confidence))
                else:m.confidence+=alpha*max(0,d.confidence-m.confidence)
            else:
                self.counter+=1;strength=d.confidence if cfg.evidence_model=='legacy' else alpha*d.confidence
                m=ObjectMemory(f'object-{self.counter}',list(d.bbox),dict(d.probabilities),list(d.appearance),center,float(strength),d.salience,self.time);self.objects.append(m)
            m.observation_count+=1;m.exposure_ms+=elapsed_ms;m.visits+=int(central);m.familiarity=min(1,m.familiarity+(.15 if central else .03)*d.confidence*alpha)
            m.last_seen=self.time;m.visibility=d.confidence;m.state='observed';m.novelty=1-m.familiarity;m.ior=1. if central else m.ior;m.extent=[m.bbox[2]-m.bbox[0],m.bbox[3]-m.bbox[1]];m.sensory_ceiling=d.confidence;m.partial=d.partial;m.source=d.source
            m.identity_uncertainty=entropy(m.probabilities);m.position_uncertainty=float(np.clip((1-m.confidence)*(.65 if d.partial else .35),0,1));spread=(m.position_uncertainty*max(m.extent))**2
            m.image_covariance_px=np.diag([spread,spread]).tolist()
        self.objects=[m for m in self.objects if m.confidence>.02][-128:]
        for cell in self.grid:
            x,y=int(cell['x']),int(cell['y'])
            if observation.central[min(y,self.height-1),min(x,self.width-1)]:cell['coverage']=min(1,cell['coverage']+elapsed_ms/300);cell['last_seen']=self.time;cell['visits']+=1
    def snapshot(self):return dict(time_ms=self.time,objects=[m.to_dict() for m in self.objects],regions=[dict(c) for c in self.grid],registry_limit=128,frame='image-anchored relative camera frame',consecutive_stays=self.consecutive_stays)
