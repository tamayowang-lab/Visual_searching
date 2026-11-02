"""Hierarchical evidence memory with separate perceptual and geometric uncertainty."""
from dataclasses import dataclass,field,asdict
import numpy as np
from .association import associate

@dataclass
class ReconstructionEstimate:
    relative_depth: float | None = None
    center: list | None = None
    dimensions: list = field(default_factory=lambda:[None,None,None])
    depth_uncertainty: float = 1.
    dimension_uncertainty: list = field(default_factory=lambda:[1.,1.,1.])
    scale_id: str | None = None
    frame_id: str | None = None
    orientation: list | None = None
    aggregation: str = 'latest observation only; history contains independent, unaligned hypotheses'
    coordinate_frame: str = 'image-anchored relative units; not metric'
    geometry_kind: str = 'partial cuboid hypothesis; unseen thickness unknown'
    observation_history: list = field(default_factory=list)

@dataclass
class ObjectMemory:
    id: str
    bbox: list
    probabilities: dict
    appearance: list
    centroid: list
    confidence: float
    identity_uncertainty: float
    localization_uncertainty: float
    last_seen: float
    observation_count: int = 1
    extent_is_partial: bool = True
    reconstruction: ReconstructionEstimate = field(default_factory=ReconstructionEstimate)
    history: list = field(default_factory=list)

    def to_dict(self):
        result=asdict(self)
        result['perceptual_uncertainty']=dict(identity=result.pop('identity_uncertainty'),localization_2d=result.pop('localization_uncertainty'))
        return result


def entropy(probabilities):
    p=np.array(list(probabilities.values()),float);p=p/p.sum() if p.sum() else np.ones(len(p))/max(1,len(p))
    return float(-np.sum(p*np.log(np.maximum(p,1e-12)))/np.log(max(2,len(p))))


class World:
    def __init__(self,width,height,config):
        self.width=width;self.height=height;self.config=config;self.objects=[];self.ensembles=[];self.time=0.;self.counter=0

    def elapse(self,dt):
        self.time+=dt
        for memory in self.objects:
            memory.confidence*=float(np.exp(-dt/self.config.memory_decay_ms))
            # Recognition decay does not change depth or dimension uncertainty.
            memory.identity_uncertainty=min(1,memory.identity_uncertainty+.02*dt/self.config.memory_decay_ms)

    def update(self,detections,observation,dt,ensemble):
        self.elapse(dt)
        pairs=associate(self.objects,detections,np.hypot(self.width,self.height));matched={j:i for i,j in pairs}
        consistency=[]
        for j,d in enumerate(detections):
            center=[(d.bbox[0]+d.bbox[2])/2,(d.bbox[1]+d.bbox[3])/2]
            alpha=float(1-np.exp(-dt/self.config.evidence_time_ms))
            if j in matched:
                m=self.objects[matched[j]]
                displacement=float(np.linalg.norm(np.array(center)-m.centroid)/np.hypot(self.width,self.height))
                prior_label=max(m.probabilities,key=m.probabilities.get);label=max(d.probabilities,key=d.probabilities.get)
                consistency.append(dict(object_id=m.id,normalized_centroid_displacement=displacement,identity_agreement=prior_label==label))
                keys=set(m.probabilities)|set(d.probabilities)
                m.probabilities={k:(1-alpha)*m.probabilities.get(k,0)+alpha*d.probabilities.get(k,0) for k in keys}
                m.confidence+=alpha*max(0,d.confidence-m.confidence);m.observation_count+=1
                m.bbox=list(d.bbox);m.centroid=center;m.appearance=list(d.appearance)
            else:
                self.counter+=1
                m=ObjectMemory(f'object-{self.counter}',list(d.bbox),dict(d.probabilities),list(d.appearance),center,alpha*d.confidence,entropy(d.probabilities),1-d.confidence,self.time)
                self.objects.append(m)
            m.extent_is_partial=bool(d.partial)
            m.identity_uncertainty=entropy(m.probabilities);m.localization_uncertainty=1-d.confidence;m.last_seen=self.time
            m.history.append(dict(time_ms=self.time,bbox=list(d.bbox),confidence=d.confidence))
        # One scene-level summary is retained, not expanded into hypothetical objects.
        self.ensembles.append(dict(time_ms=self.time,summary=ensemble.to_dict()))
        return dict(kind='observable_2d_consistency_proxy',matched_observations=consistency,
                    actual_reconstruction_improvement=None,depth_stability=None,
                    note='Agreement/stability can be measured without proving geometric correctness; cross-fixation depth scale is not yet aligned')

    def snapshot(self):
        return dict(time_ms=self.time,objects=[m.to_dict() for m in self.objects],ensembles=list(self.ensembles),
                    reconstruction_validated=False,frame='single-image anchor; metric scale and hidden extent unidentified')
