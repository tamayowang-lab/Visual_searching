"""Aggregate scene evidence, deliberately separate from individual identities.

Connected coarse structures are NOT an estimate of how many physical objects
exist. Count and size-distribution estimation remain explicit missing features.
"""
from dataclasses import dataclass, asdict, field
import cv2
import numpy as np

@dataclass
class EnsembleSummary:
    region: list
    mean_rgb: list
    rgb_std: list
    edge_density: float
    structural_group_count: int
    mean_reliability: float = 0.
    local_regions: list = field(default_factory=list)
    estimated_object_count: int | None = None
    estimated_object_size_distribution: list | None = None
    member_object_ids: list | None = None
    source: str = 'bounded coarse visual input; aggregate scene statistics'
    uncertainty: str = 'uncalibrated; structural groups are not individual objects'

    def to_dict(self):return asdict(self)


def summarize(observation,width):
    h,w=observation.rgb.shape[:2];cw=min(width,w);ch=max(1,round(h*cw/w))
    coarse=cv2.resize(observation.rgb,(cw,ch),interpolation=cv2.INTER_AREA)
    gray=cv2.cvtColor(coarse,cv2.COLOR_RGB2GRAY)
    edges=cv2.Canny(gray,60,120)
    mask=cv2.dilate(edges,np.ones((3,3),np.uint8))
    count,labels,stats,centers=cv2.connectedComponentsWithStats(mask)
    reliability=cv2.resize(observation.reliability,(cw,ch),interpolation=cv2.INTER_AREA)
    regions=[]
    for label in range(1,count):
        x,y,bw,bh,area=stats[label]
        if area<3:continue
        use=labels==label
        regions.append(dict(region=[float(x*w/cw),float(y*h/ch),float((x+bw)*w/cw),float((y+bh)*h/ch)],
                            mean_rgb=(coarse[use].mean(axis=0)/255).tolist(),
                            edge_density=float((edges[use]>0).mean()),mean_reliability=float(reliability[use].mean()),
                            structural_group_count=1,estimated_object_count=None,estimated_object_size_distribution=None,
                            member_object_ids=None,interpretation='localized coarse structural component, not an individual object'))
    return EnsembleSummary([0,0,w,h],(coarse.mean(axis=(0,1))/255).tolist(),
                           (coarse.std(axis=(0,1))/255).tolist(),float((edges>0).mean()),len(regions),
                           mean_reliability=float(reliability.mean()),local_regions=regions)
