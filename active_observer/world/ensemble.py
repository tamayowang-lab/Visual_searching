"""Aggregate scene evidence, deliberately separate from individual identities.

Connected coarse structures are NOT an estimate of how many physical objects
exist. Count and size-distribution estimation remain explicit missing features.
"""
from dataclasses import dataclass, asdict
import cv2
import numpy as np

@dataclass
class EnsembleSummary:
    region: list
    mean_rgb: list
    rgb_std: list
    edge_density: float
    structural_group_count: int
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
    groups=int(sum(area>=3 for area in stats[1:,cv2.CC_STAT_AREA]))
    return EnsembleSummary([0,0,w,h],(coarse.mean(axis=(0,1))/255).tolist(),
                           (coarse.std(axis=(0,1))/255).tolist(),float((edges>0).mean()),groups)
