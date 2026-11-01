"""Gated one-to-one association with semantic conflict and partial-overlap checks."""
import numpy as np
from scipy.optimize import linear_sum_assignment

def iou(a,b):
    inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]));area=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/max(area,1e-9)

def known(p):
    items=[(k,v) for k,v in p.items() if k!='unknown' and not k.endswith('-region')]
    return max(items,key=lambda x:x[1]) if items else ('unknown',0)

def associate(memories: list,detections: list,diagonal: float,gate: float=.15) -> list[tuple[int,int]]:
    if not memories or not detections:return []
    cost=np.full((len(memories),len(detections)),1e6)
    for i,m in enumerate(memories):
        for j,d in enumerate(detections):
            center=np.array([(d.bbox[0]+d.bbox[2])/2,(d.bbox[1]+d.bbox[3])/2]);dist=np.linalg.norm(np.array(m.centroid)-center)/diagonal;overlap=iou(m.bbox,d.bbox)
            if dist>gate:continue
            mk,mp=known(m.probabilities);dk,dp=known(d.probabilities)
            if mp>.6 and dp>.6 and mk!=dk:continue
            appearance=np.linalg.norm(np.array(m.appearance)-d.appearance)/np.sqrt(3)
            if appearance>.55:continue
            # Prevent a new unrelated location from merging merely due to similar color.
            if overlap==0 and dist>.045:continue
            category=sum(min(v,d.probabilities.get(k,0)) for k,v in m.probabilities.items())
            cost[i,j]=.35*dist/gate+.3*(1-overlap)+.2*(1-category)+.15*appearance
    rows,cols=linear_sum_assignment(cost)
    return [(int(i),int(j)) for i,j in zip(rows,cols) if cost[i,j]<.85]
