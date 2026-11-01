"""Timing-only sensory proxy; does not alter the Phase-2 world update or detector."""
import numpy as np
from .timing import ProcessingContext

def processing_context(world,observation,detections,tau_ms):
    h,w=observation.rgb.shape[:2];x,y=observation.gaze;r=observation.radius_px
    crop=observation.rgb[max(0,int(y-r)):min(h,int(y+r+1)),max(0,int(x-r)):min(w,int(x+r+1))].astype(float)/255
    image_quality=float(np.clip(crop.std()/ .25,0,1));central=[]
    for d in detections:
        cx,cy=(d.bbox[0]+d.bbox[2])/2,(d.bbox[1]+d.bbox[3])/2
        if np.hypot(cx-x,cy-y)>r:continue
        size=float(np.clip(np.sqrt((d.bbox[2]-d.bbox[0])*(d.bbox[3]-d.bbox[1])/(np.pi*r*r)),.1,1))
        visibility=float(observation.reliability[min(h-1,int(cy)),min(w-1,int(cx))]);quality=visibility*(.45+.3*size+.25*image_quality)
        ceiling=float(np.clip(.8*quality+.2*d.confidence,0,1));probs=np.array(list(d.probabilities.values()));difficulty=float(-np.sum(probs*np.log(np.maximum(probs,1e-12)))/np.log(max(2,len(probs))))
        nearby=[m for m in world.objects if np.hypot(m.centroid[0]-cx,m.centroid[1]-cy)<r/2];prior=max([m.confidence for m in nearby],default=0.)
        central.append(ProcessingContext(min(prior,ceiling),ceiling,tau_ms*(1+.8*difficulty+.5*(1-size)),'central visibility + size + local quality; detector confidence is only one term'))
    if central:return max(central,key=lambda c:c.sensory_ceiling-c.initial_evidence)
    coverage=max([g['coverage'] for g in world.grid if np.hypot(g['x']-x,g['y']-y)<r],default=0.)
    ceiling=.25+.5*image_quality
    return ProcessingContext(min(coverage,ceiling),ceiling,tau_ms,'central texture evidence; no invented semantic identity')
