"""Adaptive proposals use current sensory structure, never future gaze labels."""
from copy import deepcopy
import numpy as np
from active_observer.attention.policy import feature_at

def refine_candidates(candidates,observation,world,cfg,mask,perception):
    if perception is None or not perception.current:return candidates
    support=np.asarray(perception.current['support_values']);mh,mw=support.shape;h,w=mask.mask.shape;yy,xx=np.mgrid[:mh,:mw];xs=(xx+.5)*w/mw;ys=(yy+.5)*h/mh;valid_small=mask.mask[np.minimum(h-1,ys.astype(int)),np.minimum(w-1,xs.astype(int))];proposals=[];global_color=observation.rgb.mean(axis=(0,1))/255
    def emit(template,x,y,area,index):
        c=deepcopy(template);c.update(id=template['id']+f':refine-{index}',parent_id=template['id'],x=float(x),y=float(y),proposal_area=float(area),adaptive=True);c['terms']['salience']=feature_at(observation,x,y,global_color);c['salience']=c['terms']['salience'];c['terms']['move']=float(np.hypot(x-observation.gaze[0],y-observation.gaze[1])/np.hypot(w,h));c['persistence']=.08*float(np.hypot(x-observation.gaze[0],y-observation.gaze[1])<observation.radius_px/2);return c
    for c in candidates:
        if c['kind']=='region' and c['id'].startswith('region-'):
            _,row,col=c['id'].split('-');row=int(row);col=int(col);x1=col*w/cfg.grid_columns;x2=(col+1)*w/cfg.grid_columns;y1=row*h/cfg.grid_rows;y2=(row+1)*h/cfg.grid_rows
        elif c['kind']=='object':
            memory=next((m for m in world.objects if m.id==c['id']),None)
            if memory is None:proposals.append(c);continue
            x1,y1,x2,y2=memory.bbox
        else:proposals.append(c);continue
        full_valid=mask.mask[max(0,int(y1)):min(h,int(np.ceil(y2))),max(0,int(x1)):min(w,int(np.ceil(x2)))];valid_fraction=float(full_valid.mean()) if full_valid.size else 0.
        if valid_fraction<=0:continue
        use=(xs>=x1)&(xs<x2)&(ys>=y1)&(ys<y2);use &= valid_small
        ids=np.argwhere(use)
        if not len(ids):proposals.append(c);continue
        order=sorted(ids.tolist(),key=lambda p:float(support[p[0],p[1]]),reverse=True);chosen=[]
        for iy,ix in order:
            if support[iy,ix]<.25:break
            point=(xs[iy,ix],ys[iy,ix])
            if all(np.hypot(point[0]-a[0],point[1]-a[1])>=observation.radius_px/2 for a in chosen):chosen.append(point)
            if len(chosen)>=2:break
        if not chosen:
            if mask.valid(c['x'],c['y']):proposals.append(dict(c,proposal_area=valid_fraction))
            else:
                iy,ix=ids[len(ids)//2];proposals.append(emit(c,xs[iy,ix],ys[iy,ix],valid_fraction,0))
        else:
            anchors=[(c['x'],c['y'])] if mask.valid(c['x'],c['y']) else []
            chosen=[p for p in chosen if not any(np.hypot(p[0]-a[0],p[1]-a[1])<1 for a in anchors)]
            count=len(anchors)+len(chosen)
            if anchors:proposals.append(dict(c,proposal_area=valid_fraction/count,parent_id=c['id'],adaptive=False))
            for i,(x,y) in enumerate(chosen):proposals.append(emit(c,x,y,valid_fraction/count,i))
    return proposals
