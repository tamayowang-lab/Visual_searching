"""Current-observation features and explicitly uncertain coarse semantic cues.

No raw RGB argument is accepted. The pretrained network receives only the
current foveated observation, additionally downsampled before semantic analysis.
"""
from dataclasses import asdict
from pathlib import Path
import hashlib,json
import numpy as np
import cv2

class ActivePerception:
    def __init__(self,objects,options,mask,cache_directory=None):
        self.objects=objects;self.options=options;self.mask=mask;self.cache=Path(cache_directory) if cache_directory else None
        if self.cache:self.cache.mkdir(parents=True,exist_ok=True)
        self.current={};self.hypotheses=[];self.events=[];self.counter=0
    def observe(self,observation,central_detections,coarse_sensor=None):
        rgb=observation.rgb;h,w=rgb.shape[:2];feature_rgb=rgb
        if coarse_sensor is not None:
            feature_rgb=coarse_sensor.copy();feature_rgb[observation.central]=rgb[observation.central]
        rw=min(w,80);rh=max(1,round(h*rw/w));a=cv2.resize(feature_rgb.astype(np.float32)/255,(rw,rh),interpolation=cv2.INTER_AREA);gray=cv2.cvtColor(a,cv2.COLOR_RGB2GRAY);maps=[]
        for sigma in [1.,2.,4.]:
            smooth=cv2.GaussianBlur(gray,(0,0),sigma);gradient=cv2.magnitude(cv2.Sobel(smooth,cv2.CV_32F,1,0),cv2.Sobel(smooth,cv2.CV_32F,0,1));color=np.linalg.norm(a-cv2.GaussianBlur(a,(0,0),sigma*2),axis=2);v=gradient+color
            scale=max(float(np.quantile(v,.95)),.03);maps.append(np.clip(v/scale,0,1))
        support=np.mean(maps,axis=0);small_mask=cv2.resize(self.mask.mask.astype(np.uint8),(rw,rh),interpolation=cv2.INTER_NEAREST).astype(bool)
        if self.options.valid_stimulus:
            for m in maps:m[~small_mask]=0
            support[~small_mask]=0
        cues=[];coarse_hash=None;source='disabled'
        if self.options.peripheral_guidance:
            cw=min(w,self.options.coarse_width);ch=max(1,round(h*cw/w));coarse=coarse_sensor if coarse_sensor is not None else cv2.resize(cv2.resize(rgb,(cw,ch),interpolation=cv2.INTER_AREA),(w,h),interpolation=cv2.INTER_LINEAR);coarse_hash=hashlib.sha256(coarse.tobytes()).hexdigest();cues,source=self.coarse_semantics(coarse,observation)
        self.current=dict(feature_maps=[dict(scale=s,values=np.round(m,4).tolist()) for s,m in zip([1,2,4],maps)],support_values=np.round(support,4).tolist(),map_width=rw,map_height=rh,semantic_cues=cues,peripheral_input_sha256=coarse_hash,peripheral_source=source,peripheral_sensor=self.options.peripheral_sensor,feature_input_sha256=hashlib.sha256(feature_rgb.tobytes()).hexdigest(),peripheral_width=self.options.coarse_width,central_detections=[asdict(d) for d in central_detections],proto_objects=[],proto_events=[])
        return self.current
    def coarse_semantics(self,coarse,obs):
        base=self.objects
        while hasattr(base,'provider'):base=base.provider
        if hasattr(base,'coarse_semantics'):return base.coarse_semantics(coarse), 'injected test/replaceable coarse provider'
        if not hasattr(base,'model') or not hasattr(base,'torch') or not hasattr(base,'labels'):return [],'unavailable with classical provider; no semantic labels invented'
        meta=base.metadata();key=hashlib.sha256(coarse.tobytes()+json.dumps(meta,sort_keys=True).encode()).hexdigest();path=self.cache/(key+'.json') if self.cache else None
        if path and path.exists():raw=json.loads(path.read_text(encoding='utf-8'))
        else:
            torch=base.torch
            with torch.inference_mode():found=base.model([torch.from_numpy(coarse.transpose(2,0,1).copy()).float()/255])[0]
            raw=[dict(bbox=b,label=base.labels[label],detector_score=p) for b,label,p in zip(found['boxes'].tolist(),found['labels'].tolist(),found['scores'].tolist()) if p>=.15]
            if path:path.write_text(json.dumps(raw,allow_nan=False),encoding='utf-8')
        cues=[];h,w=coarse.shape[:2]
        for d in raw:
            x1,y1,x2,y2=d['bbox'];cx=(x1+x2)/2;cy=(y1+y2)/2
            if self.options.valid_stimulus and not self.mask.valid(cx,cy):continue
            reliability=float(obs.reliability[min(h-1,max(0,int(cy))),min(w-1,max(0,int(cx)))]);quality=float(np.clip(.2+.8*reliability,0,1));score=float(min(self.options.peripheral_cap,d['detector_score']*self.options.peripheral_cap*quality))
            cues.append(dict(**d,weighted_evidence=score,reliability=reliability,status='tentative peripheral hypothesis, not calibrated probability',source='pretrained detector on downsampled current foveated RGB'))
        return cues,('verified SSDLite on bounded multiscale RGB sensor (maximum 160-wide source)' if self.options.peripheral_sensor=='bounded_rgb' else 'verified SSDLite on downsampled current foveated RGB only')
    def support_at(self,x,y):
        if not self.current:return 0.
        a=np.asarray(self.current['support_values']);return float(a[min(a.shape[0]-1,max(0,int(y/self.mask.height*a.shape[0]))),min(a.shape[1]-1,max(0,int(x/self.mask.width*a.shape[1])))])
    def target_at(self,x,y,target):
        if not target or not self.current:return 0.
        value=0.
        for cue in self.current['semantic_cues']:
            if cue['label']!=target:continue
            a,b,c,d=cue['bbox'];dx=max(a-x,0,x-c);dy=max(b-y,0,y-d);spread=max(8,.15*np.hypot(c-a,d-b));value=max(value,cue['weighted_evidence']*np.exp(-.5*(dx*dx+dy*dy)/(spread*spread)))
        return float(value)
    def snapshot(self,target=''):
        value=dict(self.current);h=self.current.get('map_height',1);w=self.current.get('map_width',1)
        yy,xx=np.mgrid[:h,:w];xx=(xx+.5)*self.mask.width/w;yy=(yy+.5)*self.mask.height/h;field=np.zeros((h,w))
        for cue in self.current.get('semantic_cues',[]):
            if cue['label']!=target:continue
            a,b,c,d=cue['bbox'];dx=np.maximum(np.maximum(a-xx,0),xx-c);dy=np.maximum(np.maximum(b-yy,0),yy-d);spread=max(8,.15*np.hypot(c-a,d-b));field=np.maximum(field,cue['weighted_evidence']*np.exp(-.5*(dx*dx+dy*dy)/(spread*spread)))
        if self.options.valid_stimulus:field[~cv2.resize(self.mask.mask.astype(np.uint8),(w,h),interpolation=cv2.INTER_NEAREST).astype(bool)]=0
        value['target_relevance_map']=np.round(field,4).tolist();value['proto_objects']=__import__('copy').deepcopy(self.hypotheses);value['proto_events']=self.events
        return value


def bounded_peripheral_sensor(image,gaze,radius,width=160):
    """Discard fine pixels BEFORE semantic inference; no full-resolution bank.

    Two bounded resolutions (160/80 wide by default) mixed with eccentricity.
    Central object recognition still uses the original foveated central channel.
    """
    h,w=image.shape[:2];cw=min(max(1,w//4),width,160);ch=max(1,round(h*cw/w));low=cv2.resize(image,(cw,ch),interpolation=cv2.INTER_AREA);outer=cv2.resize(cv2.resize(low,(max(1,cw//2),max(1,ch//2)),interpolation=cv2.INTER_AREA),(cw,ch),interpolation=cv2.INTER_LINEAR);yy,xx=np.mgrid[:ch,:cw];distance=np.hypot((xx+.5)*w/cw-gaze[0],(yy+.5)*h/ch-gaze[1]);alpha=np.exp(-distance/(3*radius))[...,None];mixed=np.clip(low*alpha+outer*(1-alpha),0,255).astype(np.uint8);return cv2.resize(mixed,(w,h),interpolation=cv2.INTER_LINEAR)


def overlap(a,b):
    inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]));union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter;return inter/max(union,1e-9)


def update_workspace(perception,observation,detections,dwell,world):
    """Tentative structures persist separately from centrally recognized memory.

    Confidence and semantic supports are model states, not calibrated posterior
    probabilities. No ground-truth boxes or human future locations are accepted.
    """
    from copy import deepcopy
    o=perception;options=o.options;o.events=[];h,w=observation.rgb.shape[:2];support=np.asarray(o.current['support_values']);mh,mw=support.shape;valid=cv2.resize(o.mask.mask.astype(np.uint8),(mw,mh),interpolation=cv2.INTER_NEAREST).astype(bool);threshold=max(.4,float(np.quantile(support[valid],.8)));binary=((support>=threshold)&valid).astype(np.uint8);count,labels,stats,centroids=cv2.connectedComponentsWithStats(binary,8);proposals=[]
    for i in range(1,count):
        x,y,bw,bh,area=map(int,stats[i]);
        if area<3:continue
        region=labels==i;iy,ix=np.unravel_index(np.argmax(np.where(region,support,-1)),support.shape);point=[float((ix+.5)*w/mw),float((iy+.5)*h/mh)];box=[float(x*w/mw),float(y*h/mh),float((x+bw)*w/mw),float((y+bh)*h/mh)];quality=float(support[region].mean());proposals.append(dict(bbox=box,point=point,quality=quality,source='multiscale feature component',cue=None))
    for cue in o.current['semantic_cues']:
        b=cue['bbox'];point=[float((b[0]+b[2])/2),float((b[1]+b[3])/2)]
        if not o.mask.valid(*point):continue
        proposals.append(dict(bbox=list(b),point=point,quality=float(min(.9,.35+cue['weighted_evidence'])),source='uncertain coarse semantic structure',cue=cue))
    proposals=sorted(proposals,key=lambda p:p['quality'],reverse=True)[:24];old=deepcopy(o.hypotheses) if options.proto_persistence else [];used=set();new=[]
    for proposal in proposals:
        point=proposal['point'];box=proposal['bbox'];px=min(w-1,int(point[0]));py=min(h-1,int(point[1]));appearance=(observation.rgb[max(0,py-4):min(h,py+5),max(0,px-4):min(w,px+5)].mean(axis=(0,1))/255).tolist();matches=[]
        for j,p in enumerate(old):
            distance=np.linalg.norm(np.asarray(p['point'])-point);color=np.linalg.norm(np.asarray(p['appearance'])-appearance)
            if j not in used and (overlap(p['bbox'],box)>.2 or distance<observation.radius_px) and color<.6:matches.append((distance/np.hypot(w,h)+color*.2,j))
        if matches:
            _,j=min(matches);used.add(j);p=deepcopy(old[j]);event='updated'
        else:
            o.counter+=1;p=dict(id='proto-'+str(o.counter),confidence=0.,semantic_support={'unknown':1.},central_exposure_ms=0.,central_verified=False,observations=0,missed_ms=0.,status='tentative',linked_memory_id=None);event='created'
        p['bbox']=box;p['point']=point;p['appearance']=appearance;p['source']=proposal['source'];p['observations']+=1;p['missed_ms']=0.;reliability=float(observation.reliability[py,px]);ceiling=float(np.clip(proposal['quality']*(.3+.7*reliability),.05,.85));p['confidence']=float(p['confidence']+(1-np.exp(-dwell/300))*max(0,ceiling-p['confidence']));cue=proposal['cue']
        if cue:p['semantic_support'][cue['label']]=float(p['semantic_support'].get(cue['label'],0)+cue['weighted_evidence']*dwell/250)
        central=[]
        for d in detections:
            if d.source!='semantic' or overlap(box,d.bbox)<.1:continue
            cx=(d.bbox[0]+d.bbox[2])/2;cy=(d.bbox[1]+d.bbox[3])/2
            if np.hypot(cx-observation.gaze[0],cy-observation.gaze[1])<=observation.radius_px:central.append(d)
        if central:
            d=max(central,key=lambda d:overlap(box,d.bbox)*d.confidence);p['central_exposure_ms']+=float(dwell);p['central_verified']=True
            for name,prob in d.probabilities.items():
                if name!='unknown':p['semantic_support'][name]=float(p['semantic_support'].get(name,0)+prob*dwell/150)
        total=sum(p['semantic_support'].values());p['probabilities']={k:float(v/total) for k,v in p['semantic_support'].items()};best=max(p['probabilities'],key=p['probabilities'].get);p['status']='recognized' if p['central_verified'] and p['central_exposure_ms']>=120 and best!='unknown' and p['probabilities'][best]>=.5 else 'tentative';linked=[m for m in world.objects if overlap(m.bbox,box)>.2 and m.source=='semantic'];p['linked_memory_id']=max(linked,key=lambda m:overlap(m.bbox,box)).id if linked else None
        new.append(p);o.events.append(dict(id=p['id'],event=event,source=proposal['source'],confidence=float(p['confidence']),central_verified=bool(p['central_verified']),status=p['status']))
    if options.proto_persistence:
        for j,p in enumerate(old):
            if j in used:continue
            p['missed_ms']+=float(dwell);p['confidence']=float(p['confidence']*np.exp(-dwell/1200))
            if p['confidence']>.05 and p['missed_ms']<2000:new.append(p);o.events.append(dict(id=p['id'],event='remembered_decay',confidence=p['confidence']))
            else:o.events.append(dict(id=p['id'],event='discarded',reason='insufficient continuing support'))
    merged=[]
    for p in sorted(new,key=lambda p:p['confidence'],reverse=True):
        match=next((a for a in merged if overlap(a['bbox'],p['bbox'])>.65 and np.linalg.norm(np.array(a['appearance'])-p['appearance'])<.3),None)
        if match is None:merged.append(p);continue
        for key,value in p['semantic_support'].items():match['semantic_support'][key]=float(max(match['semantic_support'].get(key,0),value))
        match['central_verified']=bool(match['central_verified'] or p['central_verified']);match['central_exposure_ms']=max(match['central_exposure_ms'],p['central_exposure_ms']);total=sum(match['semantic_support'].values());match['probabilities']={k:float(v/total) for k,v in match['semantic_support'].items()};o.events.append(dict(id=match['id'],event='merged',removed_id=p['id'],reason='overlapping similarly colored observed structures'))
    for p in merged:
        best=max(p['probabilities'],key=p['probabilities'].get);p['status']='recognized' if p['central_verified'] and p['central_exposure_ms']>=120 and best!='unknown' and p['probabilities'][best]>=.5 else 'tentative'
    o.hypotheses=merged[:32]
