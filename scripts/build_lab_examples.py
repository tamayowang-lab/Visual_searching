"""Generate actual diagnostic runs on a real photograph; mock cases are explicit."""
from pathlib import Path
import argparse,json,sys
from dataclasses import replace
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from active_observer.config import Config
from active_observer.io.image import load_image
from active_observer.perception.objects import Detection,descriptor
from active_observer.perception.depth import DepthFrame
from active_observer.simulation.runner import run
from active_observer.simulation.recording import save_run
from active_observer.viz.replay import write_replay

class MockObjects:
    def __init__(self,count=1,follow_gaze=False):self.count=count;self.follow_gaze=follow_gaze
    def metadata(self):return dict(provider='MOCK central patch detector; not object recognition')
    def detect(self,obs):
        h,w=obs.rgb.shape[:2];result=[]
        for i in range(self.count):
            x=(obs.gaze[0] if self.follow_gaze else w/2)+(i-(self.count-1)/2)*w*.065;y=obs.gaze[1] if self.follow_gaze else h/2
            if np.hypot(x-obs.gaze[0],y-obs.gaze[1])>obs.radius_px:continue
            box=[max(0,x-w*.018),max(0,y-h*.035),min(w,x+w*.018),min(h,y+h*.035)]
            result.append(Detection(box,{'mock-patch-'+str(i):.95,'unknown':.05},descriptor(obs.rgb,box),.95,.5,source='semantic'))
        return result

class MockDepth:
    def __init__(self,alternating=False):self.count=0;self.alternating=alternating
    def predict(self,rgb):
        self.count+=1;gray=rgb.mean(axis=2)/255
        inverse=.1+.8*gray
        if self.alternating and self.count%2==0:inverse=.9-.6*gray
        valid=np.ones(gray.shape,bool);valid[:3,:3]=False
        return DepthFrame(inverse.astype(np.float32),valid,np.full(gray.shape,.2,np.float32),
                          dict(estimator='MOCK intensity-derived depth response',normalization='independent diagnostic frame',
                               calibrated_uncertainty=False,synthetic_3d_world=False,alternating_inconsistent_scales=self.alternating))


def export(image,result,folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    saved=save_run(result,folder/'run.json')
    # Export from persisted NPZ to exercise the same path users use to reopen runs.
    write_replay(image,saved,folder/'inspector.html',asset_root=folder)
    return dict(path=folder.name+'/inspector.html',run=folder.name+'/run.json',
                label=result['image_id'],kind=result['example_kind'],fixations=len(result['fixations']),
                outcome=result['termination']['outcome'],depth=result['capabilities']['gaze_dependent_depth'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image',default='examples/photos/porto-square.jpg')
    parser.add_argument('--out',default='outputs/laboratory')
    parser.add_argument('--mock-only',action='store_true')
    args=parser.parse_args();image=load_image(args.image);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    entries=[]
    if not args.mock_only:
        for name,config_path,ms in [('real-depth','configs/with-depth.yaml',1800.),('real-no-depth','configs/default.yaml',1000.)]:
            cfg=replace(Config.load(config_path),administrative_timeout_ms=ms,central_fraction=.12)
            result=run(image,cfg,image_id='Porto / '+name)
            result['example_kind']='Real pretrained SSDLite'+(' + MiDaS, each on gaze-dependent RGB' if name=='real-depth' else '; depth explicitly disabled')
            entries.append(export(image,result,out/name));print(name,entries[-1],flush=True)
    cases=[('mock-one',1,False,1200.),('mock-many',3,False,1200.),('mock-independent-scales',1,True,1800.),('mock-empty',0,False,700.),('mock-early-timeout',1,False,100.),('mock-saccade-timeout',0,False,255.)]
    for name,count,alternating,ms in cases:
        cfg=Config(administrative_timeout_ms=ms,central_fraction=.16)
        result=run(image,cfg,image_id=name,object_provider=MockObjects(count,follow_gaze=alternating),depth_provider=MockDepth(alternating))
        result['example_kind']='MOCK object/depth providers on a real photograph; diagnostic contract fixture only'+('; deliberately inconsistent depth responses' if alternating else '')
        entries.append(export(image,result,out/name));print(name,entries[-1],flush=True)
    (out/'manifest.json').write_text(json.dumps(entries,indent=2))
    links=''.join('<li><a href="'+e['path']+'">'+e['label']+'</a><p>'+e['kind']+' · '+str(e['fixations'])+' fixations · '+e['outcome']+'</p></li>' for e in entries)
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Gaze Lab · Recorded reconstruction laboratory</title><style>body{font:17px/1.6 system-ui;background:#101722;color:#dce8f2;max-width:1000px;margin:6vh auto;padding:24px}a{color:#92dac2}p{color:#aebfd2}li{margin:22px 0}h1{line-height:1.2}</style><h1>Gaze-dependent reconstruction laboratory</h1><p>Actual recorded simulation outputs. Independent depth scales are never fused. Open a recording to inspect sensory input, depth, memory, priority, and stopping. The default real run uses pretrained providers; every mock fixture is explicitly labeled.</p><ul>'+links+'</ul><p>Each inspector is standalone and bundles Three.js locally; no browser inference or CDN is required. The adjoining run.json and NPZ assets preserve exact arrays and provenance.</p>',encoding='utf-8')

if __name__=='__main__':main()
