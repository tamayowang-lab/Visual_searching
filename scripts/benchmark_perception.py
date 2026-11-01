"""Run real CPU inference, not mocked adapters; record timings and process memory."""
from pathlib import Path
import json,argparse,time,sys,os,platform
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from active_observer.config import Config
from active_observer.io.image import load_image
from active_observer.perception.factory import ProviderFactory
from active_observer.perception.foveation import observe

def memory():
    try:
        import resource
        return dict(peak_process_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024 if sys.platform!='darwin' else 1024**2),scope='whole process including Python and both model weights, not isolated tensor memory')
    except ImportError:return dict(peak_process_rss_mb=None,scope='resource API unavailable; measure process working set externally on Windows')

def main():
    p=argparse.ArgumentParser();p.add_argument('image');p.add_argument('--config',required=True);p.add_argument('--out',required=True);a=p.parse_args();cfg=Config.load(a.config);im=load_image(a.image);h,w=im.shape[:2];factory=ProviderFactory();start=time.perf_counter();objects,depth=factory.get(cfg);load=time.perf_counter()-start
    t=time.perf_counter();frame=depth.predict(im) if depth else None;depth_cold=time.perf_counter()-t;t=time.perf_counter();warm=depth.predict(im) if depth else None;depth_warm=time.perf_counter()-t
    records=[]
    for x,y in [(.2,.65),(.5,.5),(.8,.5),(.5,.8)]:
        o=observe(im,(x*(w-1),y*(h-1)),cfg);t=time.perf_counter();found=objects.detect(o);cold=time.perf_counter()-t;t=time.perf_counter();again=objects.detect(o);cached=time.perf_counter()-t
        records.append(dict(gaze=o.gaze,semantic=[d.probabilities for d in found if d.source=='semantic'],cold_seconds=cold,cached_seconds=cached,cache_reproduces=[d.__dict__ for d in found]==[d.__dict__ for d in again]))
    report=dict(image=a.image,image_dimensions=[w,h],platform=platform.platform(),load_seconds=load,depth_first_seconds=depth_cold,depth_cached_seconds=depth_warm,depth_metadata=frame.metadata if frame else None,object_provider=objects.metadata() if hasattr(objects,'metadata') else type(objects).__name__,fixations=records,memory=memory(),warning='Real model inference verification, not semantic accuracy, depth ground-truth accuracy or human validation')
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    if frame is not None:np.savez_compressed(out.with_suffix('.npz'),depth=frame.inverse_depth,valid=frame.valid,quality=frame.quality)
    print(json.dumps(dict(semantic_detections=sum(len(f['semantic']) for f in records),load_seconds=load,depth_first_seconds=depth_cold,memory=report['memory'])))
if __name__=='__main__':main()
