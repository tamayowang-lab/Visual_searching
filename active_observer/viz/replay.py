"""Standalone offline laboratory: recorded arrays, hashes, no model inference."""
from copy import deepcopy
from pathlib import Path
import base64,gzip,json,hashlib
import cv2
from active_observer.config import Config
from active_observer.perception.foveation import observe
from active_observer.simulation.recording import depth_arrays


def image_uri(rgb):
    ok,encoded=cv2.imencode('.png',cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
    if not ok:raise ValueError('Image encoding failed')
    return 'data:image/png;base64,'+base64.b64encode(encoded).decode()


def write_replay(image,result,path,*,asset_root=None):
    if hashlib.sha256(image.tobytes()).hexdigest()!=result['image_sha256']:raise ValueError('Wrong scene for result')
    cfg=Config(**result['config']);frames=[];reliability=[];export=deepcopy(result)
    path=Path(path);root=Path(asset_root) if asset_root is not None else path.parent
    for f in export['fixations']:
        obs=observe(image,f['gaze'],cfg)
        if hashlib.sha256(obs.rgb.tobytes()).hexdigest()!=f['sensory_sha256']:raise ValueError('Sensory replay mismatch')
        frames.append(image_uri(obs.rgb))
        rel=(obs.reliability*255).round().astype('uint8');reliability.append(image_uri(cv2.cvtColor(rel,cv2.COLOR_GRAY2RGB)))
        if f.get('depth',{}).get('available'):
            if f['depth']['input_sha256']!=f['sensory_sha256']:raise ValueError('Depth/sensory provenance mismatch')
            arrays=depth_arrays(f['depth'],root)
            f['depth']['arrays']={name:dict(dtype=a.dtype.str,values=a.tolist()) for name,a in arrays.items()}
    payload=json.dumps(dict(result=export,original=image_uri(image),frames=frames,reliability=reliability),allow_nan=False,separators=(',',':')).encode()
    encoded=base64.b64encode(gzip.compress(payload,compresslevel=6,mtime=0)).decode()
    assets=Path(__file__).parent;template=(assets/'viewer.html').read_text(encoding='utf-8')
    for marker,filename in [('__STYLE__','viewer.css'),('__THREE__','vendor/three.min.js'),('__ORBIT__','vendor/OrbitControls.js'),('__SCRIPT__','viewer.js')]:
        content=(assets/filename).read_text(encoding='utf-8')
        if filename=='vendor/three.min.js':content='/*\n'+(assets/'vendor/LICENSE.three.txt').read_text()+'\n*/\n'+content
        template=template.replace(marker,content.replace('</script','<\\/script'))
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(template.replace('__DATA__',encoded),encoding='utf-8')
