"""Small diagnostic viewer, explicitly labeled as an incomplete foundation."""
from pathlib import Path
import base64,json,hashlib
import cv2
from active_observer.config import Config
from active_observer.perception.foveation import observe


def image_uri(rgb):
    ok,encoded=cv2.imencode('.png',cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
    if not ok:raise ValueError('Image encoding failed')
    return 'data:image/png;base64,'+base64.b64encode(encoded).decode()


def write_replay(image,result,path):
    if hashlib.sha256(image.tobytes()).hexdigest()!=result['image_sha256']:raise ValueError('Wrong scene for result')
    cfg=Config(**result['config']);frames=[]
    for f in result['fixations']:
        obs=observe(image,f['gaze'],cfg)
        if hashlib.sha256(obs.rgb.tobytes()).hexdigest()!=f['sensory_sha256']:raise ValueError('Sensory replay mismatch')
        frames.append(image_uri(obs.rgb))
    data=json.dumps(dict(result=result,original=image_uri(image),frames=frames),allow_nan=False).replace('<','\\u003c')
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(Path(__file__).with_name('viewer.html').read_text(encoding='utf-8').replace('__DATA__',data),encoding='utf-8')
