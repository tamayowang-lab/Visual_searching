"""Self-contained decision inspector and reusable exact-input display assets."""
from pathlib import Path
import base64,hashlib,json
import cv2,numpy as np
from active_observer.config import Config
from active_observer.active.options import ActiveOptions
from active_observer.active.perception import bounded_peripheral_sensor
from active_observer.perception.foveation import observe

ASSETS=Path(__file__).with_name('lab')

def encoded_image(rgb,thumbnail=False):
    rgb=np.asarray(rgb,np.uint8)
    if thumbnail and rgb.shape[1]>384:rgb=cv2.resize(rgb,(384,max(1,round(rgb.shape[0]*384/rgb.shape[1]))),interpolation=cv2.INTER_AREA)
    extension='.jpg' if thumbnail else '.png';params=[cv2.IMWRITE_JPEG_QUALITY,83] if thumbnail else [];ok,b=cv2.imencode(extension,cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),params)
    if not ok:raise ValueError('Cannot encode visualization input')
    return ('image/jpeg' if thumbnail else 'image/png'),b.tobytes()

def frames(image,result,sink):
    """Verify full sensor RGB hashes before encoding display thumbnails."""
    cfg=Config(**result['config']);opt=ActiveOptions(**cfg.active);central=[];peripheral=[]
    for f in result['fixations']:
        obs=observe(image,(f['x_px'],f['y_px']),cfg);trace=f['decision_trace'];digest=hashlib.sha256(obs.rgb.tobytes()).hexdigest()
        if digest!=trace['observation']['input_rgb_sha256']:raise ValueError('Reconstructed central observation does not match actual recorded input')
        central.append(sink(obs.rgb,True));state=trace['perceptual_state']
        if state.get('peripheral_input_sha256'):
            if opt.peripheral_sensor=='bounded_rgb':coarse=bounded_peripheral_sensor(image,obs.gaze,obs.radius_px,opt.coarse_width)
            else:
                h,w=image.shape[:2];cw=min(w,opt.coarse_width);ch=max(1,round(h*cw/w));coarse=cv2.resize(cv2.resize(obs.rgb,(cw,ch),interpolation=cv2.INTER_AREA),(w,h),interpolation=cv2.INTER_LINEAR)
            if hashlib.sha256(coarse.tobytes()).hexdigest()!=state['peripheral_input_sha256']:raise ValueError('Reconstructed peripheral sensor does not match logged input')
            peripheral.append(sink(coarse,True))
        else:peripheral.append(None)
    return dict(frames=central,peripheral_frames=peripheral,input_frames_verified=True)

def html(bundle,path):
    markup=(ASSETS/'index.html').read_text(encoding='utf-8');css=(ASSETS/'style.css').read_text(encoding='utf-8');js=(ASSETS/'app.js').read_text(encoding='utf-8');payload=json.dumps(bundle,separators=(',',':'),allow_nan=False).replace('<','\\u003c')
    markup=markup.replace('<link rel="stylesheet" href="style.css">','<style>'+css+'</style>').replace('<script type="module" src="app.js"></script>','<script>window.OBSERVER_BUNDLE='+payload+';</script><script type="module">'+js+'</script>');Path(path).write_text(markup,encoding='utf-8')

def write_active_html(image,result,path):
    def sink(rgb,thumbnail=False):
        mime,b=encoded_image(rgb,thumbnail);return 'data:'+mime+';base64,'+base64.b64encode(b).decode()
    entry=dict(id='local',label='Recorded simulation',variant=result['variant'],mode=result['config']['active'].get('behavior_mode','legacy'),result=result,**frames(image,result,sink));scene=dict(id='local',name=result['image_id'],dataset='local',image=sink(image),runs=[entry]);bundle=dict(datasets=[dict(id='local',label='Local simulation')],scenes=[scene],default_variant=result['variant']);html(bundle,path)
