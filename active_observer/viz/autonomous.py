"""Portable inspector using frames captured from the actual simulation input."""
import base64
import gzip
import json
from pathlib import Path
from active_observer.viz.active import encoded_image


def image_uri(rgb):
    mime, data = encoded_image(rgb)
    return 'data:'+mime+';base64,'+base64.b64encode(data).decode('ascii')


def write_demo(bundle, path, *, external=False):
    template = Path(__file__).with_name('autonomous.html').read_text(encoding='utf-8')
    compressed = gzip.compress(json.dumps(bundle, allow_nan=False, separators=(',', ':')).encode(), compresslevel=6, mtime=0)
    payload = base64.b64encode(compressed).decode()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if external:
        (path.parent/'replay.json.gz').write_bytes(compressed)
        payload='external:replay.json.gz'
    path.write_text(template.replace('__BUNDLE__', payload), encoding='utf-8')


def write_autonomous_html(image, result, path):
    """For run-one: reconstruct and verify exact sensory frames, never approximate."""
    from active_observer.config import Config
    from active_observer.perception.foveation import observe
    import hashlib
    cfg = Config(**result['config'])
    frames = []
    for f in result['fixations']:
        obs = observe(image, (f['x_px'], f['y_px']), cfg, strict_periphery=True)
        if hashlib.sha256(obs.rgb.tobytes()).hexdigest() != f['decision_trace']['observation']['rgb_sha256']:
            raise ValueError('Sensory replay does not match recorded model input')
        frames.append(image_uri(obs.rgb))
    entry = dict(scene=result['image_id'], name=result['image_id'], image=image_uri(image),
                 mode=result['experimental_control']['behavior_mode'], seed=cfg.seed, result=result, frames=frames)
    write_demo(dict(entries=[entry], initial_condition=cfg.experiment.get('observation_condition','until_finished'), fixed_time_ms=cfg.experiment.get('fixed_time_ms',15000), regeneration='active-observer run-one IMAGE --out outputs/run'), path)
