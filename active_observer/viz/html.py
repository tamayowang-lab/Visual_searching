"""Self-contained offline step-through inspection; no external CDNs or server."""
from pathlib import Path
import base64,cv2,json
import numpy as np
from active_observer.config import Config
from .contracts import validate_result
from active_observer.perception.foveation import observe


def data_url(image: np.ndarray) -> str:
    ok,b=cv2.imencode('.png',cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
    if not ok: raise ValueError("PNG encoding failed")
    return 'data:image/png;base64,'+base64.b64encode(b).decode()


def write_html(image: np.ndarray,result: dict,path: str | Path) -> None:
    if result.get("schema_version")==3 or result.get("variant") and result.get("config",{}).get("active",{}).get("enabled"):
        from .active import write_active_html
        return write_active_html(image,result,path)
    validate_result(image,result)
    cfg=Config(**result['config']);frames=[]
    for f in result['fixations']:
        o=observe(image,(f['x_px'],f['y_px']),cfg);frames.append(dict(image=data_url(o.rgb),radius=o.radius_px))
    payload=json.dumps(dict(result=result,frames=frames,original=data_url(image)),allow_nan=False).replace('<',chr(92)+'u003c')
    template=Path(__file__).with_name('viewer.html').read_text(encoding='utf-8');Path(path).write_text(template.replace('__OBSERVER_DATA__',payload), encoding='utf-8')
