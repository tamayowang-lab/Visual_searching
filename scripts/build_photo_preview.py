"""Build the Porto demonstration from actual Python runs, never synthetic trajectories.

Usage: python scripts/build_photo_preview.py INPUT_PHOTO OUTPUT_DIRECTORY
No object/depth outputs from another model are used. The photo may be shared.
"""
from pathlib import Path
import argparse,json,hashlib,shutil,sys
from dataclasses import replace
import cv2,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from active_observer.config import Config
from active_observer.io.image import load_image
from active_observer.simulation.runner import run
from active_observer.perception.foveation import observe
from active_observer.viz.dashboard import dashboard


def build(image_path: str,out_path: str,config_path: str | None=None,timing_fit: str | None=None):
    out=Path(out_path);(out/'data').mkdir(parents=True,exist_ok=True);(out/'frames').mkdir(exist_ok=True)
    image=load_image(image_path);scale=min(1.,640/image.shape[1]);image=cv2.resize(image,(round(image.shape[1]*scale),round(image.shape[0]*scale)),interpolation=cv2.INTER_AREA)
    h,w=image.shape[:2];cv2.imwrite(str(out/'scene.png'),cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
    cfg=Config.load(config_path) if config_path else Config(seed=42,max_fixations=20,time_budget_ms=7500)
    from active_observer.perception.factory import ProviderFactory
    factory=ProviderFactory();depth_file=None;runs=[];radius=w*cfg.central_fraction
    fit=json.loads(Path(timing_fit).read_text(encoding='utf-8')) if timing_fit else None
    variants=list(fit['models']) if fit else ['phase2']
    base_cfg=cfg
    for variant in variants:
        cfg=replace(base_cfg,timing_model=variant,timing=fit['models'][variant]['parameters']) if fit else base_cfg
        for row,yf in enumerate([.2,.5,.8]):
            for col,xf in enumerate([.2,.5,.8]):
                key=(variant+'--' if fit else '')+f'start-{row}-{col}';start=(xf*(w-1),yf*(h-1));result=run(image,cfg,image_id='porto-riverside-rgb',start=start,provider_factory=factory)
                reference=result['depth'].get('reference')
                if reference is not None:
                    depth_file='depth.json';(out/depth_file).write_text(json.dumps(reference,separators=(',',':')),encoding='utf-8');result['depth']['reference']=None;result['depth']['reference_path']=depth_file
                raw=json.dumps(result,separators=(',',':'),allow_nan=False);(out/'data'/f'{key}.json').write_text(raw,encoding='utf-8')
                frames=[]
                for f in result['fixations']:
                    obs=observe(image,(f['x_px'],f['y_px']),cfg);digest=hashlib.sha256(obs.rgb.tobytes()).hexdigest()[:20];path=out/'frames'/f'{digest}.jpg'
                    if not path.exists():cv2.imwrite(str(path),cv2.cvtColor(obs.rgb,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,88])
                    frames.append('frames/'+path.name)
                runs.append(dict(timing_model=variant,id=key,label=f'{row*3+col+1}',start=list(start),frames=frames,fixations=len(result['fixations']),duration_ms=result['duration_ms'],object_count=len(result['fixations'][-1]['memory']['objects']),data='data/'+key+'.json'))
                if row==col==1:
                    original_reference=result['depth'].get('reference');result['depth']['reference']=reference;dashboard(image,result,out/'dashboard.png');result['depth']['reference']=original_reference
                print(key,len(result['fixations']),flush=True)
    manifest=dict(title='Porto riverside',width=w,height=h,radius_px=radius,model='RGB Active Observer V2',config=cfg.to_dict(),runs=runs,default_run=(fit['selected_variant']+'--' if fit else '')+'start-1-1',data_kind='Real-photograph model demonstration, not human eye tracking',execution='Nine independently generated Python runs; selecting a start replays its prepared run. There is no live Python inference in this static viewer.',perception=result['perception'],depth=result['depth'].get('metadata') or 'No depth supplied',depth_file=depth_file,source=dict(name='Porto panorama perspective photograph, also used as the Gaze Lab input',repository='https://github.com/David-Ef/salient360Toolbox',filename='1_PortoRiverside.png',note='Repository-level MIT license; stimulus-specific license not separately documented. The demonstration crop is not matched to human panorama-viewing trials.'),independence='Only the RGB photograph is shared. Observer code, trajectories and scores come from the new Python implementation.')
    if fit:
        manifest.update(timing_variants=variants,selected_on_validation=fit['selected_variant'],model='RGB Active Observer Phase 3 timing',execution='Three timing variants × nine prepared starts; fixed spatial policy, seed 42. No live Python inference in this static viewer.')
    root=Path(__file__).resolve().parents[1]
    manifest['code_sha256']={str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in (root/'active_observer').rglob('*.py')}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    for name in ['index.html','style.css','app.js']:
        source=Path(__file__).resolve().parents[1]/'preview'/name
        if source.exists():shutil.copy2(source,out/name)
    return manifest

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('image');p.add_argument('out');p.add_argument('--config');p.add_argument('--timing-fit');a=p.parse_args();build(a.image,a.out,a.config,a.timing_fit)
