"""CLI entrypoints; outputs are explicitly reproducible JSON and headless PNG."""
import argparse,json
from pathlib import Path
from .config import Config
from .io.image import load_image
from .simulation.runner import run
from .viz.dashboard import dashboard
from .viz.html import write_html
from .perception.depth import ConstantDepth,PrecomputedDepth,CachedDepth,TransformersDepth
from .perception.objects import FovealTransformersObjects
from .io.human import convert_coco


def main():
    p=argparse.ArgumentParser(prog='active-observer');sub=p.add_subparsers(dest='command',required=True)
    one=sub.add_parser('run-one');one.add_argument('image');one.add_argument('--config');one.add_argument('--out',default='outputs/run');one.add_argument('--target',default='');one.add_argument('--seed',type=int);one.add_argument('--start',nargs=2,type=float);one.add_argument('--depth',help='constant or an aligned .npy/.npz relative-depth file');one.add_argument('--depth-kind',choices=['inverse_depth','relative_range'],default='inverse_depth');one.add_argument('--depth-checkpoint');one.add_argument('--detector-checkpoint');one.add_argument('--revision',default='main');one.add_argument('--allow-model-download',action='store_true');one.add_argument('--stimulus-bounds',nargs=4,type=float,metavar=('X1','Y1','X2','Y2'));one.add_argument('--valid-mask',help='Explicit aligned binary mask (.npy/.npz or image); never inferred from input darkness')
    vis=sub.add_parser('visualize');vis.add_argument('image');vis.add_argument('result');vis.add_argument('--out',default='outputs/dashboard.png');vis.add_argument('--step',type=int)
    batch=sub.add_parser('run-batch');batch.add_argument('images',nargs='+');batch.add_argument('--config');batch.add_argument('--target',default='');batch.add_argument('--out',default='outputs/batch')
    ev=sub.add_parser('evaluate');ev.add_argument('manifest');ev.add_argument('--config');ev.add_argument('--out',default='outputs/evaluation');ev.add_argument('--split',choices=['validation','test'],default='test');ev.add_argument('--seeds',nargs='+',type=int,default=[11,23,37])
    cc=sub.add_parser('import-coco');cc.add_argument('raw');cc.add_argument('image_manifest');cc.add_argument('out')
    a=p.parse_args()
    if a.command=='import-coco':convert_coco(a.raw,a.image_manifest,a.out);return
    if a.command=='evaluate':
        from .evaluation.batch import evaluate
        r=evaluate(a.manifest,Config.load(a.config),a.out,split=a.split,seeds=a.seeds);print(json.dumps({k:r[k] for k in ['status','images','participants','depth_tested']}));return
    if a.command=='run-batch':
        cfg=Config.load(a.config);root=Path(a.out);root.mkdir(parents=True,exist_ok=True)
        for name in a.images:
            im=load_image(name);r=run(im,cfg,image_id=Path(name).stem,target=a.target);folder=root/Path(name).stem
            if folder.exists():raise ValueError(f'Batch output collision: {folder}')
            folder.mkdir()
            if cfg.active.get('enabled'):(folder/'decision-traces.jsonl').write_text(''.join(json.dumps(f['decision_trace'],allow_nan=False)+'\n' for f in r['fixations']),encoding='utf-8')
            (folder/'run.json').write_text(json.dumps(r,allow_nan=False), encoding='utf-8');dashboard(im,r,folder/'dashboard.png');write_html(im,r,folder/'inspector.html')
        return
    image=load_image(a.image)
    if a.command=='run-one':
        cfg=Config.load(a.config)
        if a.seed is not None: cfg.seed=a.seed
        if a.depth and a.depth_checkpoint: raise ValueError('Choose precomputed/dummy depth OR pretrained checkpoint')
        cfg.perception=dict(cfg.perception)
        if a.depth:cfg.perception['depth']={'provider':'constant'} if a.depth=='constant' else {'provider':'precomputed','path':a.depth,'kind':a.depth_kind}
        if a.depth_checkpoint:cfg.perception['depth']=dict(provider='transformers',model_id=a.depth_checkpoint,revision=a.revision,allow_download=a.allow_model_download,output_kind=a.depth_kind)
        if a.detector_checkpoint:cfg.perception['objects']=dict(provider='transformers',model_id=a.detector_checkpoint,revision=a.revision,allow_download=a.allow_model_download)
        if cfg.perception.get('objects',{}).get('provider','classical')=='classical':
            import warnings
            warnings.warn('Classical color-region provider selected: no semantic object recognition. Use configs/pretrained-cpu.yaml for verified pretrained providers.',UserWarning)
        kwargs={}
        if a.stimulus_bounds is not None or a.valid_mask:
            if not cfg.active.get('enabled'):raise ValueError('Explicit masks require an opt-in Phase 4B configuration')
            if a.stimulus_bounds is not None:kwargs['stimulus_bounds']=a.stimulus_bounds
            if a.valid_mask:
                import numpy as np,cv2
                if a.valid_mask.endswith('.npy'):mask=np.load(a.valid_mask,allow_pickle=False)
                elif a.valid_mask.endswith('.npz'):
                    with np.load(a.valid_mask,allow_pickle=False) as archive:mask=archive['mask']
                else:mask=cv2.imread(a.valid_mask,cv2.IMREAD_GRAYSCALE)
                if mask is None:raise ValueError('Cannot read validity mask')
                kwargs['valid_mask']=mask.astype(bool)
        result=run(image,cfg,image_id=Path(a.image).stem,target=a.target,start=a.start,**kwargs);out=Path(a.out);out.mkdir(parents=True,exist_ok=True);(out/'run.json').write_text(json.dumps(result,indent=2,allow_nan=False), encoding='utf-8');dashboard(image,result,out/'dashboard.png');
        if cfg.active.get('enabled'):(out/'decision-traces.jsonl').write_text(''.join(json.dumps(f['decision_trace'],allow_nan=False)+'\n' for f in result['fixations']),encoding='utf-8')
        if cfg.active.get('enabled'):
            from .viz.active import write_active_html
            write_active_html(image,result,out/'inspector.html')
        else:write_html(image,result,out/'inspector.html')
        print(json.dumps(dict(result=str(out/'run.json'),dashboard=str(out/'dashboard.png'),fixations=len(result['fixations']),reason=result['stop_reason'])))
    else: dashboard(image,json.loads(Path(a.result).read_text(encoding='utf-8')),a.out,a.step)

if __name__=='__main__': main()
