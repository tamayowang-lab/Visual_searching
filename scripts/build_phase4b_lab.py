"""Build an internal-research replay site from private evaluation outputs.

Never commit the generated empirical images, coordinates or replay bundles to a
public repository. Selects the first trial by stable ID, never by model outcome.
"""
from pathlib import Path
import argparse,gzip,json,hashlib,shutil,sys
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from active_observer.io.image import load_image
from active_observer.io.human import load_manifest
from active_observer.viz.active import ASSETS,frames,encoded_image

def build(manifest,work,out,demo=None,partial=False):
    work=Path(work);out=Path(out);out.mkdir(parents=True,exist_ok=True);(out/'data').mkdir(exist_ok=True);(out/'images').mkdir(exist_ok=True)
    for p in ASSETS.iterdir():shutil.copy2(p,out/p.name)
    def sink(rgb,thumbnail=False):
        mime,data=encoded_image(rgb,thumbnail);name='images/'+hashlib.sha256(data).hexdigest()[:24]+('.jpg' if thumbnail else '.png');p=out/name
        if not p.exists():p.write_bytes(data)
        return name
    def save(run,name):
        if run.get('depth',{}).get('reference') is not None:
            run={**run,'depth':{**run['depth'],'reference':None,'reference_note':'Large optional depth field omitted from browser replay. Original result remains local; fixation decisions and complete decision traces are unchanged.'}}
        path='data/'+name+'.json.gz'
        with gzip.open(out/path,'wt',encoding='utf-8',compresslevel=6) as f:json.dump(run,f,separators=(',',':'),allow_nan=False)
        return path
    data=load_manifest(manifest);records=sorted([r for r in data['records'] if r['split']=='test'],key=lambda r:(r['image_id'],r['task'],r['participant']));scenes=[]
    for image_id in sorted({r['image_id'] for r in records}):
        group=[r for r in records if r['image_id']==image_id];image=load_image(group[0]['image_path']);human={}
        for r in group:human.setdefault(r['task'],[]).extend([[f['x'],f['y'],f['duration_ms']] for f in r['fixations'][1:]])
        scene=dict(id=image_id,name=image_id,dataset='coco',image=sink(image),human_by_target=human,runs=[],comparison=[],comparison_note='Image-level matched means across all tasks, participants and three seeds; not the single replay selected above. Replay examples are the first stable trial ID per image and task, with no outcome-based selection.');chosen={}
        for r in group:chosen.setdefault(r['task'],r)
        for v in 'ABCDEFSN':
            directory=work/(f'test-{v}-revision2' if v in 'NF' else f'test-{v}');df=pd.read_csv(directory/'trials-private.csv',dtype={'participant':str}) if (directory/'trials-private.csv').exists() or not partial else None;prefix=work/(f'prefix-{v}-revision2' if v in 'NF' else f'prefix-{v}');index=json.loads((prefix/'index-private.json').read_text(encoding='utf-8'));by=pd.read_csv(directory/'by-image.csv').set_index('image_id').loc[image_id] if df is not None else None
            if by is not None:scene['comparison'].append(dict(variant=v,target_acquisition=float(by.model_target_ever),count_nss=float(by.count_nss),amplitude_px=float(by.model_amplitude_all_px),padding_fraction=float(by.model_padding_fraction)))
            for target,r in chosen.items():
                stem=hashlib.sha256((image_id+target+r['participant']+v).encode()).hexdigest()[:20];base=dict(variant=v,mode='search',target=target,human_path=[[f['x'],f['y']] for f in r['fixations']])
                if df is not None:
                    row=df[(df.image_id==image_id)&(df.participant==r['participant'])&(df.task==target)&(df.seed==11)].iloc[0]
                    with gzip.open(directory/'predictions'/row.prediction,'rt',encoding='utf-8') as f:run=json.load(f)
                    scene['runs'].append(dict(**base,id=stem,label=target+' · autonomous · seed 11',kind='autonomous',path=save(run,stem),**frames(image,run,sink)))
                idx=next(a for a in index if a['image_id']==image_id and str(a['participant'])==r['participant'] and a['task']==target)
                with gzip.open(prefix/idx['path'],'rt',encoding='utf-8') as f:p=json.load(f)
                scene['runs'].append(dict(**base,id=stem+'-prefix',label=target+' · human prefix',kind='human_prefix',path=save(p['result'],stem+'-prefix'),next_human=p['next_human'],prefix_metrics=p['prefix_metrics'],**frames(image,p['result'],sink)))
        scenes.append(scene);print('Built',image_id,flush=True)
    if demo:
        demo=json.loads(Path(demo).read_text(encoding='utf-8'))
        for s in demo:
            image=load_image(s.pop('image_path'));s['image']=sink(image)
            for entry in s['runs']:
                run=json.loads(Path(entry.pop('source')).read_text(encoding='utf-8'));entry['path']=save(run,entry['id']);entry.update(frames(image,run,sink))
            scenes.append(s)
    bundle=dict(datasets=[dict(id='coco',label='COCO-Search18 · six new test scenes'),dict(id='porto',label='Real photograph · explore/search demonstration')],scenes=scenes,default_variant='D',report_url='../reports/phase4b-report.html',privacy='Internal single-site research only; do not redistribute empirical images or coordinates.')
    (out/'manifest.json').write_text(json.dumps(bundle,separators=(',',':'),allow_nan=False),encoding='utf-8');print('Saved',out,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--work',required=True);p.add_argument('--out',required=True);p.add_argument('--demo');p.add_argument('--partial',action='store_true');a=p.parse_args();build(a.manifest,a.work,a.out,a.demo,a.partial)
