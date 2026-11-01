from pathlib import Path
import argparse,gzip,json,sys,hashlib
import numpy as np,pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from active_observer.config import Config
from active_observer.io.human import load_manifest
from active_observer.perception.factory import ProviderFactory
from active_observer.evaluation.active_prefix import replay
from active_observer.attention.timing import TimingParameters,ProcessingContext,simulate_timing
from evaluate_phase4b import config,load_raw

def evaluate_prefix(manifest,base,fit,raw_path,variant,split,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);data=load_manifest(manifest);records=[r for r in data['records'] if r['split']==split];raw=load_raw(raw_path);factory=ProviderFactory();cfg=config(base,fit,variant);rows=[];timing=[];index=[]
    for i,r in enumerate(records):
        t=raw[r['image_id'],r['participant'],r['task']];x,y,w,h=np.array(t['bbox'])*640/1680;result,metrics,labels,contexts=replay(r,cfg,factory,[x,y,x+w,y+h]);key=hashlib.sha256((r['image_id']+r['participant']+r['task']).encode()).hexdigest()[:20];path=out/(key+'.json.gz')
        with gzip.open(path,'wt',encoding='utf-8') as f:json.dump(dict(result=result,prefix_metrics=metrics,next_human=labels),f,allow_nan=False)
        rows.extend(dict(variant=variant,**row) for row in metrics);index.append(dict(image_id=r['image_id'],participant=r['participant'],task=r['task'],path=path.name))
        for c in contexts:
            for mode in ['heuristic_lognormal','accumulator','accumulator_hold']:
                samples=[];params=TimingParameters(**fit['models'][mode]['parameters'])
                for rep in range(6):
                    seed=int(hashlib.sha256((r['image_id']+r['participant']+r['task']+str(c['step'])).encode()).hexdigest()[:8],16);value=simulate_timing(c['utilities'],ProcessingContext(**c['processing']),mode,params,np.random.default_rng(np.random.SeedSequence([719,seed,rep])),heuristic_terms=c['heuristic_terms'],keep_trace=False);samples.append(value['duration_ms'])
                a=np.asarray(samples);crps=float(np.abs(a-c['duration_ms']).mean()-.5*np.abs(a[:,None]-a[None,:]).mean());timing.append(dict(variant=variant,timing_model=mode,image_id=r['image_id'],participant=r['participant'],task=r['task'],crps_ms=crps))
        if i%15==0:print('Prefix',variant,i,'/',len(records),flush=True)
    pd.DataFrame(rows).to_csv(out/'prefix-private.csv',index=False,encoding='utf-8');pd.DataFrame(timing).to_csv(out/'timing-private.csv',index=False,encoding='utf-8');(out/'index-private.json').write_text(json.dumps(index,indent=2),encoding='utf-8')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--config',required=True);p.add_argument('--fit',required=True);p.add_argument('--raw',required=True);p.add_argument('--variant',required=True);p.add_argument('--split',default='test');p.add_argument('--out',required=True);a=p.parse_args();evaluate_prefix(a.manifest,Config.load(a.config),json.loads(Path(a.fit).read_text(encoding='utf-8')),a.raw,a.variant,a.split,a.out)
