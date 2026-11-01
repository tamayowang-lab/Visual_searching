"""Sequential Phase 4B experiments; private predictions, aggregate publication."""
from pathlib import Path
from dataclasses import replace
import argparse,gzip,json,hashlib,sys,time
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.stats import wasserstein_distance
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from active_observer.config import Config
from active_observer.io.human import load_manifest
from active_observer.perception.factory import ProviderFactory
from active_observer.evaluation.batch import predict
from active_observer.evaluation.metrics import compare,density,nss
from active_observer.evaluation.diagnostics import inside,amplitudes,angular_amplitudes,aligned_model_fixations

def options(variant,mode='search'):
    i='ABCDEF'.index(variant) if variant in 'ABCDEF' else 5 if variant=='N' else 4
    return dict(enabled=True,variant=variant,valid_stimulus=i>=1,behavior_mode=mode if i>=2 else 'legacy',peripheral_guidance=i>=2,selection='balanced' if i>=3 else 'legacy',adaptive_candidates=i>=4,dynamic_perception=i>=5,peripheral_sensor='bounded_rgb' if variant in 'FSN' else 'observed',proto_persistence=variant!='N')

def config(base,fit,variant,mode='search'):
    return replace(base,timing_model='heuristic_lognormal',timing=fit['models']['heuristic_lognormal']['parameters'],active=options(variant,mode))

def load_raw(path):
    rows=json.loads(Path(path).read_text(encoding='utf-8'));return {(r['name'],str(r['subject']),r['task']):r for r in rows if r.get('correct')==1}

def spatial_metrics(model,r,box):
    fs=aligned_model_fixations(model['fixations'],r['viewing_duration_ms']);mp=np.array([[f['x_px'],f['y_px']] for f in fs]);hp=np.array([[f['x'],f['y']] for f in r['fixations']]);out=compare(model,r);bounds=[r['valid_stimulus_bounds']];target=[box]
    for who,xy,fix in [('model',mp,fs),('human',hp,r['fixations'])]:
        a=amplitudes(xy);deg=angular_amplitudes(xy);post=xy[1:];hits=inside(post,target);hit=np.flatnonzero(hits);dwell=np.array([f['duration_ms'] for f in fix[1:]])
        out.update({who+'_amplitude_all_px':float(a.mean()) if len(a) else None,who+'_amplitude_all_deg':float(deg.mean()) if len(deg) else None,who+'_target_ever':float(hits.any()),who+'_first_target_step':int(hit[0]+1) if len(hit) else None,who+'_target_first_saccade':float(hits[0]) if len(hits) else 0.,who+'_padding_fraction':float(np.mean(~inside(post,bounds))) if len(post) else 0.,who+'_small_step_fraction':float(np.mean(a<=32)) if len(a) else None,who+'_target_dwell_fraction':float(dwell@hits/dwell.sum()) if dwell.sum()>0 else 0.})
        for weighting in ['count','duration']:
            dm=density(post.tolist(),640,400,dwell.tolist() if weighting=='duration' else None);dh=density(hp[1:].tolist(),640,400,[f['duration_ms'] for f in r['fixations'][1:]] if weighting=='duration' else None)
            if who=='model':
                out[weighting+'_cc']=float(np.corrcoef(dm.ravel(),dh.ravel())[0,1]) if dm.std()>0 and dh.std()>0 else None;out[weighting+'_sim']=float(np.minimum(dm/max(dm.sum(),1e-12),dh/max(dh.sum(),1e-12)).sum());out[weighting+'_nss']=nss(dm,hp[1:].tolist())
                yy,xx=np.mgrid[:400,:640];center=np.exp(-.5*(((xx-320)/(.22*np.hypot(640,400)))**2+((yy-200)/(.22*np.hypot(640,400)))**2));center/=center.sum();model_density=.99*dm/dm.sum()+.01/dm.size if dm.sum()>0 else np.ones_like(dm)/dm.size;ix=np.clip(hp[1:,0].astype(int),0,639);iy=np.clip(hp[1:,1].astype(int),0,399);out[weighting+'_gain_vs_center_bits']=float(np.log2(model_density[iy,ix]/center[iy,ix]).mean()) if len(ix) else None
    initial_target=bool(inside(hp[:1],target)[0]);out['initial_human_on_target']=float(initial_target)
    for who,xy in [('model',mp),('human',hp)]:
        out[who+'_target_including_initial']=float(inside(xy,target).any());out[who+'_target_acquisition_initial_off_target']=None if initial_target else out[who+'_target_ever']
    out['model_no_post_initial_fixations']=float(len(mp)<=1)
    claims=correct=0;first_true=None;cue_first=0;retention=[];proto_confirmed=[];events=[]
    from active_observer.active.perception import overlap
    for i,f in enumerate(model['fixations']):
        tr=f['decision_trace'];local_true=False
        for d in tr['observation']['detections']:
            if d['source']=='semantic' and d['probabilities'].get(r['task'],0)>0:
                claims+=1;hit=overlap(d['bbox'],box)>.1;correct+=int(hit);local_true|=hit
        if local_true and first_true is None:first_true=i
        if first_true is not None:retention.append(any(m['probabilities'].get(r['task'],0)>0 and overlap(m['bbox'],box)>.1 for m in tr['memory_after']['objects']))
        state=tr['perceptual_state'];events.extend(state.get('proto_events',[]));proto_confirmed.extend(p['status']=='recognized' for p in state.get('proto_objects',[]))
        if i==0:cue_first=int(any(c['label']==r['task'] for c in state.get('semantic_cues',[])))
    out['central_target_seen']=float(first_true is not None);out['target_detection_precision_proxy']=correct/claims if claims else None;out['target_memory_retention_proxy']=float(np.mean(retention)) if retention else None;out['early_target_cue']=float(cue_first);out['proto_recognized_fraction']=float(np.mean(proto_confirmed)) if proto_confirmed else None;out['proto_merge_events']=sum(e['event']=='merged' for e in events);out['proto_discard_events']=sum(e['event']=='discarded' for e in events)
    out['model_terminal_censored']=float(fs[-1].get('right_censored',False));out['fixations_logged']=len(model['fixations']);out['trace_complete']=float(all('decision_trace' in f for f in model['fixations']));out['probability_to_padding']=float(np.mean([sum(c['probability'] for c in f['decision_trace']['available_candidates'] if not inside([[c['x'],c['y']]],bounds)[0]) for f in model['fixations']]))
    return out

def aggregate(df,out):
    metrics=[c for c in df.select_dtypes('number').columns if c!='seed'];trials=df.groupby(['variant','image_id','participant','task'])[metrics].mean();cells=trials.groupby(['variant','image_id','participant']).mean();images=cells.groupby(['variant','image_id']).mean();subjects=cells.groupby(['variant','participant']).mean();summary=images.groupby('variant').mean()
    for name,frame in [('by-image',images),('by-participant',subjects),('summary',summary)]:frame.to_csv(out/(name+'.csv'),encoding='utf-8')
    return json.loads(summary.to_json(orient='index'))

def evaluate(manifest,base,fit,raw_path,variant,split,seeds,out,reuse_unchanged_signature=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);(out/'predictions').mkdir(exist_ok=True);data=load_manifest(manifest);raw=load_raw(raw_path);factory=ProviderFactory();rows=[];records=[r for r in data['records'] if r['split']==split];cfg=config(base,fit,variant);started=time.perf_counter();code_signature=hashlib.sha256(b''.join(p.read_bytes() for p in sorted((Path(__file__).resolve().parents[1]/'active_observer/active').glob('*.py')))).hexdigest()
    for i,r in enumerate(records):
        t=raw[r['image_id'],r['participant'],r['task']];x,y,w,h=np.array(t['bbox'])*640/1680;box=[x,y,x+w,y+h]
        for seed in seeds:
            cc=replace(cfg,seed=seed,max_fixations=100,time_budget_ms=r['viewing_duration_ms']);identity=dict(image=r['image_sha256'],target=r['task'],start=r['fixations'][0],cfg=cc.to_dict());
            if variant=='A':identity['declared_stimulus_bounds']=r['valid_stimulus_bounds']
            key=hashlib.sha256(json.dumps(dict(code_signature=code_signature,**identity),sort_keys=True).encode()).hexdigest();file=out/'predictions'/(key+'.json.gz');producer_signature=code_signature
            if not file.exists() and reuse_unchanged_signature and variant in 'ABCDES':
                oldkey=hashlib.sha256(json.dumps(dict(code_signature=reuse_unchanged_signature,**identity),sort_keys=True).encode()).hexdigest();oldfile=out/'predictions'/(oldkey+'.json.gz')
                if oldfile.exists():file=oldfile;producer_signature=reuse_unchanged_signature
            if file.exists():
                with gzip.open(file,'rt',encoding='utf-8') as f:model=json.load(f)
            else:
                model=predict(r,cc,factory)
                tmp=file.with_suffix('.tmp')
                with gzip.open(tmp,'wt',encoding='utf-8') as f:json.dump(model,f,allow_nan=False)
                tmp.replace(file)
            rows.append(dict(variant=variant,image_id=r['image_id'],participant=r['participant'],task=r['task'],seed=seed,prediction=file.name,producer_code_signature=producer_signature,**spatial_metrics(model,r,box)))
        if i%15==0:print(variant,split,i,'/',len(records),flush=True)
    df=pd.DataFrame(rows);df.to_csv(out/'trials-private.csv',index=False,encoding='utf-8');result=dict(code_signature=code_signature,numerically_unchanged_cache_signature=reuse_unchanged_signature if variant in 'ABCDES' else None,variant=variant,split=split,scenes=len({r['image_id'] for r in records}),trials=len(records),seeds=seeds,elapsed_seconds=time.perf_counter()-started,config=cfg.to_dict(),summary=aggregate(df,out),source_manifest_sha256=hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),assumptions='RT-conditioned viewing windows; no next human point or target box supplied to inference. Evaluation only target boxes. Fixed original timing fit.')
    (out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8');return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--config',required=True);p.add_argument('--fit',required=True);p.add_argument('--raw',required=True);p.add_argument('--variant',choices=list('ABCDEFSN'),required=True);p.add_argument('--split',choices=['validation','test'],required=True);p.add_argument('--seeds',type=int,nargs='+',default=[11,23,37]);p.add_argument('--out',required=True);a=p.parse_args();evaluate(a.manifest,Config.load(a.config),json.loads(Path(a.fit).read_text(encoding='utf-8')),a.raw,a.variant,a.split,a.seeds,a.out)
