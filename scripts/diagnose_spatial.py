"""Phase 4A evaluation-only audit, prefix replay, and frozen autonomous baselines.

Full private predictions, raw gaze and image overlays remain in the work directory.
No fitting, production policy changes, or oracle inference occur here.
"""
from pathlib import Path
from dataclasses import replace
from collections import defaultdict
import argparse,gzip,json,hashlib,sys
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from active_observer.config import Config
from active_observer.io.human import load_manifest
from active_observer.io.image import load_image,Camera
from active_observer.world.model import World
from active_observer.perception.factory import ProviderFactory
from active_observer.perception.foveation import observe
from active_observer.perception.depth import update_observed_depth
from active_observer.attention.policy import score
from active_observer.evaluation.batch import predict
from active_observer.evaluation.metrics import compare,sequence_dtw,refixation_rate
from active_observer.evaluation.diagnostics import TOLERANCES,lattice,multiscale,amplitudes,inside,selection_diagnostics,aligned_model_fixations,angular_distance,angular_amplitudes,content_box

POLICIES={'full':'full','center_prior':'center','salience':'salience','movement_prior':'random_walk'}

def dump(path,data):Path(path).write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8')

def raw_lookup(source):
    rows=json.loads((Path(source)/'coco_search18/coco_search18_fixations_TP_train_split1.json').read_text(encoding='utf-8'));lookup={}
    for t in rows:
        if t.get('correct')!=1:continue
        key=(t['name'],str(t['subject']),t['task'])
        if key in lookup:raise ValueError('Ambiguous original trial key')
        lookup[key]=t
    return lookup

def boxes_for(records,lookup):
    by_image=defaultdict(list)
    for r in records:
        t=lookup[(r['image_id'],r['participant'],r['task'])];x,y,w,h=np.asarray(t['bbox'])*640/1680;box=[x,y,x+w,y+h]
        if box not in by_image[r['image_id']]:by_image[r['image_id']].append(box)
    return dict(by_image)

def aggregate(frame,group,root,stem):
    keys=['image_id','participant','task'];other=[k for k in group if k not in keys]
    numeric=list(frame.select_dtypes(include='number').columns);numeric=[k for k in numeric if k not in set(other+keys+['seed','step','replicate'])]
    trials=frame.groupby(other+keys,dropna=False)[numeric].mean()  # seeds / steps within actual task trial
    cells=trials.groupby(other+['image_id','participant']).mean()  # tasks within observer/scene
    img=cells.groupby(other+['image_id']).mean();sub=cells.groupby(other+['participant']).mean();summary=img.groupby(other).mean() if other else img.mean().to_frame().T
    img.to_csv(root/(stem+'-by-image.csv'),encoding='utf-8');sub.to_csv(root/(stem+'-by-participant.csv'),encoding='utf-8');summary.to_csv(root/(stem+'-summary.csv'),encoding='utf-8')
    return summary

def audit(data,lookup,root):
    rows=[];meta=data['metadata'];scale=640/1680
    for r in data['records']:
        t=lookup[(r['image_id'],r['participant'],r['task'])];xy=np.column_stack([t['X'],t['Y']]);mapped=np.array([[f['x'],f['y']] for f in r['fixations']]);a=amplitudes(mapped);b=amplitudes(xy*scale);rt=t['RT'];elapsed=0;recreated=[]
        for x,y,d in zip(t['X'],t['Y'],t['T']):
            if elapsed>=rt:break
            recreated.append([x*scale,y*scale,min(d,rt-elapsed)]);elapsed+=min(d,rt-elapsed)
        actual=np.array([[f['x'],f['y'],f['duration_ms']] for f in r['fixations']]);error=float(np.max(np.abs(np.asarray(recreated)-actual)))
        rows.append(dict(image_id=r['image_id'],participant=r['participant'],task=r['task'],split=r['split'],roundtrip_coordinate_error_px=float(np.max(np.abs(mapped/scale-xy[:len(mapped)]))),adapter_reconstruction_error=error,raw_count=len(xy),window_count=len(mapped),rt_ms=rt,raw_duration_sum_ms=sum(t['T']),duration_minus_rt_ms=sum(t['T'])-rt,window_duration_ms=sum(f['duration_ms'] for f in r['fixations']),terminal_duration_modified=float(actual[-1,2]!=t['T'][len(actual)-1]),removed_fixations=len(xy)-len(mapped),initial_distance_from_center_px=float(np.linalg.norm(mapped[0]-[320,200])),amplitude_all_px=float(a.mean()) if len(a) else np.nan,amplitude_exclude_first_px=float(a[1:].mean()) if len(a)>1 else np.nan,raw_amplitude_all_px=float(b.mean()) if len(b) else np.nan,raw_amplitude_exclude_first_px=float(b[1:].mean()) if len(b)>1 else np.nan,scaling_amplitude_error_px=float(np.max(np.abs(amplitudes(xy)*scale-b))) if len(b) else 0.))
    df=pd.DataFrame(rows);df.to_csv(root/'audit-private.csv',index=False,encoding='utf-8');aggregate(df,['split'],root,'audit')
    result=dict(trials=len(df),images=df.image_id.nunique(),max_adapter_error=float(df.adapter_reconstruction_error.max()),max_roundtrip_error=float(df.roundtrip_coordinate_error_px.max()),max_amplitude_scaling_error=float(df.scaling_amplitude_error_px.max()),raw_T_exceeds_RT_trials=int((df.duration_minus_rt_ms>0).sum()),raw_T_below_RT_trials=int((df.duration_minus_rt_ms<0).sum()),terminal_modified_trials=int(df.terminal_duration_modified.sum()),removed_fixation_total=int(df.removed_fixations.sum()),duration_minus_rt_quantiles=df.duration_minus_rt_ms.quantile([0,.25,.5,.75,1]).to_dict(),display=[1680,1050],inference=[640,400],scale=scale,coordinate_units='display pixels mapped to inference pixels; evaluation ray geometry uses published 54 x 35 degree field',calibration_note='Published supplement: 22-inch, 1680x1050 display, 47 cm viewing distance, 54 x 35 degree field. Evaluation uses normalized flat-display rays from those rounded angles, not participant-specific calibration. COCO-Periph 16 pixels/degree and the observer pseudo-camera are not human calibration.',geometry=meta['geometry_audit'],limitations=['Padding rule matches the mirror helper; official 1680x1050 stimulus files not locally available for pixelwise verification.','No per-fixation onset or saccade durations in the local JSON. Cumulative T clipping at RT is an approximation, not verified wall-clock alignment.','Image+participant alone is not a unique trial key across search categories; this phase uses image+participant+task.'])
    dump(root/'audit.json',result);return result

def prefix_replay(data,cfg,lookup,root):
    factory=ProviderFactory();rows=[];coverage=[];examples={};snapshots=[];known_boxes=boxes_for(data['records'],lookup)
    for number,r in enumerate(data['records']):
        im=load_image(r['image_path']);h,w=im.shape[:2];world=World(w,h,cfg);objects,depth=factory.get(cfg,r);d=depth.predict(im) if depth else None;camera=Camera(w,h);t=lookup[(r['image_id'],r['participant'],r['task'])];x,y,bw,bh=np.asarray(t['bbox'])*640/1680;target_box=[[x,y,x+bw,y+bh]]
        for i,f in enumerate(r['fixations'][:-1]):
            gaze=(f['x'],f['y']);o=observe(im,gaze,cfg);det=objects.detect(o);world.update(det,o,f['duration_ms'])
            if d is not None:update_observed_depth(world,o,d,camera)
            if i and np.linalg.norm(np.array(gaze)-[r['fixations'][i-1]['x'],r['fixations'][i-1]['y']])<1:world.consecutive_stays+=1
            else:world.consecutive_stays=0
            candidates=score(world,o,r['task'],cfg)  # frozen BEFORE accessing the next label
            nxt=r['fixations'][i+1];next_xy=[nxt['x'],nxt['y']];details=selection_diagnostics(candidates,next_xy);xy=np.array([[c['x'],c['y']] for c in candidates]);p=np.array([c['probability'] for c in candidates]);dist=np.linalg.norm(xy-gaze,axis=1)
            row=dict(image_id=r['image_id'],participant=r['participant'],task=r['task'],split=r['split'],step=i,first_transition=float(i==0),human_amplitude_px=float(np.linalg.norm(np.asarray(next_xy)-gaze)),expected_amplitude_px=float(p@dist),greedy_amplitude_px=float(dist[np.argmax(p)]),probability_on_target_box=float(p@inside(xy,target_box)),next_on_target_box=float(inside([next_xy],target_box)[0]),current_on_target_box=float(inside([gaze],target_box)[0]),candidate_on_target_box=float(inside(xy,target_box).any()),semantic_memory_count=sum(m.source=='semantic' for m in world.objects),central_target_detected=float(any(dd.probabilities.get(r['task'],0)>0 for dd in det)),expected_novelty=float(sum(c['probability']*c['utility_components'].get('novelty',0) for c in candidates)),expected_info=float(sum(c['probability']*c['utility_components'].get('info',0) for c in candidates)),expected_movement_cost=float(sum(c['probability']*-c['utility_components'].get('move',0) for c in candidates)),expected_task=float(sum(c['probability']*c['utility_components'].get('task',0) for c in candidates)),probability_annotated_object=float(p@inside(xy,known_boxes[r['image_id']])),**details)
            photograph=[content_box(data['metadata']['geometry_audit'][r['image_id']])]
            row.update(probability_on_padding=float(p@(~inside(xy,photograph))),candidate_padding_fraction=float(np.mean(~inside(xy,photograph))),human_next_on_padding=float(~inside([next_xy],photograph)[0]))
            rows.append(row);snapshots.append(dict(image_id=r['image_id'],participant=r['participant'],task=r['task'],step=i,gaze=gaze,next_xy=next_xy,candidates=candidates))
            sets={'observer':xy,'uniform_grid_40':lattice(w,h),'multiscale_40':multiscale(w,h,gaze),'dense_grid_160':lattice(w,h,16,10)}
            rng=np.random.default_rng(np.random.SeedSequence([492,number,i]));random_sets=[rng.uniform([0,0],[w,h],size=(40,2)) for _ in range(32)]
            for name,points in sets.items():
                nearest=float(np.linalg.norm(points-next_xy,axis=1).min());coverage.append(dict(image_id=r['image_id'],participant=r['participant'],task=r['task'],split=r['split'],step=i,candidate_set=name,count=len(points),nearest_error_px=nearest,nearest_error_deg=float(angular_distance(points,[next_xy]).min()),**{f'coverage_deg_{a}':float(angular_distance(points,[next_xy]).min()<=a) for a in [1,2,3,4]},**{f'coverage_{a}':float(nearest<=a) for a in TOLERANCES}))
            nearest=np.array([np.linalg.norm(points-next_xy,axis=1).min() for points in random_sets]);degrees=np.array([angular_distance(points,[next_xy]).min() for points in random_sets]);coverage.append(dict(image_id=r['image_id'],participant=r['participant'],task=r['task'],split=r['split'],step=i,candidate_set='uniform_random_40',count=40,nearest_error_px=float(nearest.mean()),nearest_error_deg=float(degrees.mean()),**{f'coverage_deg_{a}':float((degrees<=a).mean()) for a in [1,2,3,4]},**{f'coverage_{a}':float((nearest<=a).mean()) for a in TOLERANCES}))
            if r['image_id'] not in examples:examples[r['image_id']]=dict(record=r,step=i,candidates=candidates,next_xy=next_xy,target_box=target_box)
        if number%30==0:print('Prefix',number,'/',len(data['records']),flush=True)
    with gzip.open(root/'prefix-snapshots-private.json.gz','wt',encoding='utf-8') as f:json.dump(snapshots,f,allow_nan=False)
    df=pd.DataFrame(rows);df.to_csv(root/'prefix-private.csv',index=False,encoding='utf-8');cov=pd.DataFrame(coverage);cov.to_csv(root/'coverage-private.csv',index=False,encoding='utf-8');aggregate(df,['split'],root,'selection');aggregate(cov,['split','candidate_set'],root,'coverage');dump(root/'examples-private.json',examples)
    # Additional aggregate diagnosis by phase and candidate type, never per-trial coordinates.
    df.groupby(['split','first_transition']).mean(numeric_only=True).to_csv(root/'selection-by-transition-private.csv',encoding='utf-8')
    df.groupby(['split','nearest_kind']).agg(transitions=('nearest_kind','size'),distance_px=('nearest_error_px','mean'),rank=('nearest_rank_min','mean'),probability=('nearest_probability','mean')).to_csv(root/'nearest-types-private.csv',encoding='utf-8')
    return df

def full_image_reference(im,objects):
    """Evaluation-only un-gated SSDLite boxes, NOT ground-truth object/empty labels."""
    from active_observer.perception.pretrained import TorchvisionFoveal
    base=getattr(objects,'inner',objects)
    # Cache wrappers expose the provider as .provider in older versions.
    while not isinstance(base,TorchvisionFoveal):
        child=next((getattr(base,k) for k in ['provider','wrapped','delegate'] if hasattr(base,k)),None)
        if child is None:raise TypeError('Cannot locate verified reference detector')
        base=child
    torch=base.torch
    with torch.inference_mode():r=base.model([torch.from_numpy(im.transpose(2,0,1).copy()).float()/255])[0]
    return [dict(bbox=b,label=base.labels[label],confidence=p) for b,label,p in zip(r['boxes'].tolist(),r['labels'].tolist(),r['scores'].tolist()) if p>=.3]

def autonomous(data,cfg,fit,lookup,root):
    records=[r for r in data['records'] if r['split']=='test'];factory=ProviderFactory();rows=[];human_rows=[];events=[];references={};(root/'predictions').mkdir(exist_ok=True);known_boxes=boxes_for(data['records'],lookup)
    for variant,policy in POLICIES.items():
        for i,r in enumerate(records):
            image=r['image_id'];im=load_image(r['image_path']);h,w=im.shape[:2];objects,_=factory.get(cfg,r)
            if image not in references:
                reference_path=root/('reference-'+image+'.json')
                if reference_path.exists():references[image]=json.loads(reference_path.read_text(encoding='utf-8'))
                else:references[image]=full_image_reference(im,objects);dump(reference_path,references[image])
            boxes=[d['bbox'] for d in references[image]];t=lookup[(image,r['participant'],r['task'])];x,y,bw,bh=np.asarray(t['bbox'])*640/1680;target_box=[[x,y,x+bw,y+bh]]
            for seed in [11,23,37]:
                config=replace(cfg,policy=policy,seed=seed,max_fixations=100,time_budget_ms=r['viewing_duration_ms'],timing_model='heuristic_lognormal',timing=fit['models']['heuristic_lognormal']['parameters'])
                key=hashlib.sha256(json.dumps(dict(image=r['image_sha256'],task=r['task'],participant=r['participant'],config=config.to_dict(),start=r['fixations'][0]),sort_keys=True).encode()).hexdigest();path=root/'predictions'/(key+'.json.gz')
                if path.exists():
                    with gzip.open(path,'rt',encoding='utf-8') as f:model=json.load(f)
                else:
                    model=predict(r,config,factory)
                    with gzip.open(path,'wt',encoding='utf-8') as f:json.dump(model,f,allow_nan=False)
                mf=aligned_model_fixations(model['fixations'],r['viewing_duration_ms']);mp=np.array([[f['x_px'],f['y_px']] for f in mf]);hp=np.array([[f['x'],f['y']] for f in r['fixations']]);metrics=compare(model,r)
                row=dict(image_id=image,participant=r['participant'],task=r['task'],variant=variant,seed=seed,**metrics)
                for label,xy,fix,output in [('model',mp,mf,row),('human',hp,r['fixations'],row)]:
                    a=amplitudes(xy);later=a[1:];post=xy[1:];center=np.linalg.norm(post-[w/2,h/2],axis=1)/np.hypot(w,h);hit=inside(post,target_box);obj=inside(post,boxes);annotated=inside(post,known_boxes[image]);dur=np.array([f['duration_ms'] for f in fix[1:]])
                    output.update({label+'_amplitude_all_px':float(a.mean()) if len(a) else np.nan,label+'_amplitude_exclude_first_px':float(later.mean()) if len(later) else np.nan,label+'_center_distance_diag':float(center.mean()) if len(center) else np.nan,label+'_center_half_fraction':float(np.mean((post[:,0]>=w*.25)&(post[:,0]<=w*.75)&(post[:,1]>=h*.25)&(post[:,1]<=h*.75))) if len(post) else np.nan,label+'_target_fraction':float(hit.mean()) if len(hit) else np.nan,label+'_target_ever':float(hit.any()),label+'_reference_object_fraction':float(obj.mean()) if len(obj) else np.nan,label+'_known_annotation_fraction':float(annotated.mean()) if len(annotated) else np.nan,label+'_target_dwell_fraction':float(dur@hit/dur.sum()) if dur.sum()>0 else np.nan,label+'_object_dwell_fraction':float(dur@obj/dur.sum()) if dur.sum()>0 else np.nan,label+'_post_initial_count':len(post)})
                row['amplitude_w1_all_px']=float(wasserstein_distance(amplitudes(mp),amplitudes(hp))) if len(mp)>1 else np.nan
                row['model_target_first_saccade']=float(inside(mp[1:2],target_box).any());row['human_target_first_saccade']=float(inside(hp[1:2],target_box).any())
                # Ordinal first K sensitivity avoids pretending RT-derived timing is exact.
                lower=aligned_model_fixations(model['fixations'],sum(f['duration_ms'] for f in r['fixations']));lower_xy=np.array([[f['x_px'],f['y_px']] for f in lower]);lower_amp=amplitudes(lower_xy)
                row['lower_window_count']=len(lower_xy)-1;row['lower_window_amplitude_all_px']=float(lower_amp.mean()) if len(lower_amp) else np.nan
                row['first3_sequence_dtw']=sequence_dtw(mp[:4].tolist(),hp[:4].tolist(),w,h)
                rows.append(row);events.append(dict(variant=variant,image_id=image,participant=r['participant'],task=r['task'],seed=seed,model_xy=mp.tolist(),human_xy=hp.tolist(),model_amplitudes_all=amplitudes(mp).tolist(),human_amplitudes_all=amplitudes(hp).tolist(),model_amplitudes_later=amplitudes(mp,True).tolist(),human_amplitudes_later=amplitudes(hp,True).tolist(),prediction=path.name))
            if i%20==0:print('Autonomous',variant,i,'/',len(records),flush=True)
    df=pd.DataFrame(rows);df.to_csv(root/'spatial-private.csv',index=False,encoding='utf-8');aggregate(df,['variant'],root,'spatial');dump(root/'events-private.json',events);dump(root/'references-private.json',references)
    return df

def uncertainty(root):
    """Image units only. Shared scenes are paired; seed repetition adds no scenes."""
    rng=np.random.default_rng(7401);result={}
    for stem,index,group in [('selection',['split'],['image_id']),('coverage',['split','candidate_set'],['image_id']),('spatial',['variant'],['image_id'])]:
        df=pd.read_csv(root/(stem+'-by-image.csv'));out={}
        for key,part in df.groupby(index):
            numeric=part.select_dtypes(include='number');cols=numeric.columns;values=numeric.to_numpy();samples=[]
            for _ in range(2000):
                a=values[rng.integers(len(values),size=len(values))];count=np.isfinite(a).sum(axis=0);samples.append(np.divide(np.nansum(a,axis=0),count,out=np.full(len(cols),np.nan),where=count>0))
            z=np.asarray(samples);out[key if isinstance(key,str) else '|'.join(map(str,key))]={col:dict(mean=float(np.nanmean(values[:,i])) if np.isfinite(values[:,i]).any() else None,ci95=np.nanquantile(z[:,i],[.025,.975]).tolist() if np.isfinite(z[:,i]).any() else None) for i,col in enumerate(cols)}
        result[stem]=out
    df=pd.read_csv(root/'spatial-by-image.csv');paired={}
    for variant in POLICIES:
        if variant=='full':continue
        a=df[df.variant==variant].set_index('image_id');b=df[df.variant=='full'].set_index('image_id');paired[variant]={}
        for metric in ['nss','spatial_dtw','amplitude_w1_all_px','model_amplitude_all_px','model_target_ever']:
            delta=(a[metric]-b[metric]).dropna().to_numpy();v=np.array([rng.choice(delta,len(delta),replace=True).mean() for _ in range(2000)]);paired[variant][metric]=dict(mean=float(delta.mean()),ci95=np.quantile(v,[.025,.975]).tolist())
    result['paired_baseline_minus_full']=paired;dump(root/'uncertainty.json',result)

def finalize(root):
    """Add angular/window-independent metrics to cached predictions; never refit."""
    df=pd.read_csv(root/'spatial-private.csv');events=json.loads((root/'events-private.json').read_text(encoding='utf-8'))
    assert len(events)==len(df)
    read_audit=json.loads((root/'audit.json').read_text(encoding='utf-8'))
    for i,e in enumerate(events):
        assert df.iloc[i]['variant']==e['variant'] and df.iloc[i]['participant']==int(e['participant'])
        photograph=[content_box(read_audit['geometry'][e['image_id']])]
        for label in ['model','human']:
            xy=np.asarray(e[label+'_xy']);a=angular_amplitudes(xy);later=a[1:];px=amplitudes(xy)
            df.loc[i,label+'_padding_fraction']=float(np.mean(~inside(xy[1:],photograph))) if len(xy)>1 else np.nan
            df.loc[i,label+'_amplitude_all_deg']=float(a.mean()) if len(a) else np.nan
            df.loc[i,label+'_amplitude_exclude_first_deg']=float(later.mean()) if len(later) else np.nan
            df.loc[i,label+'_small_step_fraction']=float(np.mean(px[1:]<=32)) if len(px)>1 else np.nan
            e[label+'_amplitudes_all_deg']=a.tolist();e[label+'_amplitudes_later_deg']=later.tolist()
        a=angular_amplitudes(e['model_xy']);b=angular_amplitudes(e['human_xy']);df.loc[i,'amplitude_w1_all_deg']=float(wasserstein_distance(a,b)) if len(a) and len(b) else np.nan
    df.to_csv(root/'spatial-private.csv',index=False,encoding='utf-8');dump(root/'events-private.json',events);aggregate(df,['variant'],root,'spatial')
    prefix=pd.read_csv(root/'prefix-private.csv')
    for kind in ['region','object','stay']:prefix['nearest_is_'+kind]=(prefix.nearest_kind==kind).astype(float)
    prefix.to_csv(root/'prefix-private.csv',index=False,encoding='utf-8')
    for stem,group,private in [('audit',['split'],'audit'),('selection',['split'],'prefix'),('coverage',['split','candidate_set'],'coverage'),('transition',['split','first_transition'],'prefix')]:aggregate(pd.read_csv(root/(private+'-private.csv')),group,root,stem)
    paired=df.dropna(subset=['model_amplitude_exclude_first_px','human_amplitude_exclude_first_px'])[['variant','image_id','participant','task','seed','model_amplitude_exclude_first_px','human_amplitude_exclude_first_px']]
    aggregate(paired,['variant'],root,'later-paired')
    aggregate(prefix,['split','current_on_target_box'],root,'target-state')
    uncertainty(root)

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['audit','prefix','autonomous','uncertainty','finalize']);p.add_argument('--manifest',required=True);p.add_argument('--source',required=True);p.add_argument('--config',required=True);p.add_argument('--fit',required=True);p.add_argument('--out',required=True);a=p.parse_args();root=Path(a.out);root.mkdir(parents=True,exist_ok=True);data=load_manifest(a.manifest);cfg=Config.load(a.config);lookup=raw_lookup(a.source)
    if any((r['width'],r['height'])!=(640,400) for r in data['records']):raise ValueError('This COCO audit requires the declared 640x400 adapter geometry')
    if a.stage=='audit':audit(data,lookup,root)
    elif a.stage=='prefix':prefix_replay(data,cfg,lookup,root)
    elif a.stage=='autonomous':autonomous(data,cfg,json.loads(Path(a.fit).read_text(encoding='utf-8')),lookup,root)
    elif a.stage=='finalize':finalize(root)
    else:uncertainty(root)
if __name__=='__main__':main()
