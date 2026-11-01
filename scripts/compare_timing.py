"""Fit timing on train, select on validation, then compare three frozen variants.

Conditional timing contexts use observed prefixes explicitly; autonomous scanpaths
never receive future human fixations. Raw eye-tracking records remain local.
"""
from pathlib import Path
from dataclasses import replace,asdict
from collections import defaultdict
import argparse,json,hashlib,time,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from active_observer.config import Config
from active_observer.io.human import load_manifest
from active_observer.io.image import load_image
from active_observer.world.model import World
from active_observer.perception.factory import ProviderFactory
from active_observer.perception.foveation import observe
from active_observer.perception.depth import update_observed_depth
from active_observer.attention.policy import score
from active_observer.attention.processing import processing_context
from active_observer.attention.timing import TimingParameters,ProcessingContext,simulate_timing
from active_observer.evaluation.batch import predict
from active_observer.evaluation.metrics import compare,merged_model_fixations

VARIANTS=['heuristic_lognormal','accumulator','accumulator_hold']

def build_contexts(manifest,config,splits,out):
    data=load_manifest(manifest);factory=ProviderFactory();images={};contexts=[]
    for number,r in enumerate(data['records']):
        if r['split'] not in splits:continue
        if r['image_path'] not in images:images[r['image_path']]=load_image(r['image_path'])
        im=images[r['image_path']];h,w=im.shape[:2];world=World(w,h,config);objects,depth=factory.get(config,r);d=depth.predict(im) if depth else None
        for i,f in enumerate(r['fixations']):
            o=observe(im,(f['x'],f['y']),config);detections=objects.detect(o);candidates=score(world,o,r.get('task',''),config)
            if 0<i<len(r['fixations'])-1:
                processing=processing_context(world,o,detections,250.);nearest=min(candidates,key=lambda c:(c['x']-f['x'])**2+(c['y']-f['y'])**2)
                contexts.append(dict(split=r['split'],image_id=r['image_id'],participant=r['participant'],duration_ms=f['duration_ms'],utilities=[c['utility'] for c in candidates],heuristic_terms=nearest['terms'],processing=asdict(processing)))
            world.update(detections,o,f['duration_ms'])
            if d is not None:update_observed_depth(world,o,d,config_camera(w,h))
        if number%20==0:print('Context extraction',number,flush=True)
    Path(out).write_text(json.dumps(contexts,allow_nan=False),encoding='utf-8');print('Contexts', {s:sum(c['split']==s for c in contexts) for s in splits},flush=True);return contexts


def config_camera(w,h):
    from active_observer.io.image import Camera
    return Camera(w,h)


def draw_samples(context,variant,parameters,repetitions=6,seed=808):
    processing=ProcessingContext(**context['processing']);samples=[]
    # A stable key ensures common random numbers across competing model variants.
    key=int(hashlib.sha256((context['image_id']+':'+context['participant']+':'+str(context['utilities'])).encode()).hexdigest()[:8],16)
    for rep in range(repetitions):
        rng=np.random.default_rng(np.random.SeedSequence([seed,key,rep]));r=simulate_timing(context['utilities'],processing,variant,parameters,rng,heuristic_terms=context['heuristic_terms'],keep_trace=False)
        samples.append(r['duration_ms'])
    return np.asarray(samples)


def score_contexts(contexts,variant,parameters,repetitions=6):
    rows=[];cells=defaultdict(list)
    for c in contexts:
        samples=draw_samples(c,variant,parameters,repetitions);truth=c['duration_ms'];crps=float(np.abs(samples-truth).mean()-.5*np.abs(samples[:,None]-samples[None,:]).mean())
        cells[(c['image_id'],c['participant'])].append(crps);rows.append(dict(image_id=c['image_id'],participant=c['participant'],crps_ms=crps,human_ms=truth,predicted_mean_ms=float(samples.mean())))
    images=defaultdict(list)
    for (image,participant),values in cells.items():images[image].append(float(np.mean(values)))
    losses={k:float(np.mean(v)) for k,v in images.items()};values=list(losses.values())
    return dict(mean=float(np.mean(values)),image_se=float(np.std(values,ddof=1)/np.sqrt(len(values))) if len(values)>1 else 0.,image_losses=losses,rows=rows)


def fit_models(contexts,out):
    training=[c for c in contexts if c['split']=='train'];validation=[c for c in contexts if c['split']=='validation']
    if not training or not validation:raise ValueError('Training and validation contexts required')
    configs={};search={}
    for variant in VARIANTS:
        grid=[]
        if variant=='heuristic_lognormal':
            for scale in [.6,.8,1.]:
                for sigma in [.18,.35,.55]:grid.append(TimingParameters(heuristic_scale=scale,heuristic_sigma=sigma))
        else:
            for urgency in [2.5,4.,6.]:
                for diffusion in [.1,.25,.4]:
                    for prep in [20.,40.]:
                        for hold in ([0.,.3,.6] if variant=='accumulator_hold' else [0.]):grid.append(TimingParameters(urgency=urgency,diffusion=diffusion,preparation_ms=prep,hold_strength=hold))
        candidates=[]
        for i,p in enumerate(grid):
            value=score_contexts(training,variant,p);candidates.append(dict(parameters=asdict(p),training_crps_ms=value['mean']))
            if i%10==0:print('Fit',variant,i+1,'/',len(grid),flush=True)
        best=min(candidates,key=lambda v:v['training_crps_ms']);p=TimingParameters(**best['parameters']);val=score_contexts(validation,variant,p);best.update(validation_crps_ms=val['mean'],validation_image_se_ms=val['image_se'],validation_image_losses=val['image_losses']);configs[variant]=best;search[variant]=candidates
    best_variant=min(VARIANTS,key=lambda v:configs[v]['validation_crps_ms']);cutoff=configs[best_variant]['validation_crps_ms']+configs[best_variant]['validation_image_se_ms'];selected=next(v for v in VARIANTS if configs[v]['validation_crps_ms']<=cutoff)
    result=dict(scope='Timing comparison only; spatial/peripheral/EIG policy unchanged',fit='Population-level train CRPS; validation one-standard-error simplicity rule; no held-out test fitting',threshold_fixed=1.,priority_gain_fixed=2.,sensory_tau_fixed_ms=250.,repetitions=6,training_contexts=len(training),validation_contexts=len(validation),training_images=sorted({c['image_id'] for c in training}),validation_images=sorted({c['image_id'] for c in validation}),models=configs,search=search,best_validation_variant=best_variant,one_se_cutoff_ms=cutoff,selected_variant=selected,selection_order=VARIANTS)
    Path(out).write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8');print('Selected',selected,'validation best',best_variant,flush=True);return result


def empirical_comparison(manifest,config,fit,contexts,out,seeds):
    import pandas as pd
    records=[r for r in load_manifest(manifest)['records'] if r['split']=='test'];factory=ProviderFactory();rows=[];events=[];conditional={};root=Path(out);root.mkdir(parents=True,exist_ok=True)
    test_contexts=[c for c in contexts if c['split']=='test']
    if test_contexts:
        for variant in VARIANTS:conditional[variant]=score_contexts(test_contexts,variant,TimingParameters(**fit['models'][variant]['parameters']))
    for variant in VARIANTS:
        for i,r in enumerate(records):
            for seed in seeds:
                cfg=replace(config,seed=seed,timing_model=variant,timing=fit['models'][variant]['parameters'],time_budget_ms=float(r.get('viewing_duration_ms',config.time_budget_ms)),max_fixations=100)
                model=predict(r,cfg,factory);metrics=compare(model,r);fix=merged_model_fixations(model['fixations'])[1:]
                # Preserve censoring when the final observation is a merged stay segment.
                if fix:fix[-1]['right_censored']=model['fixations'][-1].get('right_censored',False)
                human=r['fixations'][1:];complete=[f['duration_ms'] for f in fix if not f.get('right_censored',False)]
                rows.append(dict(variant=variant,image_id=r['image_id'],participant=r['participant'],seed=seed,censored_model_fraction=float(np.mean([f.get('right_censored',False) for f in fix])) if fix else None,completed_model_mean_ms=float(np.mean(complete)) if complete else None,**metrics))
                events.append(dict(variant=variant,image_id=r['image_id'],participant=r['participant'],seed=seed,model_durations_ms=[f['duration_ms'] for f in fix],model_censored=[f.get('right_censored',False) for f in fix],human_durations_ms=[f['duration_ms'] for f in human],human_censored=[j==len(human)-1 for j in range(len(human))],model_amplitudes_px=np.linalg.norm(np.diff(np.array([[f['x_px'],f['y_px']] for f in fix]),axis=0),axis=1).tolist() if len(fix)>1 else [],human_amplitudes_px=np.linalg.norm(np.diff(np.array([[f['x'],f['y']] for f in human]),axis=0),axis=1).tolist() if len(human)>1 else []))
            if i%20==0:print('Generative',variant,i,'/',len(records),flush=True)
    df=pd.DataFrame(rows);df.to_csv(root/'trials.csv',index=False,encoding='utf-8');metrics=[k for k in df.select_dtypes(include='number').columns if k!='seed'];cells=df.groupby(['variant','image_id','participant'])[metrics].mean();images=cells.groupby(['variant','image_id']).mean();participants=cells.groupby(['variant','participant']).mean();summary=images.groupby('variant').mean();images.to_csv(root/'by-image.csv',encoding='utf-8');participants.to_csv(root/'by-participant.csv',encoding='utf-8');summary.to_csv(root/'summary.csv',encoding='utf-8')
    (root/'events-private.json').write_text(json.dumps(events,allow_nan=False),encoding='utf-8')
    # Paired uncertainty over shared images and subjects; seeds already averaged.
    rng=np.random.default_rng(8492);intervals={};subjects=sorted(df.participant.unique());image_ids=sorted(df.image_id.unique())
    for variant in VARIANTS[1:]:
        intervals[variant]={}
        for metric in ['duration_wasserstein_ms','amplitude_wasserstein_px','nss','spatial_dtw','model_refixation_rate']:
            delta=(cells.loc[variant][metric]-cells.loc['heuristic_lognormal'][metric]).dropna()
            if delta.empty:
                intervals[variant][metric]=dict(delta=None,ci95=None);continue
            matrix=delta.unstack().reindex(index=image_ids,columns=subjects).to_numpy()
            values=[]
            for rep in range(2000):
                sample=matrix[rng.integers(len(image_ids),size=len(image_ids))][:,rng.integers(len(subjects),size=len(subjects))]
                count=np.isfinite(sample).sum(axis=1);means=np.divide(np.nansum(sample,axis=1),count,out=np.full(len(count),np.nan),where=count>0)
                if np.isfinite(means).any():values.append(float(np.nanmean(means)))
            intervals[variant][metric]=dict(delta=float(delta.groupby(level=0).mean().mean()),ci95=np.quantile(values,[.025,.975]).tolist() if values else None)
    report=dict(status='Exploratory timing-only comparison on the previously examined internal COCO split; no confirmatory human-likeness claim',images=len(image_ids),participants=len(subjects),trials=len(records),seeds=seeds,variants=VARIANTS,selected_on_validation=fit['selected_variant'],parameters=fit['models'],conditional_timing={k:{a:b for a,b in v.items() if a!='rows'} for k,v in conditional.items()},generative_summary=json.loads(summary.to_json(orient='index')),paired_intervals=intervals,scope='No spatial weights, perception, peripheral guidance, EIG expression or world update modified. Primary spatial selector remains softmax; races control departure readiness only.',timing_observability='Dataset provides fixation events. Decision/preparation decomposition and pixel-based execution durations are model assumptions. RT is supplied as exposure budget.',uncertainty='2000 crossed image/participant bootstrap resamples, after averaging seeds; small-sample approximate pointwise intervals, not multiplicity-adjusted significance tests.',conditional_protocol='Train/validation conditional contexts replay past human observations; test conditional CRPS uses known current fixation only and is separate from autonomous generative trajectories. Final human fixation excluded from conditional fitting/scoring as potentially censored.')
    (root/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8');return report


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['contexts','fit','evaluate']);p.add_argument('--manifest',required=True);p.add_argument('--config',required=True);p.add_argument('--out',required=True);p.add_argument('--contexts');p.add_argument('--fit');p.add_argument('--splits',nargs='+',default=['train','validation']);p.add_argument('--seeds',nargs='+',type=int,default=[11,23,37]);a=p.parse_args();cfg=Config.load(a.config)
    if a.stage=='contexts':build_contexts(a.manifest,cfg,a.splits,a.out)
    elif a.stage=='fit':fit_models(json.loads(Path(a.contexts).read_text(encoding='utf-8')),a.out)
    else:empirical_comparison(a.manifest,cfg,json.loads(Path(a.fit).read_text(encoding='utf-8')),json.loads(Path(a.contexts).read_text(encoding='utf-8')),a.out,a.seeds)
if __name__=='__main__':main()
