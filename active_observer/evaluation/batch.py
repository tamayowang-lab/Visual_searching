"""Held-out-image evaluation, explicit baselines, per-image/participant summaries."""
from dataclasses import replace
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from active_observer.config import Config
from active_observer.io.image import load_image
from active_observer.io.human import load_manifest
from active_observer.perception.depth import PrecomputedDepth
from active_observer.simulation.runner import run
from .metrics import compare
from active_observer.perception.factory import ProviderFactory

VARIANTS={"full":{},"random_walk":{"policy":"random_walk"},"without_semantics":{"semantic_enabled":False},"without_information":{"information_enabled":False},"random_center":{"policy":"random_center"},"center":{"policy":"center"},"salience":{"policy":"salience"},"salience_ior":{"policy":"salience_ior"},"without_depth":{"depth_enabled":False},"without_memory":{"memory_enabled":False},"without_uncertainty":{"uncertainty_enabled":False},"without_novelty":{"novelty_enabled":False},"without_history":{"history_enabled":False},"without_exploration":{"exploration_enabled":False}}


def predict(record: dict,config: Config,provider_factory=None) -> dict:
    """Only the declared initial fixation may enter prediction; never future fixations."""
    start=(record['fixations'][0]['x'],record['fixations'][0]['y']) if record['initial_fixation_included'] else None
    objects,provider=(provider_factory or ProviderFactory()).get(config,record)
    kwargs={}
    if config.active.get('enabled'):
        if 'valid_stimulus_bounds' not in record:raise ValueError('Phase 4B empirical evaluation requires declared stimulus bounds; never infer validity from RGB darkness')
        kwargs['stimulus_bounds']=record['valid_stimulus_bounds']
    return run(load_image(record['image_path']),config,image_id=record['image_id'],target=record.get('task',''),start=start,depth_provider=provider,object_provider=objects,include_depth_reference=False,**kwargs)


def evaluate(manifest: str,config: Config,out: str,*,split: str='test',seeds=(11,23,37),ablations: bool=True) -> dict:
    dataset=load_manifest(manifest);records=[r for r in dataset['records'] if r['split']==split]
    if not records:raise ValueError(f"No records in held-out split {split}")
    if split=='train':raise ValueError("evaluate refuses training-split claims; use validation or test")
    factory=ProviderFactory()
    out=Path(out);out.mkdir(parents=True,exist_ok=True);rows=[];prediction_hashes=[]
    variants=VARIANTS if ablations else {'full':{}}
    for name,changes in variants.items():
        print('Evaluating',name,'on',len(records),'trials ×',len(seeds),'seeds',flush=True)
        for rec in records:
            for seed in seeds:
                cfg=replace(config,seed=int(seed),**changes)
                if rec.get('viewing_duration_ms') is not None:cfg=replace(cfg,time_budget_ms=float(rec['viewing_duration_ms']),max_fixations=max(100,cfg.max_fixations))
                model=predict(rec,cfg,factory)
                digest=hashlib.sha256(json.dumps(model,sort_keys=True,allow_nan=False).encode()).hexdigest();prediction_hashes.append(dict(variant=name,image_id=rec['image_id'],participant=rec['participant'],seed=seed,sha256=digest))
                rows.append(dict(variant=name,image_id=rec['image_id'],participant=rec['participant'],seed=seed,**compare(model,rec)))
    frame=pd.DataFrame(rows);frame.to_csv(out/'trials.csv',index=False, encoding='utf-8')
    metrics=[k for k in frame.select_dtypes(include='number').columns if k!='seed']
    # Repeated model seeds are averaged before image- and participant-level summaries.
    cells=frame.groupby(['variant','image_id','participant'])[metrics].mean();image_means=cells.groupby(['variant','image_id']).mean();participant_means=cells.groupby(['variant','participant']).mean()
    summary=image_means.groupby('variant').mean();summary.to_csv(out/'summary.csv', encoding='utf-8');image_means.to_csv(out/'by-image.csv', encoding='utf-8');participant_means.to_csv(out/'by-participant.csv', encoding='utf-8')
    depth_supplied=any(r.get('depth_path') for r in records) or config.perception.get('depth',{}).get('provider','none')!='none'
    report=dict(status='Pipeline verification only: synthetic reference scanpaths, not empirical human results' if dataset['metadata'].get('synthetic',False) else 'Held-out evaluation of user-supplied records; inspect dataset provenance',dataset=dataset['metadata'],split=split,images=len({r['image_id'] for r in records}),participants=len({r['participant'] for r in records}),trials=len(records),seeds=list(seeds),variants=list(variants),depth_tested=depth_supplied,depth_note='No supplied depth: full and without_depth are identical by construction' if not depth_supplied else 'Relative-depth hypothesis; behavioral agreement does not establish physical accuracy',aggregation='Seeds averaged within image/participant; summary weights images equally. Participant summaries separate. No significance claims from seed repetitions.',metrics={'NSS':'higher better; undefined for constant maps','AUC':'deterministic all nonfixated pixels as negatives; not branded AUC-Judd','spatial_DTW':'custom, lower better, not MultiMatch','amplitude':'pixels; angular amplitudes require calibrated display geometry'},perception_configuration=config.perception,initial_convention=dataset['metadata']['initial_fixation_included'],timing='record viewing_duration_ms if supplied; otherwise configured budget; never next-fixation coordinates',summary=json.loads(summary.to_json(orient='index')),prediction_hashes=prediction_hashes)
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False), encoding='utf-8');return report
