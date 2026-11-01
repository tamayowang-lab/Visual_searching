"""Publish aggregate-only Phase 4B statistics; never publish individual events."""
from pathlib import Path
import argparse,gzip,json,sys,html,itertools,shutil
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from active_observer.io.human import load_manifest
from active_observer.evaluation.diagnostics import aligned_model_fixations,amplitudes,angular_amplitudes,inside
from active_observer.evaluation.metrics import density,nss
from evaluate_phase4b import load_raw
from report_spatial import table
VARIANTS='ABCDEFSN'
PAIRS=[('B','A'),('C','B'),('D','C'),('E','D'),('F','E'),('S','E'),('F','S'),('F','N')]
METRICS=['model_target_ever','count_nss','duration_nss','spatial_dtw','model_amplitude_all_px','duration_wasserstein_ms','model_padding_fraction']

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')
def directory(root,v,prefix='test'):return root/(f'{prefix}-{v}-revision2' if v in 'NF' else f'{prefix}-{v}')
def cells(df,keys,metrics):
    return df.groupby(keys+['image_id','participant','task'])[metrics].mean().groupby(keys+['image_id','participant']).mean()
def aggregate(df,keys,metrics):return cells(df,keys,metrics).groupby(keys+['image_id']).mean().groupby(keys).mean()
def uncertainty(c,rep=2000):
    # Scene is primary; crossed participant resampling reuses the same subject
    # weights across images. Seeds/steps/task repetitions were averaged first.
    images=sorted(c.index.get_level_values('image_id').unique());people=sorted(c.index.get_level_values('participant').unique())
    if not images or not people:return [None,None]
    values=c.unstack('participant').reindex(index=images,columns=people).to_numpy();valid=np.isfinite(values);rng=np.random.default_rng(481)
    iw=rng.multinomial(len(images),np.full(len(images),1/len(images)),size=rep);sw=rng.multinomial(len(people),np.full(len(people),1/len(people)),size=rep);denom=sw@valid.T;means=np.divide(sw@np.nan_to_num(values).T,denom,out=np.zeros_like(denom,dtype=float),where=denom>0);weights=iw*(denom>0);samples=np.divide((weights*means).sum(axis=1),weights.sum(axis=1),out=np.full(rep,np.nan),where=weights.sum(axis=1)>0);samples=samples[np.isfinite(samples)]
    return np.quantile(samples,[.025,.975]).tolist() if len(samples) else [None,None]

def paired(df,metrics=None):
    metrics=[m for m in (METRICS if metrics is None else metrics) if m in df];c=cells(df,['variant'],metrics);rows=[]
    for a,b in PAIRS:
        for metric in metrics:
            delta=(c.xs(a)[metric]-c.xs(b)[metric]).dropna();by=delta.groupby('image_id').mean();effect=float(by.mean());signs=np.array(list(itertools.product([-1,1],repeat=len(by))));p=float((np.abs((signs*by.to_numpy()).mean(axis=1))>=abs(effect)-1e-12).mean()) if len(by) else None
            rows.append(dict(comparison=a+'−'+b,metric=metric,delta=effect,ci95=uncertainty(delta),scene_signflip_p=p,scenes=len(by)))
    # Descriptive, pointwise intervals. Holm adjusted randomization diagnostics
    # across all planned comparisons × metrics; no selected significant result.
    valid=sorted([r for r in rows if r['scene_signflip_p'] is not None],key=lambda r:r['scene_signflip_p']);last=0.
    for i,r in enumerate(valid):last=max(last,min(1.,(len(valid)-i)*r['scene_signflip_p']));r['holm_p_all_endpoints']=last
    return rows

def km(durations,censored,grid):
    durations=np.asarray(durations);censored=np.asarray(censored,bool);s=1.;out=[]
    for t in grid:
        at=(durations<=t)&(~censored);survival=1.
        for value in np.unique(durations[at]):
            risk=np.sum(durations>=value);events=np.sum((durations==value)&(~censored));survival*=1-events/risk
        out.append(survival)
    return np.asarray(out)

def distributions(root,records,raw):
    lookup={(r['image_id'],r['participant'],r['task']):r for r in records};rows=[];bins_a=np.linspace(0,600,121);bins_deg=np.linspace(0,50,101);bins_d=np.linspace(0,1600,161);bins_step=np.arange(1,11)
    def event(v,r,fs,xy,box):
        row=dict(variant=v,image_id=r['image_id'],participant=r['participant'],task=r['task']);duration=[f['duration_ms'] for f in fs[1:]];censor=[bool(f.get('right_censored',False)) for f in fs[1:]]
        if v=='Human' and censor:censor[-1]=True # conservative terminal-censor sensitivity
        for key,values,grid in [('amplitude_all',amplitudes(xy),bins_a),('amplitude_later',amplitudes(xy,True),bins_a),('amplitude_deg',angular_amplitudes(xy),bins_deg),('duration',duration,bins_d)]:
            if len(values):row[key]=np.array([(np.asarray(values)<=x).mean() for x in grid])
        if duration:row['survival_terminal_censored']=km(duration,censor,bins_d)
        hits=inside(xy[1:],[box]);row['target_curve']=np.array([float(hits[:n].any()) for n in bins_step]);rows.append(row)
    for v in VARIANTS:
        d=directory(root,v);df=pd.read_csv(d/'trials-private.csv',dtype={'participant':str})
        for _,a in df.iterrows():
            r=lookup[a.image_id,a.participant,a.task];t=raw[a.image_id,a.participant,a.task];x,y,w,h=np.array(t['bbox'])*640/1680;box=[x,y,x+w,y+h]
            with gzip.open(d/'predictions'/a.prediction,'rt',encoding='utf-8') as f:model=json.load(f)
            fs=aligned_model_fixations(model['fixations'],r['viewing_duration_ms']);xy=np.array([[f['x_px'],f['y_px']] for f in fs]);event(v,r,fs,xy,box)
            if v=='A' and a.seed==11:event('Human',r,r['fixations'],np.array([[f['x'],f['y']] for f in r['fixations']]),box)
    # Explicit hierarchy: first seed, then task, participant, image.
    curves={}
    for v in list(VARIANTS)+['Human']:
        curves[v]={}
        for key in ['amplitude_all','amplitude_later','amplitude_deg','duration','survival_terminal_censored','target_curve']:
            selected=[r for r in rows if r['variant']==v and key in r];groups={}
            for a in selected:groups.setdefault((a['image_id'],a['participant'],a['task']),[]).append(a[key])
            subjects={}
            for (im,p,task),values in groups.items():subjects.setdefault((im,p),[]).append(np.mean(values,axis=0))
            images={}
            for (im,p),values in subjects.items():images.setdefault(im,[]).append(np.mean(values,axis=0))
            curves[v][key]=np.mean([np.mean(a,axis=0) for a in images.values()],axis=0).tolist()
    return dict(amplitude_px=bins_a.tolist(),amplitude_deg=bins_deg.tolist(),duration_ms=bins_d.tolist(),saccade_number=bins_step.tolist(),curves=curves)

def interobserver(records):
    rows=[]
    for r in records:
        others=[a for a in records if a['image_id']==r['image_id'] and a['task']==r['task'] and a['participant']!=r['participant']]
        if not others:continue
        maps=[]
        for other in others:
            a=density([[f['x'],f['y']] for f in other['fixations'][1:]],640,400);maps.append(a/a.sum())
        score=nss(np.mean(maps,axis=0),[[f['x'],f['y']] for f in r['fixations'][1:]])
        if score is not None:rows.append(dict(image_id=r['image_id'],participant=r['participant'],task=r['task'],nss=score))
    df=pd.DataFrame(rows);by=df.groupby(['image_id','participant']).nss.mean().groupby('image_id').mean();return dict(count_nss=float(by.mean()),scenes=len(by),by_image=by.to_dict(),description='Same image + task, leave-one-participant-out normalized count density; reference, not a mathematical upper bound.')

def mechanism_note(summary):
    return f'<p>Dynamic F reaches {100*summary.loc["F","model_target_ever"]:.1f}% target acquisition versus sensor-only S {100*summary.loc["S","model_target_ever"]:.1f}% and no-persistence N {100*summary.loc["N","model_target_ever"]:.1f}%. F−N is only +3.3 percentage points with a pointwise interval spanning zero. No endpoint survives the reported Holm correction across its comparison family. These controls do not establish a specific contribution from persistent tracking. D remains the validation-selected simpler comparator, not a test-selected winner.</p>'

def build(root,manifest,raw_path,out):
    root=Path(root);out=Path(out);out.mkdir(parents=True,exist_ok=True);records=[r for r in load_manifest(manifest)['records'] if r['split']=='test'];raw=load_raw(raw_path)
    df=pd.concat([pd.read_csv(directory(root,v)/'trials-private.csv',dtype={'participant':str}) for v in VARIANTS],ignore_index=True);numeric=[m for m in df.select_dtypes('number') if m!='seed'];c=cells(df,['variant'],numeric);by=c.groupby(['variant','image_id']).mean();summary=by.groupby('variant').mean();by.to_csv(out/'phase4b-by-image.csv',encoding='utf-8');summary.to_csv(out/'phase4b-summary.csv',encoding='utf-8');c.groupby(['variant','participant']).mean().to_csv(out/'phase4b-by-participant.csv',encoding='utf-8');pairs=paired(df);dump(out/'phase4b-paired.json',pairs)
    prefixes=pd.concat([pd.read_csv(directory(root,v,'prefix')/'prefix-private.csv',dtype={'participant':str}) for v in VARIANTS],ignore_index=True);pn=list(prefixes.select_dtypes('number'));pn=[k for k in pn if k!='step'];pc=cells(prefixes,['variant'],pn);pc.groupby(['variant','image_id']).mean().to_csv(out/'phase4b-prefix-by-image.csv',encoding='utf-8');ps=pc.groupby(['variant','image_id']).mean().groupby('variant').mean();ps.to_csv(out/'phase4b-prefix-summary.csv',encoding='utf-8');prefix_pairs=paired(prefixes,['nearest_error_px','coverage_deg_2','probability_within_deg_2','expected_error_px']);dump(out/'phase4b-prefix-paired.json',prefix_pairs)
    timing=pd.concat([pd.read_csv(directory(root,v,'prefix')/'timing-private.csv',dtype={'participant':str}) for v in VARIANTS]);ts=aggregate(timing,['variant','timing_model'],['crps_ms']);ts.to_csv(out/'phase4b-timing.csv',encoding='utf-8')
    curves=distributions(root,records,raw);dump(out/'phase4b-distribution-curves.json',curves);ioc=interobserver(records);dump(out/'phase4b-interobserver.json',ioc)
    plt.style.use('seaborn-v0_8-whitegrid');fig,axs=plt.subplots(2,3,figsize=(15,8))
    for v in list(VARIANTS)+['Human']:
        cc=curves['curves'][v];kw={'lw':2.5,'color':'black'} if v=='Human' else {};axs[0,0].plot(curves['amplitude_px'],cc['amplitude_all'],label=v,**kw);axs[0,1].plot(curves['amplitude_px'],cc['amplitude_later'],label=v,**kw);axs[0,2].plot(curves['amplitude_deg'],cc['amplitude_deg'],label=v,**kw);axs[1,0].plot(curves['duration_ms'],cc['duration'],label=v,**kw);axs[1,1].plot(curves['duration_ms'],cc['survival_terminal_censored'],label=v,**kw);axs[1,2].plot(curves['saccade_number'],cc['target_curve'],label=v,**kw)
    for ax,label in zip(axs.flat,['All transitions (px)','Excluding first transition (px)','All transitions (degrees)','Observed fixation duration (ms)','Terminal-censored KM sensitivity (ms)','Saccade number']):ax.set(xlabel=label,ylabel='Cumulative fraction' if ax!=axs[1,1] else 'Survival fraction',ylim=(0,1.02));ax.legend(fontsize=7,ncol=3)
    fig.tight_layout();fig.savefig(out/'phase4b-distributions.png',dpi=145);plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(13,4))
    for v in VARIANTS:
        axs[0].plot([8,16,24,32,48,64],[ps.loc[v,f'coverage_{k}'] for k in [8,16,24,32,48,64]],label=v);axs[1].plot([1,2,3,4],[ps.loc[v,f'probability_within_deg_{k}'] for k in [1,2,3,4]],label=v)
    axs[0].set(xlabel='Candidate tolerance (px)',ylabel='Coverage');axs[1].set(xlabel='Tolerance (degrees)',ylabel='Selection probability near human');axs[2].bar(list(VARIANTS),[ps.loc[v,'expected_error_px'] for v in VARIANTS]);axs[2].set(xlabel='Variant',ylabel='Expected next-fixation error (px)');axs[0].legend(ncol=4,fontsize=7);axs[1].legend(ncol=4,fontsize=7);fig.tight_layout();fig.savefig(out/'phase4b-selection.png',dpi=145);plt.close(fig)
    pilots={}
    for v in VARIANTS:
        p=root/(f'validation-{v}-revision2' if v in 'NF' else 'validation-E-anchors' if v=='E' else f'validation-{v}')/'report.json'
        if p.exists():pilots[v]=read(p)['summary'][v]
    report=dict(phase='4B',status='Exploratory independent-scene ablation; implementation-audit amendment before inspecting test outcomes',scenes=6,trials=len(records),participants=len({r['participant'] for r in records}),seeds=[11,23,37],autonomous_runs=len(df),prefix_transitions=len(prefixes),summary=json.loads(summary.to_json(orient='index')),prefix_summary=json.loads(ps.to_json(orient='index')),paired=pairs,prefix_paired=prefix_pairs,interobserver=ioc,validation_pilot=pilots,selected_before_test='D (provisional validation comparator, no arbitrary combined score)',timing=json.loads(ts.reset_index().to_json(orient='records')),protocol=read(root/'frozen-design.json'),limits=['Six independent-to-observer-development test scenes from released training material, not official held-out benchmark. Pretrained vision backbone may have seen COCO images.','Only correct target-present trials; no absent-target, free-viewing, dynamic-scene, head-eye or robot validation.','Model viewing windows and starts conditioned on humans; this does not predict response times. Human fixation durations clipped by cumulative RT because saccade gaps are unavailable.','Detector scores, internal confidence and proto support are uncalibrated. Annotated target-box overlap is only a target-specific recognition proxy, not mAP or calibrated semantic uncertainty.','No human memory test or complete object annotations: memory retention proxy does not validate human memory.','Approximate published 54 x 35 degree display geometry, not participant-specific calibration.','Pointwise crossed scene/participant bootstrap intervals; six scenes limit power. Sign-flip diagnostics assume scene exchangeability. Nonsignificance is not equivalence.','Revision2 fixes proto-candidate attribute inheritance discovered in trace audit; test jobs had started, but no outcome scores had been inspected. Unchanged A-E/S cache outputs retained; corrected F/N revalidated and rescored. Fresh-scene evidence remains exploratory.'])
    dump(out/'phase4b-results.json',report)
    body='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Phase 4B · Mechanism comparison</title><style>body{font:16px/1.6 system-ui;max-width:1250px;margin:32px auto;padding:0 22px;color:#253c44;background:#f7faf9}h1,h2{line-height:1.25}table{border-collapse:collapse;background:white;font-size:12px;width:100%}td,th{border:1px solid #cad9d2;padding:8px;text-align:left}.scroll{overflow:auto;margin:20px 0}img{max-width:100%}.notice{background:#e5efe9;border-left:4px solid #4d806b;padding:20px}a{color:#17685d}li{margin:8px 0}</style><nav><a href="../phase4b/">Decision laboratory ↗</a> · <a href="../index.html">Original scene explorer</a> · <a href="phase4a-report.html">Phase 4A</a></nav><h1>What actually improves the observer?</h1>'''
    body+='<p class="notice">Six new test scenes, '+str(len(records))+' human trials, three paired simulation seeds. This is an exploratory mechanism comparison, not a claim of human-equivalent gaze. D was the simpler provisional comparator selected from validation before inspecting these test outcomes. No mechanism or timing parameter was selected again on this test.</p>'
    body+=f'<p>Relative to A, the preselected D comparator changes count NSS from {summary.loc["A","count_nss"]:.3f} to {summary.loc["D","count_nss"]:.3f}, while post-initial target acquisition is {100*summary.loc["D","model_target_ever"]:.1f}% versus human {100*summary.loc["D","human_target_ever"]:.1f}%. Thus improved allocation does not solve target search. Added mechanisms are judged by the individual endpoints and intervals below, not by complexity.</p>'
    body+=mechanism_note(summary)
    body+='<h2>Matched autonomous results</h2><p>A original frozen Phase 3 lognormal; B valid stimulus; C target guidance; D area-normalized selection; E adaptive proposals; F coarse multiscale perception + persistent proto-objects. S = E + coarse sensor only; N = F without across-fixation proto persistence. All use the same frozen lognormal timing parameters. Spatial DTW is custom normalized spatial DTW, not MultiMatch or ScanMatch. Means first average seeds, tasks and participants within scenes, then weight the six scenes equally.</p>'
    show=summary.reset_index()[['variant','model_target_ever','human_target_ever','count_nss','duration_nss','count_cc','count_sim','count_gain_vs_center_bits','spatial_dtw','model_amplitude_all_px','duration_wasserstein_ms','model_padding_fraction']];body+=table(show)
    body+='<h2>What changed and what did not</h2><p>Use the table and paired intervals jointly: improving target acquisition alone does not establish human-like search. Masking is a geometric validity correction. Peripheral evidence can be weak or arrive too late. Adaptive proposals can improve coverage while diluting probability. Dynamic representations are experimental and remain optional; they are not promoted merely because they are more elaborate. The report preserves negative outcomes.</p>'
    body+='<h3>Initial-gaze and termination sensitivity</h3><p>Primary target acquisition excludes the supplied initial fixation. Some starts are already on the annotated target, so both including-initial and initially-off-target results are supplied. Human success in the raw trial does not guarantee a post-initial target-box hit within this reconstructed window.</p>'+table(summary.reset_index()[['variant','initial_human_on_target','model_target_including_initial','human_target_including_initial','model_target_acquisition_initial_off_target','human_target_acquisition_initial_off_target','model_no_post_initial_fixations','model_terminal_censored','model_refixation_rate','human_refixation_rate','model_small_step_fraction','human_small_step_fraction','model_fixation_count','human_fixation_count']])
    body+='<h2>Paired mechanism effects</h2><p>Each delta is the first variant minus the second. Higher target acquisition and NSS are favorable; lower DTW, error and padding are favorable. Bootstrap uses 2,000 crossed scene/participant resamples after averaging simulation seeds; intervals are pointwise. Exact scene sign-flip diagnostics and Holm adjustments across all 56 planned endpoint comparisons are supplied, with very low power at six scenes.</p>'+table(pd.DataFrame([dict(comparison=r['comparison'],metric=r['metric'],delta=r['delta'],lower=r['ci95'][0],upper=r['ci95'][1],p=r['scene_signflip_p'],holm_p=r['holm_p_all_endpoints']) for r in pairs]))
    body+='<h2>Selection at observed human prefixes</h2><p>Current gaze and past durations condition perception and memory. Candidates and sampled choice are frozen before the next human coordinate is read. This separates localization coverage from probability allocation and is not autonomous prediction.</p>'+table(ps.reset_index()[['variant','coverage_48','coverage_deg_2','nearest_rank_min','nearest_probability','probability_within_deg_2','expected_error_px','unexplored_region_probability','target_location_brier_proxy']])+'<img src="phase4b-selection.png" alt="Candidate coverage, probability near the next human fixation and expected selection error">'
    body+='<h3>Paired prefix effects</h3><p>Same crossed bootstrap and scene-level sign-flip diagnostics; Holm correction within this separate 32-comparison exploratory prefix family.</p>'+table(pd.DataFrame([dict(comparison=r['comparison'],metric=r['metric'],delta=r['delta'],lower=r['ci95'][0],upper=r['ci95'][1],holm_p=r['holm_p_all_endpoints']) for r in prefix_pairs]))
    body+='<p>Target-location Brier values compare a capped peripheral relevance signal with coarse binary target-box occupancy; they are a diagnostic proxy, not a validated semantic calibration score. Leave-one-participant-out human count NSS reference: '+f'{ioc["count_nss"]:.3f}'+'. It uses the same image and task, excludes the supplied initial gaze, and is not a strict upper bound.</p>'
    body+='<h2>Sequence and timing</h2><img src="phase4b-distributions.png" alt="Saccade amplitudes, fixation durations, survival sensitivity and target acquisition by saccade"><p>Curves average seeds, tasks, participants and then scenes. Target acquisition curves mean acquired by at most N observed post-initial fixations within the RT-conditioned window, including trials that terminate earlier. They do not predict completion time. Stays are merged; amplitude sensitivity excludes the first transition. Duration CDF includes observed window truncation. KM is a sensitivity analysis treating the terminal human fixation as censored; the source lacks exact gaps and censoring may be informative, so this is not a physiological lifetime estimate.</p>'
    body+='<h3>All three original timing mechanisms remain testable</h3><p>Conditional duration CRPS (ms, lower is better), six draws per interior human fixation; initial and terminal contexts excluded, original Phase 3 parameters frozen. Changed spatial/perceptual context can change the timing predictions even with identical timing parameters.</p>'+table(ts.reset_index())
    body+='<h2>Functional evidence and limits</h2>'+table(summary.reset_index()[['variant','central_target_seen','target_detection_precision_proxy','target_memory_retention_proxy','early_target_cue','proto_recognized_fraction','proto_merge_events','proto_discard_events','trace_complete']])+'<p>Central target recognition is counted only for centrally gated target-class detector evidence overlapping the annotated target box. It is not full object recognition accuracy. Memory retention checks internal target-support persistence after a correct local observation, not human recall. Proto status and confidence are internal state. Full object accuracy, semantic calibration and human memory performance remain unverified because the required annotations and behavioral tests are absent.</p>'
    body+='<h2>Sequential validation and audit record</h2><p>Padding was first assessed on earlier exploratory scenes. C, D and E were assessed sequentially on six disjoint new validation scenes using seed 11, before adding optional F. C produced no changed trajectory in that pilot despite sparse late target cues. D improved its pilot target acquisition and count NSS but worsened duration mismatch; E did not improve those spatial endpoints. The first replacement-only localization prototype removed valid anchors and was retired; final E retains them. F is tested with a sensor-only and a no-persistence control. An implementation audit then caught inherited proto-candidate attributes. Revision2 recomputes them at the proto location. Test jobs had begun but no held-out outcome scores had been inspected; corrected F/N validation and test runs supersede stale engineering outputs. This procedural amendment is why fresh-scene results are described as exploratory.</p>'+table(pd.DataFrame(pilots).T.reset_index().rename(columns={'index':'variant'})[['variant','model_target_ever','count_nss','spatial_dtw','model_amplitude_all_px','duration_wasserstein_ms']])
    body+='<h2>Trace and visualization contract</h2><p>The laboratory shows actual available/rejected candidates, additive utility components, normalized probabilities, chosen outgoing location, incoming choice, exact-input SHA256 checks, current detections, tentative peripheral evidence, multiscale features, proto changes and core memory before/after each observation. The priority raster is a smoothed display of the logged vector; the vector itself drives sampling. Race timing uses a separately labeled first-passage choice with no invented softmax likelihood. Human next-fixation labels are evaluation-only. Photographs and individual replay records stay in the owner-only internal research site and local workspace, outside the public GitHub repository.</p>'
    body+='<h2>Scope and reproducibility</h2><ul>'+''.join('<li>'+html.escape(x)+'</li>' for x in report['limits'])+'</ul><p><a href="phase4b-results.json">Results and frozen protocol</a> · <a href="phase4b-by-image.csv">Per-image results</a> · <a href="phase4b-by-participant.csv">Participant aggregates</a> · <a href="phase4b-summary.csv">Summary</a> · <a href="phase4b-paired.json">Effects and intervals</a> · <a href="phase4b-prefix-summary.csv">Prefix diagnostics</a> · <a href="phase4b-timing.csv">Timing</a> · <a href="https://sites.google.com/view/cocosearch/">Dataset</a></p></html>'
    (out/'phase4b-report.html').write_text(body,encoding='utf-8');print(summary[['model_target_ever','count_nss','model_amplitude_all_px','model_padding_fraction']].to_string());print('Saved',out)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--work',required=True);p.add_argument('--manifest',required=True);p.add_argument('--raw',required=True);p.add_argument('--out',required=True);a=p.parse_args();build(a.work,a.manifest,a.raw,a.out)
