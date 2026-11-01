from dataclasses import replace
import numpy as np
from active_observer.config import Config
from active_observer.simulation.runner import run,run_phase2

def image():
    a=np.full((60,100,3),70,np.uint8);a[10:30,20:40]=[230,20,20];return a

def test_phase2_default_path_is_preserved():
    cfg=Config(max_fixations=4);assert run(image(),cfg)==run_phase2(image(),cfg)

def test_three_variants_have_real_traces_and_reproducible_events():
    for mode in ['heuristic_lognormal','accumulator','accumulator_hold']:
        cfg=Config(max_fixations=4,time_budget_ms=1500,timing_model=mode);a=run(image(),cfg);assert a==run(image(),cfg);assert a['fixations'][0]['timing']['trace'];assert a['duration_ms']<=1500+1e-8
        for s in a['saccades']:assert s['duration_ms']>=s['observed_duration_ms']
        for f in a['fixations']:assert abs(f['fixation_offset_ms']-f['fixation_onset_ms']-f['duration_ms'])<1e-8

def test_execution_is_not_fixation_and_budget_can_end_during_preparation():
    cfg=Config(max_fixations=20,time_budget_ms=50,timing_model='accumulator',timing={'diffusion':0,'urgency':100,'priority_gain':0,'preparation_ms':60});a=run(image(),cfg);assert len(a['fixations'])==1;assert a['fixations'][0]['right_censored'];assert not a['saccades'];assert a['duration_ms']==50
    cfg=Config(max_fixations=3,time_budget_ms=2000,timing_model='accumulator',timing={'diffusion':0});a=run(image(),cfg)
    assert abs(a['duration_ms']-sum(f['duration_ms'] for f in a['fixations'])-sum(s['observed_duration_ms'] for s in a['saccades']))<1e-8

def test_no_hold_does_not_use_processing_signal():
    from active_observer.attention.timing import simulate_timing,ProcessingContext,TimingParameters
    p=TimingParameters();a=simulate_timing([0,1],ProcessingContext(0,1,500),'accumulator',p,np.random.default_rng(1));b=simulate_timing([0,1],ProcessingContext(.9,.9,50),'accumulator',p,np.random.default_rng(1));assert a['duration_ms']==b['duration_ms']
