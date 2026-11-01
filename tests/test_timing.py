from dataclasses import replace
import numpy as np
import pytest
from active_observer.attention.timing import TimingParameters,ProcessingContext,simulate_timing,diffusion_increment


def simulate(variant='accumulator',params=None,context=None,seed=4,**kw):return simulate_timing([0.,1.],context or ProcessingContext(),variant,params or TimingParameters(),np.random.default_rng(seed),**kw)

def test_noiseless_threshold_and_preparation():
    p=TimingParameters(urgency=4,priority_gain=1,diffusion=0,preparation_ms=30)
    r=simulate(params=p);assert abs(r['decision_time_ms']-200)<1e-8;assert abs(r['duration_ms']-230)<1e-8;assert r['winner_index']==1

def test_hold_zero_is_exactly_nested():
    p=TimingParameters(hold_strength=0)
    a=simulate(params=p);b=simulate('accumulator_hold',params=p);assert a['trace']==b['trace'];assert a['duration_ms']==b['duration_ms']

def test_diffusion_variance_scales_with_time():
    rng=np.random.default_rng(1);a=diffusion_increment(rng,200000,.3,.01);b=diffusion_increment(rng,200000,.3,.04);assert abs(b.var()/a.var()-4)<.1

def test_integration_step_sensitivity_noiseless():
    a=simulate(params=TimingParameters(diffusion=0,dt_ms=20));b=simulate(params=TimingParameters(diffusion=0,dt_ms=5));assert abs(a['duration_ms']-b['duration_ms'])<1e-8

def test_bounded_processing_hold_and_evidence_saturation():
    p=TimingParameters(diffusion=0,hold_strength=.8)
    easy=simulate('accumulator_hold',p,ProcessingContext(.75,.8,200));hard=simulate('accumulator_hold',p,ProcessingContext(0,.9,400));assert hard['duration_ms']>easy['duration_ms'];assert max(x['hold'] for x in hard['trace'])<=.8
    c=ProcessingContext(.2,.8,250);assert c.at(10000)['sensory_evidence']<=.8;assert c.at(10000)['acquisition_rate_per_second']<1e-10

def test_high_priority_plan_triggers_sooner():
    p=TimingParameters(diffusion=0);c=ProcessingContext();rng=np.random.default_rng(1)
    flat=simulate_timing([0.,0.],c,'accumulator',p,rng);high=simulate_timing([0.,1.],c,'accumulator',p,rng);assert high['duration_ms']<flat['duration_ms']

def test_budget_and_long_tail_not_clipped_to_800():
    p=TimingParameters(urgency=.7,priority_gain=0,diffusion=0)
    r=simulate(params=p);assert r['duration_ms']>800
    r=simulate(params=p,remaining_ms=150);assert r['duration_ms']==150;assert r['right_censored'];assert r['decision_time_ms'] is None

def test_seeds_and_invalid_parameters():
    assert simulate()==simulate()
    with pytest.raises(ValueError):TimingParameters(dt_ms=0).validate()
    with pytest.raises(ValueError):TimingParameters(hold_strength=2).validate()

def test_stochastic_step_sensitivity_is_small_relative_to_distribution():
    p=TimingParameters(diffusion=.25);a=[];b=[]
    for seed in range(250):
        a.append(simulate(params=p,seed=seed)['duration_ms']);b.append(simulate(params=replace(p,dt_ms=10),seed=seed)['duration_ms'])
    assert abs(np.mean(a)-np.mean(b))<8
    assert abs(np.quantile(a,.9)-np.quantile(b,.9))<12
