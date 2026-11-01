from dataclasses import replace
import json
import numpy as np
import pytest
from active_observer.config import Config
from active_observer.simulation.runner import run
from active_observer.experiment.trajectory import inspect_at
from active_observer.experiment.options import ExperimentOptions

def image():
    a=np.full((60,100,3),70,np.uint8);a[20:40,35:65]=[230,20,20];return a

def cfg(**kwargs):
    return Config(timing_model='heuristic_lognormal',active={'enabled':True,'variant':'D','selection':'balanced','valid_stimulus':True},experiment={'enabled':True,'behavior_mode':'explore','observation_condition':'until_finished','fixed_time_ms':200.,'administrative_timeout_ms':1500.},**kwargs)

def test_conditions_share_one_trajectory_and_ignore_legacy_fixation_budget():
    a=cfg(max_fixations=1);b=replace(a,max_fixations=500,experiment={**a.experiment,'observation_condition':'fixed_time','fixed_time_ms':300.});x=run(image(),a);y=run(image(),b);assert len(x['fixations'])>1;assert [(f['x_px'],f['y_px'],f['duration_ms']) for f in x['fixations']]==[(f['x_px'],f['y_px'],f['duration_ms']) for f in y['fixations']];assert x['termination']==y['termination'];json.dumps(x,allow_nan=False)

def test_partial_view_never_exposes_future_memory_or_decision():
    x=run(image(),cfg());f=x['fixations'][0];t=f['duration_ms']/2;view=inspect_at(x,t);p=view['current_partial_fixation'];assert p is not None;assert p['decision_trace']['memory_after'] is None;assert view['memory']==f['decision_trace']['memory_before'];assert p['decision_trace']['selection']['action']=='not_yet_decided';assert p['decision_trace']['stopping'] is None;assert view['termination'] is None

def test_saccade_time_counts_and_terminal_state_is_preserved():
    x=run(image(),cfg());f=next(f for f in x['fixations'] if f.get('saccade_duration_ms',0)>0);t=f['fixation_offset_ms']+min(1.,f['saccade_duration_ms']/2);v=inspect_at(x,t);assert v['phase']=='saccade';assert v['elapsed_ms']==t;late=inspect_at(x,5000);assert late['elapsed_ms']==x['duration_ms'];assert late['actual_completion_time_ms']==x['duration_ms'];assert late['termination']==x['termination'];assert x['termination']['administrative_timeout']

def test_only_two_modes_and_explicit_administrative_limit():
    with pytest.raises(ValueError):ExperimentOptions(behavior_mode='legacy').validate()
    a=cfg();a.experiment['administrative_max_steps']=1;x=run(image(),a);assert len(x['fixations'])==1;assert x['termination']['administrative_timeout'];assert not x['termination']['cognitive_completion'];assert x['termination']['outcome']=='unresolved'
