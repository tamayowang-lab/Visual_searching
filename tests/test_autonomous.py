"""Focused architectural and visual-evidence checks; no statistical evaluation."""
from dataclasses import replace
import hashlib
import json
import numpy as np
import pytest
from active_observer.config import Config
from active_observer.simulation.runner import run
from active_observer.autonomous.demo import scene_image, demo_config
from active_observer.experiment.trajectory import inspect_at
from active_observer.perception.foveation import observe
from active_observer.perception.objects import Detection


def short_cfg(**kwargs):
    base=replace(demo_config(),perception={'objects':{'provider':'classical'},'depth':{'provider':'none'}})
    return replace(base,experiment={**base.experiment,'administrative_timeout_ms':1800.,'fixed_time_ms':1000.},**kwargs)


def test_default_dispatch_has_no_legacy_count_time_grid_or_race_dependence(monkeypatch):
    from active_observer.attention import policy
    from active_observer.active import candidates, policy as active_policy
    def forbidden(*args,**kwargs):raise AssertionError('Legacy candidates were used')
    monkeypatch.setattr(policy,'score',forbidden)
    monkeypatch.setattr(active_policy,'score',forbidden)
    monkeypatch.setattr(candidates,'refine_candidates',forbidden)
    image=scene_image('porto-square')
    a=run(image,short_cfg(max_fixations=1,time_budget_ms=1,grid_columns=0,grid_rows=0,timing_model='accumulator',timing={'selection_mode':'race'}))
    b=run(image,short_cfg(max_fixations=999,time_budget_ms=999999,grid_columns=31,grid_rows=27))
    assert a['model']=='autonomous-dense-field'
    assert len(a['fixations'])>1
    assert a['fixations']==b['fixations']
    assert a['termination']==b['termination']
    assert all(f['timing']['variant']=='heuristic_lognormal' and f['timing']['selection_mode']=='softmax' for f in a['fixations'])


@pytest.mark.parametrize('scene',['porto-square','porto-promenade','porto-waterfront'])
def test_real_photographs_have_valid_explainable_gaze_and_memory(scene):
    image=scene_image(scene)
    result=run(image,short_cfg())
    h,w=image.shape[:2]
    assert image.shape==(400,640,3)
    assert np.unique(image.reshape(-1,3),axis=0).shape[0]>10000
    assert all(0<=f['x_px']<w and 0<=f['y_px']<h for f in result['fixations'])
    assert any(f['memory']['objects'] for f in result['fixations'])
    for i,f in enumerate(result['fixations']):
        tr=f['decision_trace'];assert tr['stopping']['reason'];assert tr['memory_after']==f['memory']
        if not tr['selected_next']:continue
        selected=tr['selected_next'];ix,iy=selected['field_index'];field=tr['priority']
        assert abs(sum(selected['components'].values())-selected['utility'])<1e-10
        assert abs(field['total'][iy][ix]-selected['utility'])<.00006
        assert selected['selection_probability']>0
        if i+1<len(result['fixations']):
            following=result['fixations'][i+1];assert [following['x_px'],following['y_px']]==[selected['x'],selected['y']]


def test_exact_input_and_memory_are_causal():
    image=scene_image('porto-square');captured=[]
    class Provider:
        def detect(self,obs):
            captured.append(obs.rgb.copy())
            if len(captured)>1:return []
            x,y=obs.gaze
            return [Detection([x-10,y-10,x+10,y+10],{'local':.8,'unknown':.2},[.2,.2,.2],.7,.5)]
    cfg=short_cfg();result=run(image,cfg,object_provider=Provider())
    for f,pixels in zip(result['fixations'],captured):
        expected=observe(image,(f['x_px'],f['y_px']),cfg,strict_periphery=True)
        assert np.array_equal(pixels,expected.rgb)
        assert hashlib.sha256(pixels.tobytes()).hexdigest()==f['decision_trace']['observation']['rgb_sha256']
    first=result['fixations'][0];view=inspect_at(result,first['duration_ms']/2)
    assert view['memory']['objects']==[]
    assert view['current_partial_fixation']['decision_trace']['stopping'] is None
    assert view['current_partial_fixation']['decision_trace']['selection']['action']=='not_yet_decided'
    after=inspect_at(result,first['fixation_offset_ms']);assert after['memory']['objects'][0]['probabilities']['local']==.8
    assert result['fixations'][1]['memory']['objects'][0]['observation_count']==1


def test_reproducibility_and_fixed_time_are_views_of_same_trajectory():
    cfg=short_cfg();image=scene_image('porto-promenade')
    a=run(image,cfg);assert a==run(image,cfg)
    b=run(image,replace(cfg,experiment={**cfg.experiment,'observation_condition':'fixed_time','fixed_time_ms':450.}))
    assert a['fixations']==b['fixations'];assert a['termination']==b['termination']
    for t in [0,100,250,450,1300,2000]:assert inspect_at(a,t)==inspect_at(b,t)
    json.dumps(a,allow_nan=False)


def test_administrative_timeout_and_unresolved_are_not_cognitive_completion():
    image=scene_image('porto-square');cfg=short_cfg()
    result=run(image,replace(cfg,experiment={**cfg.experiment,'administrative_timeout_ms':100.,'fixed_time_ms':50.}))
    assert result['termination']['outcome']=='administrative_timeout'
    assert not result['termination']['cognitive_completion']
    assert result['fixations'][-1]['decision_trace']['stopping']['administrative_timeout']
    unresolved=run(image,replace(cfg,experiment={**cfg.experiment,'behavior_mode':'search'},stopping=dict(observation_cost=10.,minimum_elapsed_ms=0.,persistence=1,allow_absence=False,recent_gain_rate_threshold=100.)),target='chair')
    assert unresolved['termination']['outcome']=='unresolved'
    assert not unresolved['termination']['cognitive_completion']


def test_two_modes_only():
    with pytest.raises(ValueError):run(scene_image('porto-square'),Config(experiment={'behavior_mode':'free-view'}))
    with pytest.raises(ValueError):run(scene_image('porto-square'),Config(experiment={'behavior_mode':'search'}))


def test_unobserved_fine_detail_cannot_leak_through_the_peripheral_channel():
    base=np.full((120,192,3),100,np.uint8)
    altered=base.copy()
    yy,xx=np.mgrid[:12,:12]
    altered[:12,:12]=np.repeat((((xx+yy)%2)*200)[...,None],3,axis=2)
    cfg=demo_config();gaze=(96.,60.)
    assert np.array_equal(observe(base,gaze,cfg,strict_periphery=True).rgb,
                          observe(altered,gaze,cfg,strict_periphery=True).rgb)


def test_verification_requires_central_semantics_not_peripheral_or_structure():
    from active_observer.experiment.stopping import StopController
    from active_observer.world.model import World
    from active_observer.active.masking import StimulusMask
    photo=scene_image('porto-square');h,w=photo.shape[:2];cfg=short_cfg()
    obs=observe(photo,(w/2,h/2),cfg,strict_periphery=True)
    mask=StimulusMask(w,h);world=World(w,h,cfg,spatial_grid=False)
    candidate=dict(id='local-observation',terms={'novelty':1.,'move':0.,'salience':1.},perceptual_support=1.,expected_uncertainty_reduction=1.)
    outcomes={}
    for source in ['semantic','proto']:
        controller=StopController('search','person')
        detection=Detection([w/2-20,h/2-20,w/2+20,h/2+20],{'person':.99,'unknown':.01},[.2,.3,.4],.99,.5,source=source)
        for elapsed in [250.,500.]:
            outcomes[source]=controller.assess(world,obs,[detection],[candidate],None,mask,elapsed,250.,spatial_coverage=.5)
    assert outcomes['semantic']['terminal_outcome']=='target_verified'
    assert outcomes['proto']['target_evidence_count']==0
    assert outcomes['proto']['terminal_outcome']!='target_verified'


def test_autonomous_observation_can_continue_beyond_sixteen_fixations():
    cfg=short_cfg(max_fixations=1)
    cfg=replace(cfg,experiment={**cfg.experiment,'administrative_timeout_ms':10000.})
    result=run(scene_image('porto-square'),cfg)
    assert len(result['fixations'])>16
