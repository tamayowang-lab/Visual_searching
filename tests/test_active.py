from dataclasses import replace
import numpy as np
import pytest
from active_observer.config import Config
from active_observer.simulation.runner import run
from active_observer.active.masking import StimulusMask

def image():
    a=np.full((60,100,3),70,np.uint8);a[10:30,20:40]=[230,20,20];return a

def test_active_A_preserves_phase3_numerical_trajectory():
    cfg=Config(max_fixations=5,time_budget_ms=2000,timing_model='heuristic_lognormal');a=run(image(),cfg);b=run(image(),replace(cfg,active={'enabled':True,'variant':'A'}))
    for x,y in zip(a['fixations'],b['fixations']):
        assert [x[k] for k in ['x_px','y_px','duration_ms']]==[y[k] for k in ['x_px','y_px','duration_ms']]
        assert [c['probability'] for c in x['next_candidates']]==[c['probability'] for c in y['next_candidates']]

def test_mask_never_infers_darkness_and_rejects_invalid_start():
    mask=StimulusMask(100,60,bounds=[20,0,80,60]);assert not mask.valid(5,10);assert mask.valid(30,10)
    cfg=Config(max_fixations=15,time_budget_ms=3000,timing_model='heuristic_lognormal',active={'enabled':True,'variant':'B','valid_stimulus':True})
    a=run(np.zeros((60,100,3),np.uint8),cfg,stimulus_bounds=[20,0,80,60]);assert all(20<=f['x_px']<80 for f in a['fixations'])
    with pytest.raises(ValueError):run(image(),cfg,start=(0,0),stimulus_bounds=[20,0,80,60])

def test_every_fixation_has_actual_reconstructable_decision_and_memory_trace():
    cfg=Config(max_fixations=5,timing_model='accumulator',active={'enabled':True});a=run(image(),cfg)
    for f in a['fixations']:
        t=f['decision_trace'];assert t['observation']['detections'];assert t['memory_after']==f['memory'];assert t['available_candidates']==f['next_candidates'];assert len(t['observation']['input_rgb_sha256'])==64
        if t['selection']['action']=='fixate':
            p=[c['probability'] for c in t['available_candidates']];j=min(len(p)-1,np.searchsorted(np.cumsum(p),t['selection']['uniform_draw'],side='right'));assert t['available_candidates'][j]['id']==t['selection']['chosen_candidate']
    assert a['fixations'][-1]['decision_trace']['selection']['action']=='stop'

def test_mask_applies_to_every_existing_policy():
    for policy in ['full','center','random_center','salience','salience_ior','random_walk']:
        cfg=Config(policy=policy,max_fixations=5,timing_model='heuristic_lognormal',active={'enabled':True,'variant':'B','valid_stimulus':True});r=run(image(),cfg,stimulus_bounds=[25,5,75,55])
        for f in r['fixations']:
            assert 25<=f['x_px']<75 and 5<=f['y_px']<55
            assert all(c['probability']==0 for c in f['decision_trace']['rejected_candidates'])
            assert abs(sum(c['probability'] for c in f['decision_trace']['available_candidates'])-1)<1e-10

def test_search_uses_only_coarse_current_observation_and_changes_prior_before_recognition():
    from active_observer.perception.objects import ClassicalObjects
    class Provider(ClassicalObjects):
        def coarse_semantics(self,rgb):
            self.received=rgb.copy();return [dict(bbox=[70,10,90,40],label='chair',detector_score=.8,weighted_evidence=.3,reliability=.2,status='test hypothesis',source='injected test provider')]
    p=Provider();cfg=Config(max_fixations=2,timing_model='heuristic_lognormal',active={'enabled':True,'variant':'C','behavior_mode':'search','peripheral_guidance':True,'valid_stimulus':True});r=run(image(),cfg,target='chair',object_provider=p);f=r['fixations'][0];assert any(c['terms']['task']>0 for c in f['next_candidates']);assert f['decision_trace']['perceptual_state']['peripheral_input_sha256']
    e=run(image(),replace(cfg,active={**cfg.active,'behavior_mode':'explore'}),target='chair',object_provider=p);assert all(c['terms']['task']==0 for c in e['fixations'][0]['next_candidates'])

def test_rich_search_trace_remains_strict_json_after_memory_decay():
    import json
    cfg=Config(max_fixations=5,timing_model='heuristic_lognormal',active={'enabled':True,'variant':'C','behavior_mode':'search','peripheral_guidance':True});r=run(image(),cfg,target='chair');json.dumps(r,allow_nan=False)

def test_balanced_group_normalization_resists_candidate_count_bias():
    from copy import deepcopy
    from active_observer.active.options import ActiveOptions
    from active_observer.active.policy import balanced_competition
    def c(kind):return dict(kind=kind,utility_components={},terms={'task':0.,'novelty':1.,'move':.1,'salience':.5,'history':0.},expected_uncertainty_reduction=.4,perceptual_support=.5,proposal_area=1.)
    a=balanced_competition([c('object'),c('region')],ActiveOptions(selection='balanced'),'');b=balanced_competition([c('object')]+[c('region') for _ in range(40)],ActiveOptions(selection='balanced'),'');assert abs(a[0]['probability']-b[0]['probability'])<1e-12
    for candidate in b:assert abs(sum(candidate['utility_components'].values())-candidate['utility'])<1e-12

def test_adaptive_localization_is_valid_and_data_dependent():
    cfg=Config(max_fixations=3,timing_model='heuristic_lognormal',active={'enabled':True,'variant':'E','valid_stimulus':True,'behavior_mode':'explore','selection':'balanced','adaptive_candidates':True})
    r=run(image(),cfg,stimulus_bounds=[20,0,80,60]);c=r['fixations'][0]['next_candidates'];assert any(a.get('adaptive') for a in c);assert all(20<=a['x']<80 for a in c);assert all(a.get('proposal_area',0)>0 for a in c)
    import json;json.dumps(r,allow_nan=False)

def test_bounded_sensor_discards_high_frequency_before_semantic_access():
    from active_observer.active.perception import bounded_peripheral_sensor
    a=np.indices((60,100)).sum(axis=0)%2;a=np.repeat((a*255).astype(np.uint8)[...,None],3,axis=2);b=bounded_peripheral_sensor(a,(50,30),8,width=40);assert b.shape==a.shape;assert b.std()<a.std()/4

def test_dynamic_workspace_records_tentative_states_and_strict_json():
    import json
    cfg=Config(max_fixations=5,timing_model='heuristic_lognormal',active={'enabled':True,'variant':'F','behavior_mode':'explore','valid_stimulus':True,'selection':'balanced','adaptive_candidates':True,'dynamic_perception':True,'peripheral_sensor':'bounded_rgb'})
    r=run(image(),cfg,stimulus_bounds=[20,0,80,60]);json.dumps(r,allow_nan=False);states=[f['decision_trace']['perceptual_state'] for f in r['fixations']];assert any(s['proto_events'] for s in states)
    for s in states:assert all(o['status']=='tentative' for o in s['proto_objects'])

def test_all_three_timing_variants_are_preserved_by_A():
    for timing in ['heuristic_lognormal','accumulator','accumulator_hold']:
        cfg=Config(max_fixations=4,timing_model=timing);a=run(image(),cfg);b=run(image(),replace(cfg,active={'enabled':True,'variant':'A'}));assert [(f['x_px'],f['y_px'],f['duration_ms']) for f in a['fixations']]==[(f['x_px'],f['y_px'],f['duration_ms']) for f in b['fixations']]

def test_coarse_semantics_cannot_be_configured_to_full_resolution():
    from active_observer.active.options import ActiveOptions
    with pytest.raises(ValueError):ActiveOptions(coarse_width=640).validate()

def test_adaptive_proposals_retain_valid_original_anchors():
    cfg=Config(max_fixations=1,timing_model='heuristic_lognormal',active={'enabled':True,'variant':'D','behavior_mode':'explore','selection':'balanced','valid_stimulus':True});a=run(image(),cfg);b=run(image(),replace(cfg,active={**cfg.active,'variant':'E','adaptive_candidates':True}));old={(c['x'],c['y']) for c in a['fixations'][0]['next_candidates']};new={(c['x'],c['y']) for c in b['fixations'][0]['next_candidates']};assert old<=new

def test_prefix_candidates_match_actual_state_and_future_label_never_changes_decision(tmp_path):
    from copy import deepcopy
    import cv2
    from active_observer.evaluation.active_prefix import replay
    from active_observer.perception.factory import ProviderFactory
    im=np.full((400,640,3),70,np.uint8);im[170:220,290:340]=[230,20,20];p=tmp_path/'image.png';cv2.imwrite(str(p),cv2.cvtColor(im,cv2.COLOR_RGB2BGR));cfg=Config(max_fixations=1,timing_model='heuristic_lognormal',depth_enabled=False,active={'enabled':True,'variant':'A'})
    simulation=run(im,cfg,target='chair');duration=simulation['fixations'][0]['duration_ms'];record=dict(image_id='unit',image_path=str(p),participant='unit',task='chair',valid_stimulus_bounds=[0,0,640,400],fixations=[dict(x=319.5,y=199.5,duration_ms=duration),dict(x=100.,y=100.,duration_ms=200.)]);a,_,_,_=replay(record,cfg,ProviderFactory());other=deepcopy(record);other['fixations'][1]['x']=500.;b,_,_,_=replay(other,cfg,ProviderFactory());ta=a['fixations'][0]['decision_trace'];tb=b['fixations'][0]['decision_trace'];assert ta['available_candidates']==tb['available_candidates'];assert ta['selection']==tb['selection'];assert ta['memory_after']==tb['memory_after'];assert [c['probability'] for c in ta['available_candidates']]==[c['probability'] for c in simulation['fixations'][0]['next_candidates']]

def test_dynamic_candidate_attributes_come_from_its_own_location():
    from active_observer.active.policy import decision_candidates
    from active_observer.active.options import ActiveOptions
    from active_observer.active.masking import StimulusMask
    from active_observer.perception.foveation import observe
    from active_observer.world.model import World
    from active_observer.attention.policy import feature_at
    cfg=Config(timing_model='heuristic_lognormal');im=image();obs=observe(im,(50,30),cfg);world=World(100,60,cfg)
    class Workspace:
        def support_at(self,x,y):return .5
        hypotheses=[dict(id='proto-test',point=[30.,20.],confidence=.2,probabilities={'unknown':1.},observations=1,status='tentative',linked_memory_id=None)]
    candidates,_=decision_candidates(world,obs,'',cfg,ActiveOptions(enabled=True,dynamic_perception=True),StimulusMask(100,60),Workspace());c=next(c for c in candidates if c['id']=='proto-test');assert c['uncertainty']==.8;assert c['terms']['salience']==feature_at(obs,30,20);assert abs(sum(a['probability'] for a in candidates)-1)<1e-12;assert c['depth_uncertainty']==1

def test_original_baseline_declares_padding_without_enforcing_or_changing_trajectory():
    cfg=Config(max_fixations=4,timing_model='heuristic_lognormal',active={'enabled':True,'variant':'A'})
    a=run(image(),cfg);b=run(image(),cfg,stimulus_bounds=[25,5,75,55]);assert [(f['x_px'],f['y_px'],f['duration_ms']) for f in a['fixations']]==[(f['x_px'],f['y_px'],f['duration_ms']) for f in b['fixations']];assert b['validity']['valid_fraction']<1;assert any(not c['valid_stimulus'] and c['probability']>0 for c in b['fixations'][0]['decision_trace']['available_candidates'])

def test_active_inspector_is_standalone_utf8_under_windows_default(tmp_path,monkeypatch):
    from pathlib import Path
    from active_observer.viz.html import write_html
    original=Path.open
    def windows_open(self,mode='r',buffering=-1,encoding=None,errors=None,newline=None):
        if 'b' not in mode and encoding in (None,'locale'):encoding='cp1252'
        return original(self,mode,buffering,encoding,errors,newline)
    monkeypatch.setattr(Path,'open',windows_open);r=run(image(),Config(max_fixations=1,timing_model='heuristic_lognormal',active={'enabled':True,'variant':'A'}),image_id='视觉</script>');path=tmp_path/'active.html';write_html(image(),r,path);s=path.read_bytes().decode('utf-8');assert 'window.OBSERVER_BUNDLE=' in s;assert '<script type="module" src=' not in s;assert '视觉</script>' not in s;assert 'decision_trace' in s
