import numpy as np
from active_observer.config import Config
from active_observer.world.model import World
from active_observer.perception.foveation import observe
from active_observer.perception.objects import Detection
from active_observer.attention.policy import score
from active_observer.evaluation.metrics import merged_model_fixations

def fixture():
    im=np.full((80,120,3),50,np.uint8);im[20:40,30:50]=[230,20,20];o=observe(im,(40,30),Config());w=World(120,80,Config());w.update([Detection([35,25,45,35],{'cup':.8,'unknown':.2},[1,0,0],.8,.9)],o,100);return im,o,w

def test_information_is_separate_from_novelty_and_switchable():
    _,o,w=fixture();a=score(w,o,'',Config(novelty_enabled=False));assert all(c['terms']['novelty']==0 for c in a);assert any(c['terms']['info']>0 for c in a)
    assert all(c['terms']['info']==0 for c in score(w,o,'',Config(information_enabled=False)))

def test_stay_exists_only_when_further_evidence_has_value():
    _,o,w=fixture();assert any(c['kind']=='stay' for c in score(w,o,'',Config()))
    w.consecutive_stays=2;assert not any(c['kind']=='stay' for c in score(w,o,'',Config()))
    w.consecutive_stays=0;w.objects[0].confidence=.8;assert not any(c['kind']=='stay' for c in score(w,o,'',Config()))

def test_equal_candidates_and_movement_cost():
    _,o,w=fixture();cfg=Config();sc=score(w,o,'',cfg);near=min(sc,key=lambda c:c['terms']['move']);far=max(sc,key=lambda c:c['terms']['move']);assert near['terms']['move']<far['terms']['move']
    cfg.weights={**cfg.weights,'move':4};changed=score(w,o,'',cfg);by={c['id']:c for c in changed};assert by[far['id']]['utility']-far['utility']<by[near['id']]['utility']-near['utility']

def test_additional_observation_segments_merge_for_metrics():
    a=[dict(x_px=10,y_px=10,duration_ms=100,mode='shift'),dict(x_px=10,y_px=10,duration_ms=100,mode='stay')];m=merged_model_fixations(a);assert len(m)==1 and m[0]['duration_ms']==200;assert a[0]['duration_ms']==100

def test_expected_gain_accounts_for_observation_cost():
    _,o,w=fixture()
    for cell in w.grid:cell['coverage']=0.
    sc=score(w,o,'',Config(stay_enabled=False));regions=[c for c in sc if c['kind']=='region'];near=min(regions,key=lambda c:c['terms']['move']);far=max(regions,key=lambda c:c['terms']['move']);assert near['observation_cost']<far['observation_cost'];assert near['terms']['info']>far['terms']['info'];assert 0<near['expected_reliability']<=1
