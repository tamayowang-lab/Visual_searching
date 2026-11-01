import numpy as np
from active_observer.config import Config
from active_observer.simulation.runner import run
from active_observer.viz.html import write_html
from active_observer.viz.dashboard import dashboard
from active_observer.perception.objects import ManualObjects

def test_signed_decomposition_matches_utility():
    im=np.zeros((40,60,3),np.uint8)
    for mode in ['full','center','random_center','salience','salience_ior']:
        r=run(im,Config(max_fixations=2,policy=mode))
        for c in r['fixations'][0]['next_candidates']:assert abs(sum(c['utility_components'].values())-c['utility'])<1e-12

def test_visualizations_work_without_server(tmp_path):
    im=np.zeros((40,60,3),np.uint8);r=run(im,Config(max_fixations=2));dashboard(im,r,tmp_path/'a.png');write_html(im,r,tmp_path/'a.html');assert (tmp_path/'a.png').stat().st_size>1000
    s=(tmp_path/'a.html').read_text(encoding='utf-8');assert 'data:image/png;base64,' in s;assert '__OBSERVER_DATA__' not in s;assert 'https://cdn' not in s

def test_private_semantic_labels_do_not_change_peripheral_search():
    im=np.zeros((60,100,3),np.uint8);cfg=Config(max_fixations=1)
    a=run(im,cfg,start=(80,40),object_provider=ManualObjects([dict(bbox=[5,5,20,20],label='cup')]))
    b=run(im,cfg,start=(80,40),object_provider=ManualObjects([dict(bbox=[5,5,20,20],label='secret')]))
    assert a==b

def test_stays_are_bounded_by_budget_and_target_stop_is_explicit():
    im=np.zeros((40,60,3),np.uint8);r=run(im,Config(max_fixations=3,exploration_enabled=False));assert len(r['fixations'])==3
    p=ManualObjects([dict(bbox=[20,10,40,30],label='cup')]);r=run(im,Config(stop_on_target=True,target_threshold=.7,evidence_model="legacy"),target='cup',object_provider=p);assert len(r['fixations'])==1;assert r['stop_reason']=='explicit target detection'

def test_target_stop_requires_current_central_evidence():
    from active_observer.perception.objects import Detection
    class PeripheralFixture:
        def detect(self,o):return [Detection([0,0,5,5],{'cup':1.},[0.,0.,0.],1.,.5)]
    r=run(np.zeros((80,120,3),np.uint8),Config(max_fixations=1,stop_on_target=True),target='cup',object_provider=PeripheralFixture())
    assert r['stop_reason']=='fixation budget'
