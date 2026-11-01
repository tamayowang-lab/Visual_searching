import numpy as np
import pytest
from active_observer.config import Config
from active_observer.io.image import Camera
from active_observer.perception.foveation import observe
from active_observer.perception.objects import ManualObjects,Detection
from active_observer.simulation.runner import run
from active_observer.world.model import World
from active_observer.world.association import associate

@pytest.fixture
def rgb():
    image=np.full((80,120,3),60,np.uint8);image[10:30,10:30]=[220,30,30];return image

def test_camera_center_and_offscreen():
    c=Camera(120,80);assert np.allclose(c.bearing(59.5,39.5),[0,0,1]);assert c.bearing(100,40)[0]>0
    with pytest.raises(ValueError):c.bearing(-1,40)
    assert 'never meters' in c.metadata()['xyz_units']

def test_foveation_does_not_fabricate_pixels(rgb):
    o=observe(rgb,(60,40),Config());assert o.rgb.shape==rgb.shape;assert o.reliability[40,60]>o.reliability[0,0];assert 'pixel-based' in o.geometry

def test_private_manual_proposals_do_not_leak(rgb):
    provider=ManualObjects([dict(bbox=[10,10,30,30],label='secret')])
    assert provider.detect(observe(rgb,(90,60),Config()))==[]
    assert provider.detect(observe(rgb,(20,20),Config()))[0].probabilities['secret']>.8

def test_gated_one_to_one_association(rgb):
    w=World(120,80,Config());obs=observe(rgb,(20,20),Config());d=Detection([10,10,30,30],{'red':1},[1,0,0],.8,.8)
    w.update([d],obs,200);identity=w.objects[0].id;w.update([d],obs,200);assert len(w.objects)==1;assert w.objects[0].id==identity
    far=Detection([100,60,119,79],{'blue':1},[0,0,1],.8,.8);assert associate(w.objects,[far],144)==[]

def test_unobserved_memory_decays(rgb):
    w=World(120,80,Config());o=observe(rgb,(20,20),Config());d=Detection([10,10,30,30],{'red':1},[1,0,0],.8,.8);w.update([d],o,200);before=w.objects[0].confidence;w.update([],o,1000);assert w.objects[0].confidence<before;assert w.objects[0].state=='remembered'

def test_seeded_scanpath_and_explicit_budgets(rgb):
    c=Config(max_fixations=5);a=run(rgb,c);b=run(rgb,c);assert a==b;assert len(a['fixations'])==5;assert all(0<=p['x_px']<120 for p in a['fixations']);assert a['depth']['available'] is False
    assert all(p['candidate_kind'] in {'initial','region','object'} for p in a['fixations'])

def test_time_budget_and_invalid_start(rgb):
    a=run(rgb,Config(time_budget_ms=120));assert a['duration_ms']==120;assert a['stop_reason']=='time budget'
    with pytest.raises(ValueError):run(rgb,start=(120,0))

def test_weights_are_inspectable_and_probabilities_normalized(rgb):
    a=run(rgb,Config(max_fixations=2));cs=a['fixations'][0]['next_candidates'];assert abs(sum(c['probability'] for c in cs)-1)<1e-12
    assert set(cs[0]['terms'])=={'salience','task','context','novelty','info','move','history'}
