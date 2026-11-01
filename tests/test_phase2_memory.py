import numpy as np
import pytest
from active_observer.world.model import World
from active_observer.perception.objects import Detection
from active_observer.perception.foveation import observe
from active_observer.config import Config
from active_observer.world.frames import transform_point

def fixture(duration):
    cfg=Config();w=World(100,60,cfg);o=observe(np.zeros((60,100,3),np.uint8),(50,30),cfg);d=Detection([45,25,55,35],{'cup':.8,'unknown':.2},[.5,.5,.5],.8,.5,source='semantic');w.update([d],o,duration);return w,o,d

def test_duration_dependent_bounded_evidence():
    short,_,_=fixture(100);long,o,d=fixture(600);assert long.objects[0].confidence>short.objects[0].confidence
    for i in range(100):long.update([d],o,300)
    assert long.objects[0].confidence<=.8;assert long.objects[0].observation_count==101;assert long.objects[0].identity_uncertainty>0

def test_semantic_conflict_does_not_merge():
    w,o,d=fixture(300);other=Detection(d.bbox,{'dog':.9,'unknown':.1},d.appearance,.9,.5,source='semantic');w.update([other],o,300);assert len(w.objects)==2

def test_partial_observation_keeps_identity_and_observed_extent():
    w,o,d=fixture(300);identity=w.objects[0].id;part=Detection([48,26,54,34],d.probabilities,d.appearance,.8,.5,partial=True,source='semantic');w.update([part],o,300)
    assert len(w.objects)==1;assert w.objects[0].id==identity;assert w.objects[0].bbox==d.bbox;assert w.objects[0].partial

def test_pose_transform_is_separate_and_rigid():
    t=np.eye(4);t[:3,3]=[1,2,3];assert transform_point([0,0,0],t)==[1,2,3]
    with pytest.raises(ValueError):transform_point([0,0,0],np.ones((4,4)))
