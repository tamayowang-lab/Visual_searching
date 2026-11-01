from dataclasses import asdict
import numpy as np
import pytest
from active_observer.config import Config
from active_observer.perception.foveation import observe
from active_observer.perception.objects import ManualObjects,Detection
from active_observer.perception.pretrained import CachedObjects,verify
from active_observer.world.model import World
from active_observer.attention.policy import score

def test_single_salient_object_and_balanced_salient_objects():
    im=np.full((100,160,3),70,np.uint8);im[30:50,30:50]=[240,20,20];o=observe(im,(80,50),Config());w=World(160,100,Config())
    candidates=score(w,o,'',Config(policy='salience'))
    strongest=max(candidates,key=lambda c:c['utility']);assert strongest['x']<80
    # Symmetric scene and symmetric starting gaze give equal local salience.
    im[:,80:]=im[:,:80][:,::-1];o=observe(im,(79.5,49.5),Config());a=score(w,o,'',Config(policy='salience'));left=min(a,key=lambda c:(c['x']-30)**2+(c['y']-30)**2);right=min(a,key=lambda c:(c['x']-130)**2+(c['y']-30)**2);assert abs(left['terms']['salience']-right['terms']['salience'])<.08

def test_central_peripheral_identity_and_unexplored_candidates():
    im=np.zeros((100,160,3),np.uint8);p=ManualObjects([{'bbox':[20,20,40,40],'label':'cup'}]);central=observe(im,(30,30),Config());peripheral=observe(im,(120,70),Config());assert len(p.detect(central))==1;assert p.detect(peripheral)==[]
    w=World(160,100,Config());assert all(c['kind']=='region' for c in score(w,peripheral,'',Config()))

def test_familiarity_reduces_novelty_and_ambiguous_identity_remains_uncertain():
    im=np.zeros((100,160,3),np.uint8);o=observe(im,(80,50),Config());w=World(160,100,Config());d=Detection([75,45,85,55],{'cup':.5,'bottle':.5},[.4,.4,.4],.8,.5)
    w.update([d],o,300);before=w.objects[0].novelty
    for i in range(8):w.update([d],o,300)
    assert w.objects[0].novelty<before;assert w.objects[0].identity_uncertainty>.99

def test_cached_observations_do_not_reinfer_and_cache_tracks_gaze(tmp_path):
    class Stub:
        def __init__(self):self.calls=0
        def metadata(self):return {'provider':'test-only'}
        def detect(self,o):self.calls+=1;return [Detection([1,1,5,5],{'unknown':1},[0,0,0],.2,.1)]
    p=Stub();cache=CachedObjects(p,tmp_path);im=np.zeros((30,40,3),np.uint8);o=observe(im,(10,10),Config());a=cache.detect(o);b=cache.detect(o);assert p.calls==1 and asdict(a[0])==asdict(b[0]);cache.detect(observe(im,(11,10),Config()));assert p.calls==2

def test_missing_or_wrong_checkpoint_does_not_fabricate_predictions(tmp_path):
    with pytest.raises(RuntimeError,match='No classical fallback'):verify(tmp_path/'missing','x')
    p=tmp_path/'wrong';p.write_bytes(b'incorrect')
    with pytest.raises(ValueError,match='SHA256'):verify(p,'0'*64)

def test_factory_can_use_the_same_constant_depth_contract_in_evaluation(tmp_path):
    from active_observer.perception.factory import ProviderFactory
    cfg=Config(perception={'objects':{'provider':'classical'},'depth':{'provider':'constant'},'cache_directory':str(tmp_path)})
    objects,depth=ProviderFactory().get(cfg);assert depth.predict(np.zeros((10,10,3),np.uint8)).valid.all()
    cfg.depth_enabled=False;assert ProviderFactory().get(cfg)[1] is None
