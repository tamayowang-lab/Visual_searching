import numpy as np
import pytest
from active_observer.perception.depth import normalize,ConstantDepth,ArrayDepth,CachedDepth
from active_observer.perception.objects import ManualObjects
from active_observer.simulation.runner import run
from active_observer.config import Config

def test_robust_normalization_mask_and_no_metric_claim():
    a=normalize(np.array([[1,2,np.nan],[3,4,5.]]));assert not a.valid[0,2];assert a.inverse_depth.min()>=0;assert a.inverse_depth.max()<=1;assert 'not meters' in a.metadata['units'];assert not a.metadata['calibrated_uncertainty']

def test_degenerate_and_no_valid_flags():
    assert 'constant_or_degenerate' in normalize(np.ones((3,3))).metadata['quality_flags']
    assert not normalize(np.full((2,2),np.nan)).valid.any()

def test_dimensions_and_kind_are_checked():
    with pytest.raises(ValueError):ArrayDepth(np.ones((3,3))).predict(np.zeros((5,5,3),np.uint8))
    with pytest.raises(ValueError):normalize(np.ones((2,2)),kind='meters')

def test_cache_identity_and_repeated_prediction(tmp_path):
    class Counting:
        def __init__(self):self.calls=0
        def predict(self,image):self.calls+=1;return ConstantDepth().predict(image)
    p=Counting();image=np.zeros((8,9,3),np.uint8);c=CachedDepth(p,tmp_path,'dummy-v1');c.predict(image);c.predict(image);assert p.calls==1
    CachedDepth(p,tmp_path,'dummy-v2').predict(image);assert p.calls==2

def test_central_only_depth_memory_and_depth_ablation():
    image=np.full((60,100,3),50,np.uint8);provider=ManualObjects([dict(bbox=[5,5,20,20],label='cup')]);depth=ArrayDepth(np.tile(np.linspace(0,1,100),(60,1)))
    a=run(image,Config(max_fixations=1),start=(80,40),object_provider=provider,depth_provider=depth);assert not a['fixations'][0]['memory']['objects']
    b=run(image,Config(max_fixations=1),start=(12,12),object_provider=provider,depth_provider=depth);m=b['fixations'][0]['memory']['objects'][0];assert m['relative_depth'] is not None;assert len(m['xyz'])==3
    c=run(image,Config(max_fixations=1,depth_enabled=False),start=(12,12),object_provider=provider,depth_provider=depth);assert c['fixations'][0]['memory']['objects'][0]['relative_depth'] is None

def test_remote_depth_changes_do_not_change_single_fixation_beliefs():
    image=np.full((60,100,3),50,np.uint8);base=np.tile(np.linspace(0,1,100),(60,1));a=base.copy();a[40:50,80:90]=a[40:50,80:90][::-1,::-1]
    o=ManualObjects([dict(bbox=[5,5,20,20],label='cup')]);cfg=Config(max_fixations=1)
    x=run(image,cfg,start=(12,12),object_provider=o,depth_provider=ArrayDepth(base));y=run(image,cfg,start=(12,12),object_provider=o,depth_provider=ArrayDepth(a));assert x['fixations'][0]['memory']==y['fixations'][0]['memory']
