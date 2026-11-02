"""Focused scientific-boundary tests; test sensors never become demo states."""
from dataclasses import replace
from pathlib import Path
import hashlib,json
import numpy as np
import pytest
from active_observer.config import Config
from active_observer.io.image import load_image,Camera
from active_observer.perception.foveation import observe
from active_observer.perception.objects import Detection
from active_observer.perception.depth import normalize
from active_observer.world.model import World
from active_observer.world.ensemble import summarize
from active_observer.simulation.runner import run
from active_observer.simulation.replay import inspect_at
from active_observer.simulation.stopping import StopController

PHOTO=Path(__file__).resolve().parents[1]/'examples/photos/porto-square.jpg'

class EmptyObjects:
    def detect(self,observation):return []

class LocalObjects:
    def detect(self,observation):
        x,y=observation.gaze;h,w=observation.rgb.shape[:2]
        return [Detection([max(0,x-4),max(0,y-4),min(w,x+4),min(h,y+4)],{'object':.99,'unknown':.01},[.3,.4,.5],.99,.5,source='semantic')]

class DepthSpy:
    def __init__(self):self.inputs=[]
    def predict(self,rgb):
        self.inputs.append(rgb.copy())
        h,w=rgb.shape[:2]
        return normalize(np.tile(np.linspace(.1,1,w),(h,1)),quality=np.full((h,w),.1),estimator='test sensor')

@pytest.fixture
def photo():
    import cv2
    return cv2.resize(load_image(PHOTO),(160,100),interpolation=cv2.INTER_AREA)

@pytest.fixture
def cfg():return Config(administrative_timeout_ms=1200.)


def test_depth_and_objects_receive_actual_gaze_input_on_every_fixation(photo,cfg):
    spy=DepthSpy();seen=[]
    class Objects(EmptyObjects):
        def detect(self,o):seen.append(o.rgb.copy());return []
    result=run(photo,cfg,object_provider=Objects(),depth_provider=spy)
    assert len(spy.inputs)==len(result['fixations'])==len(seen)>1
    assert not np.array_equal(spy.inputs[0],photo)
    for f,depth_rgb,obj_rgb in zip(result['fixations'],spy.inputs,seen):
        actual=observe(photo,f['gaze'],cfg).rgb
        assert np.array_equal(depth_rgb,actual)
        assert np.array_equal(obj_rgb,actual)
        assert f['depth_input_sha256']==hashlib.sha256(actual.tobytes()).hexdigest()


def test_recognition_does_not_reduce_reconstruction_uncertainty(photo,cfg):
    observation=observe(photo,(80,50),cfg);world=World(160,100,cfg);objects=LocalObjects().detect(observation)
    for _ in range(8):world.update(objects,observation,250,summarize(observation,32))
    m=world.objects[0]
    assert m.confidence>.9 and m.identity_uncertainty<.1
    assert m.reconstruction.depth_uncertainty==1
    assert m.reconstruction.dimension_uncertainty==[1,1,1]
    assert m.reconstruction.center is None
    world.elapse(1000)
    assert m.reconstruction.depth_uncertainty==1


def test_cuboid_hidden_extent_and_scale_stay_unknown(photo,cfg):
    result=run(photo,cfg,object_provider=LocalObjects(),depth_provider=DepthSpy())
    m=result['fixations'][0]['memory_after']['objects'][0]
    assert m['perceptual_uncertainty']['identity']<.1
    geometry=m['reconstruction']
    assert geometry['depth_uncertainty']>=.9
    assert geometry['dimensions'][:2]!=[None,None]
    assert geometry['dimensions'][2] is None
    assert geometry['dimension_uncertainty'][2]==1
    assert geometry['scale_id'] is None
    assert geometry['center'] is not None
    assert result['capabilities']['complete_cuboid_extent'] is False


def test_ensembles_never_expand_into_individual_objects(photo,cfg):
    result=run(photo,cfg,object_provider=EmptyObjects())
    snapshot=result['fixations'][-1]['memory_after']
    assert snapshot['objects']==[] and snapshot['ensembles']
    ensemble=snapshot['ensembles'][-1]['summary']
    assert ensemble['estimated_object_count'] is None
    assert ensemble['member_object_ids'] is None
    assert len(ensemble['mean_rgb'])==3
    assert ensemble['structural_group_count']>=0


def test_predicted_gain_is_not_measured_improvement(photo,cfg):
    result=run(photo,cfg,object_provider=LocalObjects(),depth_provider=DepthSpy())
    for f in result['fixations']:
        assert f['predicted_information_gain']>=0
        assert f['measured_proxies']['actual_reconstruction_improvement'] is None
        assert f['measured_proxies']['depth_stability'] is None
        assert f['stopping']['reconstruction_validated'] is False


def test_peripheral_detections_cannot_become_object_memory(photo,cfg):
    class Remote:
        def detect(self,o):return [Detection([0,0,8,8],{'remote':.99,'unknown':.01},[.3]*3,.99,.5,source='semantic')]
    result=run(photo,replace(cfg,administrative_timeout_ms=200),object_provider=Remote())
    assert result['fixations'][0]['memory_after']['objects']==[]
    assert result['fixations'][0]['memory_after']['ensembles']


def test_no_fixation_limit_or_candidate_grid(photo,cfg):
    with pytest.raises(TypeError):Config(max_fixations=16)
    with pytest.raises(TypeError):Config(grid_columns=8)
    with pytest.raises(TypeError):Config(execution_path='archived')
    result=run(photo,replace(cfg,administrative_timeout_ms=4000,timing_mean_ms=100,timing_sigma=.01),object_provider=EmptyObjects())
    assert len(result['fixations'])>16
    for f in result['fixations']:
        x,y=f['gaze'];assert 0<=x<160 and 0<=y<100
        if f['selected_next']:
            choice=f['selected_next'];ix,iy=choice['field_index']
            assert abs(sum(choice['components'].values())-choice['utility'])<1e-10
            assert abs(f['priority']['total'][iy][ix]-choice['utility'])<.00006


def test_reproducible_trajectory_and_causal_time_views(photo,cfg):
    result=run(photo,cfg,object_provider=LocalObjects(),depth_provider=DepthSpy())
    assert result==run(photo,cfg,object_provider=LocalObjects(),depth_provider=DepthSpy())
    serialized=json.dumps(result,allow_nan=False)
    first=result['fixations'][0]
    early=inspect_at(result,first['duration_ms']/2)
    assert early['memory']['objects']==[]
    assert early['stopping'] is None and early['next_choice'] is None and early['termination'] is None
    committed=inspect_at(result,first['offset_ms'])
    assert committed['memory']['objects'][0]['reconstruction']['center'] is not None
    for t in [0,100,250,600,1500]:inspect_at(result,t)
    assert json.dumps(result,allow_nan=False)==serialized


def test_timeout_is_not_validated_completion(photo,cfg):
    result=run(photo,replace(cfg,administrative_timeout_ms=100),object_provider=EmptyObjects())
    assert result['termination']['outcome']=='administrative_timeout'
    assert result['termination']['administrative_timeout'] is True
    assert result['termination']['reconstruction_validated'] is False


@pytest.mark.parametrize('depth_available',[False,True])
def test_low_value_stop_is_an_estimate_not_geometric_validation(cfg,depth_available):
    c=replace(cfg,low_value_patience=2,minimum_observation_ms=0)
    controller=StopController(c);priority={'gain':np.zeros((2,2))};valid=np.ones((2,2),bool)
    assert controller.assess(priority,valid,1,500,depth_available)['decision']=='continue'
    final=controller.assess(priority,valid,1,750,depth_available)
    assert final['decision']=='stop'
    assert final['outcome']==('low_estimated_benefit' if depth_available else 'reconstruction_unresolved')
    assert final['observed_reconstruction_improvement'] is None
    assert final['reconstruction_validated'] is False


def test_missing_depth_does_not_fabricate_geometry(photo,cfg):
    result=run(photo,cfg,object_provider=LocalObjects())
    assert result['capabilities']['gaze_dependent_depth'] is False
    assert result['fixations'][0]['memory_after']['objects'][0]['reconstruction']['center'] is None


def test_wrong_depth_shape_is_rejected(photo,cfg):
    class WrongDepth:
        def predict(self,rgb):return normalize(np.zeros((10,10)))
    with pytest.raises(ValueError,match='align'):
        run(photo,cfg,object_provider=EmptyObjects(),depth_provider=WrongDepth())


def test_precomputed_full_scene_depth_is_not_a_runtime_provider(cfg):
    from active_observer.perception.factory import ProviderFactory
    with pytest.raises(ValueError,match='precomputed'):
        ProviderFactory().get_depth(replace(cfg,depth={'provider':'precomputed','path':'unused.npy'}))


def test_unobserved_fine_detail_is_removed(cfg):
    a=np.full((120,192,3),100,np.uint8);b=a.copy();yy,xx=np.mgrid[:12,:12]
    b[:12,:12]=np.repeat((((xx+yy)%2)*200)[...,None],3,axis=2)
    assert np.array_equal(observe(a,(96,60),cfg).rgb,observe(b,(96,60),cfg).rgb)


def test_replay_uses_exact_recorded_input_and_rejects_wrong_scene(photo,cfg,tmp_path):
    from active_observer.viz.replay import write_replay
    result=run(photo,cfg,object_provider=EmptyObjects());path=tmp_path/'view.html'
    write_replay(photo,result,path);assert path.is_file()
    wrong=photo.copy();wrong[0,0]=255-wrong[0,0]
    with pytest.raises(ValueError,match='Wrong scene'):write_replay(wrong,result,path)


def test_invalid_inputs_fail_explicitly(photo,cfg):
    with pytest.raises(ValueError):run(photo,cfg,start=(-1,0),object_provider=EmptyObjects())
    with pytest.raises(ValueError):Config(spatial_temperature=float('nan')).validate()
    with pytest.raises(ValueError):Config(priority_width=0).validate()
    with pytest.raises(ValueError):run(photo,cfg,camera=Camera(100,100),object_provider=EmptyObjects())
