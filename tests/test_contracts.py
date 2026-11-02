"""Phase A acceptance contracts, independent of pretrained models and the viewer."""
from dataclasses import replace
from contextlib import nullcontext
import hashlib,json
import numpy as np
import pytest
from active_observer.config import Config
from active_observer.io.image import Camera
from active_observer.perception.foveation import observe
from active_observer.perception.depth import DepthFrame,record_depth,update_observed_depth,CachedDepth
from active_observer.perception.objects import Detection
from active_observer.world.model import World,ReconstructionEstimate
from active_observer.world.ensemble import summarize
from active_observer.attention.priority import PriorityField
from active_observer.simulation.runner import run,central_detections
from active_observer.simulation.recording import save_run,depth_arrays
from active_observer.simulation.replay import inspect_at

class Empty:
    def detect(self,o):return []
class Fixed:
    def detect(self,o):
        h,w=o.rgb.shape[:2];x,y=w/2,h/2
        return [Detection([x-3,y-3,x+3,y+3],{'patch':.9,'unknown':.1},[.3,.4,.5],.9,.4,source='semantic')]
class Depth:
    def __init__(self):self.calls=[]
    def predict(self,rgb):
        self.calls.append(rgb.copy());h,w=rgb.shape[:2]
        a=np.tile(np.linspace(.1,.9,w,dtype=np.float32),(h,1))
        return DepthFrame(a,np.ones((h,w),bool),np.full((h,w),.4,np.float32),{'estimator':'explicit mock','scale_id':'unverified-provider-claim'})

def photo():
    yy,xx=np.mgrid[:60,:100]
    return np.stack(((xx*9+yy)%256,(yy*7+xx)%256,(xx+yy*3)%256),axis=-1).astype(np.uint8)

def cfg(**kwargs):return Config(administrative_timeout_ms=900,central_fraction=.2,**kwargs)

@pytest.mark.parametrize('gaze',[(0.,0.),(99.,0.),(0.,59.),(99.,59.),(.5,30.25)])
def test_edge_input_gate_and_reliability(gaze):
    o=observe(photo(),gaze,cfg());assert o.rgb.shape==photo().shape
    assert np.array_equal(o.rgb[o.central],photo()[o.central])
    assert np.isfinite(o.reliability).all() and o.reliability.min()>=.05
    assert o.reliability.max()<=1
    gx,gy=gaze;d=Detection([max(0,gx-5),max(0,gy-5),min(100,gx+5),min(60,gy+5)],{'patch':1},[.1]*3,.9,.5,source='semantic')
    assert central_detections([d],o)

def test_continuous_gate_does_not_use_rounded_pixel_for_semantics():
    o=observe(photo(),(50.,30.),replace(cfg(),central_fraction=.1))
    d=Detection([59.9,29.9,60.5,30.1],{'patch':1},[.1]*3,.9,.5,source='semantic')
    assert central_detections([d],o)==[]

def test_detector_bbox_is_clipped_and_marked_partial():
    o=observe(photo(),(50.,30.),replace(cfg(),central_fraction=.1))
    d=Detection([0,0,100,60],{'patch':1},[.1]*3,.9,.5,source='semantic')
    local=central_detections([d],o)[0]
    assert local.bbox==[40.,20.,60.,40.] and local.partial

def test_projection_uses_z_not_range_and_accepts_half_open_edges():
    c=Camera(100,60,fx=100.,fy=80.,cx=50.,cy=30.)
    assert np.allclose(c.unproject_z(100,60,2),[1.,.75,2.])
    assert np.allclose(c.unproject_z(50,30,2),[0,0,2])
    with pytest.raises(ValueError):Camera(100,60,cx=float('nan'))

def test_independent_geometry_hypotheses_survive_inconsistent_depth_and_miss():
    image=photo();c=cfg();o=observe(image,(50,30),c);w=World(100,60,c);camera=Camera(100,60)
    boxes=Fixed().detect(o)
    for step,value in enumerate([.15,.85]):
        w.update(boxes,o,250,summarize(o,32))
        frame=DepthFrame(np.full((60,100),value),np.ones((60,100),bool),np.full((60,100),.4),{'estimator':'mock','scale_id':'same unverified claim'})
        record=record_depth(frame,hashlib.sha256(o.rgb.tobytes()).hexdigest(),step)
        update_observed_depth(w,o,frame,camera,record)
    assert len(w.objects)==1
    r=w.objects[0].reconstruction;history=r.observation_history
    assert len(history)==2 and history[0]['frame_id']!=history[1]['frame_id']
    assert history[0]['center'][2]!=history[1]['center'][2]
    assert all(h['scale_id'] is None and h['orientation'] is None and h['dimensions'][2] is None for h in history)
    assert np.isclose(history[0]['dimensions'][0],6*history[0]['center'][2]/camera.fx)
    old=json.dumps(history)
    w.update([],o,250,summarize(o,32))
    assert w.objects[0].id=='object-1' and json.dumps(w.objects[0].reconstruction.observation_history)==old

def test_localized_ensemble_proxies_do_not_enumerate_objects():
    o=observe(photo(),(50,30),cfg());e=summarize(o,32)
    assert e.mean_reliability<=1 and e.local_regions
    for region in e.local_regions:
        a,b,c,d=region['region'];assert 0<=a<c<=100 and 0<=b<d<=60
        assert region['estimated_object_count'] is None and region['member_object_ids'] is None
        assert 0<=region['mean_reliability']<=1

def test_predicted_gain_diagnostic_is_not_summed_twice_and_thickness_cannot_force_revisit():
    c=cfg();o=observe(photo(),(50,30),c);world=World(100,60,c);world.update(Fixed().detect(o),o,250,summarize(o,32))
    world.objects[0].reconstruction=ReconstructionEstimate(depth_uncertainty=0,dimension_uncertainty=[0,0,1])
    p=PriorityField(100,60,np.ones((60,100),bool));field=p.score(o,world,summarize(o,32))
    assert 'estimated_reconstruction_gain' not in field['components']
    assert np.allclose(field['total'],sum(field['components'].values()))
    assert not np.any(field['components']['reconstruction_information_need'])
    assert 'grouping' in field['components']

def test_recording_npz_is_lossless_and_hash_verified(tmp_path):
    result=run(photo(),cfg(),object_provider=Fixed(),depth_provider=Depth())
    path=tmp_path/'run.json';saved=save_run(result,path)
    assert saved['schema_version']==2
    for raw,persisted in zip(result['fixations'],saved['fixations']):
        assert 'arrays' not in persisted['depth']
        a=depth_arrays(raw['depth']);b=depth_arrays(persisted['depth'],tmp_path)
        for name in a:assert a[name].dtype==b[name].dtype and np.array_equal(a[name],b[name])
        assert persisted['depth']['coordinate_contract']['shared_scale_id'] is None
        assert persisted['depth']['input_sha256']==persisted['sensory_sha256']
    bad=saved['fixations'][0]['depth'];bad['array_hashes']['quality']='bad'
    with pytest.raises(ValueError,match='hash'):depth_arrays(bad,tmp_path)

def test_cache_keys_are_only_actual_gaze_input_and_provider_identity(tmp_path):
    provider=Depth();cache=CachedDepth(provider,tmp_path,'mock-provider')
    a=observe(photo(),(10,10),cfg()).rgb;b=observe(photo(),(90,50),cfg()).rgb
    cache.predict(a);cache.predict(a);cache.predict(b)
    assert len(provider.calls)==2 and np.array_equal(provider.calls[0],a) and np.array_equal(provider.calls[1],b)

def test_replay_zero_mid_fixation_saccade_terminal_and_no_future_depth():
    result=run(photo(),cfg(),object_provider=Fixed(),depth_provider=Depth());f=result['fixations'][0]
    initial=inspect_at(result,0);assert initial['memory']['objects']==[] and initial['depth'] is None and initial['committed_index']==-1
    partial=inspect_at(result,f['offset_ms']/2);assert partial['next_choice'] is None and partial['stopping'] is None
    t=f['offset_ms']+f['saccade_duration_ms']/2;view=inspect_at(result,t)
    assert view['phase']=='saccade' and view['sensory_available'] is False
    assert np.allclose(view['gaze'],(np.array(f['gaze'])+f['saccade_to'])/2)
    assert view['memory']['time_ms']==t
    second=result['fixations'][1];v=inspect_at(result,second['onset_ms']+1)
    assert v['committed_index']==0 and v['depth']['frame_id']==f['depth']['frame_id']
    assert all(h['time_ms']<=v['elapsed_ms'] for m in v['memory']['objects'] for h in m['reconstruction']['observation_history'])
    assert inspect_at(result,1e6)['termination']==result['termination']

def test_saccade_timeout_does_not_teleport_to_destination():
    result=run(photo(),Config(administrative_timeout_ms=255),object_provider=Empty());f=result['fixations'][0]
    assert f['saccade_duration_ms']<f['saccade_planned_ms']
    v=inspect_at(result,255)
    assert not np.allclose(v['gaze'],f['saccade_to']) and v['termination']['administrative_timeout']

@pytest.mark.parametrize('kind',['valid','quality','inverse_depth'])
def test_invalid_depth_arrays_rejected(kind):
    f=Depth().predict(photo())
    if kind=='valid':f.valid=np.ones((2,2),bool)
    elif kind=='quality':f.quality[0,0]=np.nan
    else:f.inverse_depth[0,0]=2
    with pytest.raises(ValueError):f.validate(photo().shape[:2])


def test_npz_provenance_mismatch_and_missing_asset_are_explicit(tmp_path):
    result=run(photo(),cfg(),object_provider=Fixed(),depth_provider=Depth())
    saved=save_run(result,tmp_path/'run.json');record=saved['fixations'][0]['depth']
    record['metadata']['estimator']='tampered'
    with pytest.raises(ValueError,match='provenance'):depth_arrays(record,tmp_path)
    record['metadata']['estimator']='explicit mock'
    (tmp_path/record['asset']).unlink()
    with pytest.raises(ValueError,match='Missing'):depth_arrays(record,tmp_path)


def test_export_rehydrates_npz_without_calling_any_model(tmp_path):
    from active_observer.viz.replay import write_replay
    result=run(photo(),cfg(),object_provider=Fixed(),depth_provider=Depth())
    saved=save_run(result,tmp_path/'run.json')
    write_replay(photo(),saved,tmp_path/'inspector.html',asset_root=tmp_path)
    html=(tmp_path/'inspector.html').read_text()
    assert 'THREE' in html and 'OBSERVER_PAYLOAD' in html
    assert '<script src=' not in html
    bad=json.loads(json.dumps(saved));bad['fixations'][0]['depth']['input_sha256']='wrong'
    with pytest.raises(ValueError,match='provenance'):write_replay(photo(),bad,tmp_path/'bad.html',asset_root=tmp_path)
