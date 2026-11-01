import copy,json
import cv2,numpy as np,pytest
from active_observer.io.human import load_manifest
from active_observer.evaluation.batch import predict,evaluate
from active_observer.evaluation.metrics import nss,auc_pixels,sequence_dtw
from active_observer.config import Config

def manifest(tmp_path):
    im=np.full((30,50,3),60,np.uint8);cv2.imwrite(str(tmp_path/'im.png'),im)
    d=dict(dataset_name='test fixture',synthetic=True,coordinate_units='pixels',initial_fixation_included=True,records=[dict(image_id='a',image='im.png',width=50,height=30,task='',participant='fixture',split='test',fixations=[dict(x=25,y=15,duration_ms=200),dict(x=10,y=10,duration_ms=300)])]);p=tmp_path/'manifest.json';p.write_text(json.dumps(d), encoding='utf-8');return p,d

def test_duplicate_pixels_cannot_cross_splits(tmp_path):
    p,d=manifest(tmp_path);other=copy.deepcopy(d['records'][0]);other.update(image_id='new-name',split='train');d['records'].append(other);p.write_text(json.dumps(d), encoding='utf-8')
    with pytest.raises(ValueError,match='leakage'):load_manifest(p)

def test_units_dimensions_and_initial_convention_checked(tmp_path):
    p,d=manifest(tmp_path);d['records'][0]['width']=51;p.write_text(json.dumps(d), encoding='utf-8')
    with pytest.raises(ValueError,match='dimensions'):load_manifest(p)
    d['records'][0]['width']=50;d['coordinate_units']='unknown';p.write_text(json.dumps(d), encoding='utf-8')
    with pytest.raises(ValueError):load_manifest(p)

def test_future_human_fixations_cannot_change_predictions(tmp_path):
    p,d=manifest(tmp_path);r=load_manifest(p)['records'][0];s=copy.deepcopy(r);s['fixations'][1]=dict(x=49,y=29,duration_ms=10000)
    assert predict(r,Config(max_fixations=3))==predict(s,Config(max_fixations=3))

def test_metrics_defined_and_undefined_cases():
    assert nss(np.ones((4,4)),[(1,1)]) is None
    assert auc_pixels(np.ones((4,4)),[(1,1)])==.5
    a=[(1,1),(2,2)];assert sequence_dtw(a,a,10,10)==0
    assert sequence_dtw([],a,10,10) is None

def test_evaluation_labels_synthetic_and_no_depth(tmp_path):
    p,d=manifest(tmp_path);r=evaluate(p,Config(max_fixations=3),tmp_path/'out',seeds=[3]);assert 'not empirical' in r['status'];assert not r['depth_tested'];assert len(r['variants'])==14;assert (tmp_path/'out/summary.csv').exists()
    assert r['summary']['full']['spatial_dtw']==r['summary']['without_depth']['spatial_dtw']

def test_standalone_viewer_escapes_embedded_labels(tmp_path):
    from active_observer.simulation.runner import run
    from active_observer.viz.html import write_html
    r=run(np.zeros((20,30,3),np.uint8),Config(max_fixations=1),target='</script><script>alert(1)</script>');p=tmp_path/'view.html';write_html(np.zeros((20,30,3),np.uint8),r,p);assert '"target": "</script>' not in p.read_text(encoding='utf-8')

def test_visualization_rejects_wrong_scene(tmp_path):
    from active_observer.simulation.runner import run
    from active_observer.viz.html import write_html
    image=np.zeros((20,30,3),np.uint8);r=run(image,Config(max_fixations=1));wrong=image.copy();wrong[0,0]=255
    with pytest.raises(ValueError,match='different RGB'):write_html(wrong,r,tmp_path/'wrong.html')

def test_mask_shape_and_relative_range_contracts():
    from active_observer.perception.depth import normalize
    with pytest.raises(ValueError):normalize(np.ones((3,3)),valid=np.ones((1,3),bool))
    a=normalize(np.array([[1.,2.],[3.,0.]]),kind='relative_range');assert not a.valid[1,1];assert a.inverse_depth[0,0]>a.inverse_depth[1,0]
