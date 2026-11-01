import numpy as np
from active_observer.evaluation.diagnostics import amplitudes,lattice,multiscale,selection_diagnostics,aligned_model_fixations

def candidate(x,u,p):return dict(x=x,y=0,utility=u,probability=p,kind='region',confidence=0,terms={'task':0,'context':0})

def test_first_transition_convention():
    xy=[[0,0],[3,4],[3,6]];assert amplitudes(xy).tolist()==[5,2];assert amplitudes(xy,True).tolist()==[2]

def test_coverage_and_selection_are_distinct_and_oracle_is_labelled():
    c=[candidate(0,2,.8),candidate(10,0,.2)];r=selection_diagnostics(c,[10,0]);assert r['nearest_error_px']==0;assert r['nearest_rank_min']==2;assert r['expected_error_px']==8;assert r['selection_excess_px']==8
    assert abs(r['oracle_added_probability']-1/6)<1e-12;assert abs(r['oracle_added_expected_error_px']-8/1.2)<1e-12
    assert c[1]['x']==10 and len(c)==2

def test_proposals_have_no_future_input_and_obey_bounds():
    assert len(lattice(640,400))==40
    a=multiscale(640,400,[2,3]);assert len(a)<=40;assert np.all(a>=0);assert np.all(a[:,0]<640);assert np.all(a[:,1]<400)

def test_merged_stay_carries_final_censoring():
    a=[dict(x_px=0,y_px=0,duration_ms=100,fixation_onset_ms=0,mode='initial',right_censored=False),dict(x_px=0,y_px=0,duration_ms=50,fixation_onset_ms=100,mode='stay',right_censored=True)]
    r=aligned_model_fixations(a,150);assert len(r)==1 and r[0]['duration_ms']==150 and r[0]['right_censored'];assert a[0]['duration_ms']==100

def test_published_display_angles_and_scale_invariance():
    from active_observer.evaluation.diagnostics import angular_distance,angular_amplitudes
    assert np.isclose(angular_distance([[0,200]],[[640,200]])[0],54)
    assert np.isclose(angular_distance([[320,0]],[[320,400]])[0],35)
    assert angular_distance([[20,30]],[[20,30]])[0]==0
    xy=np.array([[50,70],[350,300],[600,100]])
    assert np.allclose(angular_amplitudes(xy),angular_amplitudes(xy*1680/640,1680,1050))

def test_window_truncation_censors_and_excludes_future_events():
    a=[dict(x_px=0,y_px=0,duration_ms=100,fixation_onset_ms=0,mode='initial'),dict(x_px=20,y_px=20,duration_ms=100,fixation_onset_ms=130,mode='shift'),dict(x_px=30,y_px=30,duration_ms=100,fixation_onset_ms=260,mode='shift')]
    r=aligned_model_fixations(a,180);assert len(r)==2;assert r[-1]['duration_ms']==50;assert r[-1]['right_censored'];assert a[1]['duration_ms']==100

def test_padding_is_distinguished_from_unknown_scene_content():
    from active_observer.evaluation.diagnostics import content_box,inside
    box=content_box({'source_size':[640,480],'padding_xy':[64,0]})
    assert np.allclose(box,[640/12,0,640*11/12,400]);assert inside([[40,200],[320,200]],[box]).tolist()==[False,True]
