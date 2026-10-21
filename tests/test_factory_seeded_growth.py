"""Contract fixtures only; real-mask calibration establishes accuracy."""
import numpy as np
import pytest
from edge_runtime.profiles.factory_encoder import seeded_growth_mask


def test_growth_adds_only_connected_neural_pixels_and_never_edits_scores():
    scores=np.zeros((20,20),np.float32)
    scores[4:8,4:8]=.8;scores[5:7,5:7]=.9995
    scores[12:18,12:18]=.8
    original=scores.copy()
    seed=seeded_growth_mask(scores,.999,4)
    grown=seeded_growth_mask(scores,.999,4,.5)
    assert seed.sum()==4 and grown.sum()==16
    assert not grown[12:18,12:18].any()
    np.testing.assert_array_equal(scores,original)


def test_no_retained_seed_means_no_flag_even_with_weak_foreground():
    scores=np.full((20,20),.9,np.float32);scores[0,0]=.9995
    assert not seeded_growth_mask(scores,.999,4,.5).any()
    with pytest.raises(ValueError):seeded_growth_mask(scores,.999,4,1.)


def test_independent_audit_matches_runtime_growth_geometry():
    from scripts.factory_localization_audit import retained_mask
    import cv2
    scores=np.zeros((20,20),np.float32);scores[3:14,5:11]=.7;scores[6:8,6:8]=.9995
    expected=cv2.resize(seeded_growth_mask(scores,.999,4,.5),(60,40),interpolation=cv2.INTER_NEAREST).astype(bool)
    np.testing.assert_array_equal(retained_mask(scores,.999,4,(40,60),.5),expected)


def test_warmup_initializes_predict_without_inspection_state():
    from edge_runtime.profiles.factory_encoder import FactorySegmentation
    model=FactorySegmentation.__new__(FactorySegmentation)
    model.initialization_report={'compile_and_request_ms':12.,'warmup_completed':False}
    calls=[]
    def predict(frame,threshold,area):
        calls.append((frame.shape,frame.dtype,float(threshold),area))
        assert not frame.any()
        return None
    model.predict=predict
    report=model.warmup(.999,16)
    assert [value[0] for value in calls]==[(1280,500,3),(1281,500,3),(1267,500,3)]
    assert all(value[1:]==(np.dtype('uint8'),.999,16) for value in calls)
    assert report['warmup_completed'] and report['warmup_iterations']==3
    assert len(report['warmup_predict_ms'])==3
    assert not hasattr(model,'inspection_count')


def test_no_seed_skips_low_labeling_and_empty_prediction_skips_final_statistics(monkeypatch):
    import cv2
    from edge_runtime.profiles.factory_encoder import FactorySegmentation
    model=FactorySegmentation.__new__(FactorySegmentation)
    model.input='input';model.output='output';model.output_logits=True
    model.growth_threshold=.95;model.execution_devices=['GPU.0']
    logits=np.full((1,1,640,256),-20.,np.float32)
    class Request:
        def infer(self,value):return {'output':logits}
    model.request=Request()
    original=cv2.connectedComponentsWithStats;calls=[]
    def high_only(*args,**kwargs):
        calls.append(True)
        return original(*args,**kwargs)
    def forbidden(*args,**kwargs):raise AssertionError('Unneeded empty-mask low/final processing')
    monkeypatch.setattr(cv2,'connectedComponentsWithStats',high_only)
    monkeypatch.setattr(cv2,'connectedComponents',forbidden)
    monkeypatch.setattr(np,'bincount',forbidden)
    scores,candidates,metrics=model.predict(np.zeros((1280,500,3),np.uint8),.999,16)
    assert calls==[True] and not candidates and metrics['candidate_regions']==0
    assert np.all(scores<.0001)
