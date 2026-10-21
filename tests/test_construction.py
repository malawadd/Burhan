import numpy as np
import pytest
from edge_runtime.contracts import FrameContext
from edge_runtime.profiles.construction_model import decode, prepare
import edge_runtime.profiles.construction as construction


class FakeModel:
    def __init__(self, options):
        self.mode="ppe";self.spec={"capabilities":["ppe_hardhat","ppe_vest"]};self.rows=[]
    def infer(self, frame):self.seen_shape=frame.shape;return self.rows
    def describe(self):return {}


def test_preprocess_obeys_each_pinned_contract():
    f=np.zeros((100,200,3),np.uint8);f[:]=[10,20,30]
    x,t=prepare(f,"ppe");assert x.shape==(1,3,640,640);assert t==(3.2,6.4,0,0)
    np.testing.assert_allclose(x[0,:,0,0],np.array([30,20,10])/255,rtol=1e-6)
    x,t=prepare(f,"crack");assert t==(3.2,3.2,0,160);assert np.allclose(x[0,:,0,0],114/255)
    x,t=prepare(f,"ppe","letterbox");assert t==(3.2,3.2,0,160);assert np.allclose(x[0,:,0,0],114/255)


def test_explicit_no_ppe_channel_coordinates_and_nms():
    x=np.zeros((1,14,8400),np.float32);x[0,:4,0]=[320,320,320,320];x[0,6,0]=.8
    x[0,:4,1]=x[0,:4,0];x[0,6,1]=.7
    rows=decode([x],"ppe",(3.2,6.4,0,0),(100,200));assert len(rows)==1
    assert rows[0]["label"]=="NO-Hardhat";assert rows[0]["bbox"]==(50,25,100,50)
    with pytest.raises(ValueError):decode([np.zeros((1,6,8400))],"ppe",(1,1,0,0),(640,640))


def test_crack_mask_uses_coefficients_and_crops_box():
    x=np.zeros((1,37,8400),np.float32);x[0,:5,0]=[320,320,320,320,.8];x[0,5,0]=1
    proto=np.ones((1,32,160,160),np.float32)
    rows=decode([x,proto],"crack",(1,1,0,0),(640,640))
    assert rows[0]["mask"].sum()==320*320


def test_boxes_skip_dense_masks_with_exact_candidate_parity():
    x=np.zeros((1,37,8400),np.float32)
    for i,(cx,score) in enumerate([(100,.8),(400,.7),(105,.6)]):
        x[0,:5,i]=[cx,300,100,100,score];x[0,5,i]=1
    proto=np.ones((1,32,160,160),np.float32)
    masks=decode([x,proto],"crack",(1,1,0,0),(640,640),mask_rendering="masks")
    boxes=decode([x,proto],"crack",(1,1,0,0),(640,640),mask_rendering="boxes")
    assert len(boxes)==len(masks)==2
    assert boxes==[{k:v for k,v in item.items() if k!="mask"} for item in masks]
    assert all("mask" not in item for item in boxes)
    with pytest.raises(ValueError,match="mask_rendering"):
        decode([x,proto],"crack",(1,1,0,0),(640,640),mask_rendering="fake")


def test_person_never_means_missing_ppe_or_code_pass(monkeypatch):
    monkeypatch.setattr(construction,"ConstructionModel",FakeModel)
    profile=construction.ConstructionProfile({"stage":"foundations"});profile.model.rows=[{"label":"Person","bbox":(10,10,20,30),"score":.9,"class_id":5}]
    result=profile.process(np.zeros((80,100,3),np.uint8),FrameContext(1,1,"camera:0","a"))
    assert result.status=="unassessed";assert result.domain_state["code_compliance"]=="not_determined"
    assert result.domain_state["selected_stage"]=="foundations";assert result.metrics["review_candidates"]==0
    with pytest.raises(ValueError,match="simulation"):profile.process(np.zeros((80,100,3),np.uint8),FrameContext(2,2,"demo","a"))


def test_review_events_confirm_and_reset_across_sessions(monkeypatch):
    monkeypatch.setattr(construction,"ConstructionModel",FakeModel)
    profile=construction.ConstructionProfile({});profile.model.rows=[{"label":"NO-Hardhat","bbox":(10,10,20,30),"score":.9,"class_id":2}]
    f=np.zeros((80,100,3),np.uint8)
    assert not profile.process(f,FrameContext(1,1,"camera:0","a")).emit_event
    assert profile.process(f,FrameContext(2,1.1,"camera:0","a")).emit_event
    assert not profile.process(f,FrameContext(3,1.2,"camera:0","a")).emit_event
    assert not profile.process(f,FrameContext(1,2,"camera:0","b")).emit_event


def test_operator_roi_maps_boxes_and_masks_without_mutating_model_output(monkeypatch):
    monkeypatch.setattr(construction,"ConstructionModel",FakeModel)
    profile=construction.ConstructionProfile({"roi":[.1,.25,.6,.75]})
    mask=np.zeros((40,50),bool);mask[10:40,10:30]=True
    profile.model.rows=[{"label":"crack","bbox":(10,10,20,30),"score":.9,"class_id":0,"mask":mask}]
    result=profile.process(np.zeros((80,100,3),np.uint8),FrameContext(1,1,"camera:0","a"))
    assert profile.model.seen_shape==(40,50,3)
    assert result.findings[0].bbox==(20,30,20,30)
    assert result.domain_state["analysis_roi_xywh"]==[10,20,50,40]
    assert result.domain_state["code_compliance"]=="not_determined"
    assert profile.model.rows[0]["bbox"]==(10,10,20,30)
    assert result.annotated_frame[45,25].any()
    with pytest.raises(ValueError,match="positive area"):
        construction.ConstructionProfile({"roi":[.6,.2,.1,.8]})
    tiny=construction.ConstructionProfile({"roi":[0,0,.1,.1]})
    with pytest.raises(ValueError,match="32 input pixels"):
        tiny.process(np.zeros((80,100,3),np.uint8),FrameContext(1,1,"camera:0","a"))


def test_context_hidden_and_issue_has_actionable_explanation(monkeypatch):
    monkeypatch.setattr(construction,"ConstructionModel",FakeModel)
    profile=construction.ConstructionProfile({})
    profile.model.rows=[{"label":"Person","bbox":(10,10,20,30),"score":.9,"class_id":5}]
    frame=np.zeros((80,100,3),np.uint8)
    result=profile.process(frame,FrameContext(1,1,"camera:0","a"))
    assert result.findings==[]
    assert len(result.domain_state["observations"])==1
    assert result.domain_state["review_findings"]==[]
    profile.model.rows=[{"label":"crack","bbox":(10,10,20,30),"score":.9,"class_id":0}]
    result=profile.process(frame,FrameContext(2,2,"camera:0","a"))
    detail=result.domain_state["review_findings"][0]
    assert detail["label"]=="Possible surface crack"
    assert "joint" in detail["next_check"]
    assert "not verified ground truth" in detail["evidence_basis"]
    assert result.findings[0].label=="Possible surface crack"
    with pytest.raises(ValueError,match="boolean"):
        construction.ConstructionProfile({"show_context_objects":"false"})


def test_tile_anomaly_is_actionable_without_claiming_a_crack_or_probability(monkeypatch):
    monkeypatch.setattr(construction, "ConstructionModel", FakeModel)
    profile = construction.ConstructionProfile({"stage": "finishes"})
    profile.model.spec = {"capabilities": ["tile_surface_anomaly"], "diagnostic_only": True,
                          "diagnostic_label": "RESEARCH - WEAK TILE LOCALIZATION"}
    profile.model.rows = [{"label": "tile_surface_anomaly", "bbox": (10, 10, 20, 30),
                           "score": .6, "class_id": 0, "anomaly_distance": 8.5}]
    result = profile.process(np.zeros((80, 100, 3), np.uint8), FrameContext(1, 1, "camera:0", "a"))
    detail = result.domain_state["review_findings"][0]
    assert detail["label"] == "Possible tile surface texture anomaly"
    assert "unclassified" in detail["why_it_matters"]
    assert "lippage" in detail["next_check"]
    assert "probability" in detail["evidence_basis"]
    assert result.domain_state["code_compliance"] == "not_determined"


def test_joint_aware_decoder_keeps_joint_distinct_from_crack():
    x=np.zeros((1,38,8400),np.float32)
    x[0,:6,0]=[100,100,40,40,.8,.1]
    x[0,:6,1]=[400,400,40,40,.1,.9]
    proto=np.ones((1,32,160,160),np.float32)
    rows=decode([x,proto],"crack",(1,1,0,0),(640,640),mask_rendering="boxes",classes=["joint","crack"])
    assert [r["label"] for r in rows]==["crack","joint"]
    assert rows[0]["bbox"]==(380,380,40,40)
    assert all("mask" not in row for row in rows)


def test_explicit_joint_candidate_runtime_uses_pinned_classes_and_hides_joint(monkeypatch,tmp_path):
    import sys,types
    import edge_runtime.profiles.construction_model as models
    import hashlib
    candidate=tmp_path/"models/construction/crack-joints-yolov8n-seg.onnx"
    candidate.parent.mkdir(parents=True);candidate.write_bytes(b"fixture-onnx")
    monkeypatch.setattr(models,"ROOT",tmp_path)
    monkeypatch.setitem(models.CRACK_VARIANTS,"joint_aware_candidate",{**models.CRACK_VARIANTS["joint_aware_candidate"],"sha256":hashlib.sha256(candidate.read_bytes()).hexdigest()})
    class Options:
        def add_session_config_entry(self,*args):pass
    class Session:
        def __init__(self,*args,**kwargs):pass
        def get_inputs(self):return [types.SimpleNamespace(shape=[1,3,640,640],name="images")]
        def run(self,*args):
            x=np.zeros((1,38,8400),np.float32)
            x[0,:6,0]=[100,100,40,40,.8,.1]
            x[0,:6,1]=[400,400,40,40,.1,.9]
            return [x,np.ones((1,32,160,160),np.float32)]
    monkeypatch.setitem(sys.modules,"onnxruntime",types.SimpleNamespace(SessionOptions=Options,InferenceSession=Session,__version__="fixture"))
    model=models.ConstructionModel({"mode":"crack","crack_variant":"joint_aware_candidate","mask_rendering":"boxes"})
    assert model.describe()["crack_variant"]=="joint_aware_candidate"
    assert model.describe()["classes"]==["joint","crack"]
    monkeypatch.setattr(construction,"ConstructionModel",lambda options:model)
    profile=construction.ConstructionProfile({"mode":"crack"})
    result=profile.process(np.zeros((640,640,3),np.uint8),FrameContext(1,1,"camera:0","a"))
    assert len(result.domain_state["observations"])==2
    assert len(result.findings)==1
    assert result.findings[0].label=="Possible surface crack"
    with pytest.raises(ValueError,match="applies only"):
        models.ConstructionModel({"mode":"ppe","crack_variant":"joint_aware_candidate"})
