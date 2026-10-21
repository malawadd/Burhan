"""Configuration and output contracts for local OCR tuning, without expensive inference."""
import sys
import types
import numpy as np
import pytest
from edge_runtime.profiles import freight_ocr


def test_ocr_presets_arena_and_stage_times(monkeypatch):
    configs=[]
    class Engine:
        def __init__(self, params): configs.append(params)
        def __call__(self, image):
            return types.SimpleNamespace(txts=("INVOICE",),scores=(.95,),
                boxes=np.array([[[0,0],[100,0],[100,20],[0,20]]]),elapse_list=[.1,.02,.3])
    import rapidocr
    monkeypatch.setattr(rapidocr,"RapidOCR",Engine)
    monkeypatch.setattr(freight_ocr,"model_inventory",lambda:{"available":True})
    monkeypatch.setattr(freight_ocr,"model_paths",lambda:["det","cls","rec"])
    page=np.full((100,100,3),255,np.uint8)
    baseline=freight_ocr.LocalOCR("baseline")
    before=baseline.read(page)
    tuned=freight_ocr.LocalOCR()
    import copy
    info_object = tuned.info
    info_before = copy.deepcopy(tuned.info)
    assert info_before["runtime_settings"]["intra_op_threads"]==4
    assert info_before["runtime_settings"]["disable_spinning"] is True
    after=tuned.read(page)
    assert tuned.info is info_object
    assert tuned.info == info_before
    assert after["provenance"]["runtime_settings"]==info_before["runtime_settings"]
    assert configs[0]["EngineConfig.onnxruntime.enable_cpu_mem_arena"] is False
    assert configs[1]["EngineConfig.onnxruntime.enable_cpu_mem_arena"] is True
    assert configs[1]["EngineConfig.onnxruntime.intra_op_num_threads"]==4
    assert configs[1]["Rec.rec_batch_num"]==1
    assert after["ocr_stage_ms"]=={"detection":100.,"orientation":20.,"recognition":300.}
    assert before["text"]==after["text"]
    assert after["session_load_ms"]>=0
    assert after["provenance"]["performance_preset"]=="optimized"
    parsed=freight_ocr.parse_document(after)
    assert parsed["ocr_stage_ms"]==after["ocr_stage_ms"]
    assert parsed["session_load_ms"]==after["session_load_ms"]


def test_invalid_ocr_preset_rejected():
    with pytest.raises(ValueError,match="preset"):
        freight_ocr.LocalOCR("unknown")


def test_detector_cap_preserves_original_page_for_ocr_and_bounds_only_tensor(monkeypatch):
    import rapidocr
    pages=[]
    class Engine:
        def __init__(self,params):
            assert not any(key.startswith("_") for key in params)
            self.text_det=types.SimpleNamespace()
        def __call__(self,page):
            pages.append(page.shape)
            tensor=self.text_det.get_preprocess(max(page.shape[:2]))(page)
            assert max(tensor.shape[2:])<=1024
            return types.SimpleNamespace(txts=("INVOICE",),scores=(.95,),
                boxes=np.array([[[0,0],[100,0],[100,20],[0,20]]]),elapse_list=[.1,.02,.3])
    monkeypatch.setattr(rapidocr,"RapidOCR",Engine)
    monkeypatch.setattr(freight_ocr,"model_inventory",lambda:{"available":True})
    monkeypatch.setattr(freight_ocr,"model_paths",lambda:["det","cls","rec"])
    page=np.full((1600,1200,3),255,np.uint8)
    result=freight_ocr.LocalOCR(overrides={"_detector_max_side":1024}).read(page)
    assert pages==[(1600,1200,3)]
    assert result["page_size"]==[1200,1600]


def test_explicit_sessions_really_disable_helper_spinning(monkeypatch):
    import rapidocr
    import onnxruntime as ort
    seen=[]
    def make_session(path,sess_options,providers):
        seen.append((path,sess_options,providers))
        return object()
    def initialize(engine,cfg):
        assert cfg.Det.session is not None
        assert cfg.Cls.session is not None
        assert cfg.Rec.session is not None
    monkeypatch.setattr(rapidocr.RapidOCR,"_initialize",initialize)
    monkeypatch.setattr(rapidocr.RapidOCR,"__call__",lambda engine,page:types.SimpleNamespace(
        txts=("INVOICE",),scores=(.95,),boxes=np.array([[[0,0],[100,0],[100,20],[0,20]]]),
        elapse_list=[.1,.02,.3]))
    monkeypatch.setattr(ort,"InferenceSession",make_session)
    monkeypatch.setattr(freight_ocr,"model_inventory",lambda:{"available":True})
    monkeypatch.setattr(freight_ocr,"model_paths",lambda:["det","cls","rec"])
    reader=freight_ocr.LocalOCR()
    reader.read(np.zeros((100,100,3),np.uint8))
    assert len(seen)==3
    for path,options,providers in seen:
        assert providers==["CPUExecutionProvider"]
        assert options.intra_op_num_threads==4
        assert options.inter_op_num_threads==1
        assert options.enable_cpu_mem_arena
        assert options.get_session_config_entry("session.intra_op.allow_spinning")=="0"
        assert options.get_session_config_entry("session.inter_op.allow_spinning")=="0"
    assert reader.info["runtime_settings"]["disable_spinning"] is True


def test_ocr_metadata_available_before_worker_and_detached_from_caller(monkeypatch):
    monkeypatch.setattr(freight_ocr,"model_inventory",lambda:{"available":False,"error":"test missing"})
    shape=[3,48,320]
    overrides={"Rec.rec_img_shape":shape,"EngineConfig.onnxruntime.intra_op_num_threads":2}
    reader=freight_ocr.LocalOCR(overrides=overrides)
    original=dict(reader.info)
    shape[2]=48
    overrides["EngineConfig.onnxruntime.intra_op_num_threads"]=8
    assert reader.info["runtime_settings"]["recognition_image_shape"]==[3,48,320]
    assert reader.info["runtime_settings"]["intra_op_threads"]==2
    assert reader.info==original
    assert reader.engine is None
    with pytest.raises(RuntimeError,match="test missing"):
        reader.read(np.zeros((10,10,3),np.uint8))
    assert reader.info==original
