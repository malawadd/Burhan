"""Decoder and manifest mechanics; fixtures are not trained defect evidence."""
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from edge_runtime.profiles import construction_task_detector as module


def install(tmp_path, monkeypatch, architecture="yolov8n", outputs=None):
    monkeypatch.setattr(module, "MODEL_ROOT", tmp_path)
    model = tmp_path / "model.onnx"
    model.write_bytes(b"test-only-fixture-not-a-neural-network")
    spec = {"architecture": architecture, "input_size": 320, "model_file": model.name,
            "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
            "classes": ["peeling_paint"], "confidence": .5,
            "inspection_task": "walls_finishes", "preprocessing": "letterbox_rgb_0_1"}
    path = tmp_path / "model.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    class Session:
        def __init__(self, path, options, providers):
            assert providers == ["CPUExecutionProvider"]
            assert options.intra_op_num_threads == options.inter_op_num_threads == 1
        def get_inputs(self):
            return [SimpleNamespace(name="images", type="tensor(float)", shape=[1, 3, 320, 320])]
        def get_outputs(self):
            names = ["boxes", "labels", "scores"] if architecture == "ssdlite320" else ["output"]
            return [SimpleNamespace(name=name) for name in names]
        def run(self, names, inputs):
            assert inputs["images"].shape == (1, 3, 320, 320)
            return outputs
    import onnxruntime
    monkeypatch.setattr(onnxruntime, "InferenceSession", Session)
    return path, spec


def test_yolo_maps_letterbox_back_to_native_image_and_suppresses_duplicate(tmp_path, monkeypatch):
    predictions = np.array([[[160, 161], [160, 160], [160, 160], [80, 80], [.9, .8]]], np.float32)
    path, _ = install(tmp_path, monkeypatch, outputs=[predictions])
    model = module.ConstructionTaskDetector(path)
    detected = model.infer(np.zeros((100, 200, 3), np.uint8))
    assert len(detected) == 1
    assert detected[0]["bbox"] == (50, 25, 100, 50)
    assert detected[0]["label"] == "peeling_paint"
    assert model.describe()["validation_status"] == "research_only"


def test_ssdlite_one_based_labels_and_native_projection(tmp_path, monkeypatch):
    outputs = [np.array([[80, 120, 240, 200]], np.float32), np.array([1], np.int64), np.array([.9], np.float32)]
    path, _ = install(tmp_path, monkeypatch, "ssdlite320", outputs)
    detected = module.ConstructionTaskDetector(path).infer(np.zeros((100, 200, 3), np.uint8))
    assert detected[0]["bbox"] == (50, 25, 100, 50)


def test_ssdlite_does_not_apply_a_second_different_nms(tmp_path, monkeypatch):
    outputs = [np.array([[80, 120, 240, 200], [90, 120, 250, 200]], np.float32),
               np.array([1, 1], np.int64), np.array([.9, .8], np.float32)]
    path, _ = install(tmp_path, monkeypatch, "ssdlite320", outputs)
    detected = module.ConstructionTaskDetector(path).infer(np.zeros((100, 200, 3), np.uint8))
    assert len(detected) == 2


def test_hash_change_and_path_escape_are_rejected(tmp_path, monkeypatch):
    path, spec = install(tmp_path, monkeypatch, outputs=[])
    (tmp_path / "model.onnx").write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA256"):
        module.ConstructionTaskDetector(path)
    spec["model_file"] = "../outside.onnx"
    path.write_text(json.dumps(spec), encoding="utf-8")
    with pytest.raises(ValueError, match="inside"):
        module.read_manifest(path)


def test_nonfinite_output_is_never_an_observation(tmp_path, monkeypatch):
    path, _ = install(tmp_path, monkeypatch, outputs=[np.full((1, 5, 1), np.nan, np.float32)])
    with pytest.raises(ValueError, match="non-finite"):
        module.ConstructionTaskDetector(path).infer(np.zeros((50, 50, 3), np.uint8))


def test_actual_mask_decoder_preserves_background_and_shape(tmp_path, monkeypatch):
    logits = np.zeros((1, 2, 320, 320), np.float32)
    logits[0, 1, 120:200, 80:160] = 8
    path, _ = install(tmp_path, monkeypatch, "lraspp", [logits])
    detected = module.ConstructionTaskDetector(path).infer(np.zeros((100, 200, 3), np.uint8))
    assert len(detected) == 1
    assert detected[0]["mask"].shape == (100, 200)
    assert detected[0]["mask"].dtype == bool
    assert detected[0]["bbox"] == (50, 25, 50, 50)


def test_yolo_prototype_coefficients_do_not_become_class_scores(tmp_path, monkeypatch):
    prediction = np.zeros((1, 37, 1), np.float32)
    prediction[0, :5, 0] = [160, 160, 160, 80, .9]
    prediction[0, 5, 0] = 10  # mask coefficient much larger than class probability
    proto = np.ones((1, 32, 80, 80), np.float32)
    path, _ = install(tmp_path, monkeypatch, "yolov8n_seg", [prediction, proto])
    detected = module.ConstructionTaskDetector(path).infer(np.zeros((100, 200, 3), np.uint8))
    assert len(detected) == 1
    assert detected[0]["bbox"] == (50, 25, 100, 50)
    assert detected[0]["mask"].sum() == 5000
    assert detected[0]["localization"] == "native_prototype_mask"


@pytest.mark.parametrize("value", [None, [], {}, 1])
def test_malformed_manifest_path_fails_as_configuration_error(value):
    with pytest.raises(ValueError):
        module.read_manifest(value)


@pytest.mark.parametrize("field,value", [("preprocessing", []), ("preprocessing", {}),
                                        ("class_confidence", None), ("class_confidence", []),
                                        ("class_confidence", [["peeling_paint", .5]])])
def test_malformed_manifest_option_types_are_configuration_errors(tmp_path, monkeypatch, field, value):
    path, spec = install(tmp_path, monkeypatch, outputs=[])
    spec[field] = value
    path.write_text(json.dumps(spec), encoding="utf8")
    with pytest.raises(ValueError):
        module.read_manifest(path)


def test_model_runtime_load_failure_is_a_configuration_error(tmp_path, monkeypatch):
    path, _ = install(tmp_path, monkeypatch, outputs=[])
    import onnxruntime
    def fail(*args, **kwargs):
        raise onnxruntime.capi.onnxruntime_pybind11_state.InvalidProtobuf("software fixture invalid model")
    monkeypatch.setattr(onnxruntime, "InferenceSession", fail)
    with pytest.raises(ValueError, match="self-contained"):
        module.ConstructionTaskDetector(path)


def test_session_loads_the_exact_verified_bytes(tmp_path, monkeypatch):
    path, _ = install(tmp_path, monkeypatch, outputs=[])
    import onnxruntime
    original = onnxruntime.InferenceSession
    observed = []
    def record(value, *args, **kwargs):
        observed.append(value)
        return original(value, *args, **kwargs)
    monkeypatch.setattr(onnxruntime, "InferenceSession", record)
    module.ConstructionTaskDetector(path)
    assert observed == [(tmp_path/"model.onnx").read_bytes()]


def test_graph_changed_after_manifest_read_is_rejected(tmp_path, monkeypatch):
    path, _ = install(tmp_path, monkeypatch, outputs=[])
    original = module.read_manifest
    def changed(value):
        result = original(value)
        (tmp_path/"model.onnx").write_bytes(b"changed between verification and session construction")
        return result
    monkeypatch.setattr(module, "read_manifest", changed)
    with pytest.raises(ValueError, match="SHA256"):
        module.ConstructionTaskDetector(path)


def test_external_weight_bytes_are_not_implicitly_loaded_without_hash_binding(tmp_path, monkeypatch):
    onnx = pytest.importorskip("onnx")
    import onnxruntime
    from onnx import helper, numpy_helper, TensorProto
    monkeypatch.setattr(module, "MODEL_ROOT", tmp_path)
    initializer = numpy_helper.from_array(np.zeros((1, 5, 1), np.float32), name="prediction")
    graph = helper.make_graph([helper.make_node("Identity", ["prediction"], ["output0"])], "software mechanics fixture",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 320, 320])],
        [helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 5, 1])], [initializer])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10)
    model_path = tmp_path/"external.onnx"
    onnx.save_model(model, str(model_path), save_as_external_data=True, all_tensors_to_one_file=True,
                    location="unhashed-weights.bin", size_threshold=0)
    spec = {"architecture": "yolov8n", "input_size": 320, "model_file": model_path.name,
            "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(), "classes": ["fixture"]}
    manifest = tmp_path/"manifest.json"
    manifest.write_text(json.dumps(spec))
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    # A path-based ORT load accepts the sidecar although its bytes are unhashed.
    assert onnxruntime.InferenceSession(str(model_path), options, providers=["CPUExecutionProvider"])
    with pytest.raises(ValueError, match="self-contained"):
        module.ConstructionTaskDetector(manifest)
