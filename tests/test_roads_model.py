import hashlib
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from edge_runtime.profiles.roads import RoadsProfile
from edge_runtime.profiles.roads_model import decode
from edge_runtime.contracts import FrameContext


class RoadModelTests(unittest.TestCase):
    def test_letterbox_decode_clipping_classwise_nms(self):
        output = np.zeros((1, 8, 8400), np.float32)
        output[0, :4, 0] = (320, 320, 200, 100)
        output[0, 7, 0] = .8
        output[0, :4, 1] = (320, 320, 200, 100)
        output[0, 7, 1] = .7
        output[0, :4, 2] = (320, 320, 200, 100)
        output[0, 4, 2] = .6
        found = decode(output, 1, (0, 140), (360, 640))
        self.assertEqual(len(found), 2)
        self.assertEqual(found[0]["bbox"], (220, 130, 200, 100))
        self.assertEqual({d["class_id"] for d in found}, {0, 3})

    def test_invalid_export_contract_fails(self):
        with self.assertRaises(ValueError):
            decode(np.zeros((1, 84, 8400)), 1, (0, 0), (640, 640))

    def test_smaller_fixed_export_decode_maps_to_original_coordinates(self):
        output = np.zeros((1, 9, 2100), np.float32)
        output[0, :4, 0] = (160, 160, 80, 40)
        output[0, 8, 0] = .9
        found = decode(output, .5, (0, 70), (360, 640), num_classes=5, input_size=320)
        self.assertEqual(found[0]["bbox"], (240, 140, 160, 80))
        self.assertEqual(found[0]["class_id"], 4)
        with self.assertRaisesRegex(ValueError, "Unsupported road model output"):
            decode(output, 1, (0, 0), (640, 640), num_classes=5, input_size=640)

    def test_performance_options_reject_unknown_or_non_integer_contract(self):
        for options in ({"input_size": 300}, {"input_size": 320.0}, {"cpu_threads": 0},
                        {"cpu_threads": True}, {"cpu_spinning": "false"},
                        {"model_variant": "india", "input_size": 320}):
            with self.assertRaises(ValueError):
                RoadsProfile({"demo": False, **options})

    def test_fiveclass_repair_is_separate_from_pothole(self):
        output = np.zeros((1, 9, 8400), np.float32)
        output[0, :4, 0] = (320, 320, 80, 60)
        output[0, 8, 0] = .9
        found = decode(output, 1, (0, 0), (640, 640), num_classes=5)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["class_id"], 4)

    def test_real_input_selects_model_not_demo_heuristic(self):
        with patch("edge_runtime.profiles.roads_model.RoadDamageModel") as model:
            profile = RoadsProfile({"demo": False})
            self.assertEqual(profile.backend, "model")
            model.assert_called_once()

    def test_missing_model_fails_explicitly(self):
        with self.assertRaisesRegex(ValueError, "model missing"):
            RoadsProfile({"demo": False, "model_path": "no-such-model.onnx"})

    def test_unavailable_openvino_gpu_does_not_silently_fall_back(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.onnx"
            path.write_bytes(b"mock-export")
            core = MagicMock()
            core.available_devices = ["CPU"]
            fake = SimpleNamespace(Core=lambda: core, get_version=lambda: "test")
            with patch.dict("sys.modules", {"openvino": fake}):
                with self.assertRaisesRegex(ValueError, "GPU is unavailable"):
                    RoadsProfile({"demo": False, "model_path": str(path),
                                  "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                  "engine": "openvino", "openvino_device": "GPU"})
            core.compile_model.assert_not_called()

    def test_openvino_gpu_configuration_uses_supported_latency_path(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.onnx"
            path.write_bytes(b"mock-export")
            core = MagicMock()
            core.available_devices = ["CPU", "GPU"]
            core.read_model.return_value.input.return_value.shape = [1, 3, 416, 416]
            core.compile_model.return_value.get_property.return_value = ["GPU.0"]
            fake = SimpleNamespace(Core=lambda: core, get_version=lambda: "test")
            with patch.dict("sys.modules", {"openvino": fake}):
                profile = RoadsProfile({"demo": False, "model_path": str(path), "input_size": 416,
                                        "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                        "engine": "openvino", "openvino_device": "GPU"})
            configuration = core.compile_model.call_args.args[2]
            self.assertEqual(configuration, {"PERFORMANCE_HINT": "LATENCY", "INFERENCE_PRECISION_HINT": "f32"})
            self.assertEqual(profile.describe()["execution_device"], "['GPU.0']")

    def test_nonfinite_output_does_not_become_detection(self):
        output = np.zeros((1, 8, 8400), np.float32)
        output[0, 7, 0] = np.inf
        self.assertEqual(decode(output, 1, (0, 0), (640, 640)), [])

    def test_model_event_confirmation_dedup_and_source_reset(self):
        output = np.zeros((1, 8, 8400), np.float32)
        output[0, :4, 0] = (320, 320, 200, 100)
        output[0, 7, 0] = .8
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.onnx"
            path.write_bytes(b"fake-contract-export")
            with patch("cv2.dnn.readNetFromONNX") as loader:
                loader.return_value.forward.return_value = output
                profile = RoadsProfile({"demo": False, "model_path": str(path),
                                        "model_variant": "multicountry",
                                        "engine": "opencv",
                                        "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
                image = np.zeros((360, 640, 3), np.uint8)
                first = profile.process(image, FrameContext(0, 0, "road-a", "one"))
                second = profile.process(image, FrameContext(1, .1, "road-a", "one"))
                duplicate = profile.process(image, FrameContext(2, .2, "road-a", "one"))
                changed = profile.process(image, FrameContext(3, .3, "road-b", "one"))
        self.assertFalse(first.emit_event)
        self.assertTrue(second.emit_event)
        self.assertFalse(duplicate.emit_event)
        self.assertFalse(changed.emit_event)
        self.assertEqual(second.findings[0].bbox, (220, 130, 200, 100))

    def test_model_hash_mismatch_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wrong.onnx"
            path.write_bytes(b"wrong-export")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                RoadsProfile({"demo": False, "model_variant": "india", "model_path": str(path)})

    def test_original_voc_float_coordinates_and_one_based_origin(self):
        from scripts.evaluate_roads_real import truth
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "float.xml"
            path.write_text("<annotation><object><name>D40</name><bndbox><xmin>11.5</xmin><ymin>21.25</ymin><xmax>31.5</xmax><ymax>41.25</ymax></bndbox></object></annotation>")
            actual = truth(path)
        self.assertEqual(actual, [{"class_id": 3, "bbox": (10.5, 20.25, 21.0, 21.0)}])


if __name__ == "__main__":
    unittest.main()
