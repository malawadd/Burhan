"""Timing adapter tests use fake frames, not purported model/video evidence."""
import json
import threading

import cv2
import numpy as np
import pytest

from edge_runtime.construction_catalog import load_catalog
from edge_runtime.sources import LocalSource
from edge_runtime.contracts import InspectionResult
from edge_runtime.runtime import EdgeRuntime, RuntimeConfig


class VideoCapture:
    def __init__(self):
        self.position = 0
        self.released = False
    def isOpened(self): return True
    def getBackendName(self): return "test_video"
    def get(self, key):
        return {cv2.CAP_PROP_FPS: 10., cv2.CAP_PROP_FRAME_COUNT: 100.,
                cv2.CAP_PROP_POS_MSEC: max(0,self.position-1)*100.}.get(key,0.)
    def grab(self):
        if self.position >= 100: return False
        self.position += 1
        return True
    def read(self):
        if self.position >= 100: return False,None
        frame = np.full((8,8,3),self.position,np.uint8)
        self.position += 1
        return True,frame
    def release(self): self.released = True


@pytest.mark.parametrize("mode,expected,skipped",[("realtime",3,2),("every_frame",1,0)])
def test_video_clock_skips_only_in_explicit_realtime_mode(tmp_path,monkeypatch,mode,expected,skipped):
    path = tmp_path / "fixture.mp4"; path.touch()
    capture = VideoCapture()
    monkeypatch.setattr("edge_runtime.sources.cv2.VideoCapture",lambda *args:capture)
    now = [100.]
    monkeypatch.setattr("edge_runtime.sources.time.perf_counter",lambda:now[0])
    source = LocalSource("file:"+str(path),"construction",15)
    source.video_mode = mode
    try:
        source.open()
        assert source.read()[0][0,0,0] == 0
        now[0] += .31
        frame,timestamp = source.read()
        assert int(frame[0,0,0]) == expected
        assert timestamp == expected*100
        assert source.source_skipped == skipped
        assert source.video_properties["source_skipped"] == skipped
        assert source.video_properties["duration_ms"] == 10000.
        assert source.video_properties["source_frame_index"] == expected
    finally:
        source.close()
    assert capture.released


def test_catalog_never_reuses_an_approval_from_static_data(tmp_path):
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps({"stages":[{"id":"structure","assessment":"approved"}],"inspections":[
        {"id":"crack","stage":"structure","assessment":"approved",
         "default_assessment":"approved","code_compliance":"passed"}]}),encoding="utf-8")
    result = load_catalog(path)
    assert result["inspections"][0]["assessment"] == "not_assessed"
    assert result["compliance_determined"] is False
    assert result["stages"][0]["assessment"] == "not_assessed"
    assert result["inspections"][0]["default_assessment"] == "unassessed"
    assert result["inspections"][0]["code_compliance"] == "not_determined"
    result["inspections"][0]["assessment"] = "approved"
    assert load_catalog(path)["inspections"][0]["assessment"] == "not_assessed"


def test_real_video_adapter_bounds_4k_memory_and_records_coordinate_scale(tmp_path,monkeypatch):
    path = tmp_path / "large.mp4"; path.touch()
    capture = VideoCapture()
    monkeypatch.setattr(capture,"read",lambda:(True,np.zeros((2160,3840,3),np.uint8)))
    monkeypatch.setattr("edge_runtime.sources.cv2.VideoCapture",lambda *args:capture)
    source = LocalSource("file:"+str(path),"construction",15)
    try:
        source.open()
        frame,_ = source.read()
        assert frame.shape == (1080,1920,3)
        assert source.video_properties["original_dimensions"] == [3840,2160]
        assert source.video_properties["analysis_dimensions"] == [1920,1080]
        assert source.video_properties["scale_to_source"] == [2.,2.]
    finally:
        source.close()


def test_catalog_unknown_stage_fails_instead_of_hiding_row(tmp_path):
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps({"stages":[],"inspections":[{"id":"x","stage":"unknown"}]}),encoding="utf-8")
    with pytest.raises(ValueError,match="unknown stage"):
        load_catalog(path)


@pytest.mark.parametrize("reported", [float("nan"), -1., 0.])
def test_unusable_video_timestamps_fall_back_to_original_frame_index(tmp_path, monkeypatch, reported):
    path = tmp_path / "bad-timestamp.mp4"; path.touch()
    capture = VideoCapture()
    original_get = capture.get
    monkeypatch.setattr(capture, "get", lambda key: reported if key == cv2.CAP_PROP_POS_MSEC else original_get(key))
    monkeypatch.setattr("edge_runtime.sources.cv2.VideoCapture", lambda *args: capture)
    source = LocalSource("file:" + str(path), "construction", 15)
    source.video_mode = "every_frame"
    try:
        source.open()
        source.read()
        _, timestamp = source.read()
        assert timestamp == 100.
        assert source.video_properties["source_frame_index"] == 1
        assert "fallback" in source.video_properties["timestamp_semantics"]
    finally:
        source.close()


def test_replay_lag_preserves_nonzero_container_origin_on_timestamp_fallback(tmp_path, monkeypatch):
    path = tmp_path / "offset-timestamps.mp4"; path.touch()
    capture = VideoCapture()
    original_get = capture.get
    monkeypatch.setattr(capture, "get", lambda key: 1000. if key == cv2.CAP_PROP_POS_MSEC else original_get(key))
    monkeypatch.setattr("edge_runtime.sources.cv2.VideoCapture", lambda *args: capture)
    now = [100.]
    monkeypatch.setattr("edge_runtime.sources.time.perf_counter", lambda: now[0])
    source = LocalSource("file:" + str(path), "construction", 10)
    try:
        source.open()
        assert source.read()[1] == 1000.
        now[0] += .35
        _, timestamp = source.read()
        assert timestamp == 1300.
        assert source.video_properties["source_frame_index"] == 3
        assert source.video_properties["source_skipped"] == 2
        assert source.video_properties["replay_lag_ms"] == pytest.approx(50.)
        assert "fallback" in source.video_properties["timestamp_semantics"]
    finally:
        source.close()


def test_capture_schedule_starts_after_open_and_factory_reports_source_gaps(monkeypatch):
    now = [100.]
    packets, waits = [], []

    class Stop:
        def is_set(self): return False
        def wait(self, delay):
            waits.append(delay)
            now[0] += delay

    class Source:
        backend = "timing_test"
        capture_properties = {}
        def __init__(self, *args):
            self.source_skipped = 0
            self.video_properties = {"source_fps": 10.}
            self.calls = 0
        def open(self): now[0] += 5.
        def read(self):
            self.calls += 1
            if self.calls == 3: return None, None
            if self.calls == 2: self.source_skipped = 2
            self.video_properties = {**self.video_properties, "source_frame_index": 0 if self.calls == 1 else 3,
                                     "timestamp_semantics": "test source timestamp"}
            return np.zeros((8,8,3), np.uint8), 0. if self.calls == 1 else 300.
        def close(self): pass

    class Buffer:
        def put(self, packet): packets.append(packet); return []

    edge = EdgeRuntime.__new__(EdgeRuntime)
    edge.config = RuntimeConfig(profile="factory", source="file:fixture.mp4", input_fps=10)
    edge._lock = threading.RLock(); edge._stop = Stop(); edge._capture_done = threading.Event()
    edge._counts = {"captured": 0, "source_skipped": 0, "coverage_gaps": 0}
    edge.description = {}; edge.session_id = "test-session"; edge.buffer = Buffer()
    edge._event = lambda *args: None
    monkeypatch.setattr("edge_runtime.runtime.LocalSource", Source)
    monkeypatch.setattr("edge_runtime.runtime.time.perf_counter", lambda: now[0])
    edge._capture()
    assert waits[0] == pytest.approx(.1)
    assert edge._counts["source_skipped"] == 2
    assert edge._counts["coverage_gaps"] == 2
    assert [p.context.source_frame_index for p in packets] == [0,3]
    # A later capture metadata update must not relabel an earlier result.
    edge.video = {"source_frame_index": 99}
    payload = edge._serialize_result(InspectionResult("unassessed", "test"), packets[0].context,
                                     {"mode":"ppe", "preprocessing":"letterbox", "supported_capabilities":["ppe_hardhat"]}, 1., 1., False)
    assert payload["source_frame_index"] == 0
    assert payload["source_timestamp_semantics"] == "test source timestamp"
    assert payload["model"]["ppe_preprocess"] == "letterbox"
    assert payload["model"]["preprocessing"] == "letterbox"
    assert payload["model"]["supported_capabilities"] == ["ppe_hardhat"]


@pytest.mark.parametrize("stage", ["invented-approval", [], None])
def test_invalid_construction_stage_rejected_before_model_load(monkeypatch, stage):
    edge = EdgeRuntime.__new__(EdgeRuntime)
    monkeypatch.setattr("edge_runtime.runtime.create_profile", lambda *args: pytest.fail("Model should not load"))
    with pytest.raises(ValueError, match="Unknown construction stage"):
        edge.start(RuntimeConfig(profile="construction", source="file:fixture.mp4", profile_options={"stage":stage}))
