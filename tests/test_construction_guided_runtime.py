"""Queue/image pairing mechanics; generated test tensors are not model evidence."""
import time

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from edge_runtime.app import create_app
from edge_runtime.contracts import Finding, InspectionResult
from edge_runtime.runtime import EdgeRuntime, RuntimeConfig


def wait_for(predicate, timeout=4):
    until = time.perf_counter() + timeout
    while time.perf_counter() < until:
        if predicate():
            return
        time.sleep(.01)
    raise AssertionError("Runtime did not reach expected state")


class Source:
    backend = "unit_test"
    capture_properties = {}
    video_properties = {}
    source_skipped = 0

    def __init__(self, *args):
        self.i = 0

    def open(self):
        pass

    def read(self):
        self.i += 1
        return np.full((48, 64, 3), self.i % 200, np.uint8), self.i * 33.3

    def close(self):
        pass


class SlowProfile:
    def describe(self):
        return {"profile": "construction", "scope": "Unit test, not validation"}

    def process(self, frame, context):
        time.sleep(.16)
        annotated = frame.copy()
        annotated[0:10] = 255
        return InspectionResult("unassessed", "test", annotated_frame=annotated,
                                findings=[Finding("test", "test", task_id="walls_finishes",
                                                  evidence_timestamp_ms=context.source_time_ms)])


@pytest.mark.parametrize("performed,neural", [(False, False), (True, False), (True, True)])
def test_guidance_and_geometry_are_not_counted_as_neural_inspection(tmp_path, monkeypatch, performed, neural):
    class CounterProfile:
        def describe(self):
            return {"inspection_kind": "model" if neural else "geometry" if performed else "guidance"}

        def process(self, frame, context):
            return InspectionResult("unassessed", "counter mechanics only", annotated_frame=frame,
                                    metrics={"inspection_performed": performed, "model_inference_ran": neural})

    monkeypatch.setattr("edge_runtime.runtime.LocalSource", Source)
    monkeypatch.setattr("edge_runtime.runtime.create_profile", lambda *args: CounterProfile())
    edge = EdgeRuntime(tmp_path)
    try:
        edge.start(RuntimeConfig(profile="construction", source="file:mechanics-only", input_fps=30,
                                 preview_fps=30, deadline_ms=1000,
                                 profile_options={"inspection_task": "walls_finishes"}))
        wait_for(lambda: edge.snapshot()["metrics"]["processed"] >= 4)
        metrics = edge.snapshot()["metrics"]
        assert metrics["inspected_frames"] == metrics["processed"] * int(performed)
        assert metrics["model_inference_frames"] == metrics["processed"] * int(neural)
        assert metrics["guidance_frames"] == metrics["processed"] * int(not performed)
        assert (metrics["inspected_fps"] > 0) == performed
        assert (metrics["model_inference_fps"] > 0) == neural
        assert (metrics["inspection_latency_p95_ms"] is not None) == performed
        assert (metrics["model_inference_p95_ms"] is not None) == neural
    finally:
        edge.close()


def test_live_preview_does_not_borrow_inspection_boxes(tmp_path, monkeypatch):
    monkeypatch.setattr("edge_runtime.runtime.LocalSource", Source)
    monkeypatch.setattr("edge_runtime.runtime.create_profile", lambda *args: SlowProfile())
    edge = EdgeRuntime(tmp_path)
    try:
        edge.start(RuntimeConfig(profile="construction", source="file:mechanics-only", input_fps=30,
                                 preview_fps=30, deadline_ms=1000,
                                 profile_options={"inspection_task": "walls_finishes"}))
        wait_for(lambda: edge.snapshot()["metrics"]["processed"] >= 4)
        state = edge.snapshot()
        assert state["preview_kind"] == "live_unannotated"
        assert state["metrics"]["preview_fps"] > state["metrics"]["inspected_fps"] * 2
        assert state["metrics"]["dropped"] > 0
        assert state["metrics"]["queue_depth"] <= 1
        raw = cv2.imdecode(np.frombuffer(edge.preview(), np.uint8), cv2.IMREAD_COLOR)
        assert raw[3, 3, 0] < 200
        pair = state["inspection_result"]
        assert pair["source_time_ms"] > 0
        assert pair["findings"][0]["task_id"] == "walls_finishes"
        image = edge.inspection_preview(pair["sequence"], pair["session_id"])
        inspected = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
        assert inspected[3, 3, 0] > 240
        with pytest.raises(ValueError, match="frame changed"):
            edge.inspection_preview(-1, pair["session_id"])
        with pytest.raises(ValueError, match="session changed"):
            edge.inspection_preview(pair["sequence"], "another-session")
    finally:
        edge.close()
    assert not edge.media.status()["worker_alive"]
    assert not edge.inspection_media.status()["worker_alive"]


def test_inspection_endpoint_has_no_image_before_inspection(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        assert client.get("/api/inspection.jpg").status_code == 503


def test_inspection_pair_survives_state_to_image_request_race_but_cache_is_bounded(tmp_path):
    edge = EdgeRuntime(tmp_path)
    try:
        edge.session_id = "test-session"
        for sequence in range(40):
            edge._inspection_payloads[sequence] = {"sequence": sequence, "captured_monotonic": time.perf_counter()}
            edge._set_inspection(edge.session_id, bytes([sequence]), sequence)
        assert edge.inspection_preview(10, "test-session") == bytes([10])
        assert edge.inspection_preview(39, "test-session") == bytes([39])
        assert edge.snapshot()["inspection_cache"]["frames"] == 32
        with pytest.raises(ValueError, match="frame changed"):
            edge.inspection_preview(7, "test-session")
    finally:
        edge.close()


def test_guided_pair_is_cleared_when_session_changes(tmp_path, monkeypatch):
    monkeypatch.setattr("edge_runtime.runtime.LocalSource", Source)
    monkeypatch.setattr("edge_runtime.runtime.create_profile", lambda *args: SlowProfile())
    edge = EdgeRuntime(tmp_path)
    config = RuntimeConfig(profile="construction", source="file:mechanics-only", input_fps=30,
                           preview_fps=0, deadline_ms=1000,
                           profile_options={"inspection_task": "walls_finishes"})
    try:
        edge.start(config)
        wait_for(lambda: edge.snapshot()["inspection_result"] is not None)
        session = edge.session_id
        edge.stop()
        edge.start(config)
        assert edge.session_id != session
        assert edge.snapshot()["inspection_result"] is None
        assert edge.inspection_preview() is None
    finally:
        edge.close()
