import socket
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from edge_runtime.app import create_app
from edge_runtime.buffer import FrameBuffer
from edge_runtime.contracts import InspectionResult
from edge_runtime.runtime import EdgeRuntime, RuntimeConfig


def until(predicate, timeout=5):
    end = time.perf_counter() + timeout
    while time.perf_counter() < end:
        if predicate():
            return
        time.sleep(.02)
    raise AssertionError("Condition not reached before timeout")


def test_queue_overload_policies():
    fifo = FrameBuffer(2, "fifo")
    assert fifo.put(1) == []
    assert fifo.put(2) == []
    assert fifo.put(3) == [3]
    assert [fifo.get(), fifo.get()] == [1, 2]
    latest = FrameBuffer(4, "latest")
    latest.put(1)
    assert latest.put(2) == [1]
    assert latest.close() == [2]
    assert latest.put(3) == [3]


@pytest.mark.parametrize("profile", ["factory", "freight", "roads"])
def test_profiles_process_without_network(tmp_path, monkeypatch, profile):
    def deny_network(*args, **kwargs):
        raise AssertionError("Inspection attempted network access")
    monkeypatch.setattr(socket.socket, "connect", deny_network)
    edge = EdgeRuntime(tmp_path / profile)
    try:
        edge.start(RuntimeConfig(profile=profile, input_fps=8, deadline_ms=2000))
        until(lambda: edge.snapshot()["metrics"]["processed"] >= 3)
        state = edge.snapshot()
        assert state["metrics"]["inspection_errors"] == 0
        assert state["last_result"]["provenance"] == "synthetic_demo"
        assert state["cloud"]["enabled"] is False
        assert edge.preview().startswith(b"\xff\xd8")
    finally:
        edge.close()
    assert not list((tmp_path / profile / "evidence").glob("*.jpg"))


class StubSource:
    backend = "test"
    def __init__(self, *args):
        pass
    def open(self):
        pass
    def read(self):
        return np.zeros((64, 64, 3), np.uint8), None
    def close(self):
        pass


class ConfirmationProfile:
    def __init__(self, slow=False):
        self.hits = 0
        self.slow = slow
    def process(self, frame, context):
        if self.slow:
            time.sleep(.09)
        self.hits += 1
        return InspectionResult("pending" if self.hits < 3 else "fault", "Test observation")
    def describe(self):
        return {"test_stub": True}
    def reset(self):
        self.hits = 0


def test_factory_trigger_allows_confirmation_before_deadline(tmp_path, monkeypatch):
    monkeypatch.setattr("edge_runtime.runtime.LocalSource", StubSource)
    monkeypatch.setattr("edge_runtime.runtime.create_profile", lambda *args: ConfirmationProfile())
    edge = EdgeRuntime(tmp_path)
    try:
        edge.start(RuntimeConfig(triggered=True, input_fps=20, deadline_ms=1000))
        unit = edge.action("trigger")["unit_id"]
        until(lambda: unit in edge._completed_units)
        assert unit not in edge._missed_units
        assert edge.snapshot()["metrics"]["missed_units"] == 0
    finally:
        edge.close()


def test_factory_late_trigger_is_not_a_clear_inspection(tmp_path, monkeypatch):
    monkeypatch.setattr("edge_runtime.runtime.LocalSource", StubSource)
    monkeypatch.setattr("edge_runtime.runtime.create_profile", lambda *args: ConfirmationProfile(slow=True))
    edge = EdgeRuntime(tmp_path)
    try:
        edge.start(RuntimeConfig(triggered=True, input_fps=20, deadline_ms=25))
        unit = edge.action("trigger")["unit_id"]
        until(lambda: unit in edge._missed_units)
        assert unit not in edge._completed_units
        assert edge.snapshot()["metrics"]["missed_units"] == 1
    finally:
        edge.close()


def test_missing_file_reports_source_error(tmp_path):
    edge = EdgeRuntime(tmp_path)
    try:
        edge.start(RuntimeConfig(source="file:" + str(tmp_path / "absent.avi")))
        until(lambda: edge.snapshot()["state"] == "source_error")
        assert edge.snapshot()["metrics"]["processed"] == 0
        assert not edge.snapshot()["running"]
        source_error = edge.snapshot()["error"]
        edge.stop()
        assert edge.snapshot()["error"] == source_error
    finally:
        edge.close()


def test_local_api_control_and_export(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        assert client.post("/api/start", json={"source": "https://example.com/video"}).status_code == 422
        assert client.post("/api/start", json={}, headers={"origin": "https://external.example"}).status_code == 403
        assert client.post("/api/stop", content="{}").status_code == 415
        assert client.get("/api/preview.jpg").status_code == 503
        assert client.post("/api/start", json={"profile": "freight", "input_fps": 8}).status_code == 200
        until(lambda: client.get("/api/state").json()["metrics"]["processed"] > 1)
        assert client.get("/api/preview.jpg").headers["content-type"] == "image/jpeg"
        assert client.post("/api/actions/capture", json={}).status_code == 200
        assert client.post("/api/actions/reset", json={}).status_code == 200
        assert client.post("/api/actions/calibrate", json={}).status_code == 409
        exported = client.get("/api/export")
        assert exported.status_code == 200
        assert "events" in exported.json()
        assert client.post("/api/stop", json={}).json()["state"] == "stopped"


def test_stopped_metrics_do_not_decay(tmp_path):
    edge = EdgeRuntime(tmp_path)
    try:
        edge.start(RuntimeConfig(profile="roads", input_fps=8))
        until(lambda: edge.snapshot()["metrics"]["processed"] >= 2)
        edge.stop()
        before = edge.snapshot()["metrics"]["uptime_s"]
        time.sleep(.15)
        assert edge.snapshot()["metrics"]["uptime_s"] == before
    finally:
        edge.close()


def test_background_profile_must_close_before_another_station_starts(tmp_path, monkeypatch):
    class ClosingProfile(ConfirmationProfile):
        ready_to_close = False
        def close(self, timeout=3):
            return self.ready_to_close
    profile = ClosingProfile()
    creations = []
    def create(*args):
        creations.append(True)
        return profile
    monkeypatch.setattr("edge_runtime.runtime.LocalSource", StubSource)
    monkeypatch.setattr("edge_runtime.runtime.create_profile", create)
    edge = EdgeRuntime(tmp_path)
    try:
        edge.start(RuntimeConfig())
        until(lambda: edge.snapshot()["metrics"]["processed"] > 0)
        with pytest.raises(RuntimeError, match="Document worker is still closing"):
            edge.start(RuntimeConfig(profile="freight"))
        assert len(creations) == 1
        assert edge.snapshot()["state"] == "stopping"
        profile.ready_to_close = True
        edge.start(RuntimeConfig(profile="freight"))
        assert len(creations) == 2
    finally:
        profile.ready_to_close = True
        edge.close()


def test_blocked_media_keeps_inference_running_and_blocks_switch_until_drained(tmp_path, monkeypatch):
    import threading
    import cv2
    from edge_runtime.media import MediaWorker

    entered, release = threading.Event(), threading.Event()
    original_encode, original_close = cv2.imencode, MediaWorker.close
    creations = []

    class EventProfile(ConfirmationProfile):
        def process(self, frame, context):
            self.hits += 1
            return InspectionResult("fault", "Stub observation", annotated_frame=frame,
                                    event_key=f"stub:{context.sequence}", emit_event=True)

    def create(*args):
        creations.append(args[0])
        return EventProfile()

    def blocked_encode(*args, **kwargs):
        entered.set()
        assert release.wait(5), "Test encoder was not released"
        return original_encode(*args, **kwargs)

    # Preserve real MediaWorker behavior, while shortening only its bounded join
    # for the test. No heavyweight model or camera is involved.
    monkeypatch.setattr("edge_runtime.runtime.LocalSource", StubSource)
    monkeypatch.setattr("edge_runtime.runtime.create_profile", create)
    monkeypatch.setattr("edge_runtime.media.cv2.imencode", blocked_encode)
    monkeypatch.setattr(MediaWorker, "close", lambda worker, timeout=3: original_close(worker, .01))
    edge = EdgeRuntime(tmp_path)
    try:
        edge.start(RuntimeConfig(source="camera:0", input_fps=30, deadline_ms=1000,
                                 save_evidence=True, preview_fps=15))
        assert entered.wait(1)
        original_session, original_media = edge.session_id, edge.media
        initial_processed = edge.snapshot()["metrics"]["processed"]
        until(lambda: edge.snapshot()["metrics"]["processed"] >= initial_processed + 8)
        while_blocked = edge.snapshot()
        assert not release.is_set()
        assert while_blocked["metrics"]["inspection_errors"] == 0
        assert while_blocked["media"]["in_flight"] == 1
        assert while_blocked["media"]["pending_evidence"] <= 4
        assert while_blocked["media"]["evidence_rejected"] > 0
        assert while_blocked["media"]["pending_bytes"] <= while_blocked["media"]["byte_limit"]

        with pytest.raises(RuntimeError, match="Media worker is still closing"):
            edge.stop()
        stopped_processed = edge.snapshot()["metrics"]["processed"]
        with pytest.raises(RuntimeError, match="Media worker is still closing"):
            edge.start(RuntimeConfig(profile="freight", source="camera:0"))
        assert creations == ["factory"]
        assert edge.session_id == original_session
        assert edge.media is original_media
        assert edge.config.profile == "factory"
        assert edge.snapshot()["state"] == "stopping"
        assert not edge.snapshot()["running"]
        assert edge.snapshot()["metrics"]["processed"] == stopped_processed

        release.set()
        until(lambda: not original_media.status()["worker_alive"])
        edge.stop()
        assert edge.snapshot()["state"] == "stopped"
        assert edge.snapshot()["error"] is None
        assert original_media.status()["pending_bytes"] == 0
        until(lambda: edge.store.status()["pending"] == 0)
        old_images = [item for item in edge.store.list(limit=100) if item["media_name"]]
        assert old_images
        assert all(item["profile"] == "factory" and item["session_id"] == original_session
                   for item in old_images)
        fallback = [item for item in edge.store.list(limit=100)
                    if item["payload"].get("evidence_image_error")]
        assert fallback  # Backpressure retains finding metadata when images cannot queue.

        edge.start(RuntimeConfig(profile="freight", source="camera:0", input_fps=30,
                                 deadline_ms=1000))
        until(lambda: edge.snapshot()["metrics"]["processed"] >= 2)
        until(lambda: edge.preview() is not None)
        assert creations == ["factory", "freight"]
        assert edge.session_id != original_session
        assert edge.media is not original_media
        assert edge.snapshot()["metrics"]["inspection_errors"] == 0
        assert edge.preview().startswith(b"\xff\xd8")
    finally:
        release.set()
        edge.close()


def test_blocked_media_does_not_delay_terminal_outcome_for_stopped_trigger(tmp_path, monkeypatch):
    import threading
    import cv2
    from edge_runtime.media import MediaWorker
    entered, release = threading.Event(), threading.Event()
    original_encode, original_close = cv2.imencode, MediaWorker.close

    class PendingProfile(ConfirmationProfile):
        def process(self, frame, context):
            return InspectionResult("pending", "Awaiting confirmation")

    def blocked_encode(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return original_encode(*args, **kwargs)

    monkeypatch.setattr("edge_runtime.runtime.LocalSource", StubSource)
    monkeypatch.setattr("edge_runtime.runtime.create_profile", lambda *args: PendingProfile())
    monkeypatch.setattr("edge_runtime.media.cv2.imencode", blocked_encode)
    monkeypatch.setattr(MediaWorker, "close", lambda worker, timeout=3: original_close(worker, .01))
    edge = EdgeRuntime(tmp_path)
    try:
        edge.start(RuntimeConfig(source="camera:0", input_fps=30, deadline_ms=1000, triggered=True))
        assert entered.wait(1)
        unit = edge.action("trigger")["unit_id"]
        until(lambda: edge.snapshot()["metrics"]["processed"] >= 1)
        assert unit in edge._trigger_times
        with pytest.raises(RuntimeError, match="Media worker is still closing"):
            edge.stop()
        assert not release.is_set()
        assert unit in edge._missed_units
        assert unit not in edge._trigger_times
        assert edge.snapshot()["metrics"]["missed_units"] == 1
        until(lambda: edge.store.status()["pending"] == 0)
        missed = [item for item in edge.store.list(limit=100) if item["kind"]=="inspection_missed"]
        assert len(missed)==1
        assert missed[0]["payload"]["unit_id"]==unit
        assert missed[0]["payload"]["reason"]=="station_stopped"
    finally:
        release.set()
        edge.close()
    assert edge.snapshot()["metrics"]["missed_units"] == 1
