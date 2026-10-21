"""Source completion and accepted-trigger outcomes, independent of model accuracy."""
from __future__ import annotations

import threading
import time

import numpy as np

from edge_runtime.contracts import InspectionResult
from edge_runtime.runtime import EdgeRuntime, RuntimeConfig


def wait_until(predicate, timeout=3):
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if predicate():
            return
        time.sleep(.01)
    raise AssertionError("Runtime completion condition was not reached")


class FiniteSource:
    backend = "completion_test"

    def __init__(self, frames=3):
        self.frames = frames
        self.index = 0
        self.closed = threading.Event()

    def open(self):
        pass

    def read(self):
        if self.index >= self.frames:
            return None, None
        self.index += 1
        return np.full((64, 64, 3), self.index, np.uint8), float(self.index)

    def close(self):
        self.closed.set()


class GatedSource(FiniteSource):
    """One frame followed by a blocked read, released explicitly as EOF."""

    def __init__(self):
        super().__init__(frames=1)
        self.second_read = threading.Event()
        self.release_eof = threading.Event()

    def read(self):
        if self.index == 0:
            return super().read()
        self.second_read.set()
        if not self.release_eof.wait(5):
            raise RuntimeError("Test did not release the blocked source")
        return None, None


class ControlledProfile:
    def __init__(self, block_first=False):
        self.entered = threading.Event()
        self.release = threading.Event()
        if not block_first:
            self.release.set()
        self.completed = []

    def describe(self):
        return {"profile_version": "completion_test"}

    def process(self, frame, context):
        self.entered.set()
        if not self.release.wait(3):
            raise RuntimeError("Test did not release inference")
        # Preserve measurable in-flight work after the finite capture finishes.
        time.sleep(.02)
        self.completed.append((context.sequence, time.perf_counter()))
        return InspectionResult(status="clear", title="Completion test")


def missed_events(edge):
    return [row for row in edge.store.list(100) if row["kind"] == "inspection_missed"]


def test_finite_fifo_source_drains_all_frames_before_eof_and_end_time(tmp_path, monkeypatch):
    source, profile = FiniteSource(3), ControlledProfile(block_first=True)
    monkeypatch.setattr("edge_runtime.runtime.LocalSource", lambda *args: source)
    monkeypatch.setattr("edge_runtime.runtime.create_profile", lambda *args: profile)
    edge = EdgeRuntime(tmp_path)
    try:
        edge.start(RuntimeConfig(profile="factory", source="file:finite-test", input_fps=30, deadline_ms=5000))
        assert profile.entered.wait(2)
        assert source.closed.wait(2)
        wait_until(edge._capture_done.is_set)
        state = edge.snapshot()
        assert state["state"] == "draining"
        assert edge._ended is None
        assert state["metrics"]["captured"] == 3
        assert state["metrics"]["processed"] == 0
        assert state["metrics"]["queue_depth"] == 2

        profile.release.set()
        wait_until(lambda: edge.snapshot()["state"] == "eof")
        state = edge.snapshot()
        assert [sequence for sequence, _ in profile.completed] == [1, 2, 3]
        assert state["metrics"]["processed"] == 3
        assert state["metrics"]["queue_depth"] == 0
        assert state["metrics"]["dropped"] == state["metrics"]["late"] == 0
        assert state["metrics"]["inspection_errors"] == 0
        assert state["last_result"]["sequence"] == 3
        assert edge._ended >= profile.completed[-1][1]
        ended, uptime = edge._ended, state["metrics"]["uptime_s"]
        time.sleep(.05)
        assert edge._ended == ended
        assert edge.snapshot()["metrics"]["uptime_s"] == uptime
    finally:
        profile.release.set()
        edge.close()


def test_trigger_accepted_after_last_frame_settles_at_eof_without_stop(tmp_path, monkeypatch):
    source, profile = GatedSource(), ControlledProfile()
    monkeypatch.setattr("edge_runtime.runtime.LocalSource", lambda *args: source)
    monkeypatch.setattr("edge_runtime.runtime.create_profile", lambda *args: profile)
    edge = EdgeRuntime(tmp_path)
    try:
        edge.start(RuntimeConfig(profile="factory", source="file:finite-trigger-test", input_fps=30,
                                 triggered=True, deadline_ms=5000))
        assert source.second_read.wait(2)  # The final frame has already passed without a trigger.
        unit = edge.action("trigger")["unit_id"]
        source.release_eof.set()
        wait_until(lambda: edge.snapshot()["state"] == "eof")
        state = edge.snapshot()
        assert state["metrics"]["missed_units"] == 1
        assert unit in edge._missed_units
        assert unit not in edge._trigger_times
        assert unit not in edge._completed_units
        assert state["metrics"]["processed"] == 0
        assert profile.completed == []
        wait_until(lambda: len(missed_events(edge)) == 1)
        event = missed_events(edge)[0]["payload"]
        assert event["unit_id"] == unit
        assert event["reason"] == "source_ended_before_final_unit_result"
    finally:
        source.release_eof.set()
        edge.close()


def test_idle_inference_expires_trigger_while_capture_is_blocked(tmp_path, monkeypatch):
    source, profile = GatedSource(), ControlledProfile()
    monkeypatch.setattr("edge_runtime.runtime.LocalSource", lambda *args: source)
    monkeypatch.setattr("edge_runtime.runtime.create_profile", lambda *args: profile)
    edge = EdgeRuntime(tmp_path)
    try:
        edge.start(RuntimeConfig(profile="factory", source="file:blocked-trigger-test", input_fps=30,
                                 triggered=True, deadline_ms=100))
        assert source.second_read.wait(2)
        accepted = time.perf_counter()
        unit = edge.action("trigger")["unit_id"]
        wait_until(lambda: unit in edge._missed_units, timeout=2)
        state = edge.snapshot()
        assert time.perf_counter() >= accepted + .1
        assert state["state"] == "running"  # No EOF or Stop was needed to settle the trigger.
        assert not edge._capture_done.is_set()
        assert state["metrics"]["captured"] == 1
        assert state["metrics"]["processed"] == 0
        assert state["metrics"]["missed_units"] == 1
        assert unit not in edge._trigger_times
        wait_until(lambda: len(missed_events(edge)) == 1)
        event = missed_events(edge)[0]["payload"]
        assert event["unit_id"] == unit
        assert event["reason"] == "trigger_deadline_expired_without_result"
    finally:
        source.release_eof.set()
        edge.close()
