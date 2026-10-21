"""Camera property negotiation is tested with a driver mock, never real hardware."""
import time

import cv2
import numpy as np

from edge_runtime.contracts import InspectionResult
from edge_runtime.runtime import EdgeRuntime, RuntimeConfig
from edge_runtime.sources import LocalSource


class MockCapture:
    def __init__(self):
        self.requests=[]
        self.released=False
    def set(self, prop, value):
        self.requests.append((prop,value))
        return prop==cv2.CAP_PROP_FRAME_HEIGHT
    def get(self, prop):
        return {cv2.CAP_PROP_FRAME_WIDTH:640.,cv2.CAP_PROP_FRAME_HEIGHT:480.,
                cv2.CAP_PROP_FPS:15.,cv2.CAP_PROP_BUFFERSIZE:0.}.get(prop,0.)
    def isOpened(self):return True
    def getBackendName(self):return "mock_camera_driver"
    def read(self):return True,np.zeros((360,640,3),np.uint8)
    def release(self):self.released=True


def test_camera_requests_fps_and_reports_rejection_driver_values_and_observed_size(monkeypatch):
    capture=MockCapture()
    monkeypatch.setattr(cv2,"VideoCapture",lambda *args:capture)
    source=LocalSource("camera:0","factory",30,1280,720)
    try:
        source.open()
        initial=source.capture_properties
        assert (cv2.CAP_PROP_FPS,30) in capture.requests
        assert initial["fps_requested"]==30
        assert initial["fps_request_accepted"] is False
        assert initial["driver_fps"]==15
        assert initial["driver_fps_is_measured"] is False
        assert initial["requested_dimensions"]==[1280,720]
        assert initial["width_request_accepted"] is False
        assert initial["height_request_accepted"] is True
        assert initial["driver_dimensions"]==[640,480]
        assert initial["effective_dimensions"]==[640,480]
        assert initial["buffer_size_requested"]==1
        assert initial["buffer_request_accepted"] is False
        assert initial["driver_buffer_size"] is None
        assert initial["exposure_latency_ms"] is None
        frame,timestamp=source.read()
        assert frame.shape==(360,640,3)
        assert timestamp is None
        assert source.capture_properties["effective_dimensions"]==[640,360]
        assert source.capture_properties["dimensions_source"]=="observed_frame"
        assert source.capture_properties is not initial
        assert initial["effective_dimensions"]==[640,480]
        assert initial["dimensions_source"]=="driver_reported"
    finally:source.close()
    assert capture.released


def test_unsupported_fps_request_and_nonfinite_driver_report_do_not_fail_open(monkeypatch):
    class UnsupportedCapture(MockCapture):
        def set(self, prop, value):
            if prop==cv2.CAP_PROP_FPS:raise cv2.error("FPS negotiation unsupported")
            return super().set(prop,value)
        def get(self,prop):
            return float("nan") if prop==cv2.CAP_PROP_FPS else super().get(prop)
    capture=UnsupportedCapture()
    monkeypatch.setattr(cv2,"VideoCapture",lambda *args:capture)
    source=LocalSource("camera:0","factory",12)
    try:
        source.open()
        assert source.capture_properties["fps_request_accepted"] is False
        assert source.capture_properties["driver_fps"] is None
        assert "unsupported" in source.capture_properties["property_errors"]["fps"]
        assert source.read()[0] is not None
    finally:source.close()


def test_runtime_camera_properties_survive_profile_reset(tmp_path,monkeypatch):
    class Profile:
        def describe(self):return {"profile_version":"mock_v1"}
        def reset(self):pass
        def process(self,frame,context):return InspectionResult("clear","Mock observation")
    capture=MockCapture()
    monkeypatch.setattr(cv2,"VideoCapture",lambda *args:capture)
    monkeypatch.setattr("edge_runtime.runtime.create_profile",lambda *args:Profile())
    edge=EdgeRuntime(tmp_path)
    try:
        edge.start(RuntimeConfig(source="camera:0",input_fps=12,width=1280,height=720,
                                 preview_fps=0,deadline_ms=1000))
        end=time.monotonic()+2
        while edge.snapshot()["metrics"]["processed"]<2 and time.monotonic()<end:
            time.sleep(.01)
        assert edge.snapshot()["metrics"]["processed"]>=2
        description=edge.snapshot()["description"]
        props=description["capture_properties"]
        assert props["fps_requested"]==12
        assert props["driver_fps"]==15
        assert props["effective_dimensions"]==[640,360]
        assert props["dimensions_source"]=="observed_frame"
        assert description["capture_backend"]=="mock_camera_driver"
        edge.action("reset")
        assert edge.snapshot()["description"]["capture_properties"]==props
        assert edge.snapshot()["description"]["capture_backend"]=="mock_camera_driver"
        assert edge.snapshot()["metrics"]["inspection_errors"]==0
    finally:edge.close()


def test_action_does_not_overwrite_first_frame_properties_published_during_describe(tmp_path,monkeypatch):
    import threading
    read_ready, describe_ready, finish_describe=threading.Event(),threading.Event(),threading.Event()
    class Capture(MockCapture):
        def read(self):
            assert read_ready.wait(2)
            return super().read()
    class Profile:
        calls=0
        def reset(self):pass
        def describe(self):
            self.calls+=1
            if self.calls>1:
                describe_ready.set()
                assert finish_describe.wait(2)
            return {"profile_version":"mock_v1"}
        def process(self,frame,context):return InspectionResult("clear","Mock observation")
    capture=Capture()
    monkeypatch.setattr(cv2,"VideoCapture",lambda *args:capture)
    monkeypatch.setattr("edge_runtime.runtime.create_profile",lambda *args:Profile())
    edge=EdgeRuntime(tmp_path)
    action_errors=[]
    def reset():
        try:edge.action("reset")
        except Exception as exc:action_errors.append(exc)
    action=None
    try:
        edge.start(RuntimeConfig(source="camera:0",input_fps=12,preview_fps=0,deadline_ms=1000))
        end=time.monotonic()+2
        while edge.snapshot()["state"]!="running" and time.monotonic()<end:time.sleep(.01)
        assert edge.snapshot()["description"]["capture_properties"]["dimensions_source"]=="driver_reported"
        action=threading.Thread(target=reset)
        action.start()
        assert describe_ready.wait(1)
        read_ready.set()
        end=time.monotonic()+2
        while edge.snapshot()["metrics"]["captured"]<1 and time.monotonic()<end:time.sleep(.01)
        assert edge.snapshot()["metrics"]["captured"]>=1
        finish_describe.set()
        action.join(2)
        assert not action.is_alive()
        assert not action_errors
        props=edge.snapshot()["description"]["capture_properties"]
        assert props["effective_dimensions"]==[640,360]
        assert props["dimensions_source"]=="observed_frame"
    finally:
        read_ready.set()
        finish_describe.set()
        if action:action.join(2)
        edge.close()
