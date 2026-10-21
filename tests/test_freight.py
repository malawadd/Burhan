import cv2
import numpy as np

from edge_runtime.contracts import FrameContext
from edge_runtime.profiles.freight import FreightProfile, demo_frame


def context(sequence=0):
    return FrameContext(sequence, sequence / 20, source_id="demo", session_id="test")


def test_no_page():
    result = FreightProfile().process(np.zeros((540, 960, 3), np.uint8), context())
    assert not result.metrics["page_detected"]
    assert not result.emit_event
    assert len(result.domain_state["missing_pages"]) == 2


def test_blurred_page_not_captured_even_on_manual_request():
    profile = FreightProfile({"stable_frames": 1})
    frame = cv2.GaussianBlur(demo_frame(40), (31, 31), 10)
    profile.request_capture()
    result = profile.process(frame, context())
    assert result.metrics["page_detected"]
    assert not result.metrics["quality_acceptable"]
    assert not result.emit_event


def test_stability_capture_duplicate_and_complete_checklist():
    profile = FreightProfile({"stable_frames": 3})
    frame = demo_frame(40)
    first = profile.process(frame, context(1))
    assert not first.emit_event
    profile.process(frame, context(2))
    accepted = profile.process(frame, context(3))
    assert accepted.emit_event
    assert accepted.domain_state["captured_pages"] == ["delivery_note"]
    duplicate = profile.process(frame, context(4))
    assert duplicate.status == "duplicate"
    assert duplicate.metrics["duplicate"]
    assert not duplicate.emit_event
    second = demo_frame(160)
    for i in range(5, 8):
        result = profile.process(second, context(i))
    assert result.emit_event
    assert result.domain_state["complete"]
    assert result.domain_state["missing_pages"] == []
    assert not profile.process(second, context(8)).emit_event


def test_manual_capture_and_reset():
    profile = FreightProfile({"stable_frames": 100})
    profile.request_capture()
    result = profile.process(demo_frame(50), context())
    assert result.emit_event
    profile.reset()
    assert not profile.process(demo_frame(50), context()).domain_state["complete"]
    assert not profile.accepted


def test_unknown_page_still_gets_geometry_and_quality():
    frame = demo_frame(60)
    # Remove saturated marker pixels while preserving geometry and printed content.
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    frame[hsv[:, :, 1] > 100] = (245, 245, 245)
    profile = FreightProfile({"stable_frames": 1})
    result = profile.process(frame, context())
    assert result.metrics["page_detected"]
    assert result.metrics["quality_acceptable"]
    assert result.domain_state["detected_template"] is None
    assert result.status == "needs_review"
    assert not result.emit_event


def test_detector_uses_pixels_not_sequence():
    frame = demo_frame(160)
    for sequence in [1, 40, 120, 10000]:
        result = FreightProfile({"stable_frames": 1}).process(frame, context(sequence))
        assert result.domain_state["detected_template"] == "packing_list"


def test_large_camera_motion_restarts_stability_count():
    profile = FreightProfile({"stable_frames": 3})
    frame = demo_frame(50)
    profile.process(frame, context(1))
    profile.process(frame, context(2))
    shifted = cv2.warpAffine(frame, np.array([[1, 0, 65], [0, 1, 0]], np.float32),
                             (frame.shape[1], frame.shape[0]), borderValue=(48, 42, 36))
    result = profile.process(shifted, context(3))
    assert not result.emit_event
    assert result.metrics["stable_frames"] == 1


def test_page_touching_camera_edge_is_rejected():
    profile = FreightProfile({"stable_frames": 1})
    frame = demo_frame(50)
    # Cutting through the document creates a quadrilateral at the sensor boundary.
    cropped = frame[:, 370:].copy()
    result = profile.process(cropped, context())
    assert not result.emit_event
    assert result.metrics.get("cropped", True)
