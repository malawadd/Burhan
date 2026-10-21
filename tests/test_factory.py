import cv2
import numpy as np
import pytest

from edge_runtime.contracts import FrameContext
from edge_runtime.profiles.factory import FactoryProfile, demo_frame


def context(sequence=0, unit_id=None):
    return FrameContext(sequence, float(sequence), source_id="test", session_id="unit", unit_id=unit_id)


def confirm(profile, frame):
    return [profile.process(frame, context(i)) for i in range(3)]


def test_requires_explicit_calibration_and_never_learns_input():
    profile = FactoryProfile()
    for i in range(4):
        assert profile.process(demo_frame(60), context(i)).status == "needs_input"
    assert not profile.describe()["calibrated"]
    assert profile.calibrate(demo_frame(0))["ok"]
    assert profile.process(demo_frame(0), context()).status == "clear"


def test_clean_and_pixel_defects_with_unrelated_sequence():
    profile = FactoryProfile({"demo": True})
    assert profile.process(demo_frame(0), context(99999)).status == "clear"
    for image_index in (60, 120, 180):
        profile.reset()
        results = confirm(profile, demo_frame(image_index))
        assert results[0].status == "pending"
        assert results[-1].status == "fault"
        assert results[-1].findings
        assert results[-1].emit_event
        assert not profile.process(demo_frame(image_index), context(888)).emit_event
        for finding in results[-1].findings:
            x, y, w, h = finding.bbox
            assert 0 <= x < 960 and 0 <= y < 540 and w > 0 and h > 0
            assert x + w <= 960 and y + h <= 540


def test_defect_drawn_directly_on_pixels_is_detected():
    profile = FactoryProfile({"demo": True, "confirmation_frames": 1})
    frame = demo_frame(0)
    cv2.rectangle(frame, (475, 310), (510, 345), (245, 245, 245), -1)
    result = profile.process(frame, context(0))
    assert result.status == "fault"
    assert any(x < 490 < x + w and y < 325 < y + h for x, y, w, h in [f.bbox for f in result.findings])


@pytest.mark.parametrize("frame", [np.zeros((540, 960, 3), np.uint8), cv2.GaussianBlur(demo_frame(0), (41, 41), 0)])
def test_blank_and_blur_are_not_clear(frame):
    profile = FactoryProfile({"demo": True})
    assert profile.process(frame, context()).status == "needs_input"


def test_bad_calibration_does_not_replace_approved_reference():
    profile = FactoryProfile({"demo": True})
    assert not profile.calibrate(np.zeros((540, 960, 3), np.uint8))["ok"]
    assert profile.process(demo_frame(0), context()).status == "clear"


def test_small_camera_translation_registered_and_boxes_in_original_coordinates():
    profile = FactoryProfile({"demo": True, "confirmation_frames": 1})
    transform = np.float32([[1, 0, 9], [0, 1, 5]])
    clean = cv2.warpAffine(demo_frame(0), transform, (960, 540), borderMode=cv2.BORDER_REPLICATE)
    assert profile.process(clean, context()).status == "clear"
    faulty = cv2.warpAffine(demo_frame(60), transform, (960, 540), borderMode=cv2.BORDER_REPLICATE)
    result = profile.process(faulty, context())
    assert result.status == "fault"
    assert any(x <= 670 + 9 < x + w and y <= 215 + 5 < y + h for x, y, w, h in [f.bbox for f in result.findings])


def test_unrelated_textured_scene_registration_is_inconclusive():
    profile = FactoryProfile({"demo": True})
    noise = np.random.default_rng(89).integers(0, 255, (540, 960, 3), dtype=np.uint8)
    assert profile.process(noise, context()).status in {"needs_input", "inconclusive"}


def test_triggered_units_are_independent_and_resolution_requires_recalibration():
    profile = FactoryProfile({"demo": True})
    assert profile.process(demo_frame(60), context(unit_id="one")).emit_event
    assert not profile.process(demo_frame(60), context(unit_id="one")).emit_event
    assert profile.process(demo_frame(60), context(unit_id="two")).emit_event
    assert profile.process(demo_frame(0, 640, 360), context()).status == "needs_input"


def test_invalid_reference_path_is_reported_not_silently_replaced():
    profile = FactoryProfile({"demo": True, "reference_path": "does-not-exist.png"})
    assert profile.process(demo_frame(0), context()).status == "needs_input"
