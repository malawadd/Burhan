"""Synthetic unit fixtures only; never inspection data or application videos."""
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from export_cubit_author_development_videos import render


def fixture(empty=False):
    raw = np.full((260, 346, 3), 10, np.uint8)
    native = np.full_like(raw, 80)
    mask = np.zeros((260, 346), bool)
    if not empty:
        mask[15, 15:25] = True
    prediction = {'bbox': [10, 10, 20, 10], 'label': 'crack', 'score': .7,
                  'matched_annotation_id': 'external', 'native_mask_empty': empty}
    record = {'source_frame_index': 0, 'video_time_s': 0., 'input_exposure_gain': 8.,
              'expected': [{'id': 'external', 'label': 'crack', 'bbox': [100, 100, 20, 10]}],
              'predictions': [prediction], 'evaluated': True, 'localized_matches': 1,
              'missed_defects': 0, 'false_alerts': 0}
    return raw, native, mask, record


def test_actual_mask_on_native_gain_input_and_gt_stays_separate():
    raw, native, mask, record = fixture()
    before = mask.copy()
    actual = [{'bbox': [10, 10, 20, 10], 'label': 'crack', 'score': .7, 'mask': mask}]
    canvas = render(raw, native, record, actual, 'unit-only', 'FAILED CALIBRATION')
    assert canvas.shape == (1300, 1420, 3)
    np.testing.assert_array_equal(canvas[660 + 30, 716 + 40], [72, 130, 84])
    # An external GT rectangle here must not be copied into the model pane.
    np.testing.assert_array_equal(canvas[660 + 200, 716 + 200], [80, 80, 80])
    np.testing.assert_array_equal(canvas[100 + 30, 8 + 40], [10, 10, 10])
    np.testing.assert_array_equal(mask, before)
    assert np.all(raw == 10) and np.all(native == 80)


def test_empty_model_mask_retains_box_without_inventing_foreground():
    raw, native, mask, record = fixture(empty=True)
    actual = [{'bbox': [10, 10, 20, 10], 'label': 'crack', 'score': .7, 'mask': mask}]
    canvas = render(raw, native, record, actual, 'unit-only', 'FAILED CALIBRATION')
    np.testing.assert_array_equal(canvas[660 + 30, 716 + 40], [80, 80, 80])
    assert not mask.any()
