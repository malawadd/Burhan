"""Geometry and metric fixtures only; they are not evidence of model accuracy."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from factory_localization_audit import aggregate, mask_metrics, retained_mask, runtime_box_metrics
from edge_runtime.contracts import Finding
import pytest


def test_wrong_location_is_not_a_localized_defect():
    truth = np.zeros((20, 20), np.uint8)
    truth[1:5, 1:5] = 1
    prediction = np.zeros_like(truth)
    prediction[10:15, 10:15] = 1
    measured = mask_metrics(prediction, truth)
    assert measured["flagged"] and not measured["localized_any_pixel"]
    assert measured["iou"] == 0


def test_one_pixel_touch_is_separate_from_useful_coverage():
    truth = np.zeros((20, 20), np.uint8)
    truth[2:12, 2:12] = 1
    prediction = np.zeros_like(truth)
    prediction[2, 2] = 1
    measured = mask_metrics(prediction, truth)
    assert measured["localized_any_pixel"]
    assert not measured["localized_10pct_coverage"]
    assert measured["region_matches_at_iou"]["0.1"] == 0


def test_whole_frame_prediction_does_not_get_good_iou():
    truth = np.zeros((20, 20), np.uint8)
    truth[2:4, 2:4] = 1
    measured = mask_metrics(np.ones_like(truth), truth)
    assert measured["pixel_recall"] == 1
    assert measured["pixel_precision"] == .01
    assert measured["iou"] == .01
    assert measured["region_matches_at_iou"]["0.1"] == 0


def test_minimum_area_is_in_model_grid_before_original_projection():
    score = np.zeros((4, 4), np.float32)
    score[1, 1:3] = 1
    assert retained_mask(score, .5, 2, (8, 8)).sum() == 8
    assert retained_mask(score, .5, 3, (8, 8)).sum() == 0


def test_one_merged_prediction_cannot_match_two_separate_defects():
    truth = np.zeros((10, 10), np.uint8)
    truth[2:4, 1:3] = 1
    truth[2:4, 7:9] = 1
    prediction = np.zeros_like(truth)
    prediction[2:4, 1:9] = 1
    measured = mask_metrics(prediction, truth)
    assert measured["gt_regions"] == 2 and measured["predicted_regions"] == 1
    assert measured["region_matches_at_iou"]["0.25"] == 1


def test_actual_runtime_box_wrong_location_and_duplicate_are_not_extra_hits():
    truth = np.zeros((20, 20), np.uint8)
    truth[2:6, 3:7] = 1
    result = runtime_box_metrics([
        Finding("defect", "match", (3, 2, 4, 4), .9),
        Finding("defect", "duplicate", (3, 2, 4, 4), .8),
        Finding("defect", "wrong", (12, 12, 3, 3), .95),
    ], truth)["runtime_box_proxy"]
    assert result["gt_boxes_xywh"] == [[3, 2, 4, 4]]
    assert result["predicted_regions"] == 3
    assert result["matches_at_iou"]["0.5"] == 1
    assert result["runtime_findings"][2]["bbox_xywh"] == [12., 12., 3., 3.]


@pytest.mark.parametrize("box", [(19, 19, 2, 2), (-1, 0, 2, 2), (0, 0, 0, 2), (0, 0, float("nan"), 2)])
def test_runtime_box_audit_rejects_invalid_geometry(box):
    with pytest.raises(ValueError):
        runtime_box_metrics([Finding("defect", "invalid", box)], np.zeros((20, 20), np.uint8))


def test_supplementary_runtime_boxes_preserve_legacy_aggregate_without_fields():
    truth = np.zeros((10, 10), np.uint8)
    truth[2:4, 1:3] = 1
    row = {"defective": True, **mask_metrics(truth, truth)}
    assert "runtime_box_proxy" not in aggregate([row])
    row.update(runtime_box_metrics([Finding("defect", "hit", (1, 2, 2, 2))], truth))
    measured = aggregate([row])
    assert measured["runtime_box_proxy"]["precision"]["0.5"] == 1
    assert measured["foreground_pixel_recall"] == 1
