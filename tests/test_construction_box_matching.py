"""Geometry fixtures, not model accuracy or demonstration footage."""
from copy import deepcopy
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_construction_defect_video import match


def test_duplicate_retains_actual_overlap_without_claiming_a_second_match():
    truth = [{"id": "crack:0", "label": "crack", "bbox": [10, 20, 30, 40]}]
    prediction = {"label": "crack", "bbox": [10, 20, 30, 40]}
    predictions = [deepcopy(prediction), deepcopy(prediction)]
    matches, missed = match(predictions, truth)
    assert matches == ["crack:0"] and missed == []
    assert predictions[0]["matched_annotation_id"] == "crack:0"
    assert predictions[1]["matched_annotation_id"] is None
    assert predictions[1]["iou"] == 1.0


def test_spalling_overlap_cannot_be_reported_as_crack_localization():
    predictions = [{"label": "crack", "bbox": [10, 20, 30, 40]}]
    truth = [{"id": "spall:0", "label": "spalling", "bbox": [10, 20, 30, 40]}]
    matches, missed = match(predictions, truth)
    assert matches == [] and missed == truth
    assert predictions[0]["matched_annotation_id"] is None
    assert predictions[0]["iou"] == 0.0
