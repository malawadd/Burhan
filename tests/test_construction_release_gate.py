from edge_runtime.construction_validation import category_release_gate


def evaluated(**updates):
    result = {"true_positives": 95, "false_positives": 5, "false_negatives": 5,
              "positive_images": 100, "negative_images": 100,
              "source_group_split_verified": True, "negative_labels_complete": True,
              "localization_verified": True, "licenses_verified": True,
              "real_video_validation_verified": True,
              "runtime": {"preview_fps": 16, "evidence_age_p95_ms": 700}}
    result.update(updates)
    return category_release_gate(result)


def test_high_image_score_cannot_replace_region_localization():
    result = evaluated(true_positives=0, false_positives=10, false_negatives=100, image_accuracy=.99)
    assert not result["released"]
    assert result["recall"] == 0


def test_photo_pass_does_not_release_video_detector():
    result = evaluated(real_video_validation_verified=False)
    assert result["photo_gate_passed"] and not result["released"]


def test_small_or_unverified_negative_set_cannot_release():
    assert not evaluated(negative_images=49)["released"]
    assert not evaluated(negative_labels_complete=False)["released"]
    assert not evaluated(source_group_split_verified=False)["released"]


def test_invalid_count_and_nan_speed_fail_closed():
    assert not evaluated(true_positives=True)["released"]
    assert not evaluated(runtime={"preview_fps": float("nan"), "evidence_age_p95_ms": 300})["released"]
    assert not category_release_gate({})["released"]


def test_metric_checks_require_real_measurement_validation():
    assert not evaluated(requires_metric_measurement=True)["released"]
    assert evaluated(requires_metric_measurement=True, independent_measurement_validation=True)["released"]


def test_thresholds_are_category_specific_and_require_both_precision_and_recall():
    assert evaluated()["released"]
    assert not evaluated(false_positives=6)["released"]
    assert not evaluated(false_negatives=11)["released"]
