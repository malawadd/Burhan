"""Category-specific release gates. Training scores never grant video validation."""
from __future__ import annotations

import math


def category_release_gate(evaluation: dict) -> dict:
    """Calculate gates from held-out localization counts, not advertised accuracy.

    Image counts must represent independently captured originals. Calling this
    helper does not verify annotation quality; provenance/independence must have
    their own auditable report, and photo evidence cannot grant video release.
    """
    reasons = []
    counts = {}
    for name in ("true_positives", "false_positives", "false_negatives", "positive_images", "negative_images"):
        value = evaluation.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            reasons.append(f"Missing/invalid {name}")
        else:
            counts[name] = value
    tp, fp, fn = (counts.get(name, 0) for name in ("true_positives", "false_positives", "false_negatives"))
    precision = tp/(tp+fp) if tp+fp else None
    recall = tp/(tp+fn) if tp+fn else None
    if precision is None or precision < .95:
        reasons.append("Localization precision below 95% or unmeasured")
    if recall is None or recall < .90:
        reasons.append("Localization recall below 90% or unmeasured")
    for name in ("positive_images", "negative_images"):
        if counts.get(name, 0) < 50:
            reasons.append(f"Fewer than 50 independent {name}")
    for name in ("source_group_split_verified", "negative_labels_complete", "localization_verified", "licenses_verified"):
        if evaluation.get(name) is not True:
            reasons.append(f"{name} is not verified")
    photo_pass = not reasons
    if evaluation.get("real_video_validation_verified") is not True:
        reasons.append("Real video event/localization validation is not verified")
    if evaluation.get("requires_metric_measurement") is True and evaluation.get("independent_measurement_validation") is not True:
        reasons.append("Independent physical measurement validation is missing")
    performance = evaluation.get("runtime", {})
    if not isinstance(performance, dict):
        performance = {}
    for name, threshold, direction in (("preview_fps", 15, "minimum"), ("evidence_age_p95_ms", 1000, "maximum")):
        value = performance.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or (value < threshold if direction == "minimum" else value > threshold):
            reasons.append(f"Runtime {name} target not passed")
    return {"released": not reasons, "status": "validated" if not reasons else "research_only",
            "photo_gate_passed": photo_pass, "precision": precision, "recall": recall,
            "positive_images": counts.get("positive_images"), "negative_images": counts.get("negative_images"),
            "reasons": reasons}
