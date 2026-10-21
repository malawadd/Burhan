"""Build a readable index from preserved results; no inference or model selection."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    experiments = [
        ("SDNET boxes", "YOLOv8n320", "output/training/construction-expansion-sdnet-v1/yolov8n-results.json", False),
        ("SDNET boxes", "SSDLite320", "output/training/construction-expansion-sdnet-v1/ssdlite320-results.json", False),
        ("SDNET masks", "LR-ASPP320", "output/evaluation/construction-expansion-sdnet-masks/lraspp-results.json", True),
        ("SDNET masks", "YOLOv8n-seg320", "output/evaluation/construction-expansion-sdnet-masks/yolov8n-seg-results.json", True),
        ("PHELE boxes", "YOLOv8n384", "output/evaluation/construction-expansion-phele-v1/yolov8n-results.json", False),
        ("PHELE boxes", "SSDLite320", "output/evaluation/construction-expansion-phele-v1/ssdlite320-results.json", False),
        ("MBDD stain boxes", "SSDLite320", "output/evaluation/construction-expansion-wall-stain-v1/ssdlite320-results.json", False),
    ]
    rows, before_rows, sources = [], [], []
    for dataset, architecture, relative, mask in experiments:
        path = ROOT / relative
        data = json.loads(path.read_text(encoding="utf-8"))
        sources.append({"path": relative, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        selected = data["after_test_cal_selected" if mask else "after_test_calibration_selected"]
        before = data["before_test_fixed_0_5" if mask else "before_test_fixed_point_0_5"]
        after = data["after_test_fixed_0_5" if mask else "after_test_fixed_point_0_5"]
        for label, metrics in selected["classes"].items():
            threshold = selected.get("thresholds", {}).get(label, selected.get("threshold"))
            rows.append((dataset, architecture, label, "mask" if mask else "box", threshold, metrics))
            before_rows.append((dataset, architecture, label, before["classes"][label], after["classes"][label]))
    extended_path = ROOT / "output/training/construction-expansion-sdnet-v2-performance/results.json"
    if extended_path.exists():
        data = json.loads(extended_path.read_text(encoding="utf-8"))
        sources.append({"path": extended_path.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(extended_path.read_bytes()).hexdigest()})
        selected = data["reused_development"]
        for label, metrics in selected["classes"].items():
            threshold = selected.get("thresholds", {}).get(label, selected.get("class_confidence", {}).get(label))
            rows.append(("SDNET reused development", "YOLOv8n512 continuation", label, "box", threshold, metrics))
    lines = ["# Construction expansion: benchmark overview", "",
        "Generated from preserved local JSON results. All models remain research; **zero released categories**. No threshold or winner is selected by this summary.", "",
        "## Frozen CAL-selected points", "",
        "TP requires native localization IoU≥0.5. Mask rows require actual mask overlap, not box overlap. Counts are objects/components, while positive/negative counts are photographs. Different datasets and fitting budgets must not be pooled or ranked as one accuracy score.", "",
        "| Dataset | Architecture | Category | Match | CAL point | TP | Unmatched | FN | Precision | Recall | Positive / target-negative photos |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for dataset, architecture, label, kind, threshold, m in rows:
        # An abstaining model has undefined precision, not 100% precision.
        precision = f"{100*m['precision']:.2f}%" if m['tp']+m['fp'] else "— (no predictions)"
        lines.append(f"| {dataset} | {architecture} | `{label}` | {kind} | {threshold} | {m['tp']} | {m['fp']} | {m['fn']} | {precision} | {100*m['recall']:.2f}% | {m.get('positive_images','?')} / {m.get('negative_images','?')} |")
    lines += ["", "SDNET opposite-class annotations are not verified exhaustive: unmatched predictions are not proven physical false alarms, and the target-negative photograph column includes opposite-class views. Publisher-Fixed-only flags are reported separately in the [fastening](construction-expansion-sdnet-benchmarks.md) and [mask](construction-expansion-sdnet-masks.md) records. Site independence is unknown. The stronger fit reuses the exposed 98-photo set as development.", "",
        "PHELE has only three Mold-positive TEST photographs; one matched region and zero unmatched regions does not establish dependable 100% precision. MBDD negatives contain other source defects, not independently confirmed healthy walls. Its native float-box counts are shown here; served integer rounding moves one borderline IoU across 0.5, documented in the [stain audit](construction-expansion-wall-stain.md).", "",
        "## Fixed 0.5 before and after training", "",
        "Before means a fresh task head on the official initializer, not a previously working installed inspection model. Both columns use the same 0.5 confidence point on the same exposed source set. CAL-selected operating points above are separate.", "",
        "| Dataset | Architecture | Category | Before TP / unmatched / FN | After TP / unmatched / FN |",
        "|---|---|---|---:|---:|"]
    for dataset, architecture, label, before, after in before_rows:
        triple = lambda m: f"{m['tp']} / {m['fp']} / {m['fn']}"
        lines.append(f"| {dataset} | {architecture} | `{label}` | {triple(before)} | {triple(after)} |")
    lines += ["", "The source records retain pixel IoU/precision/recall, individual matches, failed exports and corrected numerical verification, source sample counts, original protocols, licenses and exact training budgets. Successful parity means a model was exported faithfully, including faithfully reproducing its failures.", "",
        "## Real video, runtime and release", "",
        "[Runtime benchmarks](construction-expansion-runtime-benchmarks.md) distinguish encoded preview, unique browser frames, guidance, actual inspected/model frames, dropped/skipped frames and displayed-result age. A smooth preview is not 15 fps inspection. [Real owner video records](construction-expansion-implementation.md) retain timestamp-matched original/prediction footage and known misses/clutter; those private videos never enter fitting or calibration and have no invented frame-level ground truth.", "",
        "The [category catalog](construction-expansion-category-catalog.md) documents missing or unsuitable data. No fit meets the entire per-category precision≥95%, recall≥90%, 50 independent positives and negatives, real-video/runtime/measurement release protocol.", "",
        "## Source result receipts", ""]
    for source in sources:
        lines.append(f"- `{source['path']}` — SHA256 `{source['sha256']}`")
    lines += ["", "Reproduce with `.venv/Scripts/python.exe scripts/summarize_construction_expansion.py`. It only reads results and writes this overview plus a timestamped hash receipt; no model is fitted or evaluated.", ""]
    destination = ROOT / "docs/construction-expansion-benchmark-summary.md"
    destination.write_text("\n".join(lines), encoding="utf-8")
    now = datetime.now(timezone.utc)
    receipt = {"created_utc": now.isoformat(), "source_results": sources, "output_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
               "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "inference_or_selection_performed": False}
    out = ROOT / "output/evaluation/construction-guided-expansion/summary-receipts"
    out.mkdir(parents=True, exist_ok=True)
    (out / (now.strftime("%Y%m%dT%H%M%S.%fZ")+".json")).write_text(json.dumps(receipt, indent=2)+"\n", encoding="utf-8")
    print(f"Wrote {len(rows)} per-category result rows and {len(before_rows)} fixed-point comparisons")


if __name__ == "__main__":
    main()
