"""Serial real-video delivery checks of every installed construction research fit.

Run after fitting/export work stops. Replaying the same original separately for
each model is a comparison, not extra independent data or a looped source.
"""
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from benchmark_construction_guided import run
from edge_runtime.profiles.construction_task_detector import discover_models


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--walls-video", required=True, type=Path)
    parser.add_argument("--fastening-video", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--context", required=True, help="Concurrent workloads / quiescence evidence; not an assumption")
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "protocol-before-runs.json").exists():
        parser.error("Keep the frozen run; choose a new output directory")
    videos = {"walls_finishes": args.walls_video.resolve(), "brackets_railings": args.fastening_video.resolve()}
    if any(not path.is_file() for path in videos.values()):
        parser.error("Both actual local video files must exist")
    models = [m for m in discover_models() if m["inspection_task"] in videos]
    if not models:
        parser.error("No verified installed task model")
    protocol = {"created_utc": datetime.now(timezone.utc).isoformat(), "context": args.context,
        "scope": "Actual original owner video, once separately per pinned model; no accuracy labels or threshold/checkpoint selection. Short clips do not establish sustained field detector capacity.",
        "requested_input_fps": 30, "requested_preview_fps": 30, "deadline_ms": 1000,
        "cpu_inference_threads": 1, "maximum_run_seconds": 20,
        "videos": {task: {"path": str(path), "sha256": sha(path)} for task, path in videos.items()},
        "models": [{**model, "manifest_sha256": sha(ROOT/model["manifest"])} for model in models],
        "code_sha256": {name: sha(ROOT/name) for name in ["scripts/benchmark_construction_expansion_models.py",
            "scripts/benchmark_construction_guided.py", "edge_runtime/runtime.py", "edge_runtime/media.py",
            "edge_runtime/profiles/construction_tasks.py", "edge_runtime/profiles/construction_task_detector.py"]}}
    (out / "protocol-before-runs.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    results = []
    for index, model in enumerate(models):
        item = {"manifest": model["manifest"], "model_sha256": model["model_sha256"],
                "title": model["title"], "inspection_task": model["inspection_task"]}
        directory = out / f"model-{index+1:02d}"
        directory.mkdir()
        try:
            if sha(ROOT/model["manifest"]) != protocol["models"][index]["manifest_sha256"]:
                raise ValueError("Pinned manifest changed")
            item["result"] = run(videos[model["inspection_task"]], {"inspection_task": model["inspection_task"],
                "research_model_manifest": str((ROOT/model["manifest"]).resolve())}, 20, directory, "guided")
            item["error"] = None
        except Exception as error:
            item["error"] = f"{type(error).__name__}: {error}"
        results.append(item)
        (out / "report-in-progress.json").write_text(json.dumps({"protocol": protocol, "results": results}, indent=2), encoding="utf-8")
        print(json.dumps({"model": model["title"], "metrics": item.get("result", {}).get("metrics"), "error": item["error"]}), flush=True)
    report = {"protocol": protocol, "results": results, "accuracy_or_release_claim": False}
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# All installed construction models: real-video runtime", "", args.context, "",
        "Serial, short-clip laptop measurements after fitting/export stops; one CPU inference thread per adapter. Each original is replayed once per model. No synthetic frames, source loop, new accuracy labels, threshold tuning or checkpoint selection. Native videos and code/model hashes are pinned before the first run in protocol-before-runs.json.", "",
        "| Research model | Encoded preview fps | Actual model frames/s | Model frames / capture | Quality guidance | Queue drops / source skips | Model-frame profile p95 ms | Backend-result age p95 ms | Peak process MiB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for item in results:
        if item["error"]:
            lines.append(f"| {item['title']} | ERROR: {item['error']} | | | | | | | |")
            continue
        result = item["result"]
        m = result["metrics"]
        lines.append(f"| {item['title']} | {m['preview_fps']} | {m['model_inference_fps']} | {m['model_inference_frames']} / {m['captured']} | {m['guidance_frames']} | {m['dropped']} / {m['source_skipped']} | {m['model_inference_p95_ms']} | {result['evidence_age_p95_ms']} | {result['working_set_peak_mb']} |")
    lines += ["", "Actual inspected FPS includes geometry where applicable; this table explicitly uses model-call frames only. The model-frame profile timing includes capture-quality processing, inference, postprocessing and rendering for frames that actually ran the model, excluding external source decoding and JPEG work. Backend result age includes capture-guidance results; it is not independently measured browser display age or camera exposure age.", "",
        "Memory peaks are working sets of the same serial benchmark process and include retained native allocations. They are not isolated model memory deltas. Clips are about7–11seconds, so these data do not prove sustained15fps neural inspection or physical defect coverage. Quality gates remain failed regardless of delivery speed.", "",
        "See docs/construction-expansion-runtime-benchmarks.md for before/after delivery, the separate120-second stress run and actual unique browser/display-age checks.", ""]
    (out / "README.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
