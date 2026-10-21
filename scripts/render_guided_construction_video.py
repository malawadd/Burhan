"""Render every actual source frame beside its local research result, offline.

No frames, defects or annotations are invented. Output is an inspected replay,
not a live FPS benchmark, and a model overlay is not ground truth.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from fractions import Fraction

import cv2
import numpy as np
import av  # Media/export environment; not a runtime dependency.

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from edge_runtime.contracts import FrameContext
from edge_runtime.profiles.construction_tasks import GuidedConstructionProfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--before-manifest", type=Path, help="Optional frozen earlier model for a three-panel, same-input comparison")
    parser.add_argument("--task", required=True, choices=["walls_finishes", "brackets_railings"])
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    video, out = args.video.resolve(), args.output.resolve()
    if not video.is_file():
        parser.error("Existing real video required")
    out.mkdir(parents=True, exist_ok=True)
    if (out / "report.json").exists():
        parser.error("Preserve the completed comparison; choose a new output directory")
    profile = GuidedConstructionProfile({"inspection_task": args.task, "research_model_manifest": str(args.manifest.resolve())})
    profile.prepare_worker()
    before_profile = None
    if args.before_manifest:
        before_profile = GuidedConstructionProfile({"inspection_task": args.task, "research_model_manifest": str(args.before_manifest.resolve())})
        before_profile.prepare_worker()
    capture = cv2.VideoCapture(str(video))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    declared = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if not capture.isOpened() or not np.isfinite(fps) or not 1 <= fps <= 120:
        raise ValueError("Cannot decode source or its native FPS")
    writer, stream = None, None
    records = []
    started = time.perf_counter()
    try:
        while True:
            ok, raw = capture.read()
            if not ok:
                break
            index = len(records)
            timestamp = float(capture.get(cv2.CAP_PROP_POS_MSEC))
            if not np.isfinite(timestamp):
                raise ValueError("Source decoder returned invalid timestamp")
            if records and timestamp <= records[-1]["source_time_ms"]:
                raise ValueError("Source timestamps must increase; no synthetic timing will be substituted")
            # Inference uses the actual decoded full-resolution image. Only the
            # comparison display is resized after localization and saved.
            context = FrameContext(index+1, time.perf_counter(), "file:"+str(video), "offline-guided-diagnostic", timestamp,
                                   source_dimensions=(raw.shape[1], raw.shape[0]), analysis_dimensions=(raw.shape[1], raw.shape[0]), source_frame_index=index)
            result = profile.process(raw, context)
            previous = before_profile.process(raw, context) if before_profile else None
            height = min(640, raw.shape[0])
            width = round(raw.shape[1]*height/raw.shape[0])
            width += width % 2
            height += height % 2
            before = cv2.resize(raw, (width, height), interpolation=cv2.INTER_AREA)
            after = cv2.resize(result.annotated_frame, (width, height), interpolation=cv2.INTER_AREA)
            panels = 3 if previous else 2
            canvas = np.zeros((height+70, width*panels, 3), np.uint8)
            canvas[70:, :width] = before
            canvas[70:, width*(panels-1):] = after
            cv2.putText(canvas, "ORIGINAL", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .5, (235, 235, 235), 1)
            if previous:
                canvas[70:, width:width*2] = cv2.resize(previous.annotated_frame, (width, height), interpolation=cv2.INTER_AREA)
                cv2.putText(canvas, "BEFORE RESEARCH MODEL", (width+8, 22), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 210, 255), 1)
            cv2.putText(canvas, "AFTER RESEARCH MODEL" if previous else "LOCAL RESEARCH MODEL", (width*(panels-1)+8, 22), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 210, 255), 1)
            cv2.putText(canvas, f"Frame {index} | source {timestamp/1000:.3f}s | {result.status}", (8, 48), cv2.FONT_HERSHEY_SIMPLEX, .43, (235, 235, 235), 1)
            if writer is None:
                writer = av.open(str(out / "original-vs-research.mp4"), "w", options={"movflags": "+faststart"})
                stream = writer.add_stream("libx264", rate=Fraction(str(fps)).limit_denominator(100000))
                stream.width, stream.height = canvas.shape[1], canvas.shape[0]
                stream.pix_fmt = "yuv420p"
                stream.options = {"crf": "20", "preset": "fast", "threads": "1"}
                stream.time_base = Fraction(1, 1_000_000)
                stream.codec_context.time_base = stream.time_base
            encoded_frame = av.VideoFrame.from_ndarray(canvas, format="bgr24")
            encoded_frame.time_base = Fraction(1, 1_000_000)
            first_timestamp = records[0]["source_time_ms"] if records else timestamp
            encoded_frame.pts = round((timestamp-first_timestamp)*1000)
            for packet in stream.encode(encoded_frame):
                writer.mux(packet)
            observations = result.domain_state["observations"]
            records.append({"source_frame_index": index, "source_time_ms": timestamp,
                "input_sha256": hashlib.sha256(raw.tobytes()).hexdigest(), "shape": list(raw.shape),
                "status": result.status, "observations": observations, "quality": result.domain_state["quality"],
                "before_observations": previous.domain_state["observations"] if previous else None,
                "before_status": previous.status if previous else None})
            if index in {0, declared//2, max(0, declared-1)}:
                cv2.imwrite(str(out / f"frame-{index:05d}.jpg"), canvas)
            if len(records) % 100 == 0:
                print("ACTUAL_FRAMES_INSPECTED", len(records), flush=True)
    finally:
        capture.release()
        if writer is not None:
            for packet in stream.encode():
                writer.mux(packet)
            writer.close()
    if len(records) != declared:
        raise ValueError(f"Source decode incomplete: {len(records)} vs declared {declared}")
    check = cv2.VideoCapture(str(out / "original-vs-research.mp4"))
    output_count = 0
    output_timestamps = []
    while check.read()[0]:
        output_timestamps.append(float(check.get(cv2.CAP_PROP_POS_MSEC)))
        output_count += 1
    check.release()
    if output_count != len(records):
        raise ValueError("Comparison output frame coverage mismatch")
    expected_times = np.array([row["source_time_ms"]-records[0]["source_time_ms"] for row in records])
    timestamp_error = float(np.max(np.abs(np.asarray(output_timestamps)-expected_times)))
    if not np.isfinite(timestamp_error) or timestamp_error > 1:
        raise ValueError(f"Encoded presentation timestamps differ from source by {timestamp_error:.6f}ms")
    report = {"scope": "Every decoded genuine video frame, original beside actual local research prediction; offline rendering, not live speed or accuracy. No training/calibration or owner-video threshold selection.",
        "render_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "adapter_sha256": hashlib.sha256((ROOT / "edge_runtime/profiles/construction_task_detector.py").read_bytes()).hexdigest(),
        "source": str(video), "source_sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
        "profile": profile.describe(), "native_fps": fps, "source_frames": declared, "output_frames": output_count,
        "before_profile": before_profile.describe() if before_profile else None,
        "timing": {"scope": "Original decoded presentation timestamps relative to first source frame, including variable frame intervals",
                   "maximum_encoded_timestamp_error_ms": timestamp_error, "tolerance_ms": 1},
        "offline_processing_seconds": time.perf_counter()-started,
        "candidate_frames": sum(bool(row["observations"]) for row in records),
        "ground_truth": "No independent frame-level labels; no precision/recall inferred", "frames": records}
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("source_frames", "output_frames", "candidate_frames", "offline_processing_seconds")}))


if __name__ == "__main__":
    main()
