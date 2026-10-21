"""Sequential synthetic profile checks and process-only CPU timing.

No downloads, network, camera or cloud calls. Scenario indices render pixels;
profiles receive only the BGR frame and source-clock FrameContext.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from edge_runtime.contracts import FrameContext
from edge_runtime.profiles import factory, freight, roads


def source_context(sequence: int, fps: float, profile: str, session: str) -> FrameContext:
    """Synthetic source clock, intentionally separate from processing wall time."""
    return FrameContext(sequence=sequence, captured_monotonic=1000.0 + sequence / fps,
                        source_id=f"synthetic-{profile}", session_id=session,
                        source_time_ms=1000.0 * sequence / fps)


def profile_instance(name: str, width: int, height: int):
    if name == "factory":
        return factory.FactoryProfile({"demo": True, "reference": factory.demo_frame(0, width, height)})
    return freight.FreightProfile() if name == "freight" else roads.RoadsProfile()


def sample_rss() -> int | None:
    try:
        import psutil
        return int(psutil.Process().memory_info().rss)
    except (ImportError, OSError):
        return None


def functional_checks(width: int, height: int, fps: float) -> dict:
    """Known synthetic cases check behavior; labels never enter profile APIs."""
    checks = {}
    f = profile_instance("factory", width, height)
    checks["factory_clean"] = f.process(factory.demo_frame(0, width, height), source_context(0, fps, "factory", "checks")).status == "clear"
    for label, index in [("omitted_component", 60), ("foreign_scratch", 120), ("displaced_component", 180)]:
        f.reset()
        results = [f.process(factory.demo_frame(index, width, height), source_context(i, fps, "factory", "checks")) for i in range(3)]
        checks[f"factory_{label}"] = results[-1].status == "fault" and bool(results[-1].findings)
    uncalibrated = factory.FactoryProfile()
    checks["factory_uncalibrated_not_clear"] = uncalibrated.process(factory.demo_frame(60, width, height), source_context(0, fps, "factory", "checks")).status == "needs_input"
    packet = freight.FreightProfile()
    for i in range(12):
        result = packet.process(freight.demo_frame(30, width, height), source_context(i, fps, "freight", "checks"))
    checks["freight_first_page_captured"] = "delivery_note" in result.domain_state.get("captured_pages", [])
    for i in range(12):
        result = packet.process(freight.demo_frame(150, width, height), source_context(i + 12, fps, "freight", "checks"))
    checks["freight_two_page_checklist_complete"] = bool(result.domain_state.get("complete"))
    empty = np.zeros((height, width, 3), np.uint8)
    checks["freight_blank_not_accepted"] = freight.FreightProfile().process(empty, source_context(0, fps, "freight", "checks")).status != "accepted"
    road = roads.RoadsProfile()
    for i in range(3):
        result = road.process(roads.demo_frame(i, width, height), source_context(i, fps, "roads", "checks"))
    checks["roads_dark_candidate_review"] = result.status == "review" and bool(result.findings)
    clean_road = roads.RoadsProfile().process(roads.demo_frame(99, width, height), source_context(0, fps, "roads", "checks"))
    checks["roads_candidate_outside_view_not_review"] = clean_road.status == "clear"
    return {"passed": all(checks.values()), "cases": checks,
            "scope": "Self-authored synthetic functional checks; not recall, precision or field accuracy."}


def scenario_index(name: str, index: int, frames: int) -> tuple[int, str]:
    if name == "factory":
        scenes = [(0, "clean"), (60, "omitted_component"), (0, "clean"),
                  (120, "foreign_scratch"), (0, "clean"), (180, "displaced_component"), (0, "clean")]
        part = min(6, index * 7 // frames)
        return scenes[part]
    if name == "freight":
        half = frames // 2
        if index < half:
            return min(index, 30), "delivery_note_capture_sequence"
        return 120 + min(index - half, 30), "packing_list_capture_sequence"
    return index, "moving_synthetic_dark_region_and_elongated_shadow"


def benchmark(name: str, frames: int, warmup: int, width: int, height: int, fps: float) -> dict:
    module = {"factory": factory, "freight": freight, "roads": roads}[name]
    # Warm a separate instance; measured event/checklist/tracking state starts empty.
    warmer = profile_instance(name, width, height)
    for i in range(warmup):
        warmer.process(module.demo_frame(0, width, height), source_context(i, fps, name, "warmup"))
    profile = profile_instance(name, width, height)
    times_ms, rss_values = [], []
    statuses, scenarios = Counter(), Counter()
    events = findings = 0
    event_keys = []
    final_state = {}
    started = time.perf_counter()
    for i in range(frames):
        scene_index, label = scenario_index(name, i, frames)
        frame = module.demo_frame(scene_index, width, height)
        context = source_context(i, fps, name, "measured")
        before = time.perf_counter_ns()
        result = profile.process(frame, context)
        times_ms.append((time.perf_counter_ns() - before) / 1_000_000.0)
        statuses[result.status] += 1
        scenarios[label] += 1
        findings += len(result.findings)
        if result.emit_event:
            events += 1
            event_keys.append(result.event_key)
        final_state = result.domain_state
        rss = sample_rss()
        if rss is not None:
            rss_values.append(rss)
    whole_loop_seconds = time.perf_counter() - started
    duration = sum(times_ms) / 1000.0
    return {"frames": frames, "warmup_frames_separate_instance": warmup,
            "input_dimensions": {"width": width, "height": height},
            "synthetic_source_fps": fps, "source_clock_duration_seconds": frames / fps,
            "paced_replay": False, "sequence_labels_for_evaluation_only": dict(scenarios),
            "status_counts": dict(statuses), "event_count": events,
            "event_keys": event_keys, "finding_count_across_frames_not_unique_defects": findings,
            "processing_ms": {"p50": round(float(np.percentile(times_ms, 50)), 3),
                              "p95": round(float(np.percentile(times_ms, 95)), 3),
                              "p99": round(float(np.percentile(times_ms, 99)), 3),
                              "maximum": round(max(times_ms), 3),
                              "mean": round(float(np.mean(times_ms)), 3)},
            "sum_process_seconds": round(duration, 4),
            "process_only_frames_per_second": round(frames / duration, 3),
            "whole_loop_wall_seconds_including_generation": round(whole_loop_seconds, 4),
            "maximum_sampled_process_rss_bytes": max(rss_values) if rss_values else None,
            "rss_note": "Optional psutil sampling after each frame; whole Python process, not native peak or per-profile allocation.",
            "final_domain_state": final_state,
            "profile_description": profile.describe()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=240, help="Measured frames per profile (minimum 64).")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--source-fps", type=float, default=30.0, help="Synthetic source timestamps only; replay is unpaced.")
    parser.add_argument("--output", type=Path, default=Path("output/evaluation/profile-baselines.json"))
    args = parser.parse_args()
    if args.frames < 64 or args.warmup < 0 or min(args.width, args.height) < 240 or not np.isfinite(args.source_fps) or args.source_fps <= 0:
        parser.error("Use frames >=64, warmup >=0, dimensions >=240 and a finite positive source-fps.")
    checks = functional_checks(args.width, args.height, args.source_fps)
    report = {"generated_at_utc": datetime.now(timezone.utc).isoformat(),
              "evidence_type": "self_authored_synthetic_only",
              "no_external_dataset_or_real_accuracy_claim": True,
              "timing_scope": "Profile.process only, including its annotations. Excludes generation, capture, queue, API, storage, AWS and actuation.",
              "limitations": ["Unpaced serial replay is not a production real-time, deadline, end-to-end latency or thermal test.",
                              "Scenario source timestamps are synthetic and independent of measured wall-clock process duration.",
                              "No model receives evaluation labels or scenario indices; only rendered BGR pixels and source-clock context.",
                              "All scenes are self-authored; event counts reflect each profile's persistence/deduplication policy, not comparable domain accuracy.",
                              "Sequential profiles in one process may share native caches; host load and OpenCV threading affect results."],
              "environment": {"python": platform.python_version(), "opencv": cv2.__version__,
                              "numpy": np.__version__, "platform": platform.platform(),
                              "processor": platform.processor(), "logical_cpu_count": os.cpu_count(),
                              "opencv_threads": cv2.getNumThreads()},
              "functional_checks": checks, "profiles": {}}
    for name in ("factory", "freight", "roads"):
        report["profiles"][name] = benchmark(name, args.frames, args.warmup, args.width, args.height, args.source_fps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Functional checks: {'PASS' if checks['passed'] else 'FAIL'}")
    for name, data in report["profiles"].items():
        timing = data["processing_ms"]
        print(f"{name}: {data['frames']} frames; p50={timing['p50']} ms p95={timing['p95']} ms p99={timing['p99']} ms max={timing['maximum']} ms; events={data['event_count']}; statuses={data['status_counts']}")
    print(f"Saved: {args.output.resolve()}")
    return 0 if checks["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
