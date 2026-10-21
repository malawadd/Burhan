"""Measure real-video runtime delivery; no accuracy is inferred from throughput."""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import time
from datetime import datetime, timezone

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from edge_runtime.runtime import EdgeRuntime, RuntimeConfig


def memory_mb():
    if sys.platform != "win32":
        return None
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
    values = Counters()
    values.cb = ctypes.sizeof(values)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    if psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(values), values.cb):
        return round(values.WorkingSetSize / 1024**2, 2)
    return None


def run(video, options, seconds, output, name):
    samples = []
    with tempfile.TemporaryDirectory(prefix="inspectflow-guided-benchmark-") as directory:
        edge = EdgeRuntime(directory)
        try:
            started = time.perf_counter()
            edge.start(RuntimeConfig(profile="construction", source="file:"+str(video), input_fps=30,
                preview_fps=30, deadline_ms=1000, video_mode="realtime", profile_options=options))
            startup_ms = (time.perf_counter()-started)*1000
            until = time.perf_counter()+seconds
            while time.perf_counter() < until:
                state = edge.snapshot()
                samples.append({"elapsed_s": round(time.perf_counter()-started, 4),
                    "metrics": state["metrics"], "working_set_mb": memory_mb(),
                    "inspection_sequence": (state.get("inspection_result") or {}).get("sequence")})
                if state["error"]:
                    raise RuntimeError(state["error"])
                if state["state"] == "eof":
                    break
                time.sleep(.05)
            edge.stop()
            final = edge.snapshot()
            inspected = edge.inspection_preview() if options.get("inspection_task") else edge.preview()
            if inspected:
                (output / (name+"-inspected.jpg")).write_bytes(inspected)
            ages = [s["metrics"]["evidence_age_ms"] for s in samples if s["metrics"]["evidence_age_ms"] is not None]
            age_p95 = round(float(np.percentile(ages, 95)), 2) if ages else None
            return {"profile_options": options, "startup_ms": round(startup_ms, 2),
                "metrics": final["metrics"], "media": final["media"], "inspection_media": final["inspection_media"],
                "evidence_age_p95_ms": age_p95,
                "working_set_peak_mb": max((s["working_set_mb"] or 0 for s in samples), default=0),
                "preview_target_pass": final["metrics"]["preview_fps"] >= 15,
                "decision_age_target_pass": age_p95 is not None and age_p95 <= 1000,
                "last_result": final["last_result"], "samples": samples}
        finally:
            edge.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=12)
    parser.add_argument("--task", choices=["walls_finishes", "brackets_railings", "tile_installation"], default="walls_finishes")
    parser.add_argument("--manifest", type=Path, help="Optional pinned research model for the after run; before remains historical crack model")
    parser.add_argument("--context", default="Not independently isolated; consult process receipts", help="Honest concurrent-work / measurement context")
    args = parser.parse_args()
    video, output = args.video.resolve(), args.output.resolve()
    if not video.is_file() or args.seconds <= 0:
        parser.error("An existing real local video and positive duration are required")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "report.json").exists():
        parser.error("Preserve the previous receipt; choose a new output directory")
    legacy = {"mode": "crack", "engine": "onnxruntime", "openvino_device": "CPU", "cpu_threads": 1, "stage": "all"}
    after = {"inspection_task": args.task}
    if args.manifest:
        after["research_model_manifest"] = str(args.manifest.resolve())
    elif args.task == "walls_finishes":
        after["research_model"] = legacy
    results = {"scope": "Runtime delivery benchmark, not accuracy or category validation. Same actual source file, no rendered defects, loops or repeated stills. Before uses legacy crack model. After uses its explicitly recorded profile options: the identical crack model for default walls, a different research model when --manifest is set, or geometry/guidance for other tasks.",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "same_model_comparison": not args.manifest and args.task == "walls_finishes",
        "measurement_context": args.context,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "runtime_sha256": hashlib.sha256((ROOT / "edge_runtime/runtime.py").read_bytes()).hexdigest(),
        "media_sha256": hashlib.sha256((ROOT / "edge_runtime/media.py").read_bytes()).hexdigest(),
        "model_manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest() if args.manifest else None,
        "video": str(video), "video_sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
        "host": platform.platform(), "processor": platform.processor(), "logical_cpu_count": os.cpu_count(),
        "requested_input_fps": 30, "requested_preview_fps": 30,
        "before": run(video, legacy, args.seconds, output, "before"),
        "after": run(video, after, args.seconds, output, "after")}
    (output / "report.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps({key: {"metrics": results[key]["metrics"], "working_set_peak_mb": results[key]["working_set_peak_mb"],
                           "evidence_age_p95_ms": results[key]["evidence_age_p95_ms"]} for key in ("before", "after")}, indent=2))


if __name__ == "__main__":
    main()
