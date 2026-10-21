"""Verify repaired tile geometry through the actual running local HTTP app.

A licensed public photograph is an engineering input, not video/accuracy data.
Restores the previous local file configuration and leaves the app stopped.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/evaluation/construction-resumption-geometry/live-api-v1"
URL = "http://127.0.0.1:8765"


def request(path, payload=None):
    req = Request(URL+path, data=json.dumps(payload).encode() if payload is not None else None,
        headers={"Content-Type": "application/json"} if payload is not None else {})
    with urlopen(req, timeout=10) as response:
        return response.read()


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="live-api-v1")
    args = parser.parse_args()
    if not args.run.replace("-", "").isalnum():
        raise ValueError("Invalid verification version")
    OUT = OUT.parent / args.run
    if OUT.exists():
        raise ValueError("Preserve prior live verification")
    OUT.mkdir(parents=True)
    previous = json.loads((ROOT / "output/construction-resumption-app-before-restart.json").read_text())
    if previous["running"] or previous["config"]["source"].startswith("camera:"):
        raise ValueError("This diagnostic expects previous stopped local-file configuration")
    image = ROOT / "datasets/construction/installed-floor-diagnostics-v1/Cracked_tile_flooring.jpg"
    from edge_runtime.sources import LocalSource
    source = LocalSource("image:"+str(image), "construction", 10, 640, 360)
    source.open()
    try:
        actual_frame, _ = source.read()
        analysis_wh = list(actual_frame.shape[1::-1])
    finally:
        source.close()
    protocol = {"source": str(image), "source_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
        "actual_source_analysis_wh": analysis_wh,
        "profile_sha256": hashlib.sha256((ROOT / "edge_runtime/profiles/construction_tasks.py").read_bytes()).hexdigest(),
        "scope": "Actual HTTP/runtime/JPEG execution on public still; no fault/measurement accuracy or video FPS claim",
        "photo_author": "Malcolm Koo / Mk2010", "license": "CC BY-SA 4.0",
        "source_url": "https://commons.wikimedia.org/wiki/File:Cracked_tile_flooring.jpg"}
    (OUT / "protocol-before-run.json").write_text(json.dumps(protocol, indent=2))
    result = None
    try:
        request("/api/start", {"profile": "construction", "source": "image:"+str(image),
            "input_fps": 10, "preview_fps": 15, "width": 640, "height": 360,
            "deadline_ms": 1000, "save_evidence": False, "source_hold_seconds": 10,
            "profile_options": {"inspection_task": "tile_installation"}})
        started = time.monotonic()
        while time.monotonic()-started < 15:
            state = json.loads(request("/api/state"))
            if state.get("error"):
                raise RuntimeError(state["error"])
            result = state.get("inspection_result")
            if result and result.get("domain_state", {}).get("observations"):
                break
            time.sleep(.1)
        else:
            raise RuntimeError("No actual paired geometry observation within 15 seconds")
        data = request("/api/inspection.jpg?"+urlencode({"sequence": result["sequence"], "session_id": result["session_id"]}))
        (OUT / "actual-http-inspected.jpg").write_bytes(data)
        (OUT / "state-with-paired-evidence.json").write_text(json.dumps(state, indent=2))
        print("ACTUAL_PAIRED_TILE_OBSERVATIONS", len(result["domain_state"]["observations"]))
    finally:
        request("/api/stop", {})
        # Restore only the already authorized previous local-file configuration.
        request("/api/start", previous["config"])
        request("/api/stop", {})
        final = json.loads(request("/api/state"))
        if final["running"] or final["config"] != previous["config"]:
            raise RuntimeError("Failed to restore previous stopped configuration")
        (OUT / "restored-stopped-state.json").write_text(json.dumps(final, indent=2))
    print("VERIFIED_AND_RESTORED", OUT)


if __name__ == "__main__":
    main()
