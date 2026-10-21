"""Exercise the real loopback service. Camera probe stores/prints no image pixels."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import httpx


def wait_for(client, predicate, timeout=12):
    end = time.perf_counter() + timeout
    while time.perf_counter() < end:
        state = client.get("/api/state").json()
        if state["error"]:
            raise RuntimeError(state["error"])
        if predicate(state):
            return state
        time.sleep(.1)
    raise RuntimeError("Station did not reach expected state")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument("--camera", action="store_true")
    parser.add_argument("--output", default="output/evaluation/integration-smoke.json")
    args = parser.parse_args()
    report = {"recorded_at_utc": datetime.now(timezone.utc).isoformat(), "checks": [], "states": {}}
    with httpx.Client(base_url=args.base_url, timeout=20, trust_env=False) as client:
        for profile in ("factory", "freight", "roads"):
            client.post("/api/start", json={"profile": profile, "source": "demo", "input_fps": 8, "deadline_ms": 500}).raise_for_status()
            state = wait_for(client, lambda s: s["metrics"]["processed"] >= 12)
            assert client.get("/api/preview.jpg").headers["content-type"] == "image/jpeg"
            assert state["metrics"]["inspection_errors"] == 0
            assert state["last_result"]["provenance"] == "synthetic_demo"
            report["checks"].append({"name": f"{profile}_local_stream", "passed": True})
            report["states"][profile] = {"metrics": state["metrics"], "status": state["last_result"]["status"],
                                          "profile_version": state["last_result"].get("profile_version")}
            client.post("/api/stop", json={}).raise_for_status()
        if args.camera:
            client.post("/api/start", json={"profile": "freight", "source": "camera:0", "input_fps": 8, "save_evidence": False}).raise_for_status()
            state = wait_for(client, lambda s: s["metrics"]["processed"] >= 3, timeout=20)
            assert not state["recording"]
            report["checks"].append({"name": "laptop_camera_stream", "passed": True, "saved_media": False,
                                      "backend": state["description"].get("capture_backend"),
                                      "processed": state["metrics"]["processed"]})
            client.post("/api/stop", json={}).raise_for_status()
        assert client.get("/api/export").status_code == 200
        assert client.post("/api/start", json={}, headers={"origin": "https://untrusted.invalid"}).status_code == 403
        report["checks"].append({"name": "export_and_origin_guard", "passed": True})
        # Leave a labelled synthetic source for the user, not an active private camera.
        client.post("/api/start", json={"profile": "factory", "source": "demo", "input_fps": 8}).raise_for_status()
        wait_for(client, lambda s: s["metrics"]["processed"] >= 2)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"checks": report["checks"], "report": str(path)}, indent=2))


if __name__ == "__main__":
    main()
