"""Loopback-only dashboard and local control API."""
from contextlib import asynccontextmanager
from dataclasses import fields
import json
import math
import os
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .runtime import EdgeRuntime, RuntimeConfig
from .sources import create_profile


class StartRequest(BaseModel):
    profile: Literal["factory", "freight", "roads", "construction"] = "factory"
    source: str = Field(default="demo", max_length=1024)
    input_fps: float = Field(default=8, ge=1, le=30)
    width: int = Field(default=960, ge=320, le=1920)
    height: int = Field(default=540, ge=240, le=1080)
    deadline_ms: float = Field(default=350, ge=20, le=5000)
    save_evidence: bool = False
    triggered: bool = False
    profile_options: dict = Field(default_factory=dict)
    source_hold_seconds: float = Field(default=30, ge=1, le=300)
    preview_fps: float = Field(default=6, ge=0, le=30)
    video_mode: Literal["realtime", "every_frame"] = "realtime"


def create_app(data_dir=None):
    @asynccontextmanager
    async def lifespan(app):
        app.state.runtime = EdgeRuntime(data_dir or os.getenv("INSPECTFLOW_DATA_DIR", "runtime-data"))
        yield
        app.state.runtime.close()

    app = FastAPI(title="InspectFlow Edge", version="0.1.0", lifespan=lifespan)

    @app.middleware("http")
    async def local_control(request: Request, call_next):
        host = request.url.hostname
        if host not in {"localhost", "127.0.0.1", "::1", "testserver"}:
            return JSONResponse({"detail": "Local dashboard host required"}, status_code=403)
        if request.method in {"POST", "PUT", "PATCH", "DELETE"}:
            origin = request.headers.get("origin")
            if origin and urlparse(origin).netloc != request.url.netloc:
                return JSONResponse({"detail": "Cross-origin device control denied"}, status_code=403)
            if "application/json" not in request.headers.get("content-type", ""):
                return JSONResponse({"detail": "Use application/json"}, status_code=415)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Cache-Control"] = "no-store"
        response.headers["Content-Security-Policy"] = "default-src 'self'; img-src 'self' blob: data:; media-src 'self'; style-src 'self'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'"
        return response

    static = Path(__file__).parent / "static"
    static.mkdir(exist_ok=True)
    app.mount("/static", StaticFiles(directory=static), name="static")

    @app.get("/")
    def index():
        return FileResponse(static / "index.html")

    @app.get("/api/state")
    def state():
        return app.state.runtime.snapshot()

    @app.get("/api/profiles")
    def profiles():
        return [create_profile(name).describe() for name in ("factory", "freight", "roads")] + [
            {"profile":"construction", "name":"Construction visual review", "real_inputs_only":True,
             "scope":"Explicitly supported visual candidates; Saudi code compliance is not determined"}]

    @app.get("/api/construction/catalog")
    def construction_catalog():
        from .construction_catalog import load_catalog
        return load_catalog()

    @app.get("/api/construction/capabilities")
    def construction_capabilities():
        from .construction_capabilities import load_capabilities
        return load_capabilities()

    @app.get("/api/construction/tasks")
    def construction_tasks():
        from .profiles.construction_tasks import task_catalog
        from .profiles.construction_task_detector import discover_models
        catalog = task_catalog()
        catalog["research_models"] = discover_models()
        for task in catalog["tasks"]:
            supported = {label for model in catalog["research_models"] if model["inspection_task"] == task["id"] for label in model["classes"]}
            for category in task["categories"]:
                if category["id"] in supported:
                    category["status"] = "research_only"
                    category["reason"] = "A local trained research model is installed; independent real-video release gates have not passed. " + category["reason"]
        return catalog

    def construction_videos():
        root = Path(__file__).resolve().parent.parent
        registry = root / "config" / "construction-videos.json"
        return root, json.loads(registry.read_text(encoding="utf-8")) if registry.is_file() else []

    def recorded_video_path(root, row, key="path", suffix=".mp4"):
        try:
            path = (root / row[key]).resolve()
            if path.is_relative_to((root / "output").resolve()) and path.suffix.lower() == suffix and path.is_file():
                return path
        except (KeyError, TypeError, ValueError, OSError, RuntimeError):
            pass
        return None

    @app.get("/api/construction/videos")
    def video_inventory():
        root, rows = construction_videos()
        return [{**{key:value for key,value in row.items() if key not in {"path", "evidence_path"}},
                 "purpose": row.get("purpose", "failure_diagnostic"),
                 "available": recorded_video_path(root, row) is not None,
                 "evidence_available": recorded_video_path(root, row, "evidence_path", ".json") is not None,
                 "evidence_url": "/api/construction/videos/"+row["id"]+"/evidence",
                 "url": "/api/construction/videos/"+row["id"]} for row in rows]

    @app.get("/api/construction/videos/{identifier}/evidence")
    def recording_evidence(identifier: str):
        root, rows = construction_videos()
        row = next((item for item in rows if item["id"] == identifier), None)
        path = recorded_video_path(root, row or {}, "evidence_path", ".json")
        if path is None:
            raise HTTPException(404, "This recording has no available annotated evaluation")
        if path.stat().st_size > 4 * 1024 * 1024:
            raise HTTPException(409, "Recording evidence exceeds the supported size")
        try:
            evidence = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeError, ValueError, RecursionError) as error:
            raise HTTPException(409, "Recording evidence is unreadable") from error
        if not valid_recording_evidence(evidence):
            raise HTTPException(409, "Recording evidence has an unsupported structure")
        return JSONResponse(evidence)

    def valid_recording_evidence(evidence):
        """Validate display evidence without treating metadata as a verified finding.

        A claimed match additionally has to agree with the actual two boxes at
        the declared IoU threshold. Source annotations remain separate inputs.
        """
        def number(value, minimum=0, maximum=None):
            try:
                return (isinstance(value, (int, float)) and not isinstance(value, bool)
                        and math.isfinite(value) and value >= minimum
                        and (maximum is None or value <= maximum))
            except OverflowError:
                return False

        def integer(value):
            return isinstance(value, int) and not isinstance(value, bool) and value >= 0

        def identifier(value):
            return isinstance(value, str) and bool(value) or integer(value)

        def finite_tree(value, depth=0):
            if depth > 24:
                return False
            if isinstance(value, float) and not math.isfinite(value):
                return False
            if isinstance(value, list):
                return all(finite_tree(item, depth + 1) for item in value)
            if isinstance(value, dict):
                return all(finite_tree(item, depth + 1) for item in value.values())
            return True

        def dimensions(value):
            return isinstance(value, list) and len(value) == 2 and all(integer(item) and item > 0 for item in value)

        def box(value, size):
            if not isinstance(value, list) or len(value) != 4:
                return False
            if not all(number(item, -1e-6) for item in value) or value[2] <= 0 or value[3] <= 0:
                return False
            return size is None or value[0] + value[2] <= size[0] + 1e-6 and value[1] + value[3] <= size[1] + 1e-6

        def overlap(first, second):
            x = max(0, min(first[0] + first[2], second[0] + second[2]) - max(first[0], second[0]))
            y = max(0, min(first[1] + first[3], second[1] + second[3]) - max(first[1], second[1]))
            intersection = x * y
            return intersection / (first[2] * first[3] + second[2] * second[3] - intersection)

        if not isinstance(evidence, dict) or not isinstance(evidence.get("frames"), list) or not finite_tree(evidence):
            return False
        size = evidence.get("frame_size_wh")
        if size is not None and not dimensions(size):
            return False
        threshold = evidence.get("matching_iou_threshold", .5)
        if not number(threshold, 0, 1) or threshold <= 0:
            return False
        tolerance = evidence.get("frame_time_tolerance_s", .08)
        if not number(tolerance, 0, .5):
            return False
        summary = evidence.get("summary", {})
        if not isinstance(summary, dict):
            return False
        # Private camera diagnostics can have no annotation set at all. Unknown
        # metrics must remain null, rather than manufacturing zero errors/IoU.
        wholly_unscored = bool(evidence["frames"]) and all(
            isinstance(frame, dict) and frame.get("evaluated") is False
            and frame.get("expected") == [] and isinstance(frame.get("predictions"), list)
            and all(isinstance(item, dict) and item.get("evaluation_status") == "unscored"
                    and item.get("matched_annotation_id") is None for item in frame["predictions"])
            for frame in evidence["frames"])
        for key in ("annotated_frames", "localized_matches", "missed_defects", "false_alerts", "unsupported_defects", "unsupported_annotations"):
            if key in summary and not integer(summary[key]) and not (wholly_unscored and summary[key] is None):
                return False
        previous_time = -1
        for frame in evidence["frames"]:
            if not isinstance(frame, dict) or not number(frame.get("video_time_s")):
                return False
            if frame["video_time_s"] < previous_time:
                return False
            previous_time = frame["video_time_s"]
            if "source_frame_index" in frame and not integer(frame["source_frame_index"]):
                return False
            if "evaluated" in frame and not isinstance(frame["evaluated"], bool):
                return False
            frame_size = frame.get("frame_size_wh", size)
            if frame_size is not None and not dimensions(frame_size):
                return False
            expected, predictions = frame.get("expected"), frame.get("predictions")
            if not isinstance(expected, list) or not isinstance(predictions, list):
                return False
            annotations = {}
            for item in expected:
                if (not isinstance(item, dict) or not identifier(item.get("id"))
                        or item["id"] in annotations or not isinstance(item.get("label"), str)
                        or not item["label"] or not box(item.get("bbox"), frame_size)):
                    return False
                annotations[item["id"]] = item
            matched_ids = set()
            for item in predictions:
                if (not isinstance(item, dict) or not isinstance(item.get("label"), str) or not item["label"]
                        or not box(item.get("bbox"), frame_size) or not number(item.get("score"), 0, 1)):
                    return False
                unknown_match = (wholly_unscored and item.get("evaluation_status") == "unscored"
                                 and item.get("iou") is None and item.get("matched_annotation_id") is None)
                if not unknown_match and (not number(item.get("iou"), 0, 1) or "matched_annotation_id" not in item):
                    return False
                if "id" in item and not identifier(item["id"]):
                    return False
                status = item.get("evaluation_status")
                if status is not None and (not isinstance(status, str) or status not in {"matched", "false_alert", "unscored"}):
                    return False
                matched = item.get("matched_annotation_id")
                if matched is None:
                    if status == "matched":
                        return False
                    continue
                if not identifier(matched) or matched not in annotations or matched in matched_ids:
                    return False
                truth = annotations[matched]
                actual_iou = overlap(item["bbox"], truth["bbox"])
                if (frame.get("evaluated") is not True or truth.get("evaluation_status") == "unsupported"
                        or status in {"false_alert", "unscored"} or item["label"] != truth["label"]
                        or actual_iou + 1e-8 < threshold or abs(actual_iou - item["iou"]) > 1e-5):
                    return False
                matched_ids.add(matched)
        return True

    @app.get("/api/construction/videos/{identifier}")
    @app.head("/api/construction/videos/{identifier}", include_in_schema=False)
    def recorded_video(identifier: str):
        root, rows = construction_videos()
        row = next((item for item in rows if item["id"] == identifier),None)
        if row is None:
            raise HTTPException(404,"Unknown public-video artifact")
        path = recorded_video_path(root, row)
        if path is None:
            raise HTTPException(404,"Recorded public-video artifact is unavailable")
        return FileResponse(path,media_type="video/mp4")

    @app.get("/api/presets")
    def presets():
        """Explicit public-data examples, never selected or started automatically."""
        root = Path(__file__).resolve().parent.parent
        registry = root / "config" / "public-presets.json"
        if not registry.is_file():
            return []
        result = []
        construction_specs = None
        verified_construction = {}
        for preset in json.loads(registry.read_text(encoding="utf-8")):
            item = dict(preset)
            path = root / item.pop("path")
            item["source"] = item.pop("source_kind", "image") + ":" + str(path)
            item["available"] = path.exists()
            options = dict(item.get("profile_options", {}))
            for key in ("model_path", "defect_model_path", "reference_path"):
                if isinstance(options.get(key), str):
                    local = Path(options[key])
                    options[key] = str(local if local.is_absolute() else root / local)
                    item["available"] = item["available"] and Path(options[key]).is_file()
            if item.get("profile") == "construction":
                if options.get("inspection_task") is not None:
                    from .profiles.construction_tasks import TASK_CATALOG
                    from .profiles.construction_task_detector import read_manifest
                    task = options.get("inspection_task")
                    supported = isinstance(task, str) and task in TASK_CATALOG
                    model_available = False
                    if options.get("research_model_manifest") is not None:
                        try:
                            _, _, spec, _ = read_manifest(options["research_model_manifest"])
                            model_available = spec.get("inspection_task") == task
                        except (ValueError, OSError, TypeError):
                            pass
                        supported = supported and model_available
                    item["model_available"] = model_available
                    item["available"] = item["available"] and supported
                    item["profile_options"] = options
                    result.append(item)
                    continue
                if construction_specs is None:
                    from .construction_capabilities import registered_models
                    construction_specs = registered_models()
                mode = options.get("mode", "ppe")
                reference = (f"damage:{options.get('damage_variant')}" if mode == "damage" else
                             f"crack:{options.get('crack_variant', 'baseline')}" if mode == "crack" else
                             mode if isinstance(mode, str) else None)
                if reference not in verified_construction:
                    spec = construction_specs.get(reference)
                    installed = False
                    if spec:
                        import hashlib
                        model_root = (root / "models/construction").resolve()
                        asset = (model_root / spec["file"]).resolve()
                        installed = (asset.is_relative_to(model_root) and asset.is_file()
                                     and hashlib.sha256(asset.read_bytes()).hexdigest() == spec["sha256"])
                    verified_construction[reference] = installed
                item["model_available"] = verified_construction[reference]
                item["available"] = item["available"] and item["model_available"]
            item["profile_options"] = options
            result.append(item)
        return result

    @app.post("/api/start")
    def start(body: StartRequest):
        if body.source != "demo" and not body.source.startswith(("camera:", "file:", "image:", "images:", "pdf:")):
            raise HTTPException(422, "Use a local camera, video, image, image folder, PDF or demo source")
        if body.source.startswith("camera:"):
            try:
                if not 0 <= int(body.source[7:]) <= 16:
                    raise ValueError()
            except ValueError:
                raise HTTPException(422, "Camera source must be camera:0 through camera:16")
        if body.triggered and body.profile != "factory":
            raise HTTPException(422, "Triggered mode is for the factory profile")
        try:
            app.state.runtime.start(RuntimeConfig(**body.model_dump()))
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(409, str(exc))
        return app.state.runtime.snapshot()

    @app.post("/api/stop")
    def stop():
        try:
            app.state.runtime.stop()
        except RuntimeError as exc:
            raise HTTPException(409, str(exc))
        return app.state.runtime.snapshot()

    @app.post("/api/actions/{action}")
    def action(action: str):
        try:
            return app.state.runtime.action(action)
        except ValueError as exc:
            raise HTTPException(409, str(exc))

    @app.get("/api/preview.jpg")
    def preview():
        jpeg, sequence, session = app.state.runtime.preview_frame()
        if jpeg is None:
            raise HTTPException(503, "Waiting for a local frame")
        return Response(jpeg, media_type="image/jpeg", headers={
            "X-Frame-Sequence": str(sequence), "X-Session-Id": str(session)})

    @app.get("/api/inspection.jpg")
    def inspection_preview(sequence: int | None = None, session_id: str | None = None):
        try:
            jpeg = app.state.runtime.inspection_preview(sequence, session_id)
        except ValueError as exc:
            raise HTTPException(409, str(exc))
        if jpeg is None:
            raise HTTPException(503, "Waiting for an inspected frame")
        return Response(jpeg, media_type="image/jpeg")

    @app.get("/api/events")
    def events(limit: int = 30):
        return app.state.runtime.store.list(max(1, min(200, limit)))

    @app.get("/api/export")
    def export():
        return JSONResponse({"runtime": app.state.runtime.snapshot(), "events": app.state.runtime.store.list(200),
                             "note": "Local prototype export. Synthetic/demo inputs are not field validation."},
                            headers={"Content-Disposition": 'attachment; filename="inspectflow-events.json"'})

    return app


app = create_app()
