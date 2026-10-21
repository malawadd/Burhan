"""HTTP byte fixtures validate delivery only; they are not model/video evidence."""
import importlib
import json

from fastapi.testclient import TestClient
import pytest


@pytest.fixture
def recording_client(tmp_path, monkeypatch):
    app_module = importlib.import_module("edge_runtime.app")
    (tmp_path / "edge_runtime").mkdir()
    (tmp_path / "config").mkdir()
    (tmp_path / "output").mkdir()
    # No model, camera or database starts during HTTP contract tests.
    class Runtime:
        def __init__(self, *args): pass
        def close(self): pass
    monkeypatch.setattr(app_module, "__file__", str(tmp_path / "edge_runtime/app.py"))
    monkeypatch.setattr(app_module, "EdgeRuntime", Runtime)
    blob = b"0123456789abcdef"
    (tmp_path / "output/registered.mp4").write_bytes(blob)
    (tmp_path / "output/not-registered.mp4").write_bytes(b"private")
    (tmp_path / "private.mp4").write_bytes(b"outside output")
    (tmp_path / "output/not-video.txt").write_text("not a video", encoding="utf-8")
    records = [
        {"id":"allowed", "label":"test", "path":"output/registered.mp4"},
        {"id":"outside", "label":"test", "path":"private.mp4"},
        {"id":"traversal", "label":"test", "path":"output/../private.mp4"},
        {"id":"wrong-type", "label":"test", "path":"output/not-video.txt"},
        {"id":"missing", "label":"test", "path":"output/missing.mp4"},
    ]
    (tmp_path / "config/construction-videos.json").write_text(json.dumps(records), encoding="utf-8")
    with TestClient(app_module.create_app()) as client:
        yield client, blob


def test_recording_inventory_matches_delivery_allowlist(recording_client):
    client, blob = recording_client
    inventory = client.get("/api/construction/videos").json()
    assert {row["id"]: row["available"] for row in inventory} == {
        "allowed":True, "outside":False, "traversal":False, "wrong-type":False, "missing":False}
    assert all("path" not in row for row in inventory)
    for identifier in ("outside", "traversal", "wrong-type", "missing", "not-registered.mp4"):
        assert client.get("/api/construction/videos/" + identifier).status_code == 404
    response = client.get("/api/construction/videos/allowed")
    assert response.status_code == 200 and response.content == blob
    assert response.headers["content-type"] == "video/mp4"
    assert "media-src 'self'" in response.headers["content-security-policy"]


def test_registered_video_supports_browser_range_seeking_and_head(recording_client):
    client, blob = recording_client
    response = client.get("/api/construction/videos/allowed", headers={"Range":"bytes=4-7"})
    assert response.status_code == 206 and response.content == blob[4:8]
    assert response.headers["content-range"] == f"bytes 4-7/{len(blob)}"
    assert response.headers["accept-ranges"] == "bytes"
    suffix = client.get("/api/construction/videos/allowed", headers={"Range":"bytes=-4"})
    assert suffix.status_code == 206 and suffix.content == blob[-4:]
    unavailable = client.get("/api/construction/videos/allowed", headers={"Range":"bytes=100-200"})
    assert unavailable.status_code == 416
    assert unavailable.headers["content-range"] == f"*/{len(blob)}"
    head = client.head("/api/construction/videos/allowed")
    assert head.status_code == 200 and head.content == b""
    assert int(head.headers["content-length"]) == len(blob)
