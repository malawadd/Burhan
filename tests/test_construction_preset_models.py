"""A public source existing is insufficient when its pinned model is missing."""
import hashlib
import importlib
import json

from fastapi.testclient import TestClient


def test_preset_requires_exact_registered_model(tmp_path, monkeypatch):
    module = importlib.import_module("edge_runtime.app")
    registry = importlib.import_module("edge_runtime.construction_capabilities")
    for folder in ("edge_runtime", "config", "models/construction"):
        (tmp_path / folder).mkdir(parents=True, exist_ok=True)
    (tmp_path / "source.png").write_bytes(b"fixture path; not decoded")
    model = tmp_path / "models/construction/fixture.onnx"
    spec = {"file": "fixture.onnx", "sha256": hashlib.sha256(b"fixture model bytes").hexdigest(), "capabilities": []}
    monkeypatch.setattr(registry, "registered_models", lambda: {"damage:fixture": spec})
    monkeypatch.setattr(module, "__file__", str(tmp_path / "edge_runtime/app.py"))
    class Runtime:
        def __init__(self, *args): pass
        def close(self): pass
    monkeypatch.setattr(module, "EdgeRuntime", Runtime)
    (tmp_path / "config/public-presets.json").write_text(json.dumps([{
        "label": "Fixture", "profile": "construction", "path": "source.png",
        "profile_options": {"mode": "damage", "damage_variant": "fixture"}}]), encoding="utf8")
    with TestClient(module.create_app()) as client:
        assert client.get("/api/presets").json()[0]["available"] is False
        model.write_bytes(b"fixture model bytes")
        assert client.get("/api/presets").json()[0]["available"] is True
        model.write_bytes(b"changed model bytes")
        preset = client.get("/api/presets").json()[0]
        assert preset["available"] is False and preset["model_available"] is False
