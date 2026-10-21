"""Catalog claims require an actual registered model with the correct bytes."""
import hashlib
import json

from edge_runtime.construction_capabilities import load_capabilities


def test_supported_category_requires_matching_asset_and_capability(tmp_path):
    models = tmp_path / "models/construction"
    models.mkdir(parents=True)
    (models / "weights.onnx").write_bytes(b"test fixture, not a model")
    spec = {"file": "weights.onnx", "sha256": hashlib.sha256(b"test fixture, not a model").hexdigest(),
            "capabilities": ["surface_spalling"]}
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps({"categories": [
        {"id": "surface_spalling", "model_ref": "damage:fixture", "model_status": "supported_experimental"},
        {"id": "hidden_rebar", "model_ref": "damage:fixture", "model_status": "supported_experimental"},
        {"id": "moisture", "model_ref": "unknown", "model_status": "supported_experimental"},
        {"id": "research", "model_status": "research_only"},
        {"id": "surface_spalling", "model_ref": "damage:fixture", "model_status": "measured_weak"},
    ]}), encoding="utf8")
    result = load_capabilities(path, root=tmp_path, model_specs={"damage:fixture": spec})
    assert [row["effective_status"] for row in result["categories"]] == [
        "supported_experimental", "unavailable", "unavailable", "research_only", "measured_weak"]
    (models / "weights.onnx").write_bytes(b"changed bytes")
    assert load_capabilities(path, root=tmp_path, model_specs={"damage:fixture": spec})["categories"][0]["effective_status"] == "unavailable"
    assert result["code_compliance"] == "not_determined"


def test_model_registry_cannot_escape_construction_model_folder(tmp_path):
    outside = tmp_path / "private.onnx"
    outside.write_bytes(b"not an allowed model")
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps({"categories": [{"id": "crack", "model_ref": "damage:escape",
                                               "model_status": "supported_experimental"}]}), encoding="utf8")
    specs = {"damage:escape": {"file": "../../private.onnx", "sha256": hashlib.sha256(outside.read_bytes()).hexdigest(), "capabilities": ["crack"]}}
    assert not load_capabilities(path, root=tmp_path, model_specs=specs)["categories"][0]["model_installed"]


def test_failed_calibration_asset_cannot_be_promoted_by_catalog_wording(tmp_path):
    directory = tmp_path / "models/construction"
    directory.mkdir(parents=True)
    asset = directory / "fixture.onnx"
    asset.write_bytes(b"not a real NN")
    spec = {"file":asset.name, "sha256":hashlib.sha256(asset.read_bytes()).hexdigest(),
            "capabilities":["surface_spalling"], "diagnostic_only":True}
    path = tmp_path / "categories.json"
    path.write_text(json.dumps({"categories":[{"id":"surface_spalling", "model_ref":"damage:failed",
                                              "model_status":"supported_experimental"}]}))
    report = load_capabilities(path, root=tmp_path, model_specs={"damage:failed":spec,
                                                               "damage:ordinary":{**spec,"diagnostic_only":False}})
    assert report["categories"][0]["model_installed"]
    assert report["categories"][0]["effective_status"] == "research_only"
    assert [row["variant"] for row in report["models"]] == ["ordinary", "failed"]
    assert report["models"][1]["diagnostic_only"] is True


def test_tile_model_requires_both_pinned_dependencies(tmp_path):
    directory = tmp_path / "models/construction/tile"
    directory.mkdir(parents=True)
    dependencies = {}
    for name in ("encoder", "bank"):
        target = directory / name
        target.write_bytes(name.encode())
        dependencies[name] = {"file": name, "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
    asset = directory / "model.json"
    asset.write_text(json.dumps(dependencies))
    spec = {"file": "tile/model.json", "decoder": "tile_patch_bank", "diagnostic_only": True,
            "sha256": hashlib.sha256(asset.read_bytes()).hexdigest(), "capabilities": ["tile_surface_anomaly"]}
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps({"categories": [{"id": "tile_surface_anomaly", "model_status": "measured_weak",
                                               "model_ref": "damage:tile"}]}))
    def load():
        return load_capabilities(path, root=tmp_path, model_specs={"damage:tile": spec})
    assert load()["models"][0]["installed"]
    assert load()["categories"][0]["effective_status"] == "research_only"
    (directory / "bank").write_bytes(b"changed normal bank")
    assert not load()["models"][0]["installed"]
    (directory / "bank").unlink()
    assert not load()["categories"][0]["model_installed"]


def test_condition_alias_uses_exact_model_class_for_the_selector(tmp_path):
    directory = tmp_path / "models/construction"
    directory.mkdir(parents=True)
    asset = directory / "fixture.onnx"
    asset.write_bytes(b"contract fixture only")
    spec = {"file": asset.name, "sha256": hashlib.sha256(asset.read_bytes()).hexdigest(),
            "capabilities": ["surface_leakage_appearance"],
            "class_capabilities": {"leakage": "surface_leakage_appearance"}}
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps({"categories": []}))
    result = load_capabilities(path, root=tmp_path, model_specs={"damage:wall": spec})
    assert result["models"][0]["condition_classes"] == {"surface_leakage_appearance": "leakage"}
