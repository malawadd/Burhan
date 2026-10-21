"""HTTP/schema fixtures only: no model, inference, camera or real-video claims."""
from copy import deepcopy
import importlib
import json

from fastapi.testclient import TestClient
import pytest


def evidence_fixture():
    return {"schema_version": 1, "purpose": "defect_evidence", "frame_size_wh": [100, 80],
            "matching_iou_threshold": .5, "annotation_scope": "Unit-test geometry, not field data",
            "summary": {"annotated_frames": 1, "localized_matches": 1, "missed_defects": 0, "false_alerts": 0},
            "frames": [{"source_frame_index": 0, "video_time_s": 0.0, "evaluated": True,
                        "expected": [{"id": "truth:0", "label": "crack", "bbox": [10, 20, 30, 40]}],
                        "predictions": [{"id": 1, "label": "crack", "bbox": [10, 20, 30, 40], "score": .8,
                                         "matched_annotation_id": "truth:0", "iou": 1.0, "evaluation_status": "matched"}]}]}


@pytest.fixture
def evidence_client(tmp_path, monkeypatch):
    module = importlib.import_module("edge_runtime.app")
    for folder in ("edge_runtime", "config", "output"):
        (tmp_path / folder).mkdir()
    class Runtime:
        def __init__(self, *args): pass
        def close(self): pass
    monkeypatch.setattr(module, "__file__", str(tmp_path / "edge_runtime/app.py"))
    monkeypatch.setattr(module, "EdgeRuntime", Runtime)
    (tmp_path / "output/video.mp4").write_bytes(b"0123456789abcdef")
    evidence_path = tmp_path / "output/evaluation-private-name.json"
    evidence_path.write_text(json.dumps(evidence_fixture()), encoding="utf-8")
    (tmp_path / "output/unregistered.json").write_text('{"secret":"not allowlisted"}', encoding="utf-8")
    (tmp_path / "private.json").write_text('{"secret":"outside output"}', encoding="utf-8")
    (tmp_path / "output/not-json.txt").write_text("not evidence", encoding="utf-8")
    records = [
        {"id": "allowed", "label": "fixture", "path": "output/video.mp4", "evidence_path": "output/evaluation-private-name.json"},
        {"id": "outside", "label": "fixture", "path": "output/video.mp4", "evidence_path": "private.json"},
        {"id": "traversal", "label": "fixture", "path": "output/video.mp4", "evidence_path": "output/../private.json"},
        {"id": "wrong-type", "label": "fixture", "path": "output/video.mp4", "evidence_path": "output/not-json.txt"},
        {"id": "missing", "label": "fixture", "path": "output/video.mp4", "evidence_path": "output/missing.json"},
    ]
    (tmp_path / "config/construction-videos.json").write_text(json.dumps(records), encoding="utf-8")
    with TestClient(module.create_app()) as client:
        yield client, evidence_path


def test_evidence_inventory_hides_paths_and_reports_allowlist(evidence_client):
    client, _ = evidence_client
    response = client.get("/api/construction/videos")
    assert response.status_code == 200
    assert "evaluation-private-name" not in response.text and "evidence_path" not in response.text
    assert "output/" not in response.text and "private.json" not in response.text
    assert {row["id"]: row["evidence_available"] for row in response.json()} == {
        "allowed": True, "outside": False, "traversal": False, "wrong-type": False, "missing": False}
    assert response.json()[0]["evidence_url"] == "/api/construction/videos/allowed/evidence"


def test_valid_evidence_and_video_routes_are_separate(evidence_client):
    client, _ = evidence_client
    response = client.get("/api/construction/videos/allowed/evidence")
    assert response.status_code == 200 and response.json() == evidence_fixture()
    assert response.headers["content-type"] == "application/json"
    assert response.headers["x-content-type-options"] == "nosniff"
    mp4 = client.get("/api/construction/videos/allowed", headers={"Range": "bytes=4-7"})
    assert mp4.status_code == 206 and mp4.content == b"4567"
    assert mp4.headers["content-type"] == "video/mp4"
    assert client.head("/api/construction/videos/allowed").status_code == 200


@pytest.mark.parametrize("identifier", ["outside", "traversal", "wrong-type", "missing", "unregistered.json", "%2e%2e%2fprivate.json"])
def test_evidence_cannot_escape_registered_output_json(evidence_client, identifier):
    client, _ = evidence_client
    response = client.get(f"/api/construction/videos/{identifier}/evidence")
    assert response.status_code == 404
    assert "secret" not in response.text and "private.json" not in response.text


@pytest.mark.parametrize("payload", [b"not JSON", b"\xff", b"[]", b'{"frames":{}}', b'{"frames":[null]}', b'{"frames":[],"n":' + b'9' * 5000 + b'}'])
def test_unreadable_or_malformed_evidence_is_denied(evidence_client, payload):
    client, path = evidence_client
    path.write_bytes(payload)
    assert client.get("/api/construction/videos/allowed/evidence").status_code == 409


def test_oversized_evidence_is_denied_before_reading(evidence_client):
    client, path = evidence_client
    path.write_bytes(b" " * (4 * 1024 * 1024 + 1))
    assert client.get("/api/construction/videos/allowed/evidence").status_code == 409


@pytest.mark.parametrize("change", [
    lambda d: d["frames"][0].update(video_time_s=float("nan")),
    lambda d: d["frames"][0].update(video_time_s=float("inf")),
    lambda d: d["frames"][0].update(video_time_s=-1),
    lambda d: d["frames"][0].update(expected={}),
    lambda d: d["frames"][0].update(predictions={}),
    lambda d: d["frames"][0]["expected"][0].update(bbox=[1, 2, 0, 3]),
    lambda d: d["frames"][0]["expected"][0].update(bbox=[1, 2, 1000, 3]),
    lambda d: d["frames"][0]["predictions"][0].update(score=1.1),
    lambda d: d["frames"][0]["predictions"][0].update(iou=-.1),
    lambda d: d["frames"][0]["predictions"][0].update(matched_annotation_id=[]),
    lambda d: d["frames"][0]["predictions"][0].update(matched_annotation_id="nonexistent"),
    lambda d: d["frames"][0]["predictions"][0].update(evaluation_status={}),
    lambda d: d["frames"][0].update(evaluated=False),
    lambda d: d["frames"][0]["expected"][0].update(evaluation_status="unsupported"),
    lambda d: d["frames"][0]["predictions"][0].update(label="spalling"),
    lambda d: d["frames"][0]["predictions"][0].update(bbox=[60, 20, 30, 40], iou=1),
    lambda d: d["frames"][0]["predictions"].append(deepcopy(d["frames"][0]["predictions"][0])),
])
def test_invalid_nested_geometry_and_false_match_claims_are_denied(evidence_client, change):
    client, path = evidence_client
    document = evidence_fixture()
    change(document)
    path.write_text(json.dumps(document), encoding="utf-8")
    assert client.get("/api/construction/videos/allowed/evidence").status_code == 409


def test_unscored_unsupported_damage_is_kept_separate(evidence_client):
    client, path = evidence_client
    document = evidence_fixture()
    document["frames"][0]["evaluated"] = False
    document["frames"][0]["expected"][0].update(label="spalling", evaluation_status="unsupported")
    document["frames"][0]["predictions"][0].update(matched_annotation_id=None, iou=0, evaluation_status="unscored")
    document["summary"].update(localized_matches=0, missed_defects=0, unsupported_defects=1)
    path.write_text(json.dumps(document), encoding="utf-8")
    response = client.get("/api/construction/videos/allowed/evidence")
    assert response.status_code == 200
    frame = response.json()["frames"][0]
    assert not frame["evaluated"] and frame["predictions"][0]["matched_annotation_id"] is None


def unannotated_fixture():
    document = evidence_fixture()
    document['purpose'] = 'failure_diagnostic'
    document['summary'] = {key: None for key in document['summary']}
    frame = document['frames'][0]
    frame.update(evaluated=False, expected=[])
    prediction = frame['predictions'][0]
    prediction.update(evaluation_status='unscored')
    prediction.pop('iou')
    prediction.pop('matched_annotation_id')
    return document


def test_unannotated_local_diagnostic_preserves_unknown_metrics(evidence_client):
    client, path = evidence_client
    document = unannotated_fixture()
    path.write_text(json.dumps(document), encoding='utf-8')
    response = client.get('/api/construction/videos/allowed/evidence')
    assert response.status_code == 200 and response.json() == document


@pytest.mark.parametrize('change', [
    lambda d: d['frames'][0].update(evaluated=True),
    lambda d: d['frames'][0]['predictions'][0].update(evaluation_status='matched'),
    lambda d: d['frames'][0]['predictions'][0].update(matched_annotation_id='invented'),
    lambda d: d['frames'][0]['predictions'][0].update(iou=-.1),
    lambda d: d['frames'][0]['predictions'][0].update(score=1.1),
    lambda d: d['summary'].update(false_alerts='unknown'),
])
def test_unknown_metrics_do_not_relax_evaluated_match_validation(evidence_client, change):
    client, path = evidence_client
    document = unannotated_fixture()
    change(document)
    path.write_text(json.dumps(document), encoding='utf-8')
    assert client.get('/api/construction/videos/allowed/evidence').status_code == 409
