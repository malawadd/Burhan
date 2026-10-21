import hashlib
import json

import cv2
import numpy as np
import pytest
import joblib
from sklearn.ensemble import RandomForestClassifier

from edge_runtime.contracts import FrameContext
from edge_runtime.profiles.factory import FactoryProfile
from edge_runtime.profiles.factory_defect import FEATURE_VERSION, SurfaceDefectModel, extract_features, extract_features_fast


@pytest.fixture
def surface_fixture(tmp_path):
    rng = np.random.default_rng(9)
    x, y = [], []
    for n in range(8):
        noise = rng.integers(130, 190, (160, 64), dtype=np.uint8)
        frame = cv2.cvtColor(noise, cv2.COLOR_GRAY2BGR)
        mask = np.zeros((160, 64), np.uint8)
        if n % 2:
            frame[70:82, 20:45] = 15
            mask[70:82, 20:45] = 1
        features, shape = extract_features(frame, 64)
        x.append(features)
        y.append(mask.reshape(-1).astype(np.int32))
    model = RandomForestClassifier(n_estimators=8, max_depth=8, min_samples_leaf=1, random_state=9, n_jobs=1)
    model.fit(np.concatenate(x), np.concatenate(y))
    path = tmp_path / "surface.joblib"
    joblib.dump(model, path)
    metadata = {"feature_version": FEATURE_VERSION, "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "input_width": 64, "pixel_threshold": .5, "minimum_component_pixels": 4,
                "supported_height_width_ratio": [2.3, 2.7]}
    path.with_suffix(path.suffix + ".json").write_text(json.dumps(metadata))
    return path, frame, metadata


def test_real_model_roundtrip_and_original_coordinate_regions(surface_fixture):
    path, frame, _ = surface_fixture
    model = SurfaceDefectModel(path)
    score, candidates, metrics = model.predict(frame)
    assert score.shape == frame.shape[:2]
    assert candidates and metrics["candidate_regions"] >= 1
    assert any(x <= 30 < x+w and y <= 76 < y+h for x, y, w, h in [c["bbox"] for c in candidates])
    assert all(0 <= candidate["score"] <= 1 for candidate in candidates)


def test_model_temporal_confirmation_trigger_dedup_and_reset(surface_fixture):
    path, frame, _ = surface_fixture
    profile = FactoryProfile({"defect_model_path": str(path)})
    assert profile.describe()["mode"] == "local_trained_surface_model"
    results = [profile.process(frame, FrameContext(n, float(n), session_id="test")) for n in range(4)]
    assert [result.status for result in results] == ["pending", "pending", "fault", "fault"]
    assert results[2].emit_event and not results[3].emit_event
    assert profile.process(frame, FrameContext(5, 5., unit_id="unit1")).emit_event
    assert not profile.process(frame, FrameContext(6, 6., unit_id="unit1")).emit_event
    assert profile.process(frame, FrameContext(7, 7., unit_id="unit2")).emit_event
    profile.reset()
    assert profile.describe()["model_sha256"]
    assert not profile.calibrate(frame)["ok"]


def test_blank_and_wrong_geometry_are_abstentions(surface_fixture):
    path, frame, _ = surface_fixture
    profile = FactoryProfile({"defect_model_path": str(path)})
    assert profile.process(np.zeros_like(frame), FrameContext(0, 0.)).status == "needs_input"
    wide = cv2.resize(frame, (240, 160))
    assert profile.process(wide, FrameContext(1, 1.)).status == "inconclusive"


def test_explicit_missing_model_and_tampered_bytes_fail_loudly(surface_fixture):
    path, _, _ = surface_fixture
    with pytest.raises(ValueError, match="not found"):
        FactoryProfile({"demo": True, "defect_model_path": str(path.parent / "missing.xml")})
    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        FactoryProfile({"defect_model_path": str(path)})


def test_feature_version_is_enforced(surface_fixture):
    path, _, metadata = surface_fixture
    metadata["feature_version"] = "unknown"
    path.with_suffix(path.suffix + ".json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="feature version"):
        SurfaceDefectModel(path)


@pytest.mark.parametrize("threshold", [0., .5, .995, 1.])
def test_threshold_cascade_preserves_every_pixel_decision_and_candidate_score(surface_fixture, threshold):
    path, frame, _ = surface_fixture
    baseline = SurfaceDefectModel(path, backend="sklearn")
    cascade = SurfaceDefectModel(path, backend="threshold_cascade")
    baseline.pixel_threshold = cascade.pixel_threshold = threshold
    features, shape = extract_features(frame, 64)
    expected, _ = baseline.score_features(features)
    actual, metrics = cascade.score_features(features)
    np.testing.assert_array_equal(actual >= threshold, expected >= threshold)
    np.testing.assert_allclose(actual[expected >= threshold], expected[expected >= threshold], atol=1e-12)
    assert np.all(actual <= expected + 1e-12)
    assert metrics["model_tree_pixel_evaluations"] <= len(features) * len(baseline.model.estimators_)
    _, expected_candidates, _ = baseline.predict(frame)
    _, actual_candidates, _ = cascade.predict(frame)
    assert [c["bbox"] for c in actual_candidates] == [c["bbox"] for c in expected_candidates]
    np.testing.assert_allclose([c["score"] for c in actual_candidates], [c["score"] for c in expected_candidates])


def test_unknown_factory_backend_fails_explicitly(surface_fixture):
    with pytest.raises(ValueError, match="backend"):
        SurfaceDefectModel(surface_fixture[0], backend="unknown")


def test_native_cascade_matches_compiled_tree_predictions_and_validates_matrix(surface_fixture):
    path, frame, _ = surface_fixture
    try:
        native = SurfaceDefectModel(path, backend="native_cascade")
    except ValueError as exc:
        pytest.skip(f"Optional explicitly built native backend unavailable: {exc}")
    baseline = SurfaceDefectModel(path, backend="sklearn")
    features, _ = extract_features(frame, 64)
    for threshold in (0., .5, .995, 1.):
        baseline.pixel_threshold = native.pixel_threshold = threshold
        expected, _ = baseline.score_features(features)
        actual, _ = native.score_features(features)
        np.testing.assert_array_equal(actual >= threshold, expected >= threshold)
        np.testing.assert_allclose(actual[expected >= threshold], expected[expected >= threshold], atol=1e-12)
    with pytest.raises(ValueError, match="contiguous float32"):
        native.native.score(features.astype(np.float64), .5)
    with pytest.raises(ValueError, match="contiguous float32"):
        native.native.score(features[:, ::2], .5)


@pytest.mark.parametrize("kind", ["noise", "blank", "gradient"])
def test_native_quality_reductions_preserve_rounded_metrics(kind):
    rng = np.random.default_rng(17)
    if kind == "noise":
        frame = rng.integers(0, 256, (1250, 500, 3), dtype=np.uint8)
    elif kind == "gradient":
        gray = np.tile(np.linspace(0, 255, 500).astype(np.uint8), (1250, 1))
        frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    else:
        frame = np.zeros((1250, 500, 3), np.uint8)
    expected = FactoryProfile._quality(frame)
    actual = FactoryProfile._fast_surface_quality(frame)
    assert actual[1:] == expected[1:]


def test_in_place_feature_extraction_is_bitwise_identical():
    rng = np.random.default_rng(28)
    for shape in ((160, 64), (360, 640), (1250, 500)):
        frame = rng.integers(0, 256, (*shape, 3), dtype=np.uint8)
        expected, expected_shape = extract_features(frame)
        actual, actual_shape = extract_features_fast(frame)
        assert actual_shape == expected_shape
        np.testing.assert_array_equal(actual, expected)


def test_learned_region_bundle_filters_proposals_and_preserves_pixel_backend(surface_fixture):
    from edge_runtime.profiles.factory_regions import REGION_COLUMNS, REGION_FEATURE_VERSION
    path, frame, metadata = surface_fixture
    proposal = joblib.load(path)
    region = RandomForestClassifier(n_estimators=2, min_samples_leaf=4, bootstrap=False, random_state=9)
    region.fit(np.zeros((4,REGION_COLUMNS),dtype=np.float32),np.array([0,1,0,1]))
    joblib.dump({'kind':REGION_FEATURE_VERSION,'proposal':proposal,'region_classifier':region},path)
    metadata.update(feature_version=REGION_FEATURE_VERSION,region_threshold=.75,
                    model_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    path.with_suffix('.joblib.json').write_text(json.dumps(metadata),encoding='utf-8')
    model=SurfaceDefectModel(path,backend='threshold_cascade')
    _, candidates, metrics=model.predict(frame)
    assert metrics['proposal_regions']>0 and candidates==[]
    model.region_threshold=.25
    _, candidates, metrics=model.predict(frame)
    assert candidates and all(candidate['score']==.5 for candidate in candidates)
    assert metrics['candidate_score_meaning']=='region_forest_leaf_fraction_not_probability'


def test_region_bundle_requires_matching_feature_inventory(surface_fixture):
    from edge_runtime.profiles.factory_regions import REGION_FEATURE_VERSION
    path, _, metadata = surface_fixture
    proposal=joblib.load(path)
    region=RandomForestClassifier(n_estimators=1,random_state=9).fit([[0],[1]],[0,1])
    joblib.dump({'kind':REGION_FEATURE_VERSION,'proposal':proposal,'region_classifier':region},path)
    metadata.update(feature_version=REGION_FEATURE_VERSION,region_threshold=.5,
                    model_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    path.with_suffix('.joblib.json').write_text(json.dumps(metadata),encoding='utf-8')
    with pytest.raises(ValueError,match='region classifier'):
        SurfaceDefectModel(path)


def test_manual_review_policy_never_issues_fault_or_clear(surface_fixture):
    path,frame,metadata=surface_fixture
    metadata['manual_review_only']=True
    path.with_suffix('.joblib.json').write_text(json.dumps(metadata),encoding='utf-8')
    profile=FactoryProfile({'defect_model_path':str(path),'confirmation_frames':1})
    assert profile.describe()['decision_authority']=='human_review_only'
    profile._defect_model.predict=lambda value:(np.zeros((160,64)),[{'bbox':(20,50,10,10),'area':100,'score':.9}],{})
    positive=profile.process(frame,FrameContext(0,0.,unit_id='review-positive'))
    assert positive.status=='review' and positive.action=='human_review'
    assert positive.findings[0].kind=='surface_review_candidate'
    assert not positive.domain_state['automatic_rejection']
    profile._defect_model.predict=lambda value:(np.zeros((160,64)),[],{})
    negative=profile.process(frame,FrameContext(1,1.,unit_id='review-negative'))
    assert negative.status=='no_candidate' and negative.action=='human_review'
    assert not negative.domain_state['automatic_acceptance']
    assert 'does not approve' in negative.guidance


def test_effective_policy_and_metadata_hash_distinguish_identical_weights(surface_fixture):
    path,_,metadata=surface_fixture
    before=FactoryProfile({'defect_model_path':str(path)}).describe()
    metadata.update(pixel_threshold=.75,minimum_component_pixels=7,manual_review_only=True)
    path.with_suffix('.joblib.json').write_text(json.dumps(metadata),encoding='utf-8')
    after=FactoryProfile({'defect_model_path':str(path)}).describe()
    assert before['model_sha256']==after['model_sha256']
    assert before['model_metadata_sha256']!=after['model_metadata_sha256']
    assert after['model_metadata_sha256']==hashlib.sha256(path.with_suffix('.joblib.json').read_bytes()).hexdigest()
    assert after['decision_threshold']==.75 and after['minimum_component_area']==7
    assert before['decision_threshold']==.5
