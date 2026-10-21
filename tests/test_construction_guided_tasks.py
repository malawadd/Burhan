"""Mechanics tests only. Generated arrays/markers are not accuracy evidence."""
import cv2
import numpy as np
import pytest

from edge_runtime.contracts import FrameContext
from edge_runtime.profiles.construction_tasks import GuidedConstructionProfile, task_catalog


def frame():
    return np.random.default_rng(7).integers(40, 210, (240, 320, 3), dtype=np.uint8)


def context(sequence=0, *, source="real.mp4", time_ms=120):
    return FrameContext(sequence, float(sequence), source, "test", time_ms, source_frame_index=sequence)


class Detector:
    def __init__(self, outputs):
        self.outputs = outputs
        self.calls = 0

    def infer(self, image):
        self.calls += 1
        return self.outputs


def reference():
    return {"marker_size_mm": 50, "marker_id": 0, "same_plane": True,
            "camera_matrix": [[600, 0, 160], [0, 600, 120], [0, 0, 1]],
            "distortion_coefficients": [0, 0, 0, 0, 0], "image_size": [320, 240]}


def test_catalog_never_releases_unvalidated_categories():
    catalog = task_catalog()
    assert len(catalog["tasks"]) == 3
    assert catalog["released_categories"] == []
    assert {c["status"] for t in catalog["tasks"] for c in t["categories"]} <= {"unavailable", "research_only"}


def test_catalog_reasons_preserve_actual_training_and_data_gaps():
    categories = {category["id"]: category for task in task_catalog()["tasks"] for category in task["categories"]}
    assert "were trained" in categories["mold_appearance"]["reason"]
    assert "3 Mold-positive" in categories["mold_appearance"]["reason"]
    assert "torque" in categories["visibly_unseated_connection"]["reason"]
    assert "no corrosion class" in categories["corrosion_appearance"]["reason"]
    assert "independently measured" in categories["joint_width"]["reason"]
    assert "not been acquired" in categories["paint_blister"]["reason"]
    assert len({category["reason"] for category in categories.values()}) > 20


@pytest.mark.parametrize("options", [
    {"inspection_task": "all"}, {"roi": [0, 0, 0, 1]}, {"roi": [0, 0, float("nan"), 1]},
    {"nominal_joint_mm": True}, {"tolerance_mm": 1}, {"confirm_frames": 1.5},
    {"expected_fasteners": [{"bbox": [0, 0, 3, 1]}]}, {"calibration_reference": {"marker_size_mm": 50}},
    {"inspection_task": {}}, {"inspection_task": []}, {"expected_fasteners": None},
    {"expected_fasteners": "[]"}, {"expected_fasteners": {}},
    {"roi": [False, 0, 1, 1]}, {"roi": ["0", 0, 1, 1]},
    {"expected_fasteners": [{"bbox": [False, 0, 1, 1]}]},
])
def test_invalid_capture_configuration_rejected(options):
    with pytest.raises(ValueError):
        GuidedConstructionProfile(options)


@pytest.mark.parametrize("task", ["brackets_railings", "tile_installation"])
def test_legacy_wall_model_rejected_before_loading_for_other_tasks(monkeypatch, task):
    import edge_runtime.profiles.construction_model as legacy

    def forbidden_constructor(options):
        pytest.fail("An unrelated legacy wall model must not load for this task")

    monkeypatch.setattr(legacy, "ConstructionModel", forbidden_constructor)
    with pytest.raises(ValueError, match="only supports walls_finishes"):
        GuidedConstructionProfile({"inspection_task": task, "research_model": {"mode": "semantic"}})


def test_explicit_legacy_wall_adapter_still_loads_for_wall_task(monkeypatch):
    import edge_runtime.profiles.construction_model as legacy
    loaded = []
    adapter = Detector([])

    def constructor(options):
        loaded.append(options)
        return adapter

    monkeypatch.setattr(legacy, "ConstructionModel", constructor)
    profile = GuidedConstructionProfile({"inspection_task": "walls_finishes", "research_model": {"mode": "semantic"}})
    assert loaded == [{"mode": "semantic"}]
    assert profile.detector is adapter


def test_real_only_and_source_timestamp_validation():
    profile = GuidedConstructionProfile()
    with pytest.raises(ValueError, match="real video"):
        profile.process(frame(), context(source="demo:construction"))
    with pytest.raises(ValueError, match="source_time_ms"):
        profile.process(frame(), context(time_ms=float("nan")))


def test_unavailable_does_not_invent_wall_candidates():
    result = GuidedConstructionProfile().process(frame(), context())
    assert result.status == "unavailable"
    assert not result.findings and not result.emit_event
    assert result.domain_state["quality"]["capture_accepted"]


@pytest.mark.parametrize("image,options", [(np.full((240, 320, 3), 128, np.uint8), {}),
                                          (frame(), {"roi": [0, 0, .1, .1]})])
def test_bad_capture_skips_inference(image, options):
    detector = Detector([{"label": "peeling_paint", "bbox": [1, 1, 30, 30], "score": .9}])
    result = GuidedConstructionProfile(options, detector=detector).process(image, context())
    assert result.status == "needs_capture"
    assert not result.findings and detector.calls == 0


def test_exact_labels_do_not_reinterpret_generic_damage():
    detector = Detector([{"label": "wetspot", "bbox": [10, 20, 30, 40], "score": .9},
                         {"label": "bulge", "bbox": [10, 20, 30, 40], "score": .9}])
    result = GuidedConstructionProfile(detector=detector).process(frame(), context())
    assert result.status == "unassessed" and not result.findings


def test_roi_mapping_timestamp_and_repeated_event_suppression():
    detector = Detector([{"label": "peeling_paint", "bbox": [10, 20, 30, 40], "score": .9}])
    profile = GuidedConstructionProfile({"roi": [.25, .25, .75, .75]}, detector=detector)
    results = [profile.process(frame(), context(i)) for i in range(3)]
    assert [r.emit_event for r in results] == [False, True, False]
    obs = results[0].domain_state["observations"][0]
    assert obs["bbox"] == [90, 80, 30, 40]
    assert obs["evidence_timestamp_ms"] == 120
    assert obs["assessment_status"] == "research_candidate"
    assert profile.process(frame(), context(0)).emit_event is False


def test_missing_fastening_requires_positive_visible_class_and_expected_location():
    options = {"inspection_task": "brackets_railings", "expected_fasteners": [{"id": "seat-1", "bbox": [.1, .1, .3, .3]}]}
    output = {"label": "missing_bolt", "bbox": [32, 24, 64, 48], "score": .9}
    assert not GuidedConstructionProfile(options, detector=Detector([])).process(frame(), context()).findings
    assert not GuidedConstructionProfile(options, detector=Detector([output])).process(frame(), context()).findings
    output["seat_visible"] = True
    result = GuidedConstructionProfile(options, detector=Detector([output])).process(frame(), context())
    assert result.domain_state["observations"][0]["expected_location_id"] == "seat-1"
    assert not GuidedConstructionProfile({"inspection_task": "brackets_railings"}, detector=Detector([output])).process(frame(), context()).findings
    output["occluded"] = True
    assert not GuidedConstructionProfile(options, detector=Detector([output])).process(frame(), context()).findings


def test_cross_task_model_output_is_filtered():
    detector = Detector([{"label": "crack", "bbox": [10, 20, 30, 40], "score": .9}])
    result = GuidedConstructionProfile({"inspection_task": "brackets_railings"}, detector=detector).process(frame(), context())
    assert not result.findings


def test_parallel_edge_mechanics_remain_research_and_do_not_alarm(monkeypatch):
    profile = GuidedConstructionProfile({"inspection_task": "tile_installation", "nominal_joint_mm": 3, "tolerance_mm": .5})
    monkeypatch.setattr(profile, "_joint_edges", lambda image: [{"width_px": 8., "polygon": [[20, 20], [200, 20], [200, 28], [20, 28]],
                "center": [110, 24], "angle_degrees": 0., "edge_overlap_px": 180.}])
    result = profile.process(frame(), context())
    observation = result.domain_state["observations"][0]
    assert observation["measurements"]["edge_spacing"]["unit"] == "px"
    assert "reference_mm" not in observation["measurements"]["edge_spacing"]
    assert result.status == "research_measurement" and not result.emit_event
    assert result.findings[0].severity == "info"


@pytest.mark.parametrize("shape", [(2, 4), (2, 1, 4)])
def test_hough_python_output_shapes_are_normalized_before_unpacking(monkeypatch, shape):
    raw = np.array([[10, 60, 230, 60], [10, 68, 230, 68]], dtype=np.int32).reshape(shape)
    monkeypatch.setattr(cv2, "HoughLinesP", lambda *args, **kwargs: raw)
    band = np.full((240, 320, 3), 180, np.uint8)
    band[61:68] = 60
    observations = GuidedConstructionProfile._joint_edges(band)
    assert observations and observations[0]["width_px"] == pytest.approx(8)


def test_edge_endpoint_direction_does_not_change_band_measurement(monkeypatch):
    image = np.full((240, 320, 3), 180, np.uint8)
    image[61:68] = 60
    rows = np.array([[230, 60, 10, 60], [10, 68, 230, 68]], dtype=np.int32)
    monkeypatch.setattr(cv2, "HoughLinesP", lambda *args, **kwargs: rows)
    observation = GuidedConstructionProfile._joint_edges(image)[0]
    assert observation["width_px"] == pytest.approx(8)
    assert observation["support_fraction"] == 1
    assert observation["observed_edge_polarity"] == "dark_band"


def test_parallel_lines_on_uniform_surface_do_not_establish_visible_band(monkeypatch):
    monkeypatch.setattr(cv2, "HoughLinesP", lambda *args, **kwargs: np.array([[10,60,230,60],[10,68,230,68]]))
    assert GuidedConstructionProfile._joint_edges(np.full((240,320,3),180,np.uint8)) == []


def test_supported_band_statistics_reach_finding_without_fault_decision(monkeypatch):
    image = np.full((240,320,3),180,np.uint8)
    image[61:68] = 60
    monkeypatch.setattr(cv2, "HoughLinesP", lambda *args, **kwargs: np.array([[10,60,230,60],[10,68,230,68]]))
    result = GuidedConstructionProfile({'inspection_task':'tile_installation'}).process(image, context())
    measured = result.findings[0].measurements['edge_spacing']
    assert measured['unit'] == 'px' and measured['value'] == 8
    assert measured['spatial_p10'] == measured['spatial_p90'] == 8
    assert measured['sample_count'] == 24 and measured['support_fraction'] == 1
    assert measured['range_kind'] == 'spatial_line_fit_variation_not_measurement_uncertainty'
    assert result.status == 'research_measurement' and not result.emit_event


def test_missing_marker_and_dimensions_mismatch_fall_back_to_pixels():
    profile = GuidedConstructionProfile({"inspection_task": "tile_installation", "calibration_reference": reference()})
    image, scale, mapping, reason = profile._reference(frame(), (0, 0), (320, 240))
    assert scale is None and mapping is None and "marker" in reason.lower()
    _, scale, _, reason = profile._reference(frame(), (0, 0), (640, 480))
    assert scale is None and "dimensions" in reason


def test_calibration_requires_explicit_plane_and_resolution():
    for remove in ("same_plane", "image_size"):
        config = reference()
        del config[remove]
        with pytest.raises(ValueError):
            GuidedConstructionProfile({"calibration_reference": config})


@pytest.mark.parametrize("dimensions", [[True, 240], [320.5, 240], [float("nan"), 240], [320, "240"], [320], None])
def test_calibration_dimensions_are_explicit_integers(dimensions):
    config = reference()
    config["image_size"] = dimensions
    with pytest.raises(ValueError, match="image_size"):
        GuidedConstructionProfile({"calibration_reference": config})


@pytest.mark.parametrize("field,value", [("camera_matrix", {}), ("camera_matrix", [[{}, 0, 0]]), ("distortion_coefficients", {"bad": 1})])
def test_malformed_matrix_is_validation_error(field, value):
    config = reference()
    config[field] = value
    with pytest.raises(ValueError):
        GuidedConstructionProfile({"calibration_reference": config})


def test_real_adapter_boolean_mask_is_rendered_at_native_roi_and_empty_masks_abstain():
    mask = np.zeros((120, 160), bool)
    mask[22:27, 12:17] = True
    detector = Detector([{"label": "peeling_paint", "bbox": [10, 20, 30, 40], "score": .9, "mask": mask}])
    profile = GuidedConstructionProfile({"roi": [.25, .25, .75, .75]}, detector=detector)
    original = frame()
    result = profile.process(original, context())
    observation = result.domain_state["observations"][0]
    assert observation["mask_area_px"] == 25
    assert observation["localization_kind"] == "segmentation_mask"
    assert np.any(result.annotated_frame[82:87, 92:97] != original[82:87, 92:97])
    assert min(p[0] for poly in observation["mask_polygons"] for p in poly) == 92
    detector.outputs[0]["mask"] = np.zeros_like(mask)
    assert not profile.process(original, context(1)).findings


def test_bad_mask_shape_never_spreads_across_roi():
    detector = Detector([{"label": "peeling_paint", "bbox": [10, 20, 30, 40], "score": .9, "mask": np.ones((2, 2), bool)}])
    result = GuidedConstructionProfile(detector=detector).process(frame(), context())
    assert result.domain_state["observations"][0]["localization_kind"] == "bbox"
    assert "invalid_native" in result.domain_state["observations"][0]["mask_status"]


def test_stateless_warmup_discarded_and_provenance_exposed():
    detector = Detector([])
    detector.describe = lambda: {"source": "local unit mechanics fixture"}
    profile = GuidedConstructionProfile(detector=detector)
    receipt = profile.prepare_worker()
    assert detector.calls == 3
    assert receipt["worker_warmup_inspected_observations"] == 0
    assert profile._identity is None and profile._tracks == []
    assert profile.describe()["model_provenance"]["source"].startswith("local")


@pytest.mark.skipif(not hasattr(cv2, "aruco"), reason="OpenCV ArUco unavailable")
def test_marker_projection_roundtrip_is_only_unit_geometry():
    image = np.full((240, 320, 3), 220, np.uint8)
    marker = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), 0, 100)
    image[70:170, 110:210] = marker[:, :, None]
    profile = GuidedConstructionProfile({"calibration_reference": reference()})
    rectified, scale, mapping, reason = profile._reference(image, (0, 0), (320, 240))
    assert scale is not None and mapping is not None and rectified.ndim == 3
    assert np.isfinite(mapping(np.array([[10., 10.], [20., 20.]]))).all()
    assert "validation outstanding" in reason
