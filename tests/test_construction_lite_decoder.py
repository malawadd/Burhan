"""Semantic regions stay actual masks; no fragment grouping or box inflation."""
import numpy as np
import pytest
from edge_runtime.profiles.construction_lite import decode_crack_map, camera_exposure


def test_actual_separate_thin_components_and_strict_pixel_cutoff():
    probability = np.zeros((12, 14), dtype=np.float32)
    probability[1:9, 2] = .9
    probability[3:10, 10] = .8
    probability[0, 0] = .5  # equality is not above the selected threshold
    rows = decode_crack_map(probability, probability.shape, .5, 4, masks=True)
    assert [r['bbox'] for r in rows] == [(2, 1, 1, 8), (10, 3, 1, 7)]
    assert [int(r['mask'].sum()) for r in rows] == [8, 7]
    assert not rows[0]['mask'][:, 3:10].any()
    assert all(r['label'] == 'crack' for r in rows)


def test_native_pixel_area_is_applied_after_probability_projection():
    probability = np.zeros((4, 4), dtype=np.float32)
    probability[1:3, 1:3] = .9
    rows = decode_crack_map(probability, (8, 8), .5, 5, masks=True)
    assert len(rows) == 1
    assert rows[0]['mask'].shape == (8, 8)
    assert rows[0]['native_component_pixels'] == rows[0]['mask'].sum()
    with pytest.raises(ValueError):
        decode_crack_map(np.full((4, 4), np.nan), (8, 8))


def test_bounded_global_camera_gain_preserves_bright_input_and_geometry():
    dark=np.full((12,14,3),5,dtype=np.uint8)
    corrected,gain=camera_exposure(dark,'low_light_gain')
    assert gain==8 and corrected.shape==dark.shape
    assert np.all(corrected==40) and np.all(dark==5)
    bright=np.full((12,14,3),100,dtype=np.uint8)
    unchanged,gain=camera_exposure(bright,'low_light_gain')
    assert unchanged is bright and gain==1


def test_serving_camera_gain_tensor_matches_exact_prepared_native_pixels():
    from edge_runtime.profiles.construction_model import ConstructionModel,prepare
    raw=np.full((260,346,3),5,dtype=np.uint8)
    model=object.__new__(ConstructionModel)
    model.mode='damage';model.preprocessing='letterbox';model.input_size=640;model.exposure_policy='low_light_gain'
    actual,transform=model.prepare_input(raw)
    assert model.last_input_exposure_gain==8
    corrected,_=camera_exposure(raw,'low_light_gain')
    expected,expected_transform=prepare(corrected,'damage',size=640)
    assert np.array_equal(actual,expected) and transform==expected_transform
    model.exposure_policy='off'
    actual,transform=model.prepare_input(raw)
    assert model.last_input_exposure_gain==1
    expected,expected_transform=prepare(raw,'damage',size=640)
    assert np.array_equal(actual,expected) and transform==expected_transform
