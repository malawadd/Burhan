"""Offline calibration acceleration must preserve original-pixel metrics."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from factory_mask_metrics import component_mask_statistics,original_pixel_weights,at_area


def test_cached_projection_exactly_matches_native_mask_counts():
    rng=np.random.default_rng(871)
    for source_shape,grid_shape in [((31,13),(17,7)),((11,39),(7,17)),((8,8),(8,8))]:
        truth=rng.random(source_shape)>.83
        scores=rng.random(grid_shape).astype(np.float32)
        weights=original_pixel_weights(truth,grid_shape)
        for threshold in [.15,.5,.91]:
            original=component_mask_statistics(scores,threshold,truth)
            cached=component_mask_statistics(scores,threshold,truth,weights)
            for a,b in zip(original,cached):np.testing.assert_array_equal(a,b)
            for area in [1,2,4,8]:assert at_area(original,area)==at_area(cached,area)
