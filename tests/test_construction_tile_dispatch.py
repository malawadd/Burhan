"""The registry must dispatch the tile model without changing raw input."""
import hashlib
import numpy as np
from pathlib import Path
from edge_runtime.profiles import construction_model as module
from edge_runtime.profiles import construction_tile_adapter as adapter


def test_tile_dispatch_keeps_raw_frame_and_research_status(monkeypatch):
    class Delegate:
        rendering = "masks"
        def __init__(self, options, spec):
            self.options = options
        def infer(self, frame):
            self.seen = frame
            return []
        def describe(self):
            return {"diagnostic_only": False, "classes": ["tile_surface_anomaly"]}
    monkeypatch.setattr(adapter, "TileConstructionModel", Delegate)
    spec = {"decoder": "tile_patch_bank", "classes": ["tile_surface_anomaly"], "input_size": 256,
            "capabilities": ["tile_surface_anomaly"], "diagnostic_only": True,
            "wrapper_sha256": hashlib.sha256(Path(adapter.__file__).read_bytes()).hexdigest(),
            "fitted_source_sha256": hashlib.sha256(Path(module.__file__).with_name("construction_tile.py").read_bytes()).hexdigest()}
    monkeypatch.setitem(module.DAMAGE_VARIANTS, "fixture_tile", spec)
    model = module.ConstructionModel({"mode": "damage", "damage_variant": "fixture_tile"})
    raw = np.full((32, 40, 3), 37, np.uint8)
    assert model.infer(raw) == []
    assert model.delegate.seen is raw
    assert model.describe()["diagnostic_only"] is True
