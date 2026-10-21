"""Dispatch/source contracts only; these fixtures are never accuracy evidence."""
import hashlib
from pathlib import Path

import numpy as np
import pytest

from edge_runtime.profiles import construction_model as models
from edge_runtime.profiles import construction_cubit as adapter


def spec():
    return {'decoder': 'cubit_yolo_seg', 'classes': ['crack', 'spalling'], 'input_size': 640,
            'capabilities': ['surface_crack', 'surface_spalling'], 'diagnostic_only': True,
            'wrapper_sha256': hashlib.sha256(Path(adapter.__file__).read_bytes()).hexdigest()}


def test_author_dispatch_keeps_class_order_raw_input_and_diagnostic_status(monkeypatch):
    class Delegate:
        mask_rendering = 'masks'

        def __init__(self, options, definition):
            self.options, self.definition = options, definition

        def infer(self, frame):
            self.seen = frame
            self.last_input_exposure_gain = 4.0
            return [{'class_id': 0, 'label': self.definition['classes'][0], 'bbox': (1, 2, 3, 4), 'score': .6}]

        def describe(self):
            return {'classes': self.definition['classes'], 'diagnostic_only': False}

    monkeypatch.setattr(adapter, 'CubitConstructionModel', Delegate)
    monkeypatch.setitem(models.DAMAGE_VARIANTS, 'fixture_author', spec())
    model = models.ConstructionModel({'mode': 'damage', 'damage_variant': 'fixture_author',
                                      'damage_classes': ['crack']})
    raw = np.full((40, 60, 3), 37, np.uint8)
    assert model.infer(raw)[0]['label'] == 'crack'
    assert model.delegate.seen is raw
    assert model.last_input_exposure_gain == 4.0
    assert model.delegate.options['damage_classes'] == ['crack']
    assert model.describe()['classes'] == ['crack', 'spalling']
    assert model.describe()['diagnostic_only'] is True


def test_author_dispatch_rejects_unpinned_wrapper_before_model_initialization(monkeypatch):
    class Forbidden:
        def __init__(self, *args):
            raise AssertionError('Must reject source mismatch before model initialization')

    monkeypatch.setattr(adapter, 'CubitConstructionModel', Forbidden)
    monkeypatch.setitem(models.DAMAGE_VARIANTS, 'fixture_author', spec() | {'wrapper_sha256': '0' * 64})
    with pytest.raises(ValueError, match='implementation identity'):
        models.ConstructionModel({'mode': 'damage', 'damage_variant': 'fixture_author'})
