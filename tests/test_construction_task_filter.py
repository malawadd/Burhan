"""Operator-selected checks narrow inference scope; they are not model accuracy."""
import hashlib
import sys
import types

import numpy as np
import pytest

from edge_runtime.profiles import construction_model as models


@pytest.fixture
def local_damage_fixture(tmp_path, monkeypatch):
    path=tmp_path/'models/construction/fixture.onnx'
    path.parent.mkdir(parents=True)
    path.write_bytes(b'unit-test-onnx-contract')
    monkeypatch.setattr(models,'ROOT',tmp_path)
    monkeypatch.setitem(models.DAMAGE_VARIANTS,'task_fixture',{
        'file':'fixture.onnx','sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'classes':['spalling','crack'],'capabilities':['surface_spalling','surface_crack'],
        'source':'fixture','input_size':416})
    class Options:
        def add_session_config_entry(self,*args):
            pass
    class Session:
        def __init__(self,*args,**kwargs):
            pass
        def get_inputs(self):
            return [types.SimpleNamespace(shape=[1,3,416,416],name='images')]
        def run(self,*args):
            result=np.zeros((1,6,3549),np.float32)
            result[0,:,0]=[100,100,40,40,.9,.1]
            result[0,:,1]=[300,300,40,40,.1,.8]
            return [result]
    monkeypatch.setitem(sys.modules,'onnxruntime',types.SimpleNamespace(
        SessionOptions=Options,InferenceSession=Session,__version__='fixture'))
    return {'mode':'damage','damage_variant':'task_fixture'}


def test_selected_check_limits_candidates_and_declared_capabilities(local_damage_fixture):
    model=models.ConstructionModel(local_damage_fixture|{'damage_classes':['spalling']})
    assert [row['label'] for row in model.infer(np.zeros((416,416,3),np.uint8))] == ['spalling']
    assert model.describe()['supported_capabilities'] == ['surface_spalling']
    assert model.describe()['selected_damage_classes'] == ['spalling']
    assert model.describe()['classes'] == ['spalling','crack']


def test_default_still_returns_all_named_classes(local_damage_fixture):
    model=models.ConstructionModel(local_damage_fixture)
    assert {row['label'] for row in model.infer(np.zeros((416,416,3),np.uint8))} == {'spalling','crack'}


def test_declared_capability_alias_keeps_model_class_semantics(local_damage_fixture, monkeypatch):
    spec = dict(models.DAMAGE_VARIANTS['task_fixture'])
    spec['class_capabilities'] = {'spalling': 'surface_detachment', 'crack': 'surface_crack'}
    spec['capabilities'] = ['surface_detachment', 'surface_crack']
    monkeypatch.setitem(models.DAMAGE_VARIANTS, 'task_fixture', spec)
    model = models.ConstructionModel(local_damage_fixture | {'damage_classes': ['spalling']})
    assert model.describe()['supported_capabilities'] == ['surface_detachment']
    assert model.infer(np.zeros((416,416,3), np.uint8))[0]['label'] == 'spalling'


@pytest.mark.parametrize('selected',[[],['rust'],'spalling',[None]])
def test_unavailable_or_empty_task_is_rejected(local_damage_fixture,selected):
    with pytest.raises(ValueError,match='damage_classes'):
        models.ConstructionModel(local_damage_fixture|{'damage_classes':selected})
