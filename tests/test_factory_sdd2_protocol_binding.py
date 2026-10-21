"""No test-data access or NN: exact pre-opening declaration identity guards."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import audit_factory_sdd2_frozen as audit

FREEZE = 'a'*64
MODEL = 'b'*64


@pytest.mark.parametrize('protocol', [{}, {'after_model_sha256':MODEL},
                                    {'after_freeze_sha256':None},
                                    {'after_freeze_sha256':'a'*63},
                                    {'after_freeze_sha256':'g'*64}])
def test_missing_or_invalid_freeze_pin_fails_closed(protocol):
    with pytest.raises(RuntimeError, match='explicit canonical after_freeze_sha256'):
        audit.validate_protocol_selection(protocol, FREEZE, MODEL)


def test_wrong_specific_freeze_cannot_authorize_another_winner():
    with pytest.raises(RuntimeError, match='Selected freeze differs'):
        audit.validate_protocol_selection({'after_freeze_sha256':'c'*64}, FREEZE, MODEL)


@pytest.mark.parametrize('pin', [None, '', 17, 'x'*64])
def test_present_optional_model_pin_must_be_valid(pin):
    with pytest.raises(RuntimeError, match='after_model_sha256 must be an explicit'):
        audit.validate_protocol_selection({'after_freeze_sha256':FREEZE,'after_model_sha256':pin}, FREEZE, MODEL)


def test_optional_model_pin_cannot_authorize_a_different_model():
    with pytest.raises(RuntimeError, match='After model differs'):
        audit.validate_protocol_selection({'after_freeze_sha256':FREEZE,'after_model_sha256':'c'*64}, FREEZE, MODEL)


def test_specific_freeze_is_valid_with_or_without_additional_model_pin():
    assert audit.validate_protocol_selection({'after_freeze_sha256':FREEZE}, FREEZE, MODEL) is None
    assert audit.validate_protocol_selection({'after_freeze_sha256':FREEZE,'after_model_sha256':MODEL}, FREEZE, MODEL) is None


@pytest.mark.parametrize('protocol', [{'version':1}, {'after_freeze_sha256':'c'*64}])
def test_main_rejects_before_archive_hash_model_construction_or_opening(tmp_path, monkeypatch, protocol):
    protocol_path=tmp_path/'declaration.json';protocol_path.write_text(json.dumps(protocol),encoding='utf-8')
    freeze_path=tmp_path/'freeze.json';freeze_path.write_text('{}',encoding='utf-8')
    after=tmp_path/'after.onnx';after.write_bytes(b'placeholder, never loaded')
    calls=[]
    def model_hash_only(path):
        calls.append(Path(path))
        assert Path(path)==after, 'No baseline/archive hashing allowed before protocol authorization'
        return MODEL
    def forbidden(*args,**kwargs):
        pytest.fail('No model construction or ZIP opening allowed before protocol authorization')
    monkeypatch.setattr(audit,'sha256',model_hash_only)
    monkeypatch.setattr(audit,'FactoryProfile',forbidden)
    monkeypatch.setattr(audit.zipfile,'ZipFile',forbidden)
    monkeypatch.setattr(sys,'argv',['audit','--before',str(tmp_path/'before.onnx'),'--after',str(after),
                                   '--freeze',str(freeze_path),'--protocol',str(protocol_path),'--output',str(tmp_path/'result.json')])
    with pytest.raises(RuntimeError):audit.main()
    assert calls==[after]
    assert not (tmp_path/'result.json').exists()
    assert not (tmp_path/'result-opening-journal.json').exists()


def test_default_remains_preserved_v1_path():
    assert audit.DEFAULT_PROTOCOL.name=='sdd2-test-protocol-before-opening.json'
