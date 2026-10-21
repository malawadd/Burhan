"""Future registered specialists preserve their verified native input contract."""
import numpy as np
import pytest
from edge_runtime.profiles import construction_model as module
from edge_runtime.profiles import construction_lite


def install(monkeypatch):
    class FakeSpecialist:
        def __init__(self,options,spec):
            self.options=options;self.spec=spec;self.mask_rendering='masks';self.exposure_policy=options['exposure_policy'];self.observed=[]
        def infer(self,frame):self.observed.append(frame.copy());return []
        def describe(self):return {'classes':['crack'],'supported_capabilities':['surface_crack'],'exposure_policy':self.exposure_policy}
    monkeypatch.setattr(construction_lite,'LiteCrackConstructionModel',FakeSpecialist)
    monkeypatch.setitem(module.DAMAGE_VARIANTS,'fixture_lite',{'decoder':'lite_semseg','input_size':384,'classes':['crack'],
        'capabilities':['surface_crack'],'exposure_policy':'low_light_gain','diagnostic_only':True,'confidence':.35})
    return module.ConstructionModel


def test_native_raw_reaches_delegate_gain_is_not_applied_twice(monkeypatch):
    model=install(monkeypatch)({'mode':'damage','damage_variant':'fixture_lite'})
    raw=np.full((260,346,3),10,np.uint8);model.infer(raw)
    assert np.array_equal(model.delegate.observed[0],raw)
    assert model.last_input_exposure_gain==8
    assert model.describe()['diagnostic_only'] is True
    assert model.describe()['supported_capabilities']==['surface_crack']


def test_only_crack_calibration_cut_is_forwarded(monkeypatch):
    model=install(monkeypatch)({'mode':'damage','damage_variant':'fixture_lite','class_confidence':{'crack':.25}})
    assert model.delegate.options['confidence']==.25
    assert model.delegate.options['exposure_policy']=='low_light_gain'


def test_specialist_never_accepts_spalling_threshold(monkeypatch):
    factory=install(monkeypatch)
    with pytest.raises(ValueError,match='spalling'):
        factory({'mode':'damage','damage_variant':'fixture_lite','class_confidence':{'spalling':.25}})
