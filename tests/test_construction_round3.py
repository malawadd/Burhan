import numpy as np
from edge_runtime.profiles.construction_model import prepare,decode_damage_boxes
import edge_runtime.profiles.construction as construction
from edge_runtime.contracts import FrameContext


def test_damage_letterbox_obeys_actual_416_contract():
    tensor,transform=prepare(np.zeros((100,200,3),np.uint8),'damage',size=416)
    assert tensor.shape==(1,3,416,416)
    assert transform==(2.08,2.08,0,104)
    assert np.allclose(tensor[:,:,0,:],114/255)


def test_damage_class_thresholds_preserve_true_spall_and_crack_identity():
    prediction=np.zeros((1,6,3549),np.float32)
    prediction[0,:,0]=[100,100,40,40,.7,.1]
    prediction[0,:,1]=[300,300,40,40,.1,.4]
    rows=decode_damage_boxes([prediction],(1,1,0,0),(416,416),416,['spalling','crack'],.1,.45,{'spalling':.8,'crack':.3})
    assert [r['label'] for r in rows]==['crack']
    rows=decode_damage_boxes([prediction],(1,1,0,0),(416,416),416,['spalling','crack'],.1,.45,{'spalling':.6,'crack':.3})
    assert [r['label'] for r in rows]==['spalling','crack']


def test_selected_capabilities_and_inspector_marking_are_not_internal_void(monkeypatch):
    class Model:
        mode='damage';mask_rendering='boxes';spec={'capabilities':['surface_spalling','surface_hollowareas']}
        def __init__(self,options):pass
        def describe(self):return {'supported_capabilities':['surface_hollowareas']}
        def infer(self,frame):return [{'label':'hollowareas','class_id':6,'bbox':(10,10,30,30),'score':.8}]
    monkeypatch.setattr(construction,'ConstructionModel',Model)
    profile=construction.ConstructionProfile({'mode':'damage'})
    result=profile.process(np.zeros((80,100,3),np.uint8),FrameContext(1,1,'camera:0','a'))
    assert result.domain_state['supported_capabilities']==['surface_hollowareas']
    detail=result.domain_state['review_findings'][0]
    assert 'chalk' in detail['label']
    assert 'not hidden material' in detail['why_it_matters']
    assert 'sounding-test record' in detail['next_check']
    assert result.domain_state['code_compliance']=='not_determined'
