import hashlib
import numpy as np
import pytest
from edge_runtime.profiles import construction_tile_adapter as module


@pytest.fixture
def setup_tile(tmp_path,monkeypatch):
    asset=tmp_path/'models/construction/tile.json';asset.parent.mkdir(parents=True);asset.write_text('{}')
    mask=np.array([[True,False],[False,True]])
    class Backend:
        def __init__(self,path,threads):
            self.threshold=2.;self.spec={'encoder':{'sha256':'encoder'},'bank':{'sha256':'bank'}}
        def infer(self,frame):return [{'label':'tile_surface_anomaly','bbox':[1,2,3,4],'mask':mask,'anomaly_distance':6.}]
        def describe(self):return {'scope':'originalfittedmodel'}
    monkeypatch.setattr(module,'ROOT',tmp_path);monkeypatch.setattr(module,'TileSurfaceModel',Backend)
    spec={'file':'tile.json','sha256':hashlib.sha256(asset.read_bytes()).hexdigest(),'classes':['tile_surface_anomaly'],'source':'manufacturer'}
    return spec,mask


def test_wrapper_adds_score_without_changing_localization(setup_tile):
    spec,mask=setup_tile
    model=module.TileConstructionModel({},spec)
    result=model.infer(np.zeros((2,2,3),np.uint8))[0]
    assert result['score']==.75
    assert result['anomaly_distance']==6.
    assert result['bbox']==[1,2,3,4] and result['mask'] is mask
    assert model.describe()['diagnostic_only'] is True
    assert model.describe()['supported_capabilities']==['tile_surface_anomaly']


@pytest.mark.parametrize('options',[{'damage_classes':['crack']},{'confidence':.4},{'class_confidence':{'tile_surface_anomaly':.3}},
                                  {'engine':'openvino'},{'openvino_device':'GPU'},{'exposure_policy':'low_light_gain'}])
def test_invalid_semantics_or_unmeasured_execution_rejected(setup_tile,options):
    spec,_=setup_tile
    with pytest.raises(ValueError):module.TileConstructionModel(options,spec)


def test_tampered_tile_metadata_rejected(setup_tile):
    spec,_=setup_tile
    spec['sha256']='0'*64
    with pytest.raises(ValueError):module.TileConstructionModel({},spec)
