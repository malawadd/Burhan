"""Hardware-free contract checks; these fixtures are not accuracy evidence."""
import hashlib
import json
import numpy as np
import pytest
from edge_runtime.profiles.factory_defect import SurfaceDefectModel
from edge_runtime.profiles.factory_encoder import FactorySegmentation


def test_neural_model_requires_auto_and_explicit_gpu_backend(tmp_path):
    path=tmp_path/'model.onnx';path.write_bytes(b'contract-fixture')
    path.with_suffix('.onnx.json').write_text(json.dumps({'feature_version':'surface_mobilenet_seg_v1',
                                                       'model_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}),encoding='utf-8')
    with pytest.raises(ValueError,match='defect_backend=auto'):
        SurfaceDefectModel(path,backend='native_cascade')


@pytest.mark.parametrize('scale',[1,2])
def test_neural_logits_scores_and_regions_map_to_original_coordinates(scale):
    model=FactorySegmentation.__new__(FactorySegmentation)
    model.input='input';model.output='output';model.output_logits=True;model.execution_devices=['GPU.0']
    logits=np.full((1,1,640*scale,256*scale),-20.,dtype=np.float32)
    logits[0,0,200*scale:240*scale,80*scale:120*scale]=2.
    class Request:
        def infer(self,value):
            assert value['input'].shape==(1,3,1280,512)
            assert value['input'].dtype==np.uint8
            return {'output':logits}
    model.request=Request()
    frame=np.zeros((160,64,3),dtype=np.uint8)
    scores,candidates,metrics=model.predict(frame,.8,8)
    assert len(candidates)==1 and candidates[0]['bbox']==(20,50,10,10)
    assert candidates[0]['score']==pytest.approx(1/(1+np.exp(-2)),abs=1e-6)
    assert np.isfinite(scores).all() and metrics['encoder_execution_devices']==['GPU.0']
    assert metrics['model_grid_width']==256*scale and metrics['model_grid_height']==640*scale
    assert logits[0,0,0,0]==-20.  # Runtime does not mutate the model's output buffer.
