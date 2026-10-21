import numpy as np

from edge_runtime.profiles import construction


def test_worker_warmup_compiles_model_without_creating_observations(monkeypatch):
    calls=[]
    class Model:
        mode='damage'
        input_size=64
        spec={'capabilities':['surface_crack']}
        def __init__(self,options):
            pass
        def describe(self):
            return {'supported_capabilities':['surface_crack']}
        def infer(self,frame):
            calls.append(frame.copy())
            return [{'label':'crack','bbox':(1,1,10,10),'score':.9}]
    monkeypatch.setattr(construction,'ConstructionModel',Model)
    profile=construction.ConstructionProfile({'mode':'damage'})
    report=profile.prepare_worker()
    assert len(calls)==3 and all(frame.shape==(64,64,3) for frame in calls)
    assert all(np.all(frame==127) for frame in calls)
    assert report['worker_warmup_inspected_observations']==0
    assert profile._sequence is None and profile._identity is None and profile._tracks==[]
    assert profile.describe()['model_initialization']==report
