"""Preparation ordering and cleanup; fixtures never establish model accuracy."""
import threading,time
import numpy as np
import pytest
from edge_runtime.runtime import EdgeRuntime,RuntimeConfig
from edge_runtime.contracts import InspectionResult


def test_capture_waits_for_actual_inference_worker_preparation(tmp_path,monkeypatch):
    events=[]
    class Profile:
        def prepare_worker(self):events.append(('prepared',threading.current_thread().name))
        def describe(self):return {'prepared':bool(events)}
        def process(self,frame,context):return InspectionResult('no_candidate','Fixture')
    class Source:
        backend='fixture'
        def __init__(self,*args):pass
        def open(self):
            assert events==[('prepared','edge-inference')]
            events.append(('opened',threading.current_thread().name))
        def read(self):return None,None
        def close(self):pass
    monkeypatch.setattr('edge_runtime.runtime.create_profile',lambda *args:Profile())
    monkeypatch.setattr('edge_runtime.runtime.LocalSource',Source)
    edge=EdgeRuntime(tmp_path)
    try:
        edge.start(RuntimeConfig(source='file:fixture'))
        end=time.perf_counter()+2
        while len(events)<2 and time.perf_counter()<end:time.sleep(.01)
        assert events[1][0]=='opened'
        assert edge.snapshot()['metrics']['captured']==0
        assert edge.description['prepared']
    finally:edge.close()


@pytest.mark.parametrize('mode',['failure','timeout'])
def test_worker_preparation_failure_never_opens_capture_and_cleans_media(tmp_path,monkeypatch,mode):
    release=threading.Event();opened=[]
    class Profile:
        def prepare_worker(self):
            if mode=='failure':raise ValueError('fixture preparation failed')
            release.wait(2)
        def describe(self):return {}
    class Source:
        def __init__(self,*args):opened.append(True)
    monkeypatch.setattr('edge_runtime.runtime.create_profile',lambda *args:Profile())
    monkeypatch.setattr('edge_runtime.runtime.LocalSource',Source)
    monkeypatch.setattr(EdgeRuntime,'WORKER_PREPARATION_TIMEOUT_SECONDS',.02)
    edge=EdgeRuntime(tmp_path)
    try:
        with pytest.raises(RuntimeError,match='preparation'):
            edge.start(RuntimeConfig(source='file:fixture'))
        assert not opened
        assert edge.state=='startup_error'
        assert edge.snapshot()['metrics']['captured']==0
        assert not edge.media.status()['worker_alive']
    finally:
        release.set();edge.close()
