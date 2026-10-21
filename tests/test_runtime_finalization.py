import time

import numpy as np

from edge_runtime.contracts import InspectionResult
from edge_runtime.runtime import EdgeRuntime, RuntimeConfig


def until(predicate, timeout=5):
    end = time.monotonic()+timeout
    while time.monotonic()<end:
        if predicate():
            return
        time.sleep(.02)
    raise AssertionError('Runtime did not complete')


class FiniteSource:
    backend = 'test-file'
    def __init__(self, *args):
        self.index = 0
    def open(self):
        pass
    def read(self):
        self.index += 1
        return (np.zeros((64,64,3),np.uint8), 0) if self.index<=3 else (None,None)
    def close(self):
        pass


class AsyncFinalProfile:
    def __init__(self):
        self.final_calls = 0
    def describe(self):
        return {'profile_version':'test-async'}
    def process(self, frame, context):
        self.context = context
        return InspectionResult('observing','OCR pending')
    def finish_source(self, timeout=.2):
        self.final_calls += 1
        if self.final_calls<3:
            return None
        return InspectionResult('accepted','Original page finalized',
            metrics={'source_finished':True}, emit_event=True,event_key='original-page')
    def source_completion_state(self):
        return {'pending_ocr':self.final_calls<3}


def test_eof_async_completion_preserves_original_frame_and_guidance_counts(tmp_path,monkeypatch):
    profile = AsyncFinalProfile()
    monkeypatch.setattr('edge_runtime.runtime.LocalSource',FiniteSource)
    monkeypatch.setattr('edge_runtime.runtime.create_profile',lambda *args:profile)
    edge = EdgeRuntime(tmp_path)
    try:
        edge.start(RuntimeConfig(profile='freight',source='file:test',input_fps=10,deadline_ms=2000))
        until(lambda:edge.snapshot()['state']=='eof')
        state=edge.snapshot()
        assert state['metrics']['captured']==3
        assert state['metrics']['processed']==3
        assert state['metrics']['async_completions']==1
        assert len(edge._latencies)==3
        result=state['last_result']
        assert result['status']=='accepted'
        assert result['sequence']==profile.context.sequence==3
        assert result['captured_monotonic']==profile.context.captured_monotonic
        assert result['async_source_completion'] is True
        assert result['host_receipt_to_decision_ms'] is None
        assert result['late'] is None
        until(lambda:any(e['kind']=='document_completed' for e in edge.store.list(20)))
    finally:
        edge.close()
