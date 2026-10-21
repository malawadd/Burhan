import threading
import time

import cv2
import numpy as np

from edge_runtime.media import MediaWorker


def test_slow_preview_is_bounded_and_never_blocks_inspection(monkeypatch):
    entered, release = threading.Event(), threading.Event()
    original = cv2.imencode
    seen, delivered = [], []
    def encode(extension,image,options):
        seen.append(image.copy())
        entered.set()
        assert release.wait(3)
        return original(extension,image,options)
    monkeypatch.setattr('edge_runtime.media.cv2.imencode',encode)
    image=np.full((40,40,3),30,np.uint8)
    worker=MediaWorker(lambda jpeg,seq:delivered.append(seq),lambda *args:None,
                       preview_fps=10000,max_bytes=image.nbytes*2)
    try:
        assert worker.preview(image,1)
        assert entered.wait(1)
        image[:]=90
        time.sleep(.001)
        assert worker.preview(image,2)
        image[:]=150
        time.sleep(.001)
        assert worker.preview(image,3)
        assert not worker.evidence(image,'factory','session',{})
        state=worker.status()
        assert state['pending_bytes']==image.nbytes*2
        assert state['preview_replaced']==1
        assert state['evidence_rejected']==1
        assert int(seen[0][0,0,0])==30
        assert not worker.close(timeout=.001)
    finally:
        release.set()
        assert worker.close()
    assert delivered==[1,3]
    assert worker.status()['pending_bytes']==0


def test_preview_resize_does_not_change_exact_event_image():
    previews,events=[],[]
    worker=MediaWorker(lambda jpeg,seq:previews.append(jpeg),lambda *args:events.append(args))
    frame=np.full((1200,1000,3),130,np.uint8)
    payload={'sequence':7,'fields':{'value':'original'}}
    try:
        assert worker.evidence(frame,'freight','original-session',payload)
        assert worker.preview(frame,7)
        payload['fields']['value']='mutated'
        assert worker.close()
    finally:worker.close()
    assert len(previews)==len(events)==1
    assert events[0][:2]==('freight','original-session')
    assert events[0][2]['fields']['value']=='original'
    preview=cv2.imdecode(np.frombuffer(previews[0],np.uint8),cv2.IMREAD_COLOR)
    event=cv2.imdecode(np.frombuffer(events[0][3],np.uint8),cv2.IMREAD_COLOR)
    assert preview.shape[:2]==(960,800)
    assert event.shape==frame.shape


def test_evidence_encoding_failure_retains_metadata(monkeypatch):
    monkeypatch.setattr('edge_runtime.media.cv2.imencode',lambda *args:(False,None))
    records=[]
    worker=MediaWorker(lambda *args:None,lambda *args:records.append(args))
    try:
        assert worker.evidence(np.zeros((20,20,3),np.uint8),'factory','s',{'sequence':5})
        assert worker.close()
    finally:worker.close()
    assert records[0][2]['sequence']==5
    assert 'evidence_image_error' in records[0][2]
    assert records[0][3] is None
    assert worker.status()['errors']==1


def test_preview_can_be_disabled_independently_of_event_images():
    worker=MediaWorker(lambda *args:None,lambda *args:None,preview_fps=0)
    try:
        assert not worker.preview(np.zeros((20,20,3),np.uint8),1)
        assert worker.status()['pending_bytes']==0
    finally:assert worker.close()
