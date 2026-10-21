"""Bounded media work outside the inspection deadline; no model/frame skipping."""
from collections import deque
from copy import deepcopy
import threading
import time

import cv2


class MediaWorker:
    def __init__(self, on_preview, on_evidence, preview_fps=6, max_bytes=32*1024*1024):
        self.on_preview, self.on_evidence = on_preview, on_evidence
        self.preview_fps, self.max_bytes = preview_fps, max_bytes
        self._condition = threading.Condition()
        self._preview = None
        self._events = deque()
        self._bytes = self._inflight = 0
        self._closed = False
        self._next_preview = 0.
        self._counts = dict(preview_encoded=0, preview_replaced=0, preview_throttled=0,
                            evidence_encoded=0, evidence_rejected=0, errors=0)
        self._last_encode_ms = 0.
        self._thread = threading.Thread(target=self._run, name='edge-media', daemon=True)
        self._thread.start()

    def preview(self, frame, sequence=None):
        now = time.perf_counter()
        with self._condition:
            if self._closed or not self.preview_fps:
                return False
            if now < self._next_preview:
                self._counts['preview_throttled'] += 1
                return False
            self._next_preview = now + 1/self.preview_fps
            if self._preview is not None:
                self._bytes -= self._preview[0].nbytes
                self._preview = None
                self._counts['preview_replaced'] += 1
            if self._bytes + frame.nbytes > self.max_bytes:
                self._counts['preview_replaced'] += 1
                return False
            snapshot = frame.copy()
            self._preview = (snapshot, sequence)
            self._bytes += snapshot.nbytes
            self._condition.notify()
            return True

    def evidence(self, frame, profile, session, payload):
        with self._condition:
            if self._closed or len(self._events)>=4 or self._bytes+frame.nbytes>self.max_bytes:
                self._counts['evidence_rejected'] += 1
                return False
            snapshot = frame.copy()
            self._events.append((snapshot, profile, session, deepcopy(payload)))
            self._bytes += snapshot.nbytes
            self._condition.notify()
            return True

    def _run(self):
        while True:
            with self._condition:
                self._condition.wait_for(lambda:self._closed or self._events or self._preview is not None)
                if self._events:
                    work = self._events.popleft()
                    evidence = True
                elif self._preview is not None:
                    work, self._preview = self._preview, None
                    evidence = False
                elif self._closed:
                    return
                else:
                    continue
                frame = work[0]
                self._inflight += 1
            begun = time.perf_counter()
            try:
                image = frame
                if not evidence and max(image.shape[:2])>960:
                    h,w=image.shape[:2]
                    image=cv2.resize(image,(max(1,round(w*960/max(h,w))),max(1,round(h*960/max(h,w)))),interpolation=cv2.INTER_AREA)
                ok, data=cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,78])
                if not ok:
                    raise RuntimeError('JPEG encoder returned no image')
                jpeg=data.tobytes()
                if evidence:
                    self.on_evidence(work[1],work[2],work[3],jpeg)
                else:
                    self.on_preview(jpeg,work[1])
                with self._condition:
                    self._counts['evidence_encoded' if evidence else 'preview_encoded'] += 1
            except Exception as exc:
                with self._condition:
                    self._counts['errors'] += 1
                if evidence:
                    payload=dict(work[3]);payload['evidence_image_error']=f'{type(exc).__name__}: {exc}'
                    try:self.on_evidence(work[1],work[2],payload,None)
                    except Exception:pass  # storage has its own explicit loss/error counters
            finally:
                with self._condition:
                    self._last_encode_ms=(time.perf_counter()-begun)*1000
                    self._bytes -= frame.nbytes
                    self._inflight -= 1

    def close(self, timeout=3):
        with self._condition:
            self._closed=True
            self._condition.notify_all()
        self._thread.join(timeout)
        return not self._thread.is_alive()

    def status(self):
        with self._condition:
            return {**self._counts,'pending_bytes':self._bytes,'byte_limit':self.max_bytes,
                'pending_evidence':len(self._events),'pending_preview':self._preview is not None,
                'in_flight':self._inflight,'worker_alive':self._thread.is_alive(),
                'last_encode_ms':round(self._last_encode_ms,2),'preview_fps_limit':self.preview_fps}
