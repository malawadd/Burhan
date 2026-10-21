"""Pure image-only form of the existing ev-CIVIL non-neural leakage screen.

Kept independent of optional HDF5 libraries; no model or annotation access.
"""
import hashlib
import cv2
import numpy as np


def signature(frame):
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) if frame.ndim==3 else frame
    small=cv2.resize(gray,(9,8),interpolation=cv2.INTER_AREA)
    bits=np.packbits((small[:,1:]>small[:,:-1]).reshape(-1))
    reduced=cv2.resize(gray,(32,24),interpolation=cv2.INTER_AREA).astype(np.float32).reshape(-1)
    reduced-=reduced.mean();norm=np.linalg.norm(reduced)
    if norm:reduced/=norm
    digest=hashlib.sha256(str(gray.shape).encode()+gray.tobytes()).hexdigest()
    return digest,bits,reduced
