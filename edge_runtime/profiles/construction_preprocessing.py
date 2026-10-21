"""Shared deterministic camera preprocessing, independent of model architecture."""
from __future__ import annotations
import cv2
import numpy as np


def camera_exposure(frame, policy='off'):
    """Bounded full-frame global gain, with no spatial synthesis or GT region."""
    if policy == 'off':
        return frame, 1.0
    if policy != 'low_light_gain':
        raise ValueError('Unknown exposure policy')
    p99 = float(np.percentile(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 99))
    gain = min(8.0, 128.0/max(p99, 1.0)) if p99 < 64 else 1.0
    if gain == 1.0:
        return frame, gain
    return np.clip(frame.astype(np.float32)*gain, 0, 255).astype(np.uint8), gain
