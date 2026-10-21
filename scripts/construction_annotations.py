"""Publisher's timestamp association, shared by construction preparation and scoring."""
import numpy as np


def publisher_labels_at_or_after(labels,timestamp,frame_period_us):
    candidates=labels[labels[:,0]>=timestamp]
    if not len(candidates):return labels[:0]
    chosen=np.min(candidates[:,0])
    if chosen-timestamp>frame_period_us/2+1000:return labels[:0]
    return labels[labels[:,0]==chosen]
