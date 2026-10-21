"""Prepare real DAVIS camera sequences, publisher labels and timestamp-faithful VFR video.

Run in the isolated export environment (h5py and PyAV required), never in the app.
No frames are generated, repeated, interpolated or rearranged. Labels never enter the model input.
"""
from __future__ import annotations
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import av
import h5py
import numpy as np
from construction_annotations import publisher_labels_at_or_after

ROOT = Path(__file__).resolve().parents[1]


def encode_video(path, frames, timestamps, fps=40):
    """One encoded frame per original frame; timestamps use the camera microsecond clock."""
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width = frames.shape[2]; stream.height = frames.shape[1]
        stream.pix_fmt = "yuv420p"; stream.time_base = Fraction(1, 1_000_000)
        stream.codec_context.time_base = Fraction(1, 1_000_000)
        stream.options = {"crf": "18", "preset": "fast", "threads": "1", "bf": "0"}
        first = int(timestamps[0])
        for array, timestamp in zip(frames, timestamps):
            frame = av.VideoFrame.from_ndarray(array, format="gray" if array.ndim == 2 else "bgr24")
            frame.pts = int(timestamp) - first; frame.time_base = stream.time_base
            for packet in stream.encode(frame): container.mux(packet)
        for packet in stream.encode(): container.mux(packet)
    with av.open(str(path)) as check:
        actual = [float(f.time) for f in check.decode(video=0)]
    expected = (timestamps.astype(np.int64) - int(timestamps[0])) / 1e6
    error = max(abs(a-b) for a,b in zip(actual,expected)) if actual else float("inf")
    if len(actual) != len(frames) or error > .00001:
        raise RuntimeError(f"Encoded frame count or timestamp mismatch: {len(actual)} / {len(frames)}, error {error}")
    return {"encoded_frames": len(actual), "max_timestamp_error_s": error,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequences", nargs="+", default=["crack_10", "crack_89", "spalling_35"])
    args = parser.parse_args()
    for name in args.sequences:
        folder = ROOT / "datasets/construction/evcivil" / name
        with h5py.File(folder / f"{name}_frames.h5", "r") as source:
            frames = source["frames"][:]; times = source["frame_timestamp"][:].astype(np.int64)
        label_path = folder / f"{name}_frame_label.npy"
        if not label_path.exists(): label_path = folder / f"{name}_label.npy"
        labels = np.load(label_path, allow_pickle=False)
        # Same published association convention (first annotation at/after camera timestamp),
        # bounded to half one frame period plus 1ms. Never carry a distant box into a new frame.
        period = float(np.median(np.diff(times)))
        tolerance = period / 2 + 1000
        records = []
        for index, timestamp in enumerate(times):
            selected = publisher_labels_at_or_after(labels, timestamp, period)
            expected = [{"id": f"{name}:{index}:{j}", "label": ["crack", "spalling"][int(row[1])],
                         "bbox": [float(v) for v in row[2:6]], "annotation_timestamp_us": int(row[0])}
                        for j,row in enumerate(selected)]
            records.append({"source_frame_index": index, "video_time_s": (int(timestamp)-int(times[0]))/1e6,
                            "source_timestamp_us": int(timestamp), "expected": expected,
                            "annotation_status": "publisher_annotated" if expected else "unlabelled_do_not_assume_negative"})
        np.save(folder / "frames.npy", frames, allow_pickle=False)
        np.save(folder / "timestamps-us.npy", times, allow_pickle=False)
        info = encode_video(folder / f"{name}-original-frames.mp4", frames, times,
                            max(1, round(1e6/period)))
        result = {"schema_version": 1, "sequence": name, "source": "https://doi.org/10.11583/DTU.25924228",
                  "authors": "Kashita Niranjan Udayanga Gangoda Withana Gamage; Silvia Tolu; Matteo Fumagalli; Tobi Delbruck; Cesar Cadena. Annotation contributor: Xuanni Huo.", "license": "CC BY 4.0",
                  "input_shape_hw": list(frames.shape[1:]), "original_camera_frames": len(frames),
                  "timebase": "Original DAVIS grayscale camera timestamps in microseconds, shifted to zero; VFR, one frame per exposure",
                  "label_association": f"Publisher next-at-or-after convention; bounded gap <= {tolerance} microseconds",
                  "annotation_sha256": hashlib.sha256(label_path.read_bytes()).hexdigest(),
                  "annotation_scope": "Publisher defect boxes on public field sequences; no Saudi code acceptance or physical severity labels. Unlabelled frames are excluded from precision/recall.",
                  "encoding_verification": info, "frames": records}
        (folder / "annotations.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps({"sequence":name,"frames":len(frames),"labelled_frames":sum(bool(r['expected']) for r in records),**info}))


if __name__ == "__main__": main()
