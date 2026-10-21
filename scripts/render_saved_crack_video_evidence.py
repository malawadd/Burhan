"""Re-render saved real masks on visible input, with no NN or score changes."""
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from compare_construction_development_videos import load_cases
from edge_runtime.profiles.construction_preprocessing import camera_exposure

SOURCE = ROOT / "output/evaluation/construction-pixel-videos"
OUT = SOURCE / "display-v2"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def render(raw, native, raw_mask, served_mask, record, name, tag):
    canvas = np.full((1300, 1420, 3), (24, 27, 30), np.uint8)
    gt, trace = native.copy(), native.copy()
    matched = {p.get("matched_annotation_id") for p in record["predictions"]}
    for item in record["expected"]:
        x, y, w, h = map(round, item["bbox"])
        cv2.rectangle(gt, (x, y), (x + w, y + h),
                      (245, 210, 30) if item["id"] in matched else (70, 70, 245), 1)
    trace[raw_mask] = (.3 * trace[raw_mask] + .7 * np.array([0, 210, 245])).astype(np.uint8)
    trace[served_mask] = (.3 * trace[served_mask] + .7 * np.array([245, 160, 30])).astype(np.uint8)
    panes = [(raw, "RAW CAMERA - original unchanged"),
             (native, f"NATIVE MODEL INPUT - gain {record['input_exposure_gain']:.2f}x, before resize"),
             (gt, "PUBLISHER BOXES on model input - cyan hit / red missed"),
             (trace, "ACTUAL MASKS on model input - colors are NOT pixel GT")]
    for i, (array, title) in enumerate(panes):
        x, y = 8 + (i % 2) * 708, 95 + (i // 2) * 560
        canvas[y:y + 520, x:x + 692] = cv2.resize(array, (692, 520), interpolation=cv2.INTER_NEAREST)
        cv2.putText(canvas, title, (x, y - 13), cv2.FONT_HERSHEY_SIMPLEX, .48, (235, 235, 235), 1)
    for i, line in enumerate([
        f"REAL CAMERA | {tag} | {name} | FIXED >.50 | NOT QUALIFIED",
        f"Original frame {record['source_frame_index']} at {record['video_time_s']:.6f}s | complete recording, every source exposure"]):
        cv2.putText(canvas, line, (8, 25 + 25 * i), cv2.FONT_HERSHEY_SIMPLEX, .56, (235, 235, 235), 1)
    counts = (f"THIS FRAME: {record['localized_matches']} box matches / {record['missed_defects']} missed / {record['false_alerts']} unmatched"
              if record['evaluated'] else "UNSCORED EXPOSURE: missing publisher labels do not prove absence of cracks")
    footer = [counts,
              "YELLOW = raw threshold mask; BLUE = served area4/top100 mask. Neither color asserts pixel correctness.",
              "Scoring: publisher box IoU >=.50. No video pixel GT, spalling support, physical width or code acceptance.",
              "Gamage et al., ev-CIVIL/DTU, CC BY4.0. Display-only derivative of saved masks; NO new model inference."]
    for i, line in enumerate(footer):
        cv2.putText(canvas, line, (8, 1210 + 25 * i), cv2.FONT_HERSHEY_SIMPLEX, .45, (235, 235, 235), 1)
    return canvas


def main():
    import av
    cv2.setNumThreads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((SOURCE / "fixed-pair-three-video-protocol-before-inference.json").read_text(encoding="utf-8"))
    pairs = json.loads((SOURCE / "paired-summary.json").read_text(encoding="utf-8"))
    cases = {case["name"]: case for case in load_cases(protocol)}
    result = []
    for tag, rows in pairs.items():
        for row in rows:
            original = ROOT / row["report"]
            if sha(original) != row["report_sha256"]:
                raise ValueError("Original completed report changed")
            report = json.loads(original.read_text(encoding="utf-8"))
            packed_path = ROOT / report["native_mask_evidence"]["path"]
            if sha(packed_path) != report["native_mask_evidence"]["sha256"]:
                raise ValueError("Saved actual masks changed")
            stem = OUT / original.stem
            video, target = stem.with_suffix(".mp4"), stem.with_suffix(".json")
            if video.exists() or target.exists():
                raise ValueError("Preserve previous or partial presentation export")
            case = cases[row["sequence"]]
            with np.load(packed_path, allow_pickle=False) as masks:
                shape = tuple(masks["native_shape_hw"])
                records = report["frames"]
                if len(records) != len(case["frames"]) or not np.array_equal(masks["source_timestamps_us"], case["times"]):
                    raise ValueError("Actual full source timeline differs")
                first = int(case["times"][0])
                fps = max(1, round(1e6 / np.median(np.diff(case["times"]))))
                with av.open(str(video), "w") as container:
                    stream = container.add_stream("libx264", rate=fps)
                    stream.width, stream.height, stream.pix_fmt = 1420, 1300, "yuv420p"
                    stream.time_base = stream.codec_context.time_base = Fraction(1, 1_000_000)
                    stream.options = {"crf": "18", "preset": "fast", "threads": "1", "bf": "0"}
                    saved_preview = False
                    for i, (gray, record, stamp) in enumerate(zip(case["frames"], records, case["times"])):
                        raw = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                        native, gain = camera_exposure(raw, "low_light_gain")
                        if (hashlib.sha256(raw.tobytes()).hexdigest() != record["raw_native_bgr_sha256"]
                                or hashlib.sha256(native.tobytes()).hexdigest() != record["model_input_native_bgr_sha256"]
                                or gain != record["input_exposure_gain"]
                                or int(masks["source_frame_indices"][i]) != record["source_frame_index"]):
                            raise ValueError("Raw/input/source identity differs")
                        decoded = []
                        for field, expected in [("raw_packed", "raw_native_mask_sha256"), ("served_packed", "served_native_mask_sha256")]:
                            mask = np.unpackbits(masks[field][i], bitorder="little")[:int(np.prod(shape))].reshape(shape).astype(bool)
                            if hashlib.sha256(mask.tobytes()).hexdigest() != record[expected]:
                                raise ValueError("Native saved mask differs")
                            decoded.append(mask)
                        canvas = render(raw, native, *decoded, record, case["name"], tag)
                        if not saved_preview and record["expected"]:
                            if not cv2.imwrite(str(stem) + "-fixed-first-annotated.jpg", canvas):
                                raise IOError("Could not save display preview")
                            saved_preview = True
                        frame = av.VideoFrame.from_ndarray(canvas, format="bgr24")
                        frame.pts, frame.time_base = int(stamp) - first, stream.time_base
                        for packet in stream.encode(frame):
                            container.mux(packet)
                    for packet in stream.encode():
                        container.mux(packet)
            with av.open(str(video)) as check:
                pts = [Fraction(f.pts) * f.time_base * 1_000_000 for f in check.decode(video=0)]
            if pts != [int(t) - first for t in case["times"]]:
                raise ValueError("Display derivative changed original exposure count or timestamps")
            report["derived_video"] = {"encoded_frames": len(pts), "max_timestamp_error_s": 0, "sha256": sha(video)}
            report["presentation_derivative"] = {
                "original_report": str(original.relative_to(ROOT)), "original_report_sha256": sha(original),
                "original_video": row["video"], "original_video_sha256": row["video_sha256"],
                "renderer_sha256": sha(Path(__file__)), "model_inference_calls": 0,
                "change": "External publisher boxes and identical actual masks over the same visible native gain input; original RAW pane retained; footer outside panes.",
                "unchanged": "All700 frame records, model, masks, summary counts and scoring policy remain exactly as originally computed."}
            if report["frames"] != json.loads(original.read_text(encoding="utf-8"))["frames"]:
                raise ValueError("Scored records changed")
            target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            result.append({"id": "pixel_fixed50_" + tag.lower() + "-" + case["name"],
                           "video": str(video.relative_to(ROOT)), "video_sha256": sha(video),
                           "report": str(target.relative_to(ROOT)), "report_sha256": sha(target)})
            print(f"Rendered saved masks only: {tag}/{case['name']} ({len(pts)} real frames)", flush=True)
    (OUT / "all-six-presentation-manifest.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
