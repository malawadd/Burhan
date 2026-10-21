"""PHELE real-box exploratory comparison using the shared bounded detector recipe.

Only frozen public original-photo splits; private owner videos never loaded.
No operational category is released from these small scene-unverified splits.
"""
from pathlib import Path
import argparse
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/"datasets/construction/expansion-public-v1/phele"
OUT = ROOT/"output/evaluation/construction-expansion-phele-v1"
MODELS = ROOT/"models/construction/expansion-phele-v1"
CLASSES = ["peeling_paint", "mold_appearance"]
YOLO_CONFIG = {"epochs": 8, "input_size": 384, "freeze": 4, "batch": 8}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def train_yolo_phele(shared, manifest):
    """Predeclared longer PHELE fit; does not alter shared SDNET recipe."""
    import copy
    import shutil
    import time
    import yaml
    from ultralytics import YOLO
    source = shared.INITIAL/"yolov8n.pt"
    if sha(source) != "f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36":
        raise ValueError("Official initializer changed")
    wrapper = YOLO(str(source))
    config = copy.deepcopy(wrapper.model.yaml)
    config["nc"] = 2
    architecture = OUT/"yolov8n-phele-two-class.yaml"
    architecture.write_text(yaml.safe_dump(config))
    model = YOLO(str(architecture)).load(str(source))
    model.model.names = dict(enumerate(CLASSES))
    cal_rows = [row for row in manifest["records"] if row["split"] == "calibration"]
    test_rows = [row for row in manifest["records"] if row["split"] == "test"]
    def predict(rows):
        started = time.perf_counter()
        outputs = model.predict([str(ROOT/row["image"]) for row in rows], imgsz=384, conf=.01, iou=.45,
                                device="cpu", batch=8, verbose=False)
        elapsed = time.perf_counter()-started
        predictions = []
        for output in outputs:
            predictions.append([{"class_id": int(c), "score": score, "bbox": [xyxy[0], xyxy[1], xyxy[2]-xyxy[0], xyxy[3]-xyxy[1]]}
                                for xyxy, score, c in zip(output.boxes.xyxy.tolist(), output.boxes.conf.tolist(), output.boxes.cls.tolist())])
        return predictions, {"images": len(rows), "seconds": elapsed, "batch": 8, "throughput_fps": len(rows)/elapsed,
                             "input_size": 384, "includes_decode_resize_nms": True, "training_contention": "possible; not isolated runtime speed"}
    before, before_speed = predict(test_rows)
    started = time.perf_counter()
    model.train(data=str(DATA/"prepared/dataset.yaml"), epochs=8, imgsz=384, batch=8, device="cpu", workers=0,
                freeze=4, optimizer="AdamW", lr0=.002, lrf=.1, warmup_epochs=0, nbs=8, seed=20261021,
                deterministic=True, plots=False, save=True, cache=False, val=False, amp=False,
                mosaic=0, mixup=0, fliplr=0, hsv_h=0, hsv_s=0, hsv_v=0, degrees=0, translate=0,
                scale=0, shear=0, perspective=0, project=str(OUT), name="yolo-phele-fit", exist_ok=False, verbose=False)
    elapsed = time.perf_counter()-started
    model = YOLO(str(OUT/"yolo-phele-fit/weights/last.pt"))
    cal, speed = predict(cal_rows)
    test, _ = predict(test_rows)
    result, threshold = shared.summarize("yolov8n", before, None, cal, test, cal_rows, test_rows, speed)
    result.update(train_seconds=elapsed, fit_configuration=YOLO_CONFIG, before_prediction_speed=before_speed,
                  before_scope="Fresh two-class semantic head on official COCO features, not a previous deployed wall detector")
    shared.write(OUT/"yolov8n-predictions.json", {"classes": CLASSES, "before_test": before, "after_cal": cal, "after_test": test,
        "cal_members": [row["original_member"] for row in cal_rows], "test_members": [row["original_member"] for row in test_rows]})
    destination = MODELS/"yolov8n"
    destination.mkdir(parents=True, exist_ok=True)
    exported = Path(model.export(format="onnx", imgsz=384, opset=17, simplify=False, dynamic=False, batch=1, device="cpu"))
    shutil.copyfile(exported, destination/"model.onnx")
    shared.write(destination/"manifest.json", {"architecture": "yolov8n", "input_size": 384, "model_file": "model.onnx",
        "model_sha256": sha(destination/"model.onnx"), "classes": CLASSES, "confidence": threshold,
        "preprocessing": "letterbox_rgb_0_1", "source": manifest["source"], "license": "PHELE CC BY 4.0; Ultralytics AGPL-3.0",
        "validation_status": "research_only", "evaluation_report": str((OUT/"yolov8n-results.json").resolve()), "private_videos_used": False})
    shared.write(OUT/"yolov8n-results.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["yolo", "ssd", "both"], default="both")
    args = parser.parse_args()
    original = json.loads((DATA/"prepared-manifest.json").read_text())
    if original["classes"] != CLASSES or original["private_owner_videos_used"]:
        raise ValueError("Frozen PHELE source semantics changed")
    manifest = {"classes": CLASSES, "private_videos_used": False,
                "source": "https://data.mendeley.com/datasets/yzsxcvc8dp/4", "records": []}
    for row in original["records"]:
        if sha(ROOT/row["prepared_image"]) != row["prepared_image_sha256"] or sha(ROOT/row["prepared_label"]) != row["prepared_label_sha256"]:
            raise ValueError("Frozen prepared PHELE data changed")
        boxes = []
        for box in row["boxes"]:
            cx, cy, width, height = box["prepared_yolo"]
            boxes.append({"class_id": box["class_id"], "bbox": [(cx-width/2)*640, (cy-height/2)*640, width*640, height*640]})
        manifest["records"].append({"image": row["prepared_image"], "source_sha256": row["prepared_image_sha256"],
                                    "split": "calibration" if row["split"] == "cal" else row["split"], "boxes": boxes,
                                    "original_member": row["member"], "image_duplicate_group": row["image_duplicate_group"]})
    import train_sdnet_expansion as shared
    import random
    import numpy as np
    import ultralytics.utils.torch_utils as torch_utils
    from ultralytics.utils.callbacks import default_callbacks
    # Ultralytics CPU select_device otherwise resets torch threads from this
    # module constant even after a caller explicitly selected one thread.
    torch_utils.NUM_THREADS = 1
    shared.torch.set_num_threads(1)
    shared.DATA = DATA/"prepared"
    shared.OUT = OUT
    shared.MODELS = MODELS
    shared.CLASSES = CLASSES
    OUT.mkdir(parents=True, exist_ok=True)
    shared.torch.manual_seed(20261021)
    np.random.seed(20261021)
    random.seed(20261021)
    protocol = {"source_prepared_manifest_sha256": sha(DATA/"prepared-manifest.json"), "phele_trainer_sha256": sha(Path(__file__)),
                "shared_trainer_sha256": sha(Path(shared.__file__)), "classes": CLASSES,
                "source_class_mapping": {"5:Peeling paint": "peeling_paint", "9:Mold": "mold_appearance"},
                "publisher_record_ambiguity": "Record says Mold/dampness; actual annotation Mold. No damp-only detector or moisture reading claim.",
                "yolo_epochs": 8, "yolo_image_size": 384, "yolo_freeze": 4, "ssd_steps": 200, "batch": 8,
                "threads": 1, "selection": "Fixed final checkpoint; frozen CAL threshold grid only; TEST not used for selection.",
                "release_status": "research_only", "scene_independence": "not_established", "private_videos_used": False}
    shared.write(OUT/"protocol-before-training.json", protocol)
    shared.write(OUT/"training-manifest.json", manifest)
    def actual_yolo_threads(trainer):
        actual = shared.torch.get_num_threads()
        shared.write(OUT/"actual-yolo-training-threads.json", {"torch_num_threads": actual, "ultralytics_NUM_THREADS": torch_utils.NUM_THREADS})
        if actual != 1:
            raise ValueError("Bounded one-thread training configuration changed")
    default_callbacks["on_train_start"].append(actual_yolo_threads)
    results = {}
    def finish(name, result):
        result["release_blocker"] = "Physical wall/site independence unavailable; fewer than 50 independent positive and negative examples per class; photo-only box labels."
        result["source_class_mapping"] = protocol["source_class_mapping"]
        shared.write(OUT/f"{name}-results.json", result)
        asset_path = MODELS/name/"manifest.json"
        asset = json.loads(asset_path.read_text())
        asset.update(inspection_task="walls_finishes", source_class_mapping=protocol["source_class_mapping"],
                     license="PHELE public data CC BY 4.0; "+("Ultralytics AGPL-3.0 code/weights conditions" if name == "yolov8n" else "TorchVision BSD-3-Clause and COCO pretraining conditions"),
                     validation_status="research_only", release_blocker=result["release_blocker"],
                     source_manifest_sha256=sha(DATA/"prepared-manifest.json"))
        shared.write(asset_path, asset)
        results[name] = result
        shared.write(OUT/"comparison-summary.json", {"protocol": protocol, "results": results})
    if args.model in ("yolo", "both"):
        finish("yolov8n", train_yolo_phele(shared, manifest))
    shared.torch.set_num_threads(1)
    if args.model in ("ssd", "both"):
        shared.write(OUT/"actual-ssd-training-threads.json", {"torch_num_threads": shared.torch.get_num_threads()})
        finish("ssdlite320", shared.train_ssd(manifest))
    shared.write(OUT/"comparison-summary.json", {"protocol": protocol, "results": results})


if __name__ == "__main__":
    main()
