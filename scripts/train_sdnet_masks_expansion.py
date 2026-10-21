"""Actual-polygon SDNET mask comparison; final fixed budgets, CAL-only thresholds.

CPU one thread. No private video, generated damage, box-derived masks, or TEST selection.
Run .venv-roads-export/Scripts/python.exe scripts/train_sdnet_masks_expansion.py
"""
import os
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["YOLO_OFFLINE"] = "true"

import argparse
import copy
import hashlib
import json
import random
import shutil
import time
import urllib.request
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torchvision.models.segmentation import lraspp_mobilenet_v3_large

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets/construction/expansion-public-v1/sdnet2025/prepared/manifest.json"
OUT = ROOT / "output/evaluation/construction-expansion-sdnet-masks"
MODELS = ROOT / "models/construction/expansion-sdnet-masks"
CLASSES = ["visibly_unseated_connection", "fastener_absence_appearance"]
GRID = [.05, .10, .20, .35, .50, .65, .80, .90]
SEED = 20261021
SIZE = 320
MIN_AREA = 8
MEAN = np.array([.485, .456, .406], np.float32)
STD = np.array([.229, .224, .225], np.float32)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
cv2.setNumThreads(1)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf8")


def acquire(url, filename):
    path = MODELS / "initializers" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        request = urllib.request.Request(url, headers={"User-Agent": "InspectFlow-public-research/1.0"})
        with urllib.request.urlopen(request, timeout=90) as response, path.with_suffix(".part").open("wb") as handle:
            shutil.copyfileobj(response, handle)
        path.with_suffix(".part").replace(path)
    record = {"source_url": url, "filename": filename, "bytes": path.stat().st_size,
              "sha256": sha(path), "downloaded_for": "public-source initialization; no construction claims"}
    write(path.with_suffix(path.suffix + ".source.json"), record)
    return path, record


def read_image(row):
    image = cv2.imread(str(ROOT / row["image"]))
    if image is None or image.shape[:2] != (640, 640):
        raise ValueError("Frozen publisher derivative must be 640 square")
    return image


def tensor_from_image(image):
    # 640 square source -> 320 square: exactly the runtime letterbox with no padding.
    rgb = cv2.resize(image, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)[:, :, ::-1].astype(np.float32) / 255.
    return torch.from_numpy(np.ascontiguousarray(((rgb-MEAN)/STD).transpose(2, 0, 1)))


def native_truth(row):
    return cv2.imread(str(ROOT / row["mask"]), cv2.IMREAD_GRAYSCALE)


def truth_instances(row, class_id):
    instances = []
    for box in row["boxes"]:
        if box["class_id"] != class_id:
            continue
        mask = np.zeros((640, 640), np.uint8)
        for polygon in box["polygons"]:
            cv2.fillPoly(mask, [np.rint(np.asarray(polygon).reshape(-1, 2)).astype(np.int32)], 1)
        instances.append(mask.astype(bool))
    return instances


def semantic_instances(probabilities, thresholds):
    winners = probabilities.argmax(axis=0)
    instances = [[], []]
    for cls in range(2):
        binary = ((winners == cls+1) & (probabilities[cls+1] >= thresholds[cls])).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        for component in range(1, count):
            if stats[component, cv2.CC_STAT_AREA] < MIN_AREA:
                continue
            mask = labels == component
            instances[cls].append((float(probabilities[cls+1][mask].mean()), mask))
    return instances


def yolo_instances(prediction, thresholds):
    instances = [[], []]
    for item in prediction:
        if item["score"] >= thresholds[item["class_id"]] and int(item["mask"].sum()) >= MIN_AREA:
            instances[item["class_id"]].append((item["score"], item["mask"]))
    return instances


def wilson(successes, total):
    if not total:
        return None
    z = 1.96
    p = successes/total
    denominator = 1+z*z/total
    center = (p+z*z/(2*total))/denominator
    margin = z*np.sqrt(p*(1-p)/total+z*z/(4*total*total))/denominator
    return [float(center-margin), float(center+margin)]


def evaluate(rows, predictions, thresholds, kind):
    stats = [{"tp": 0, "fp": 0, "fn": 0, "pixel_tp": 0, "pixel_fp": 0, "pixel_fn": 0,
              "positive_images": 0, "negative_images": 0, "flagged_negative_images": 0, "matched_mask_ious": []} for _ in CLASSES]
    for row, prediction in zip(rows, predictions):
        proposed = semantic_instances(prediction, thresholds) if kind == "semantic" else yolo_instances(prediction, thresholds)
        truth = native_truth(row)
        for cls in range(2):
            s = stats[cls]
            true_masks = truth_instances(row, cls)
            candidates = sorted(proposed[cls], key=lambda pair: -pair[0])
            s["positive_images" if true_masks else "negative_images"] += 1
            if not true_masks and candidates:
                s["flagged_negative_images"] += 1
            union_pred = np.zeros((640, 640), bool)
            for _, mask in candidates:
                union_pred |= mask
            target = truth == cls+1
            s["pixel_tp"] += int((union_pred & target).sum())
            s["pixel_fp"] += int((union_pred & ~target).sum())
            s["pixel_fn"] += int((~union_pred & target).sum())
            matched = set()
            for _, mask in candidates:
                choices = []
                for index, target_mask in enumerate(true_masks):
                    if index in matched:
                        continue
                    intersection = int((mask & target_mask).sum())
                    overlap = intersection / max(1, int((mask | target_mask).sum()))
                    choices.append((overlap, index))
                overlap, index = max(choices, default=(0., None))
                if overlap >= .5:
                    matched.add(index)
                    s["tp"] += 1
                    s["matched_mask_ious"].append(overlap)
                else:
                    s["fp"] += 1
            s["fn"] += len(true_masks)-len(matched)
    for s in stats:
        s["precision"] = s["tp"]/max(1, s["tp"]+s["fp"])
        s["recall"] = s["tp"]/max(1, s["tp"]+s["fn"])
        s["precision_wilson95"] = wilson(s["tp"], s["tp"]+s["fp"])
        s["recall_wilson95"] = wilson(s["tp"], s["tp"]+s["fn"])
        s["pixel_precision"] = s["pixel_tp"]/max(1, s["pixel_tp"]+s["pixel_fp"])
        s["pixel_recall"] = s["pixel_tp"]/max(1, s["pixel_tp"]+s["pixel_fn"])
        s["pixel_iou"] = s["pixel_tp"]/max(1, s["pixel_tp"]+s["pixel_fp"]+s["pixel_fn"])
        s["pixel_f1"] = 2*s["pixel_tp"]/max(1, 2*s["pixel_tp"]+s["pixel_fp"]+s["pixel_fn"])
        s["mean_matched_mask_iou"] = float(np.mean(s.pop("matched_mask_ious"))) if s["matched_mask_ious"] else 0.
    return {"thresholds": dict(zip(CLASSES, thresholds)), "mask_match_iou": .5,
            "minimum_component_area_native_pixels": MIN_AREA, "native_dimensions": [640,640],
            "classes": dict(zip(CLASSES, stats))}


def select_thresholds(rows, predictions, kind):
    evaluated = {str(value): evaluate(rows, predictions, [value, value], kind) for value in GRID}
    thresholds = []
    for name in CLASSES:
        ranked = []
        for value in GRID:
            s = evaluated[str(value)]["classes"][name]
            qualified = s["precision"] >= .95 and s["tp"] > 0
            f1 = 2*s["tp"]/max(1, 2*s["tp"]+s["fp"]+s["fn"])
            ranked.append(((qualified, s["recall"] if qualified else f1, s["pixel_f1"], s["precision"], value), value))
        thresholds.append(max(ranked)[1])
    return thresholds, evaluated


def semantic_predictions(model, rows):
    model.eval()
    predictions, times = [], []
    with torch.no_grad():
        for row in rows:
            started = time.perf_counter()
            logits = model(tensor_from_image(read_image(row))[None])["out"][0]
            probability = logits.softmax(0).cpu().numpy()
            native = np.stack([cv2.resize(channel, (640,640), interpolation=cv2.INTER_LINEAR) for channel in probability])
            times.append(time.perf_counter()-started)
            predictions.append(native.astype(np.float32))
    return predictions, timing(times, includes="image read, BGR/RGB resize/normalization, Torch CPU, softmax and native probability resize; excludes threshold/component extraction")


def timing(times, includes):
    return {"images": len(times), "batch": 1, "cpu_threads": 1, "seconds": float(sum(times)),
            "throughput_fps": len(times)/max(1e-9, sum(times)), "p50_ms": float(np.percentile(times,50)*1000),
            "p95_ms": float(np.percentile(times,95)*1000), "scope": includes,
            "concurrency": "May overlap another one-thread training job; not an isolated deployment FPS qualification"}


class LRExport(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images):
        return self.model(images)["out"]


def fit_lraspp(manifest, train, cal, test):
    initializer, provenance = acquire("https://download.pytorch.org/models/lraspp_mobilenet_v3_large-d234d4ea.pth", "lraspp_mobilenet_v3_large-d234d4ea.pth")
    if not sha(initializer).startswith("d234d4ea"):
        raise ValueError("Official LR-ASPP hash prefix mismatch")
    model = lraspp_mobilenet_v3_large(weights=None, weights_backbone=None, num_classes=21)
    model.load_state_dict(torch.load(initializer, map_location="cpu", weights_only=True))
    model.classifier.low_classifier = nn.Conv2d(model.classifier.low_classifier.in_channels, 3, 1)
    model.classifier.high_classifier = nn.Conv2d(model.classifier.high_classifier.in_channels, 3, 1)
    for parameter in model.backbone.parameters():
        parameter.requires_grad = False
    before, before_speed = semantic_predictions(model, test)
    buckets = {"negative": [], CLASSES[0]: [], CLASSES[1]: []}
    for row in train:
        buckets[CLASSES[row["boxes"][0]["class_id"]] if row["boxes"] else "negative"].append(row)
    rng = random.Random(SEED)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.002, weight_decay=.0005)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([.2, 1., 1.]))
    model.train()
    model.backbone.eval()
    losses, seen = [], Counter()
    started = time.perf_counter()
    for step in range(200):
        selected = rng.sample(buckets[CLASSES[0]], 3)+rng.sample(buckets[CLASSES[1]], 3)+rng.sample(buckets["negative"], 2)
        images = torch.stack([tensor_from_image(read_image(row)) for row in selected])
        masks = torch.from_numpy(np.stack([cv2.resize(native_truth(row), (SIZE,SIZE), interpolation=cv2.INTER_NEAREST) for row in selected])).long()
        logits = model(images)["out"]
        crossentropy = loss_fn(logits, masks)
        probabilities = logits.softmax(1)[:,1:]
        target = torch.stack([masks == 1, masks == 2], dim=1).float()
        intersection = (probabilities*target).sum((0,2,3))
        dice = 1-((2*intersection+1)/(probabilities.sum((0,2,3))+target.sum((0,2,3))+1)).mean()
        loss = crossentropy+.5*dice
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite LR-ASPP loss")
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),10)
        optimizer.step()
        losses.append(float(loss.detach()))
        seen.update(row["id"] for row in selected)
        if (step+1)%25 == 0:
            print("LRASPP_STEP", step+1, "LOSS", losses[-1], "SECONDS", time.perf_counter()-started, flush=True)
    train_seconds = time.perf_counter()-started
    model.eval()
    destination = MODELS / "lraspp"
    destination.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), destination / "model-state.pt")
    cal_predictions, cal_speed = semantic_predictions(model, cal)
    thresholds, grid = select_thresholds(cal, cal_predictions, "semantic")
    write(OUT / "lraspp-calibration-thresholds-frozen.json", {"thresholds": thresholds, "grid": grid, "test_used": False})
    del cal_predictions
    after, after_speed = semantic_predictions(model, test)
    result = {"architecture": "lraspp", "validation_status": "research_only", "initializer": provenance,
              "budget": {"updates": 200, "batch": 8, "frozen_backbone": True, "train_seconds": train_seconds,
                         "sample_exposures": sum(seen.values()), "unique_train_photos_seen": len(seen),
                         "loss_first_10": float(np.mean(losses[:10])), "loss_last_10": float(np.mean(losses[-10:]))},
              "before_test_fixed_0_5": evaluate(test, before, [.5,.5], "semantic"),
              "after_test_fixed_0_5": evaluate(test, after, [.5,.5], "semantic"),
              "after_test_cal_selected": evaluate(test, after, thresholds, "semantic"),
              "before_speed": before_speed, "after_speed": after_speed, "calibration_speed": cal_speed,
              "release_pass": False, "release_blocker": "Physical connection/site independence unverified; no independent 50 positive + 50 negative category release or video evidence"}
    write(OUT / "lraspp-results.json", result)
    torch.onnx.export(LRExport(model).eval(), torch.zeros(1,3,SIZE,SIZE), str(destination / "model.onnx"),
                      input_names=["images"], output_names=["logits"], opset_version=17, dynamo=False)
    asset = {"title": "SDNET actual-polygon LR-ASPP · research", "architecture": "lraspp", "inspection_task": "brackets_railings",
             "input_size": SIZE, "classes": CLASSES, "model_file": "model.onnx", "model_sha256": sha(destination / "model.onnx"),
             "preprocessing": "letterbox_rgb_imagenet", "class_confidence": dict(zip(CLASSES, thresholds)),
             "source": manifest["source"], "license": "SDNET CC BY 4.0; TorchVision BSD-3-Clause; COCO/VOC initializer conditions apply",
             "validation_status": "research_only", "validation_summary": result["release_blocker"], "private_videos_used": False,
             "evaluation_report": str((OUT / "lraspp-results.json").resolve()), "dataset_manifest_sha256": sha(DATA),
             "training_script_sha256": sha(Path(__file__)), "minimum_component_area_native_pixels": MIN_AREA}
    write(destination / "manifest.json", asset)
    parity_and_artifacts(model, destination, test, after, thresholds)
    del before, after
    return result


def parity_and_artifacts(model, destination, rows, after, thresholds):
    import onnxruntime as ort
    model.eval()
    if any(module.training for module in model.modules()):
        raise ValueError("Every parity reference module must be in evaluation mode")
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(destination / "model.onnx"), options, providers=["CPUExecutionProvider"])
    comparisons, times = [], []
    for row in rows[:10]:
        tensor = tensor_from_image(read_image(row))[None]
        with torch.no_grad():
            reference = model(tensor)["out"].numpy()
        started = time.perf_counter()
        actual = session.run(None, {session.get_inputs()[0].name: tensor.numpy()})[0]
        times.append(time.perf_counter()-started)
        difference = float(np.abs(actual-reference).max())
        comparisons.append({"id": row["id"], "max_abs_logits_difference": difference})
        if not np.allclose(actual, reference, atol=.003, rtol=.003):
            raise ValueError("LR-ASPP ONNX parity failed")
    write(OUT / "lraspp-onnx-parity.json", {"passed": True, "comparisons": comparisons, "speed": timing(times, "ORT logits inference only; first call included")})
    # Fixed first four publisher cases per class + first four negatives; never best-case selection.
    selected = []
    for cls in range(2):
        selected += [index for index,row in enumerate(rows) if any(box["class_id"]==cls for box in row["boxes"])][:4]
    selected += [index for index,row in enumerate(rows) if not row["boxes"]][:4]
    receipts = []
    for index in dict.fromkeys(selected):
        row, probability = rows[index], after[index]
        image = read_image(row)
        truth = native_truth(row)
        instances = semantic_instances(probability, thresholds)
        predicted = np.zeros((640,640), np.uint8)
        for cls, parts in enumerate(instances):
            for _, mask in parts:
                predicted[mask]=cls+1
        panels=[]
        for title, mask in [("REAL PHOTO",np.zeros_like(truth)),("PUBLISHER POLYGON",truth),("AFTER RESEARCH",predicted)]:
            panel=image.copy()
            for cls,color in [(1,(0,190,255)),(2,(220,100,255))]:
                panel[mask==cls]=(.55*panel[mask==cls]+.45*np.array(color)).astype(np.uint8)
            cv2.putText(panel,title,(12,25),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),2)
            panels.append(panel)
        target = OUT / "photo-evidence" / (row["id"]+".jpg")
        target.parent.mkdir(parents=True,exist_ok=True)
        cv2.imwrite(str(target),np.concatenate(panels,axis=1))
        receipt={"id":row["id"],"input":row["image"],"input_sha256":sha(ROOT/row["image"]),
                 "source_mask_sha256":sha(ROOT/row["mask"]),"predicted_pixels":{name:int((predicted==cls+1).sum())for cls,name in enumerate(CLASSES)},
                 "photo_evidence":str(target.relative_to(ROOT)),"photo_evidence_sha256":sha(target),"kind":"real public still photograph; not video"}
        receipts.append(receipt)
    write(OUT / "photo-evidence/receipts.json",receipts)


def prepare_yolo_polygons(manifest):
    directory = OUT / "polygon-dataset"
    for row in manifest["records"]:
        image_path = directory / "images" / row["split"] / (row["id"]+".jpg")
        image_path.parent.mkdir(parents=True, exist_ok=True)
        if not image_path.exists():
            shutil.copyfile(ROOT / row["image"], image_path)
        label_path = directory / "labels" / row["split"] / (row["id"]+".txt")
        label_path.parent.mkdir(parents=True, exist_ok=True)
        lines=[]
        for box in row["boxes"]:
            if len(box["polygons"])!=1:
                raise ValueError("Multiple polygon components need explicit treatment; never replace with boxes")
            coordinates=np.asarray(box["polygons"][0],dtype=np.float64)/640.
            if np.any(coordinates<0) or np.any(coordinates>1):
                raise ValueError("Polygon outside image")
            lines.append(str(box["class_id"])+" "+" ".join(f"{value:.8f}"for value in coordinates))
        label_path.write_text("\n".join(lines)+("\n"if lines else ""),encoding="utf8")
    import yaml
    config={"path":directory.resolve().as_posix(),"train":"images/train","val":"images/calibration","names":dict(enumerate(CLASSES))}
    (directory/"dataset.yaml").write_text(yaml.safe_dump(config),encoding="utf8")
    return directory/"dataset.yaml"


def yolo_predictions(wrapper, rows):
    predictions, times = [], []
    for row in rows:
        started=time.perf_counter()
        result=wrapper.predict(str(ROOT/row["image"]),imgsz=SIZE,conf=.01,iou=.45,device="cpu",batch=1,
                               retina_masks=True,verbose=False,max_det=300)[0]
        times.append(time.perf_counter()-started)
        items=[]
        if result.masks is not None:
            for mask,score,cls in zip(result.masks.data.cpu().numpy(),result.boxes.conf.tolist(),result.boxes.cls.tolist()):
                if mask.shape!=(640,640):
                    raise ValueError("YOLO retina mask must match original native pixels")
                items.append({"class_id":int(cls),"score":float(score),"mask":mask>.5})
        predictions.append(items)
    return predictions,timing(times,"Single-image Ultralytics prediction including decode, letterbox, CPU model, NMS and retina native masks")


def fit_yolo(manifest, train, cal, test):
    from ultralytics import YOLO
    from ultralytics.utils import torch_utils
    import yaml
    # select_device('cpu') otherwise replaces the explicit Torch setting.
    torch_utils.NUM_THREADS=1
    torch.set_num_threads(1)
    thread_checks=[]
    def check_threads(stage):
        actual={"stage":stage,"intra_op":torch.get_num_threads(),"inter_op":torch.get_num_interop_threads(),"ultralytics_cpu_constant":torch_utils.NUM_THREADS}
        thread_checks.append(actual)
        write(OUT/"yolov8n-seg-actual-threads.json",thread_checks)
        if actual["intra_op"]!=1 or actual["inter_op"]!=1:
            raise ValueError("Actual training/inference thread count deviates from one")
    check_threads("before_initialization")
    initializer,provenance=acquire("https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-seg.pt","yolov8n-seg.pt")
    wrapper=YOLO(str(initializer))
    if wrapper.task!="segment":
        raise ValueError("Official initializer must be segmentation")
    config=copy.deepcopy(wrapper.model.yaml)
    config["nc"]=2
    architecture=OUT/"yolov8n-seg-two-class.yaml"
    architecture.write_text(yaml.safe_dump(config),encoding="utf8")
    model=YOLO(str(architecture),task="segment").load(str(initializer))
    model.model.names=dict(enumerate(CLASSES))
    before,before_speed=yolo_predictions(model,test)
    check_threads("after_baseline_predict")
    model.add_callback("on_train_start",lambda trainer:check_threads("on_train_start"))
    model.add_callback("on_train_epoch_end",lambda trainer:check_threads("on_train_epoch_end"))
    dataset=prepare_yolo_polygons(manifest)
    started=time.perf_counter()
    model.train(data=str(dataset),epochs=2,imgsz=SIZE,batch=8,device="cpu",workers=0,freeze=10,
                optimizer="AdamW",lr0=.002,lrf=.1,warmup_epochs=0,nbs=8,seed=SEED,deterministic=True,
                plots=False,save=True,cache=False,val=False,amp=False,mosaic=0,mixup=0,fliplr=0,
                hsv_h=0,hsv_s=0,hsv_v=0,degrees=0,translate=0,scale=0,shear=0,perspective=0,
                project=str(OUT),name="yolo-seg-fit",exist_ok=False,verbose=False,overlap_mask=False,mask_ratio=2)
    train_seconds=time.perf_counter()-started
    check_threads("after_training")
    model=YOLO(str(OUT/"yolo-seg-fit/weights/last.pt"))
    cal_predictions,cal_speed=yolo_predictions(model,cal)
    thresholds,grid=select_thresholds(cal,cal_predictions,"instance")
    write(OUT/"yolov8n-seg-calibration-thresholds-frozen.json",{"thresholds":thresholds,"grid":grid,"test_used":False})
    del cal_predictions
    after,after_speed=yolo_predictions(model,test)
    result={"architecture":"yolov8n_seg","validation_status":"research_only","initializer":provenance,
            "budget":{"epochs":2,"batch":8,"freeze":10,"train_seconds":train_seconds,"mask_ratio":2,"overlap_mask":False,"actual_thread_checks":thread_checks},
            "before_test_fixed_0_5":evaluate(test,before,[.5,.5],"instance"),
            "after_test_fixed_0_5":evaluate(test,after,[.5,.5],"instance"),
            "after_test_cal_selected":evaluate(test,after,thresholds,"instance"),
            "before_speed":before_speed,"after_speed":after_speed,"calibration_speed":cal_speed,
            "release_pass":False,"release_blocker":"Physical connection/site independence unverified; no independent 50 positive + 50 negative category release or video evidence"}
    write(OUT/"yolov8n-seg-results.json",result)
    destination=MODELS/"yolov8n-seg"
    destination.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(OUT/"yolo-seg-fit/weights/last.pt",destination/"last.pt")
    exported=Path(model.export(format="onnx",imgsz=SIZE,opset=17,simplify=False,dynamic=False,batch=1,device="cpu"))
    shutil.copyfile(exported,destination/"model.onnx")
    asset={"title":"SDNET actual-polygon YOLOv8n-seg · research","architecture":"yolov8n_seg","inspection_task":"brackets_railings",
           "input_size":SIZE,"classes":CLASSES,"model_file":"model.onnx","model_sha256":sha(destination/"model.onnx"),
           "preprocessing":"letterbox_rgb_0_1","class_confidence":dict(zip(CLASSES,thresholds)),"source":manifest["source"],
           "license":"SDNET CC BY 4.0; Ultralytics AGPL-3.0 code/model conditions apply","validation_status":"research_only",
           "validation_summary":result["release_blocker"],"private_videos_used":False,"evaluation_report":str((OUT/"yolov8n-seg-results.json").resolve()),
           "dataset_manifest_sha256":sha(DATA),"training_script_sha256":sha(Path(__file__)),"minimum_component_area_native_pixels":MIN_AREA,
           "runtime_decoder_status":"Requires native-mask decode parity before application use"}
    write(destination/"manifest.json",asset)
    return result


def verify_lraspp_export(manifest,splits):
    """Repair the export verifier's mode, never checkpoint weights or thresholds."""
    import onnxruntime as ort
    destination=MODELS/"lraspp"
    pending=destination/"manifest.not-verified"
    current=destination/"manifest.json"
    asset=json.loads((current if current.exists() else pending).read_text())
    model=lraspp_mobilenet_v3_large(weights=None,weights_backbone=None,num_classes=3)
    model.load_state_dict(torch.load(destination/"model-state.pt",map_location="cpu",weights_only=True))
    model.eval()
    row=splits["train"][0]
    tensor=tensor_from_image(read_image(row))[None]
    def tensor_state_hash(module):
        digest=hashlib.sha256()
        for key,value in sorted(module.state_dict().items()):
            digest.update(key.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()
    def session(path,optimize=True):
        options=ort.SessionOptions()
        options.intra_op_num_threads=options.inter_op_num_threads=1
        if not optimize:
            options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        return ort.InferenceSession(str(path),options,providers=["CPUExecutionProvider"])
    exported=session(destination/"model.onnx")
    actual=exported.run(None,{exported.get_inputs()[0].name:tensor.numpy()})[0]
    with torch.no_grad():
        reference=model(tensor)["out"].numpy()
    report={"source_split":"train","source_id":row["id"],"source_image_sha256":sha(ROOT/row["image"]),
            "unchanged_onnx_sha256":sha(destination/"model.onnx"),"unchanged_checkpoint_sha256":sha(destination/"model-state.pt"),
            "failed_run_source_sha256":sha(OUT/"source-lraspp.py"),"failed_run_log_sha256":sha(ROOT/"output/evaluation/construction-expansion-sdnet-masks-lraspp.log"),
            "verifier_source_sha256":sha(Path(__file__)),"eval_reference_max_abs":float(np.abs(actual-reference).max()),
            "allclose_guard":{"atol":.003,"rtol":.003,"unchanged":True},"eval_modules_before_export":sum(module.training for module in model.modules())}
    before_state=tensor_state_hash(model)
    wrapper=LRExport(model)  # Deliberately reproduce the old default-training wrapper bug on TRAIN.
    report["old_wrapper_training_before_export"]=wrapper.training
    torch.onnx.export(wrapper,torch.zeros(1,3,SIZE,SIZE),str(OUT/"old-wrapper-mode-probe.onnx"),
                      input_names=["images"],output_names=["logits"],opset_version=17,dynamo=False)
    report["model_training_modules_after_old_wrapper_export"]=sum(module.training for module in model.modules())
    report["state_unchanged_by_export_before_forward"]=before_state==tensor_state_hash(model)
    with torch.no_grad():
        bad_reference=model(tensor)["out"].numpy()
    report["old_training_mode_reference_max_abs"]=float(np.abs(actual-bad_reference).max())
    model.load_state_dict(torch.load(destination/"model-state.pt",map_location="cpu",weights_only=True))
    model.eval()
    with torch.no_grad(),torch.backends.mkldnn.flags(enabled=False):
        no_mkldnn=model(tensor)["out"].numpy()
    report["eval_no_mkldnn_max_abs"]=float(np.abs(actual-no_mkldnn).max())
    unoptimized=session(destination/"model.onnx",False)
    no_ort_optimizations=unoptimized.run(None,{unoptimized.get_inputs()[0].name:tensor.numpy()})[0]
    report["eval_no_ort_optimizations_max_abs"]=float(np.abs(reference-no_ort_optimizations).max())
    report["diagnosis"]="Default-training export wrapper restores child.train(True) after export; verifier incorrectly compares training-mode BatchNorm with evaluation-mode ONNX. Weights, ONNX, thresholds and tolerance are unchanged."
    report["repaired_guard_passed"]=bool(np.allclose(actual,reference,atol=.003,rtol=.003))
    write(OUT/"lraspp-initial-parity-failure-diagnosis.json",report)
    if not report["repaired_guard_passed"]:
        raise ValueError("TRAIN eval-mode guard still fails; manifest stays unpublished")
    chosen=[]
    for cls in range(2):
        chosen += [row for row in splits["train"] if any(box["class_id"]==cls for box in row["boxes"])][:4]
    chosen += [row for row in splits["train"] if not row["boxes"]][:4]
    probabilities,_=semantic_predictions(model,chosen)
    thresholds=[asset["class_confidence"][name]for name in CLASSES]
    parity_and_artifacts(model,destination,chosen,probabilities,thresholds)
    asset["export_parity_report"]=str((OUT/"lraspp-onnx-parity.json").resolve())
    asset["verification_repair_report"]=str((OUT/"lraspp-initial-parity-failure-diagnosis.json").resolve())
    asset["verification_source_sha256"]=sha(Path(__file__))
    write(destination/"manifest.json",asset)
    print("LRASPP_UNCHANGED_EXPORT_VERIFIED",json.dumps(report),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",choices=["lraspp","yolo","both"],default="both")
    parser.add_argument("--verify-lraspp",action="store_true",help="Verify unchanged checkpoint/export in eval mode; no training or threshold selection")
    args=parser.parse_args()
    if not args.verify_lraspp:
        requested=["lraspp","yolov8n-seg"] if args.model=="both" else ["lraspp" if args.model=="lraspp" else "yolov8n-seg"]
        for name in requested:
            if any((MODELS/name/file).exists() for file in ["model-state.pt","last.pt","model.onnx"]):
                raise ValueError("Preserve completed experiment artifacts; use a new experiment version to train again. --verify-lraspp only verifies unchanged assets.")
    OUT.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    manifest=json.loads(DATA.read_text())
    if manifest["classes"]!=CLASSES or manifest["private_videos_used"]:
        raise ValueError("Frozen real-data contract changed")
    splits={name:[row for row in manifest["records"]if row["split"]==name]for name in ["train","calibration","test"]}
    if args.verify_lraspp:
        verify_lraspp_export(manifest,splits)
        return
    for row in manifest["records"]:
        if sha(ROOT/row["image"])!=row["source_sha256"] or sha(ROOT/row["mask"])!=row["mask_sha256"]:
            raise ValueError("Frozen image or polygon mask changed")
    protocol={"dataset_manifest_sha256":sha(DATA),"script_sha256":sha(Path(__file__)),"seed":SEED,"cpu_threads":1,
              "split_counts":{key:len(value)for key,value in splits.items()},"mask_source":"Publisher actual polygons; never rectangles",
              "threshold_grid":GRID,"minimum_component_area_native_pixels":MIN_AREA,"region_match":"Per-class greedy mask IoU>=0.5",
              "selection":"Final fixed-budget checkpoint; per-class CAL thresholds prefer nonempty regionprecision>=.95 then recall; otherwise regionF1 then pixelF1 then precision; TEST never selects",
              "lraspp":{"updates":200,"batch":8,"frozen_backbone":True,"loss":"weightedCE[0.2,1,1]+0.5foregroundDice","input_size":SIZE},
              "yolo":{"epochs":2,"batch":8,"freeze":10,"input_size":SIZE,"augmentation":"none","mask_ratio":2},
              "scene_independence":"UNVERIFIED physical scene/site IDs; exact/near-duplicate groups only","private_videos_used":False}
    write(OUT/("protocol-before-"+args.model+".json"),protocol)
    shutil.copyfile(Path(__file__),OUT/("source-"+args.model+".py"))
    if args.model in ["lraspp","both"]:
        fit_lraspp(manifest,splits["train"],splits["calibration"],splits["test"])
    if args.model in ["yolo","both"]:
        fit_yolo(manifest,splits["train"],splits["calibration"],splits["test"])


if __name__=="__main__":
    main()
