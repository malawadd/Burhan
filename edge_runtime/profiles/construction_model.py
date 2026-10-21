"""Pinned local construction observations; no legal compliance inference."""
from __future__ import annotations

import hashlib
import math
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODELS = {
    "ppe": {"file": "ppe-yolov8n.onnx", "sha256": "1d5f3cf69883e89ba6e2b3836a0b21623044e9a682f1f00dfbacac97b99baf94",
            "source": "https://huggingface.co/mancalazure/construction-ppe-safety/tree/70cc801710f246765f70ada7248d9560ec749bb2",
            "classes": ["Hardhat", "Mask", "NO-Hardhat", "NO-Mask", "NO-Safety Vest", "Person", "Safety Cone", "Safety Vest", "machinery", "vehicle"],
            "capabilities": ["ppe_hardhat", "ppe_vest", "person", "site_equipment"]},
    "crack": {"file": "crack-yolov8n-seg.onnx", "sha256": "4b789658646d790079d3e78d9d3ed475030d740a4b8412aff52248189a114cbd",
              "source": "https://huggingface.co/beaunix/smart-crack-vision/tree/9442007ef9c1003880baa5796d253608691b37c4",
              "classes": ["crack"], "capabilities": ["surface_crack"]},
}

CRACK_VARIANTS = {
    "baseline": MODELS["crack"],
    "joint_aware_candidate": {"file": "crack-joints-yolov8n-seg.onnx",
        "sha256": "4d48232d7b1427ac468022d2cc11a344885629cac71ad48bd2ff4c6659302246",
        "source": "https://huggingface.co/levanell/yolov8n-seg-cracks-joints/tree/44e73b6b015247bde629e9e7f12efef2ba4c7f3b",
        "classes": ["joint", "crack"], "capabilities": ["surface_crack"]},
}


from edge_runtime.profiles.construction_dacl import CLASSES as DACL_CLASSES, CAPABILITIES as DACL_CAPABILITIES
from edge_runtime.profiles.construction_round4_cpu_asset import SPEC as ROUND4_CPU_SPEC, VARIANT as ROUND4_CPU_VARIANT
DAMAGE_VARIANTS = {
    ROUND4_CPU_VARIANT: ROUND4_CPU_SPEC,
    'cubit_author_v2_failed': {'capabilities': ['surface_crack', 'surface_spalling'],
     'class_capabilities': {'crack': 'surface_crack', 'spalling': 'surface_spalling'},
     'class_confidence': {'crack': 0.25, 'spalling': 0.25},
     'classes': ['crack', 'spalling'],
     'confidence': 0.25,
     'decoder': 'cubit_yolo_seg',
     'diagnostic_only': True,
     'display_name': 'Author crack/spall masks · FAILED camera calibration',
     'experimental': True,
     'exposure_policy': 'low_light_gain',
     'file': 'cubit-author-release-v1/author-yolov8n-seg-640-v2.onnx',
     'input_size': 640,
     'license': 'Author standalone checkpoint license undeclared; exact training overlap unknown. Ultralytics '
                'code AGPL-3.0. CUBIT photo dataset CC BY4.0 does not independently clear checkpoint or video '
                'rights. Research only.',
     'sha256': 'df13a679081023d9093f731631ad199dbaaa9c467bb1e3c0df7054240c51bb6f',
     'source': 'https://benyunzhao.github.io/CUBIT-Dataset/',
     'validation_scope': 'Reused122 CAL: crack12/125+81unmatched, spall0/38 (abstains), '
                         'thin7/62,5/10negativeflags. Five fixed genuine development recordings; source '
                         'checkpoint training overlap unknown. Actual prototype masks, no video '
                         'pixelGT/15FPS/SBC qualification.',
     'wrapper_sha256': 'fe41777344cd92ae6a2b9cca05db56ab5418392448ebc87076cd51511b45686d'},
    'mbdd_wall_total16_failed': {'capabilities': ['surface_crack',
                      'surface_leakage_appearance',
                      'surface_detachment',
                      'surface_corrosion_appearance',
                      'surface_bulge_appearance'],
     'class_capabilities': {'abscission': 'surface_detachment',
                            'bulge': 'surface_bulge_appearance',
                            'corrosion': 'surface_corrosion_appearance',
                            'crack': 'surface_crack',
                            'leakage': 'surface_leakage_appearance'},
     'class_confidence': {'abscission': 0.2, 'bulge': 0.15, 'corrosion': 0.1, 'crack': 0.85, 'leakage': 0.15},
     'classes': ['crack', 'leakage', 'abscission', 'corrosion', 'bulge'],
     'confidence': 0.1,
     'decoder': 'yolo_detect',
     'diagnostic_label': 'FAILED WALL PHOTO RESEARCH',
     'diagnostic_only': True,
     'display_name': 'Wall appearance research · FAILED calibration',
     'experimental': True,
     'exposure_policy': 'off',
     'file': 'mbdd-wall-five-class-v3/wall-five-class-total16-last12-640.onnx',
     'input_size': 640,
     'license': 'MBDD2025 CCBY4.0; Ultralytics/sourceweight AGPL3; inherited warmstart source-data restrictions '
                'remain, including DACLNC4.0 and source-rights uncertainty.',
     'sha256': 'e413db1d7f855140753f2343d96fd6620c62f3db7946d82c06b43ea7590da15c',
     'source': 'https://zenodo.org/records/15622584',
     'validation_scope': '181 TRAIN/48 CAL source-block-separated photos; 18 previously exposed diagnostic '
                         'photos. All five minimum guards fail. No healthy-wall negatives, video, hidden '
                         'moisture, paint-only material, depth, site-independent or SBC qualification.'},
    'evcivil_r4_last4_failed': {'capabilities': ['surface_spalling', 'surface_crack'],
     'class_confidence': {'crack': 0.1, 'spalling': 0.25},
     'classes': ['spalling', 'crack'],
     'confidence': 0.05,
     'decoder': 'yolo_detect',
     'diagnostic_only': True,
     'display_name': 'Final epoch4 · failed camera calibration research',
     'experimental': True,
     'exposure_policy': 'low_light_gain',
     'file': 'evcivil-yolov8n-r4-last4-640-gain.onnx',
     'input_size': 640,
     'license': 'AGPL-3.0 Ultralytics/initial weights; ev-CIVIL CC BY4.0; DACL10k data CC BY-NC4.0 '
                '(non-commercial restriction applies); initial merged crack/joint source-data licenses not '
                'independently cleared; LCSD author-linked distribution declares Apache-2.0; original author '
                'val/test photographs are repurposed solely as our TRAIN transfer with no LCSD evaluation claim. '
                'This is a source declaration, not universal rights clearance. Inherited '
                'Ultralytics/initial-weight AGPL lineage, ev-CIVIL CC BY4.0 and DACL CC BY-NC4.0 restrictions '
                'remain; initial source-weight data rights are not independently cleared.',
     'sha256': '66dac0cf1c578c8458cfc653ca76aa2c14f2257f4eb1ada4d85b878e6be65f6f',
     'source': 'https://doi.org/10.11583/DTU.25924228',
     'validation_scope': 'All122 reused CAL: crack8/125 matches+75unmatched; spall5/38+5unmatched; '
                         'thin1/62;5/10normalflags. Post-hoc known-hit videos are illustrations without '
                         'independent accuracy credit. No promotion.'},
    'pixel_fixed50_before': {'capabilities': ['surface_crack'],
     'classes': ['crack'],
     'confidence': 0.5,
     'decoder': 'lite_semseg',
     'diagnostic_label': 'RESEARCH - FIXED .50 / NOT QUALIFIED',
     'diagnostic_only': True,
     'display_name': 'Before fixed .50 crack mask · FAILED research',
     'experimental': True,
     'exposure_policy': 'low_light_gain',
     'file': 'thin-contingency/litecrackseg-tut-384.onnx',
     'input_size': 384,
     'license': 'Whole-model/source license undeclared; MobileViT subcomponent license does not clear '
                'whole-model use. Experimental evaluation only; commercial/redistribution clearance unresolved.',
     'sha256': '20936464f5fb6ef32cf5d84a87c84616bf1d5461364fbfb466dd7c4846d04055',
     'source': 'https://github.com/1Kaleb/LiteCrackSeg/tree/082d33111f1a3b9ac1d78e984b06fd3530bf3533',
     'validation_scope': 'Fixed original TUT versus FINAL300, exposed/calibration videos. All-three fitted '
                         'calibration guards failed. No spalling support, no video pixel GT or independent '
                         'accuracy credit.'},
    'pixel_fixed50_after': {'capabilities': ['surface_crack'],
     'classes': ['crack'],
     'confidence': 0.5,
     'decoder': 'lite_semseg',
     'diagnostic_label': 'RESEARCH - FIXED .50 / NOT QUALIFIED',
     'diagnostic_only': True,
     'display_name': 'After fixed .50 crack mask · FAILED research',
     'experimental': True,
     'exposure_policy': 'low_light_gain',
     'file': 'pixel-supervision-v2/crack-pixel-step300.onnx',
     'input_size': 384,
     'license': 'LiteCrackSeg whole-model license undeclared; experimental only. DACL CC BY-NC4.0; LCSD '
                'author-linked Apache2.0 declaration with source-lineage caveat.',
     'sha256': 'b7cad346d11cb89be6a76fff38ab26fa3e8fd9778981af47498e6f8a99c6d3d7',
     'source': 'https://github.com/1Kaleb/LiteCrackSeg/tree/082d33111f1a3b9ac1d78e984b06fd3530bf3533',
     'validation_scope': 'Fixed original TUT versus FINAL300, exposed/calibration videos. All-three fitted '
                         'calibration guards failed. No spalling support, no video pixel GT or independent '
                         'accuracy credit.'},
    'tile_surface_anomaly_research': {'capabilities': ['tile_surface_anomaly'],
     'classes': ['tile_surface_anomaly'],
     'decoder': 'tile_patch_bank',
     'diagnostic_label': 'RESEARCH - WEAK TILE LOCALIZATION',
     'diagnostic_only': True,
     'display_name': 'Tile surface anomaly · manufacturing PHOTO research',
     'experimental': True,
     'exposure_policy': 'off',
     'file': 'finishes-tile/model.json',
     'fitted_source_sha256': '24c8e6aac7a260ff79281ee3eb4eaeb20377742ad27ee61c43c1c45e6e78214e',
     'input_size': 256,
     'license': 'MVTec AD CC BY-NC-SA4.0; torchvision code BSD3, official ImageNet weight/source terms apply',
     'sha256': '1ee9ecb429369c67cdd3369cd90861661ac4fe87a0e5d1ed1a38a7a5a771f0f6',
     'source': 'https://www.mvtec.com/research-teaching/datasets/mvtec-ad',
     'validation_scope': 'All117 originaltiletest photos after48normalfit/10normalcal. '
                         'CrackpixelP31.14/R59.71/IoU25.73%,0/17derivedcrackregionsIoU.50; good3/33flags. '
                         'Notinstalledfloor/videoqualified.',
     'wrapper': 'edge_runtime.profiles.construction_tile_adapter.TileConstructionModel',
     'wrapper_sha256': '0fa1c6dcad982bb85df3754ef63da4047be80a2658e365851ae8fff3908c26d8'},
    'evcivil_r4_head_failed': {'capabilities': ['surface_spalling', 'surface_crack'],
     'class_confidence': {'crack': 0.45, 'spalling': 0.1},
     'classes': ['spalling', 'crack'],
     'confidence': 0.05,
     'decoder': 'yolo_detect',
     'diagnostic_only': True,
     'display_name': 'Before failed camera calibration · development review',
     'experimental': True,
     'exposure_policy': 'low_light_gain',
     'file': 'evcivil-yolov8n-r4-head-640-gain.onnx',
     'input_size': 640,
     'license': 'AGPL-3.0 Ultralytics/initial weights; ev-CIVIL CC BY4.0; DACL10k data CC BY-NC4.0 '
                '(non-commercial restriction applies); initial merged crack/joint source-data licenses not '
                'independently cleared',
     'sha256': '33e7ab167a93c7ee9c9e80e544951ec2e61fcdaf977377ea56e321dd4fea41ad',
     'source': 'https://doi.org/10.11583/DTU.25924228',
     'validation_scope': 'FAILED fixed122-source camera calibration. Four full calibration/exposed recordings '
                         'are development only; no independent field/physical-site/Saudi qualification. '
                         'Capture-group/global-defect-ID exclusions preserve dset6 calibration, dset5 heldout, '
                         'Nov18 night calibration and Jan2 night heldout. Physical project independence and '
                         'initial pretrained overlap remain unknown. New dset1/2 fitting makes old exposed '
                         'recordings development only.'},
    'evcivil_r4_real_lowlight_failed': {'capabilities': ['surface_spalling', 'surface_crack'],
     'class_confidence': {'crack': 0.1, 'spalling': 0.25},
     'classes': ['spalling', 'crack'],
     'confidence': 0.05,
     'decoder': 'yolo_detect',
     'diagnostic_only': True,
     'display_name': 'After failed camera calibration · development review',
     'experimental': True,
     'exposure_policy': 'low_light_gain',
     'file': 'evcivil-yolov8n-r4-real-lowlight-640-gain.onnx',
     'input_size': 640,
     'license': 'AGPL-3.0 Ultralytics/initial weights; ev-CIVIL CC BY4.0; DACL10k data CC BY-NC4.0 '
                '(non-commercial restriction applies); initial merged crack/joint source-data licenses not '
                'independently cleared; LCSD author-linked distribution declares Apache-2.0; original author '
                'val/test photographs are repurposed solely as our TRAIN transfer with no LCSD evaluation claim. '
                'This is a source declaration, not universal rights clearance. Inherited '
                'Ultralytics/initial-weight AGPL lineage, ev-CIVIL CC BY4.0 and DACL CC BY-NC4.0 restrictions '
                'remain; initial source-weight data rights are not independently cleared.',
     'sha256': '2fbff4feb9f173da3aabab3146832f78d0c200a616d71e66b482f4edb1d4199e',
     'source': 'https://doi.org/10.11583/DTU.25924228',
     'validation_scope': 'FAILED fixed122-source camera calibration. Four full calibration/exposed recordings '
                         'are development only; no independent field/physical-site/Saudi qualification. '
                         'Capture-group/global-defect-ID exclusions preserve dset6 calibration, dset5 heldout, '
                         'Nov18 night calibration and Jan2 night heldout. Physical project independence and '
                         'initial pretrained overlap remain unknown. New dset1/2 fitting makes old exposed '
                         'recordings development only. Task now explicitly includes verified '
                         'render/masonry/paving surface loss; material-specific scores and physical-project '
                         'independence remain limited.'},
    "evcivil_r3_historical_failed": {
        "display_name":"Historical failed calibration · seen-source diagnostic",
        "file":"evcivil-yolov8n-r3.onnx",
        "sha256":"19137c9ff183d65f389c27abdcc8536a4b1aa79578054c55ce8d7b64b0824410",
        "source":"https://doi.org/10.11583/DTU.25924228",
        "classes":["spalling","crack"],"capabilities":["surface_spalling","surface_crack"],
        "decoder":"yolo_detect","input_size":416,"confidence":.05,
        "class_confidence":{"crack":.65,"spalling":.2},
        "license":"AGPL-3.0 Ultralytics lineage; ev-CIVIL CC BY4.0; DACL data CC BY-NC4.0; initial merged-source dataset rights not independently cleared",
        "validation_scope":"FAILED camera calibration; seen-training and calibration development diagnostics only. Publisher spalling includes surface material loss; substrate and structural cause require inspection. No independent field or Saudi-site qualification.",
        "experimental":True,"diagnostic_only":True,
    },
    "evcivil_r3_adapted_failed": {
        "display_name":"Adapted failed calibration · research diagnostic",
        "file":"evcivil-yolov8n-r3-adapted.onnx",
        "sha256":"62e2da13a7cf072140876f2d16fb63f46560ea394c317de581cbd7e0f02dda44",
        "source":"https://doi.org/10.11583/DTU.25924228",
        "classes":["spalling","crack"],"capabilities":["surface_spalling","surface_crack"],
        "decoder":"yolo_detect","input_size":416,"confidence":.05,
        "class_confidence":{"crack":.1,"spalling":.85},
        "license":"AGPL-3.0 Ultralytics lineage; ev-CIVIL CC BY4.0; DACL data CC BY-NC4.0; initial merged-source dataset rights not independently cleared",
        "validation_scope":"FAILED camera calibration after fixed backbone adaptation; seen-training/calibration research diagnostics only. Publisher spalling includes surface material loss; substrate and structural cause require inspection. No independent field or Saudi-site qualification.",
        "experimental":True,"diagnostic_only":True,
    },
    "dacl_segformer_b1": {
        "display_name":"Photo review · multiple conditions (slow)",
        "file":"dacl19/dacl-segformer-b1-512.onnx",
        "sha256":"080fb07297bbb2841fbd5f49b9d8888da53ee1c2adc1ea33bdd2addb8cc9ea7f",
        "source":"https://huggingface.co/spaces/phiyodr/dacl-challenge/tree/f5bee34782e7e02393ba4f206a95ba5ed67eb8a5",
        "decoder":"dacl_semseg","input_size":512,"confidence":.5,
        "classes":DACL_CLASSES,"capabilities":DACL_CAPABILITIES,
        "license":"Author DACL model; DACL10k dataset CC BY-NC4.0, mirror CC BY4 label does not override author terms; author checkpoint license undeclared, commercial clearance unresolved",
        "validation_scope":"Author-trained DACL10k; public image/polygon diagnostics, author training/checkpoint-selection overlap not independently established; no Saudi-site validation",
    },
    "dacl_segformer_b1_320": {
        "display_name":"Photo review · reduced resolution (weaker)",
        "file":"dacl19/dacl-segformer-b1-320.onnx",
        "sha256":"c89aa3002de737cf19ea3043704af36837ad92267466f1234ac2bcecee60ea56",
        "source":"https://huggingface.co/spaces/phiyodr/dacl-challenge/tree/f5bee34782e7e02393ba4f206a95ba5ed67eb8a5",
        "decoder":"dacl_semseg","input_size":320,"confidence":.5,
        "classes":DACL_CLASSES,"capabilities":DACL_CAPABILITIES,
        "license":"Author DACL model; DACL10k dataset CC BY-NC4.0, mirror CC BY4 label does not override author terms; author checkpoint license undeclared, commercial clearance unresolved",
        "validation_scope":"Reduced-resolution public-photo diagnostic; weaker accuracy than 512. Author training/checkpoint-selection overlap not independently established; not validated for real-time video or Saudi sites",
    },
}

def prepare(frame, mode, ppe_preprocess="stretch", size=640):
    """Pinned serving geometry, with explicit PPE aspect-preserving alternative."""
    h, w = frame.shape[:2]
    if mode == "ppe" and ppe_preprocess == "stretch":
        image = cv2.resize(frame, (size, size))
        transform = (size / w, size / h, 0, 0)
    else:
        ratio = min(size / w, size / h)
        rw, rh = round(w * ratio), round(h * ratio)
        px, py = (size - rw) // 2, (size - rh) // 2
        image = np.full((size, size, 3), 114, np.uint8)
        image[py:py + rh, px:px + rw] = cv2.resize(frame, (rw, rh))
        transform = (ratio, ratio, px, py)
    tensor = np.ascontiguousarray(image[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
    return tensor, transform


def decode(outputs, mode, transform, shape, threshold=.25, nms_iou=.45, mask_rendering="masks", classes=None):
    if not isinstance(mode, str) or mode not in MODELS:
        raise ValueError("construction mode must be ppe or crack")
    if not isinstance(mask_rendering, str) or mask_rendering not in {"masks", "boxes"}:
        raise ValueError("construction mask_rendering must be masks or boxes")
    classes = MODELS[mode]["classes"] if classes is None else classes
    count = len(classes)
    expected = 4 + count + (32 if mode == "crack" else 0)
    prediction = np.asarray(outputs[0])
    if prediction.shape != (1, expected, 8400):
        raise ValueError(f"Construction {mode} output contract mismatch: {prediction.shape}")
    if mode == "crack" and (len(outputs) != 2 or np.asarray(outputs[1]).shape != (1, 32, 160, 160)):
        raise ValueError("Construction crack prototype contract mismatch")
    rows = prediction[0].T
    labels = rows[:, 4:4 + count].argmax(axis=1)
    scores = rows[np.arange(len(rows)), 4 + labels]
    sx, sy, px, py = transform
    h, w = shape
    detections = []
    for label in range(count):
        selected = np.flatnonzero((labels == label) & (scores >= threshold) & np.isfinite(rows).all(axis=1))
        boxes, values, indices = [], [], []
        for index in selected:
            cx, cy, bw, bh = rows[index, :4]
            x1 = max(0, min(w, round((float(cx - bw / 2) - px) / sx)))
            y1 = max(0, min(h, round((float(cy - bh / 2) - py) / sy)))
            x2 = max(0, min(w, round((float(cx + bw / 2) - px) / sx)))
            y2 = max(0, min(h, round((float(cy + bh / 2) - py) / sy)))
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2 - x1, y2 - y1]); values.append(float(scores[index])); indices.append(index)
        keep = np.asarray(cv2.dnn.NMSBoxes(boxes, values, threshold, nms_iou)).reshape(-1) if boxes else []
        for index in keep:
            index = int(index)
            item = {"bbox": tuple(boxes[index]), "score": values[index], "class_id": label,
                    "label": classes[label]}
            if mode == "crack" and mask_rendering == "masks":
                # Threshold logits at zero (equivalent to sigmoid > .5), then
                # crop to detector box. No physical width/severity is inferred.
                logits = rows[indices[index], 4 + count:] @ np.asarray(outputs[1])[0].reshape(32, -1)
                mask640 = cv2.resize(logits.reshape(160, 160), (640, 640))
                rw, rh = round(w * sx), round(h * sy)
                mask = cv2.resize(mask640[py:py + rh, px:px + rw], (w, h)) > 0
                x, y, bw, bh = item["bbox"]
                cropped = np.zeros((h, w), bool); cropped[y:y + bh, x:x + bw] = mask[y:y + bh, x:x + bw]
                item["mask"] = cropped
            detections.append(item)
    return sorted(detections, key=lambda x: -x["score"])[:100]


def decode_damage_boxes(outputs, transform, shape, size, classes, threshold=.25,nms_iou=.45,class_thresholds=None):
    count=len(classes); anchors=sum((size//stride)**2 for stride in [8,16,32])
    prediction=np.asarray(outputs[0])
    if prediction.shape!=(1,4+count,anchors):raise ValueError(f"Damage detection output contract mismatch: {prediction.shape}")
    rows=prediction[0].T;scores=rows[:,4:].max(axis=1);labels=rows[:,4:].argmax(axis=1)
    sx,sy,px,py=transform;h,w=shape;result=[]
    for category,label in enumerate(classes):
        cutoff=float((class_thresholds or {}).get(label,threshold))
        indices=np.flatnonzero((labels==category)&(scores>=cutoff)&np.isfinite(rows).all(axis=1));boxes=[];values=[]
        for index in indices:
            cx,cy,bw,bh=rows[index,:4];x1=max(0,min(w,round((float(cx-bw/2)-px)/sx)));y1=max(0,min(h,round((float(cy-bh/2)-py)/sy)))
            x2=max(0,min(w,round((float(cx+bw/2)-px)/sx)));y2=max(0,min(h,round((float(cy+bh/2)-py)/sy)))
            if x2>x1 and y2>y1:boxes.append((x1,y1,x2-x1,y2-y1));values.append(float(scores[index]))
        keep=np.asarray(cv2.dnn.NMSBoxes(boxes,values,cutoff,nms_iou)).reshape(-1) if boxes else []
        for index in keep:
            result.append({'label':label,'class_id':category,'bbox':boxes[int(index)],'score':values[int(index)]})
    return sorted(result,key=lambda x:-x['score'])[:100]


def _required_execution_guard(spec, options, delegate=None):
    """Only explicitly constrained research assets reject default/fallback backends."""
    required = spec.get('required_execution')
    if required is None:
        return
    expected = {'engine': 'onnxruntime', 'openvino_device': 'CPU', 'cpu_threads': 1}
    if (not isinstance(required, dict) or required != expected
            or type(required.get('cpu_threads')) is not int):
        raise ValueError('Invalid required_execution metadata for CPU research asset')
    if any(key not in options or type(options[key]) is not type(value) or options[key] != value
           for key, value in expected.items()):
        raise ValueError('This FAILED research asset requires explicit onnxruntime/CPU/cpu_threads=1; no default or fallback')
    if delegate is None:
        return
    actual = delegate.actual_execution_config
    if (delegate.engine != 'onnxruntime' or delegate.device != 'CPU' or type(delegate.threads) is not int or delegate.threads != 1
            or actual.get('providers') != ['CPUExecutionProvider']
            or type(actual.get('intra_op_num_threads')) is not int or actual['intra_op_num_threads'] != 1
            or type(actual.get('inter_op_num_threads')) is not int or actual['inter_op_num_threads'] != 1):
        raise ValueError('Required CPU research backend differs from actual delegate execution')
    settings = delegate.session.get_session_options()
    if (delegate.session.get_providers() != ['CPUExecutionProvider']
            or type(settings.intra_op_num_threads) is not int or settings.intra_op_num_threads != 1
            or type(settings.inter_op_num_threads) is not int or settings.inter_op_num_threads != 1
            or delegate.session.get_inputs()[0].type != 'tensor(float)'):
        raise ValueError('Required CPU research backend differs from actual provider/session threads/input type')


class ConstructionModel:
    def __init__(self, options=None):
        options = dict(options or {})
        self.mode = options.get("mode", "ppe")
        if not isinstance(self.mode, str) or self.mode not in {"ppe", "crack", "damage"}:
            raise ValueError("construction mode must be ppe, crack or damage")
        self.damage_variant = options.get("damage_variant")
        self.crack_variant = options.get("crack_variant", "baseline")
        if not isinstance(self.crack_variant, str) or self.crack_variant not in CRACK_VARIANTS:
            raise ValueError("construction crack_variant must be baseline or joint_aware_candidate")
        if self.mode != "crack" and self.crack_variant != "baseline":
            raise ValueError("construction crack_variant applies only to crack mode")
        if self.mode == "damage":
            if not isinstance(self.damage_variant,str) or self.damage_variant not in DAMAGE_VARIANTS:
                raise ValueError("Select an installed, verified construction damage_variant")
            self.spec = DAMAGE_VARIANTS[self.damage_variant]
        else:
            self.spec = CRACK_VARIANTS[self.crack_variant] if self.mode == "crack" else MODELS[self.mode]
        _required_execution_guard(self.spec, options)
        self.input_size = int(self.spec.get("input_size",640))
        self.decoder = self.spec.get("decoder","yolo_seg" if self.mode == "crack" else "yolo_detect")
        if self.mode == "damage" and self.decoder == "cubit_yolo_seg":
            from edge_runtime.profiles.construction_cubit import CubitConstructionModel
            wrapper = Path(__file__).with_name("construction_cubit.py")
            if hashlib.sha256(wrapper.read_bytes()).hexdigest() != self.spec.get("wrapper_sha256"):
                raise ValueError("Author segmentation implementation identity changed")
            self.delegate = CubitConstructionModel(options, self.spec)
            _required_execution_guard(self.spec, options, self.delegate)
            self.mask_rendering = self.delegate.mask_rendering
            return
        if self.mode == "damage" and self.decoder == "dacl_semseg":
            from edge_runtime.profiles.construction_dacl import DaclConstructionModel
            self.delegate = DaclConstructionModel(options,self.spec)
            self.mask_rendering = self.delegate.mask_rendering
            return
        if self.mode == "damage" and self.decoder == "lite_semseg":
            from edge_runtime.profiles.construction_lite import LiteCrackConstructionModel
            delegated_options=dict(options)
            delegated_options.setdefault("exposure_policy",self.spec.get("exposure_policy","off"))
            cuts=delegated_options.get("class_confidence",self.spec.get("class_confidence",{}))
            if not isinstance(cuts,dict) or any(name!="crack" for name in cuts):
                raise ValueError("This crack specialist cannot apply spalling or other class thresholds")
            if "confidence" not in delegated_options and "crack" in cuts:
                delegated_options["confidence"]=cuts["crack"]
            self.delegate=LiteCrackConstructionModel(delegated_options,self.spec)
            self.mask_rendering=self.delegate.mask_rendering
            return
        if self.mode == "damage" and self.decoder == "tile_patch_bank":
            from edge_runtime.profiles.construction_tile_adapter import TileConstructionModel
            for relative, field in [("construction_tile_adapter.py", "wrapper_sha256"),
                                    ("construction_tile.py", "fitted_source_sha256")]:
                if hashlib.sha256(Path(__file__).with_name(relative).read_bytes()).hexdigest() != self.spec[field]:
                    raise ValueError("Tile specialist implementation identity changed")
            self.delegate = TileConstructionModel(options, self.spec)
            self.mask_rendering = self.delegate.rendering
            return
        self.exposure_policy = options.get("exposure_policy", self.spec.get("exposure_policy", "off"))
        if not isinstance(self.exposure_policy, str) or self.exposure_policy not in {"off", "low_light_gain"} or (self.mode != "damage" and self.exposure_policy != "off"):
            raise ValueError("Construction camera exposure policy must be off or low_light_gain for a damage detector")
        self.selected_damage_classes = None
        if self.mode == "damage":
            selected = options.get("damage_classes", self.spec["classes"])
            if not isinstance(selected, list) or not selected or any(not isinstance(name, str) or name not in self.spec["classes"] for name in selected):
                raise ValueError("damage_classes must select this detector's supported condition names")
            self.selected_damage_classes = list(dict.fromkeys(selected))
        self.mask_rendering = options.get("mask_rendering", "boxes" if self.mode == "damage" else "masks")
        if self.mode == "damage" and self.mask_rendering != "boxes":
            raise ValueError("This damage detector supplies boxes only; pixel masks are not available")
        if not isinstance(self.mask_rendering, str) or self.mask_rendering not in {"masks", "boxes"}:
            raise ValueError("construction mask_rendering must be masks or boxes")
        self.preprocessing = options.get("ppe_preprocess", "letterbox") if self.mode == "ppe" else "letterbox"
        if not isinstance(self.preprocessing, str) or self.preprocessing not in {"stretch", "letterbox"}:
            raise ValueError("construction ppe_preprocess must be stretch or letterbox")
        self.path = (ROOT / "models/construction" / self.spec["file"]).resolve()
        if not self.path.is_file():
            raise ValueError("Construction model missing; run scripts/setup_construction.py")
        self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
        if self.sha256 != self.spec["sha256"]:
            raise ValueError("Construction model SHA256 mismatch")
        self.threshold = float(options.get("confidence", self.spec.get("confidence", .25)))
        self.class_thresholds = {} if "confidence" in options else dict(self.spec.get("class_confidence",{}))
        if any(k not in self.spec["classes"] or not math.isfinite(float(v)) or not 0 < float(v) < 1 for k,v in self.class_thresholds.items()):
            raise ValueError("Invalid damage class confidence mapping")
        if self.class_thresholds:self.threshold = min(float(v) for v in self.class_thresholds.values())
        self.nms_iou = float(options.get("nms_iou", .45))
        if not all(math.isfinite(v) and 0 < v < 1 for v in (self.threshold, self.nms_iou)):
            raise ValueError("construction confidence and nms_iou must be in (0,1)")
        self.engine = options.get("engine", "onnxruntime")
        self.device = options.get("openvino_device", "CPU")
        self.threads = options.get("cpu_threads", 4)
        if type(self.threads) is not int or not 1 <= self.threads <= 8 or not isinstance(self.device, str) or self.device not in {"CPU", "GPU"}:
            raise ValueError("Invalid construction CPU threads or OpenVINO device")
        started = time.perf_counter()
        if self.engine == "onnxruntime":
            if self.device != "CPU":
                raise ValueError("GPU requires construction engine openvino")
            import onnxruntime as ort
            settings = ort.SessionOptions(); settings.intra_op_num_threads = self.threads; settings.inter_op_num_threads = 1
            settings.add_session_config_entry("session.intra_op.allow_spinning", "0")
            self.session = ort.InferenceSession(str(self.path), sess_options=settings, providers=["CPUExecutionProvider"])
            if self.session.get_inputs()[0].shape != [1, 3, self.input_size, self.input_size]:
                raise ValueError("Construction input contract mismatch")
            self.input_name = self.session.get_inputs()[0].name; self.version = ort.__version__
        elif self.engine == "openvino":
            import openvino as ov
            core = ov.Core()
            if self.device not in core.available_devices:
                raise ValueError(f"Construction OpenVINO device {self.device} unavailable")
            model = core.read_model(str(self.path))
            if list(model.input().shape) != [1, 3, self.input_size, self.input_size]:
                raise ValueError("Construction input contract mismatch")
            config = {"PERFORMANCE_HINT": "LATENCY", "INFERENCE_PRECISION_HINT": "f32"}
            if self.device == "CPU":
                config.update(INFERENCE_NUM_THREADS=self.threads, NUM_STREAMS=1)
            self.compiled = core.compile_model(model, self.device, config); self.request = self.compiled.create_infer_request()
            self.version = ov.get_version()
        else:
            raise ValueError("construction engine must be onnxruntime or openvino")
        self.load_ms = (time.perf_counter() - started) * 1000

    def prepare_input(self, frame):
        from edge_runtime.profiles.construction_preprocessing import camera_exposure
        corrected, gain = camera_exposure(frame, getattr(self, "exposure_policy", "off"))
        self.last_input_exposure_gain = gain
        return prepare(corrected, self.mode, self.preprocessing, self.input_size)

    def infer(self, frame):
        if hasattr(self,"delegate"):
            if self.decoder=="lite_semseg":
                from edge_runtime.profiles.construction_preprocessing import camera_exposure
                # Metadata only: the actual delegate receives RAW and applies
                # its pinned policy once. Do not double-enhance its NN input.
                _,self.last_input_exposure_gain=camera_exposure(frame,self.delegate.exposure_policy)
            rows = self.delegate.infer(frame)
            if self.decoder == "cubit_yolo_seg":
                self.last_input_exposure_gain = self.delegate.last_input_exposure_gain
            return rows
        tensor, transform = self.prepare_input(frame)
        if self.engine == "onnxruntime":
            outputs = self.session.run(None, {self.input_name: tensor})
        else:
            result = self.request.infer({0: tensor})
            outputs = [result[self.compiled.output(i)] for i in range(len(self.compiled.outputs))]
        if self.mode == "damage":
            rows = decode_damage_boxes(outputs, transform, frame.shape[:2], self.input_size, self.spec["classes"], self.threshold,self.nms_iou,self.class_thresholds)
            return [row for row in rows if row["label"] in self.selected_damage_classes]
        return decode(outputs, self.mode, transform, frame.shape[:2], self.threshold, self.nms_iou, self.mask_rendering, classes=self.spec["classes"])

    def describe(self):
        if hasattr(self,"delegate"):
            description=self.delegate.describe() | {"damage_variant":self.damage_variant,"mode":"damage"}
            description["diagnostic_only"]=bool(self.spec.get("diagnostic_only",False))
            return description
        return {"name": f"Construction {self.mode} observation", "sha256": self.sha256, "model_path": str(self.path),
                "model_source": self.spec["source"], "classes": self.spec["classes"], "engine": self.engine,
                "engine_version": self.version, "device": self.device, "cpu_threads": self.threads,
                "input_shape": [1, 3, self.input_size, self.input_size], "model_load_ms": round(self.load_ms, 2),
                "confidence_threshold": self.threshold, "class_confidence":self.class_thresholds, "nms_iou": self.nms_iou,
                "selected_damage_classes":self.selected_damage_classes,
                "supported_capabilities": ([self.spec.get("class_capabilities", {}).get(name, f"surface_{name}") for name in self.selected_damage_classes] if self.mode == "damage" else self.spec["capabilities"]),
                "preprocessing": self.preprocessing, "exposure_policy":self.exposure_policy,
                "damage_variant":self.damage_variant, "crack_variant": self.crack_variant if self.mode == "crack" else "not_applicable",
                "mask_rendering": self.mask_rendering if self.mode in {"crack","damage"} else "not_applicable",
                "mask_scope": ("Box-only damage detector; no pixel-mask accuracy claim" if self.mode == "damage" else "Not applicable; PPE has a box-detection head" if self.mode != "crack" else
                               "Dense masks reconstructed; masks have no calibrated physical units" if self.mask_rendering == "masks" else
                               "Boxes only; dense mask reconstruction is skipped, with no pixel-mask accuracy claim"),
                "license_status": ("AGPL-3.0 Ultralytics lineage; PPE author declares CC BY 4.0 weights; proprietary redistribution not cleared"
                                   if self.mode == "ppe" else self.spec.get("license", "Experimental damage model; consult training provenance") if self.mode == "damage" else "AGPL-3.0 weights; merged Roboflow source licenses not independently cleared; proprietary redistribution not cleared"
                                   if self.crack_variant == "joint_aware_candidate" else "AGPL-3.0 weights and Ultralytics lineage; proprietary redistribution not cleared"),
                "experimental": True, "diagnostic_only":bool(self.spec.get("diagnostic_only",False)),
                "validation_scope": self.spec.get("validation_scope", "Public diagnostic data; training overlap unknown; no Saudi site validation"),
                "safety_validation": (f"Insufficient for safety monitoring: bare-head recall {'1/5' if self.preprocessing == 'letterbox' else '0/5'} on a tiny reused real-video diagnostic"
                                      if self.mode == "ppe" else self.spec.get("validation_scope", "Experimental supervised crack/spalling; not independent Saudi-site qualification") if self.mode == "damage" else "Experimental only: 80% aggregate box recall / 58.6% precision on two exposed ev-CIVIL sequences; challenge sequence 0/8 localized; spalling unsupported"
                                      if self.crack_variant == "joint_aware_candidate" else "Not qualified for structural inspection; no independent Saudi site or video ground truth"),
                "limitations": "Visible observations only. No code certification, structural strength, crack width or concealed works. Missing PPE is never inferred from absent detections."}
