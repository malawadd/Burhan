"""Fetch pinned public ONNX files and genuine footage; inference never downloads."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile

ROOT=Path(__file__).resolve().parents[1]
ASSETS=[
    ("models/construction/ppe-yolov8n.onnx","https://huggingface.co/mancalazure/construction-ppe-safety/resolve/70cc801710f246765f70ada7248d9560ec749bb2/best.onnx","1d5f3cf69883e89ba6e2b3836a0b21623044e9a682f1f00dfbacac97b99baf94","AGPL-3.0 metadata; author CC BY 4.0 weights declaration; see docs/construction-vision.md"),
    ("models/construction/crack-yolov8n-seg.onnx","https://huggingface.co/beaunix/smart-crack-vision/resolve/9442007ef9c1003880baa5796d253608691b37c4/crack_detection_model.onnx","4b789658646d790079d3e78d9d3ed475030d740a4b8412aff52248189a114cbd","AGPL-3.0"),
    ("datasets/construction/men-at-work-cc0.webm","https://upload.wikimedia.org/wikipedia/commons/d/d6/Men_at_work_in_building_site.webm","f7cbf03f56e0a864ddba057c47b8a67c57f176e10e04995c59e14d8ab00024d7","CC0 1.0; Akum20 own work 12 March 2024"),
    ("datasets/construction/framer-fall-protection-public-domain.webm","https://upload.wikimedia.org/wikipedia/commons/2/2b/A_construction_framer_talks_about_protecting_his_crew_from_falls.webm","017422936ccbab19830136730acec895fb788a06b6e1b195a3cb300530d2728a","US Federal public domain; National Institute for Occupational Safety and Health 2014"),
]
DATA=[
    ("datasets/construction/atharion-v1.0-public.zip","https://github.com/atharion-team/ppe-dataset-v1/releases/download/v1.0/atharion-v1.0-public.zip","f5f6f1966c4bdfff4151ee68fa0da21f446e1b788d15280e5f70028c69ec88c5","Mixed AGPL-3.0 and CC BY 4.0; retain archive NOTICE.md, attribution.csv and source links"),
    ("datasets/construction/crack-seg.zip","https://github.com/ultralytics/assets/releases/download/v0.0.0/crack-seg.zip","814b566d61e890024d15f22a3e6b14e252f781ab83f60a8d59e6684817efecdc","Public Domain as declared by Ultralytics; University crack-bphdr Roboflow source"),
]


def obtain(item):
    filename,url,expected,license_note=item;path=ROOT/filename;path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest()==expected:return
    request=urllib.request.Request(url,headers={"User-Agent":"InspectFlowResearch/1.0 (local evaluation)"})
    data=urllib.request.urlopen(request,timeout=120).read()
    if hashlib.sha256(data).hexdigest()!=expected:raise ValueError(f"SHA256 mismatch: {filename}")
    temp=path.with_suffix(path.suffix+".download");temp.write_bytes(data);temp.replace(path)


if __name__=="__main__":
    a=argparse.ArgumentParser();a.add_argument("--data",action="store_true",help="Also fetch/extract real annotated test splits (~194MB archives)");args=a.parse_args()
    for item in ASSETS+(DATA if args.data else []):obtain(item);print(item[0],flush=True)
    if args.data:
        base=(ROOT/"datasets/construction").resolve()
        for item in DATA:
            with zipfile.ZipFile(ROOT/item[0]) as archive:
                for name in archive.namelist():
                    if not ("/test/" in name or "/valid/" in name or name.endswith((".yaml",".md",".csv",".json"))):continue
                    target=(base/name).resolve()
                    if not target.is_relative_to(base):raise ValueError("Archive path escaped construction directory")
                    if not name.endswith("/"):target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(archive.read(name))
    manifest={"assets":[dict(zip(("path","url","sha256","license"),x)) for x in ASSETS+DATA],
              "inference_network":"No network calls; these are explicit setup-only downloads",
              "video_proof":"Original real recordings, no synthetic frames, generated scenes or still-image videos"}
    (ROOT/"models/construction/manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
