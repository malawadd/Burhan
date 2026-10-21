"""Explicit one-time public freight evaluation assets download; runtime never calls this."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import urllib.request
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "datasets" / "freight"
INVOICE_COMMIT = "bb2e531cfa504e6f7200243620121c523f9f7622"


def fetch(url, target):
    data = urllib.request.urlopen(url, timeout=45).read()
    if len(data)>10*1024*1024:
        raise ValueError("Unexpectedly large public sample")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return {"path": str(target.relative_to(ROOT)), "bytes":len(data),"sha256":hashlib.sha256(data).hexdigest()}


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    manifest={"downloaded_utc":datetime.now(timezone.utc).isoformat(),"purpose":"Small public smoke/evaluation subset; not training or full benchmark", "assets":[]}
    rows_url="https://datasets-server.huggingface.co/rows?dataset=naver-clova-ix%2Fcord-v2&config=default&split=test&offset=0&length=3"
    rows=json.load(urllib.request.urlopen(rows_url,timeout=45))["rows"]
    for row in rows:
        index=row["row_idx"]
        base=OUT/"cord"/f"test-{index:03d}"
        item=fetch(row["row"]["image"]["src"],base.with_suffix(".jpg"))
        base.with_suffix(".json").write_text(row["row"]["ground_truth"],encoding="utf-8")
        item.update(dataset="CORD v2",split="test",row=index,kind="real photographed receipt; out of freight type scope",license="CC-BY-4.0",source="https://huggingface.co/datasets/naver-clova-ix/cord-v2",rows_api=rows_url,attribution="Park et al., CORD: A Consolidated Receipt Dataset for Post-OCR Parsing; NAVER CLOVA")
        manifest["assets"].append(item)
    baseurl=f"https://raw.githubusercontent.com/SalorWorks/shopify-invoice-test-pack/{INVOICE_COMMIT}"
    for fixture in ["01-simple-english","11-freight-duty","14-poor-scan"]:
        for filename in ["invoice.png","expected.json"]:
            url=f"{baseurl}/fixtures/{fixture}/{filename}"
            item=fetch(url,OUT/"invoice-pack"/fixture/filename)
            item.update(dataset="Synthetic Shopify Invoice Test Pack v1.0.0",fixture=fixture,kind="public externally authored synthetic invoice; not live customer shipment",license="CC-BY-4.0",source=url,commit=INVOICE_COMMIT,attribution="SalorWorks, Synthetic Shopify Invoice Test Pack v1.0.0")
            manifest["assets"].append(item)
    fetch("https://raw.githubusercontent.com/clovaai/cord/master/LICENSE-CC-BY",OUT/"CORD-LICENSE.txt")
    fetch(f"{baseurl}/LICENSE",OUT/"INVOICE-PACK-LICENSE.txt")
    (OUT/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(json.dumps({"manifest":str(OUT/"manifest.json"),"assets":len(manifest["assets"]),"image_count":6},indent=2))

if __name__=="__main__":
    main()

