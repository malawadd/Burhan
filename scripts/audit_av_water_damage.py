"""Public AV-Water Damage access audit, without credentials or model inference.

Only publisher metadata/export routes are probed. No example images, thumbnails,
private media, login state, environment keys or cloud inference are accessed.
An HTTP 401/403 is recorded as an access boundary, never bypassed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
PROJECT = "https://universe.roboflow.com/assuralabs/av-water-damage"
VERSION = PROJECT + "/dataset/1"
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
ROUTES = {
    "project": PROJECT,
    "version": VERSION,
    "publisher_download": VERSION + "/download/yolov8",
    "version_api": "https://api.roboflow.com/assuralabs/av-water-damage/1",
    "export_api": "https://api.roboflow.com/assuralabs/av-water-damage/1/yolov8",
}


def sha_bytes(body):
    return hashlib.sha256(body).hexdigest()


def probe(name, url, out):
    request = urllib.request.Request(url, headers={"User-Agent": "InspectFlow-public-research/1.0"})
    row = {"url": url, "method": "GET", "credentials_supplied": False,
           "requested_utc": datetime.now(timezone.utc).isoformat()}
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            body = response.read(MAX_RESPONSE_BYTES + 1)
            row.update(status=response.status, content_type=response.headers.get("Content-Type"),
                       challenge=response.headers.get("Cf-Mitigated"))
    except urllib.error.HTTPError as error:
        body = error.read(MAX_RESPONSE_BYTES + 1)
        row.update(status=error.code, content_type=error.headers.get("Content-Type"),
                   challenge=error.headers.get("Cf-Mitigated"))
    except (urllib.error.URLError, TimeoutError) as error:
        return {**row, "error": str(error), "training_data_acquired": False}
    row["response_truncated"] = len(body) > MAX_RESPONSE_BYTES
    body = body[:MAX_RESPONSE_BYTES]
    # Save bounded metadata/error bodies only. Never label one as an archive.
    path = out / (name + "-response.bin")
    path.write_bytes(body)
    row.update(body_file=path.name, body_sha256=sha_bytes(body), body_bytes=len(body),
               training_data_acquired=False)
    if row["status"] == 401:
        row["access_boundary"] = "API authentication required; no key sought or supplied"
    elif row["status"] == 403:
        row["access_boundary"] = "HTTP forbidden/challenge; no bypass attempted"
    elif "zip" in (row.get("content_type") or "").lower():
        row["access_boundary"] = "Possible archive response; bounded probe is not an acquired or accepted dataset"
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=ROOT / "output/evaluation/construction-water-data/av-water-damage-v1-audit")
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists():
        raise ValueError("Preserve the previous access receipt; choose a new --out directory")
    out.mkdir(parents=True)
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": sha_bytes(Path(__file__).read_bytes()),
        "publisher_pages": {"overview": PROJECT, "version": VERSION},
        "metadata_evidence": "Primary publisher pages read through web tool; direct local access separately probed below",
        "overview_advertised": {"images": 1899, "task": "object_detection", "license": "CC BY 4.0",
                                "classes": ["crack", "stain", "mold", "damp", "dampness"],
                                "versions": 1, "published_models": 0, "citation_year": 2024},
        "version_1_advertised": {"generated_date": "2024-05-07", "total_images": 1145,
                                 "split_images": {"train": 747, "valid": 215, "test": 183},
                                 "preprocessing": ["auto_orient", "stretch_640x640"],
                                 "augmentations": "none"},
        "count_caveat": "Project image count differs from v1. Neither count establishes unique original scenes or an available annotated archive.",
        "class_semantics": {"version_yaml_verified": False, "damp_vs_dampness": "undefined; do not merge without labels/provenance audit",
                            "moisture_percentage": "not provided", "cause_or_organism": "not established by RGB labels"},
        "routes": {name: probe(name, url, out) for name, url in ROUTES.items()},
        "browser_access": {"url": VERSION + "/download/yolov8", "observed_status": 403,
                           "page_title": "Just a moment...", "challenge_bypassed": False,
                           "receipt": ".playwright-cli/page-2026-10-21T07-20-11-377Z.yml"},
        "training_status": "unavailable_no_accepted_image_annotation_archive",
        "training_images_acquired": 0, "annotation_files_acquired": 0,
        "thumbnails_used_as_dataset": False, "private_owner_pixels_accessed": False,
        "cloud_inference_used": False, "account_created": False, "credentials_accessed": False,
        "required_before_training": ["legitimate complete image/annotation export and license files",
                                     "version-specific class mapping and annotation completeness",
                                     "original-versus-derived identities and exact/perceptual duplicate audit",
                                     "physical wall/site grouping before splits",
                                     "verified healthy wall and difficult-negative annotations"],
    }
    path = out / "audit.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
    print(json.dumps({"audit": str(path), "statuses": {k: v.get("status", v.get("error")) for k, v in result["routes"].items()},
                      "training_images_acquired": 0, "training_status": result["training_status"]}, indent=2))


if __name__ == "__main__":
    main()
