"""Download pinned road assets; runtime itself never uses the network."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import struct
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVISION = "84673b89d7def3dcb2211ef00a10e3eb0318d0dd"
MODEL_URL = "https://vivekvagale.github.io/road-hazard/models/yolo11n.onnx"
MODEL_HASH = "a93a8e1735aaf6cd502d13be54684b6335eb8cca1eec270d053e43550531e1f9"
RAW = f"https://raw.githubusercontent.com/VivekVagale/road-hazard/{REVISION}"


def fetch(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        data = response.read(32 * 1024 * 1024 + 1)
    if len(data) > 32 * 1024 * 1024:
        raise ValueError("Asset exceeds 32 MiB safety bound")
    return data


def write_asset(path, url, expected=None):
    data = path.read_bytes() if path.exists() else fetch(url)
    sha = hashlib.sha256(data).hexdigest()
    if expected is not None and sha != expected:
        raise ValueError(f"SHA256 mismatch: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {"file": str(path.relative_to(ROOT)).replace("\\", "/"), "url": url, "bytes": len(data), "sha256": sha}


class RemoteRange(io.RawIOBase):
    """Seekable bounded view of an uncompressed nested ZIP using HTTP byte ranges."""
    def __init__(self, url, size, base=0):
        self.url, self.size, self.base, self.position = url, size, base, 0
        self.fetched = 0

    def seekable(self):
        return True

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        if not 0 <= self.position <= self.size:
            raise ValueError("Invalid ZIP seek")
        return self.position

    def tell(self):
        return self.position

    def read(self, size=-1):
        size = self.size - self.position if size < 0 else min(size, self.size - self.position)
        if size == 0:
            return b""
        if size > 8 * 1024 * 1024:
            raise ValueError("Refusing oversized range read")
        start = self.base + self.position
        request = urllib.request.Request(self.url + f"?edge_subset={time.time_ns()}",
                                         headers={"Range": f"bytes={start}-{start + size - 1}"})
        with urllib.request.urlopen(request, timeout=60) as response:
            if response.status != 206:
                raise ValueError("Server did not honor bounded byte range")
            content_range = response.headers.get("Content-Range", "")
            if not content_range.startswith(f"bytes {start}-{start + size - 1}/"):
                raise ValueError("Unexpected Content-Range")
            data = response.read(size + 1)
        if len(data) != size:
            raise ValueError("Incomplete ranged download")
        self.position += size
        self.fetched += size
        return data


def labeled_subset(count=16):
    """First N lexicographic Czech train images: separate country from model training.

    This deterministic convenience slice is NOT representative or a held-out route.
    Read only selected members from the official 13 GB nested ZIP, checking ZIP CRCs.
    """
    url = "https://ndownloader.figshare.com/files/38030910"
    metadata = json.loads(fetch("https://api.figshare.com/v2/articles/21431547"))
    archive = next(f for f in metadata["files"] if f["id"] == 38030910)
    remote = RemoteRange(url, archive["size"])
    with zipfile.ZipFile(remote) as outer:
        info = outer.getinfo("RDD2022/Czech.zip")
        if info.compress_type != 0:
            raise ValueError("Nested archive must be uncompressed")
        remote.seek(info.header_offset)
        header = remote.read(30)
        name_len, extra_len = struct.unpack("<HH", header[26:30])
        base = info.header_offset + 30 + name_len + extra_len
    inner_remote = RemoteRange(url, info.file_size, base)
    destination = ROOT / "datasets" / "roads-rdd2022-czech-subset"
    assets = []
    with zipfile.ZipFile(inner_remote) as inner:
        names = inner.namelist()
        images = sorted(n for n in names if "/train/images/" in n and n.lower().endswith(".jpg"))[:count]
        for index, image in enumerate(images):
            annotation = image.replace("/images/", "/annotations/xmls/").rsplit(".", 1)[0] + ".xml"
            for name in (image, annotation):
                data = inner.read(name)  # zipfile verifies the member CRC
                target = destination / ("images" if name.endswith(".jpg") else "annotations") / Path(name).name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                assets.append({"file": str(target.relative_to(ROOT)).replace("\\", "/"),
                               "archive_member": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
            print(f"Czech sample {index + 1}/{len(images)}", flush=True)
    manifest = {"source": metadata["url_public_html"], "source_archive": url,
                "dataset": "RDD2022", "license": metadata["license"],
                "citation": "Arya et al., RDD2022: A multi-national image dataset for automatic Road Damage Detection (2022/2024)",
                "selection": f"First {count} lexicographic Czech train images; no representative-sampling claim",
                "overlap": "Model author reports India-only training. Country separation is author-reported, not independently audited.",
                "downloaded_range_bytes": remote.fetched + inner_remote.fetched, "assets": assets}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labeled-subset", action="store_true")
    parser.add_argument("--count", type=int, default=16)
    args = parser.parse_args()
    if not 1 <= args.count <= 100:
        parser.error("count must be 1–100")
    assets = [write_asset(ROOT / "models/roads/yolo11n-rdd2022-india.onnx", MODEL_URL, MODEL_HASH)]
    for name in ("LICENSE", "README.md"):
        assets.append(write_asset(ROOT / f"models/roads/UPSTREAM-{name}", f"{RAW}/{name}"))
    for index in range(1, 13):
        name = f"{index:02}.jpg"
        assets.append(write_asset(ROOT / f"datasets/roads-rdd2022-samples/{name}", f"{RAW}/web/samples/{name}"))
    manifest = {"retrieved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "upstream_revision": REVISION, "model_sha256": MODEL_HASH,
                "model_note": "Model hosted outside Git; exact bytes pinned by SHA256. Refuse changed downloads.",
                "license_note": "Repository MIT; underlying YOLO11 pretrained derivation carries Ultralytics AGPL/enterprise terms. Not cleared for proprietary redistribution.",
                "samples": "Author identifies 12 samples as RDD2022 held-out test images, CC BY 4.0. No labels: inference smoke only.",
                "assets": assets}
    (ROOT / "models/roads/manifest.json").write_text(json.dumps(manifest, indent=2))
    if args.labeled_subset:
        labeled_subset(args.count)
    print(json.dumps({"assets": len(assets), "model_sha256": MODEL_HASH}))


if __name__ == "__main__":
    main()
