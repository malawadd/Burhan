"""Audit provider-test convenience cases against original RDD2022 VOC labels."""
from __future__ import annotations

import hashlib
import json
import struct
import zipfile
import zlib
from pathlib import Path

from scripts.setup_roads_assets import RemoteRange, fetch

ROOT = Path(__file__).resolve().parents[1]


def main():
    destination = ROOT / "datasets/roads-rdd2022-test-balanced"
    manifest = json.loads((destination / "manifest.json").read_text())
    metadata = json.loads(fetch("https://api.figshare.com/v2/articles/21431547"))
    size = next(f["size"] for f in metadata["files"] if f["id"] == 38030910)
    url = "https://ndownloader.figshare.com/files/38030910"
    outer_remote = RemoteRange(url, size)
    offsets = {}
    with zipfile.ZipFile(outer_remote) as outer:
        for item in outer.infolist():
            if item.filename.lower().endswith(".zip"):
                if item.compress_type != 0:
                    raise ValueError("Expected stored country archives")
                outer_remote.seek(item.header_offset)
                head = outer_remote.read(30)
                nl, el = struct.unpack("<HH", head[26:30])
                offsets[Path(item.filename).stem] = (item.file_size, item.header_offset + 30 + nl + el)
    audit, transferred = [], outer_remote.fetched
    for partition in sorted(set(r["partition"] for r in manifest["images"])):
        nested_size, base = offsets[partition]
        remote = RemoteRange(url, nested_size, base)
        with zipfile.ZipFile(remote) as inner:
            for row in (r for r in manifest["images"] if r["partition"] == partition):
                filename = row["file_name"]
                image_member = next(n for n in inner.namelist() if n.endswith("/train/images/" + filename))
                xml_member = image_member.replace("/images/", "/annotations/xmls/").rsplit(".", 1)[0] + ".xml"
                image = (destination / "images" / filename).read_bytes()
                same_image = zlib.crc32(image) == inner.getinfo(image_member).CRC
                if not same_image:
                    raise ValueError(f"Provider image differs from original ZIP image: {filename}")
                xml = inner.read(xml_member)
                target = destination / "original-annotations" / (Path(filename).stem + ".xml")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(xml)
                audit.append({"image": filename, "original_image_crc32_match": same_image,
                              "xml_archive_member": xml_member, "xml_sha256": hashlib.sha256(xml).hexdigest()})
                print("Original annotation:", filename, flush=True)
        transferred += remote.fetched
    (destination / "original-annotations-audit.json").write_text(json.dumps({"source": metadata["url_public_html"],
          "license": metadata["license"], "range_bytes_downloaded": transferred, "images": audit}, indent=2))


if __name__ == "__main__":
    main()
