"""Fetch one real CC-BY bridge video using ZIP byte ranges, not the 18 GB archive."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import time
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
RECORD = "https://zenodo.org/api/records/19268973"
ARCHIVE = "03_curated_dataset_bridge_videos.zip"


def request(url, headers=None):
    return urlopen(Request(url, headers={"User-Agent": "InspectFlow-public-data-research/1.0", **(headers or {})}), timeout=90)


class RangeFile(io.RawIOBase):
    def __init__(self, url, size):
        self.url, self.size, self.position = url, size, 0

    def seekable(self):
        return True

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        if self.position < 0:
            raise ValueError("Negative byte position")
        return self.position

    def tell(self):
        return self.position

    def read(self, count=-1):
        count = self.size - self.position if count < 0 else min(count, self.size - self.position)
        if count <= 0:
            return b""
        first, last = self.position, self.position + count - 1
        for attempt in range(3):
            try:
                with request(self.url, {"Range": f"bytes={first}-{last}"}) as response:
                    if response.status != 206:
                        raise RuntimeError("Server did not honor byte range; refusing a full-archive download")
                    if not response.headers.get("Content-Range", "").startswith(f"bytes {first}-"):
                        raise RuntimeError("Unexpected response range")
                    blob = response.read(count + 1)
                    if len(blob) != count:
                        raise RuntimeError("Truncated range response")
                self.position += len(blob)
                return blob
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(1 + attempt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--member", default="bd_vid_0009.mp4", help="Exact original video basename")
    args = parser.parse_args()
    if Path(args.member).name != args.member or not args.member.endswith(".mp4"):
        raise SystemExit("Expected an MP4 basename")
    with request(RECORD) as response:
        metadata = json.load(response)
    archive = next(f for f in metadata["files"] if f["key"] == ARCHIVE)
    if metadata["metadata"].get("license", {}).get("id") != "cc-by-4.0":
        raise SystemExit("Publisher license changed; inspect record before downloading")
    destination = ROOT / "datasets/construction/bridge_raw"
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / args.member
    if target.exists():
        raise SystemExit(f"Already present; refusing to overwrite: {target}")
    member = f"03_curated_dataset_bridge_videos/{args.member}"
    temporary = target.with_suffix(".mp4.part")
    digest, size = hashlib.sha256(), 0
    with zipfile.ZipFile(RangeFile(archive["links"]["self"], archive["size"])) as remote:
        entry = remote.getinfo(member)
        with remote.open(entry) as video, temporary.open("wb") as output:
            while block := video.read(8 * 1024 * 1024):
                output.write(block)
                digest.update(block)
                size += len(block)
                print(f"{args.member}: {size}/{entry.file_size} bytes", flush=True)
        # ZipExtFile validates the complete selected member CRC on EOF.
    if size != entry.file_size:
        raise RuntimeError("Member length mismatch")
    temporary.replace(target)
    provenance = {
        "record": "https://zenodo.org/records/19268973", "title": metadata["metadata"]["title"],
        "creator": "N. T. Tsegaye", "license": "CC BY 4.0", "archive_url": archive["links"]["self"],
        "archive_bytes": archive["size"], "archive_checksum": archive["checksum"],
        "archive_hash_scope": "Publisher metadata only; full archive was not downloaded",
        "entry": member, "entry_crc32": f"{entry.CRC:08x}", "entry_crc_verified": True,
        "bytes": size, "sha256": digest.hexdigest(), "path": str(target.relative_to(ROOT)),
        "retrieved_on": time.strftime("%Y-%m-%d", time.gmtime()),
        "method": "HTTP ranges of a single ZIP member; original MP4 preserved without re-encoding",
        "scope": "Publisher identifies raw drone bridge video; no independent frame/event labels verified",
    }
    target.with_suffix(".provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    (destination / "LICENSE.md").write_text(
        "# Attribution\n\nTsegaye, N. T. (2026). Curated Dataset and Inference Results for AI-Based Bridge Defect Detection Using Images and Videos. Zenodo. https://doi.org/10.5281/zenodo.19268973\n\n"
        "CC BY 4.0: https://creativecommons.org/licenses/by/4.0/\n\nSelected original MP4 files are unmodified. See each provenance JSON for local SHA-256, publisher archive checksum and verified member CRC.\n", encoding="utf-8")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
