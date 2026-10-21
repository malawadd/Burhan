"""Fetch the author-hosted KolektorSDD2 archive; keep test sealed by default.

No re-encoded or synthetic images are produced. Test extraction is explicit and
does not mean the released test is physically-item-disjoint (IDs not published).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
URL = "https://data.vicos.si/datasets/KSDD/KolektorSDD2.zip"
SOURCE = "https://www.vicos.si/resources/kolektorsdd2/"
EXPECTED_SIZE = 853126555
EXPECTED_SHA256 = "edcdb486809b24f1d17b785e30c52fafc5999554dd5fe18ddf77b61ceb6f36a8"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=ROOT / "datasets/factory/KolektorSDD2")
    parser.add_argument("--extract-test", action="store_true", help="Explicitly unseal the official test files")
    args = parser.parse_args()
    destination = args.destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / "KolektorSDD2.zip"
    started = time.perf_counter()
    if not archive.exists():
        partial = archive.with_suffix(".zip.partial")
        for attempt in range(5):
            offset = partial.stat().st_size if partial.exists() else 0
            if offset == EXPECTED_SIZE:
                break
            if offset > EXPECTED_SIZE:
                raise RuntimeError("Oversized partial archive; inspect instead of overwriting")
            request = urllib.request.Request(URL, headers={"Range": f"bytes={offset}-"} if offset else {})
            try:
                with urllib.request.urlopen(request, timeout=45) as response:
                    expected = int(response.headers.get("Content-Length", 0))
                    if offset:
                        if response.status != 206 or response.headers.get("Content-Range") != f"bytes {offset}-{EXPECTED_SIZE - 1}/{EXPECTED_SIZE}":
                            raise RuntimeError("Server did not honor exact partial-download range")
                    if expected != EXPECTED_SIZE - offset:
                        raise RuntimeError(f"Unexpected source byte count {expected}; inspect upstream change")
                    with partial.open("ab" if offset else "wb") as target:
                        shutil.copyfileobj(response, target, length=1024 * 1024)
            except (OSError, TimeoutError) as error:
                print(f"Download attempt {attempt + 1} interrupted: {error}; resuming retained bytes", flush=True)
        if not partial.exists() or partial.stat().st_size != EXPECTED_SIZE:
            raise RuntimeError("Incomplete archive after retries; partial file retained for resume")
        partial.replace(archive)
    if archive.stat().st_size != EXPECTED_SIZE:
        raise RuntimeError("Archive size does not match original release")
    archive_sha256 = sha256(archive)
    if archive_sha256 != EXPECTED_SHA256:
        raise RuntimeError("Archive SHA-256 differs from the acquired original author release")
    inventory = {"train": {"files": 0, "bytes": 0}, "test": {"files": 0, "bytes": 0}}
    with zipfile.ZipFile(archive) as source:
        for entry in source.infolist():
            name = PurePosixPath(entry.filename)
            if name.is_absolute() or ".." in name.parts or any(":" in part for part in name.parts):
                raise RuntimeError(f"Unsafe archive member {entry.filename!r}")
            if entry.is_dir():
                continue
            partition = name.parts[0]
            if partition not in inventory:
                continue
            inventory[partition]["files"] += 1
            inventory[partition]["bytes"] += entry.file_size
            if partition == "test" and not args.extract_test:
                continue
            target = destination.joinpath(*name.parts).resolve()
            if not target.is_relative_to(destination):
                raise RuntimeError("Archive path escapes dataset root")
            target.parent.mkdir(parents=True, exist_ok=True)
            # ZipFile.open verifies each decompressed member's CRC when read.
            with source.open(entry) as original, target.open("wb") as output:
                shutil.copyfileobj(original, output, length=1024 * 1024)
    manifest = {
        "dataset": "KolektorSDD2", "source_page": SOURCE, "archive_url": URL,
        "archive_bytes": archive.stat().st_size, "archive_sha256": archive_sha256,
        "license": "CC BY-NC-SA 4.0", "commercial_permission_required": True,
        "license_source": SOURCE, "download_date": "2026-10-20",
        "official_counts": {"train_defective": 246, "train_normal": 2085, "test_defective": 110, "test_normal": 894},
        "archive_member_inventory": inventory,
        "test_extracted": args.extract_test,
        "test_content_opened_by_this_script": args.extract_test,
        "test_seal_note": "Train-only default preserves unopened official test image/mask contents in the ZIP; filenames and byte sizes are inventoried.",
        "grouping": "Official train/test split. Physical-item/batch identities are not supplied; do not claim item-disjointness.",
        "video": "Still images captured in a controlled industrial environment; no source video is supplied.",
        "elapsed_seconds": time.perf_counter() - started,
    }
    (destination / "source-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
