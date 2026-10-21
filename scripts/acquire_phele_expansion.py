"""Bounded public PHELE ZIP audit using the shared strict HTTP range reader."""
from pathlib import Path
import argparse
import hashlib
import json
import zipfile
import struct
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed

import acquire_sdnet_expansion as remote

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "datasets/construction/expansion-public-v1/phele"
remote.OUT = OUT
remote.URL = "https://data.mendeley.com/public-files/datasets/yzsxcvc8dp/files/010f225b-ec1e-4a66-adf2-1579028acb03/file_downloaded"
remote.MAX_BYTES = 600_000_000


class PheleZip(remote.RemoteZip):
    def __init__(self):
        self.blocks = []
        super().__init__()

    def request(self, start, end):
        if start >= 0:
            for first, last, data in self.blocks:
                if first <= start and end <= last:
                    return data[start-first:end-first+1]
        return super().request(start, end)

    def prefetch(self, first, last):
        self.blocks.append((first, last, super().request(first, last)))


def inventory():
    OUT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(remote.RemoteZip()) as archive:
        rows = [{"name": i.filename, "bytes": i.file_size, "compressed_bytes": i.compress_size,
                 "crc32": i.CRC, "header_offset": i.header_offset, "compression": i.compress_type}
                for i in archive.infolist() if not i.is_dir()]
    remote.write(OUT / "zip-inventory.json", rows)
    print(json.dumps({"members": len(rows), "first_members": rows[:30]}), flush=True)


def acquire(names):
    stream = PheleZip()
    inventory = json.loads((OUT / "zip-inventory.json").read_text())
    with zipfile.ZipFile(stream) as archive:
        for split in ("train", "val", "test"):
            wanted = [row for row in inventory if row["name"] in names and row["name"].startswith(f"PHELE/{split}/labels/")]
            if wanted:
                first = min(row["header_offset"] for row in wanted)
                last = max(row["header_offset"]+row["compressed_bytes"]+len(row["name"].encode())+150 for row in wanted)
                stream.prefetch(first, last)
        receipt = []
        for name in names:
            info = archive.getinfo(name)
            if ".." in Path(name).parts or Path(name).is_absolute() or ":" in name or info.file_size > 32_000_000:
                raise ValueError("Unsafe or oversized member")
            target = OUT / "original" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                target.write_bytes(archive.read(name))
            receipt.append({"member": name, "path": target.relative_to(ROOT).as_posix(), "bytes": target.stat().st_size,
                            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(), "crc32": info.CRC})
            remote.write(OUT / "acquired-members.json", {"records": receipt, "whole_archive_checksum_verified": False})
            print("ACQUIRED", name, target.stat().st_size, flush=True)


def images_parallel():
    """Four independent bounded image ranges; validate ZIP CRC after inflation."""
    names = json.loads((OUT/"selected-image-members.json").read_text())
    inventory = {row["name"]: row for row in json.loads((OUT/"zip-inventory.json").read_text())}
    stream = PheleZip()
    missing = [inventory[name] for name in names if not (OUT/"original"/name).exists()]
    ledger = json.loads((OUT/"range-ledger.json").read_text())
    if ledger["received_bytes"]+sum(row["compressed_bytes"]+2048 for row in missing) > remote.MAX_BYTES:
        raise ValueError("Prospective entire parallel acquisition exceeds byte budget")
    def one(row):
        name = row["name"]
        if ".." in Path(name).parts or ":" in name or not name.startswith("PHELE/"):
            raise ValueError("Unsafe image member")
        first = row["header_offset"]
        last = min(stream.size-1, first+30+len(name.encode())+row["compressed_bytes"]+1023)
        request = urllib.request.Request(stream.url, headers={"Range": f"bytes={first}-{last}", "User-Agent": "InspectFlow-public-research/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response:
            if response.status != 206 or response.headers.get("Content-Range") != f"bytes {first}-{last}/{stream.size}":
                raise ValueError("Range/size mismatch")
            blob = response.read(last-first+2)
            if len(blob) != last-first+1:
                raise ValueError("Range length mismatch")
        header = struct.unpack("<4s5H3I2H", blob[:30])
        if header[0] != b"PK\x03\x04" or header[2] & 1 or header[10] > 1024:
            raise ValueError("Unsupported ZIP local record")
        start = 30+header[9]+header[10]
        compressed = blob[start:start+row["compressed_bytes"]]
        body = zlib.decompress(compressed, -15) if header[3] == 8 else compressed if header[3] == 0 else None
        if body is None or len(body) != row["bytes"] or zlib.crc32(body) != row["crc32"]:
            raise ValueError("Original member CRC/length failed")
        target = OUT/"original"/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(body)
        return {"member": name, "path": target.relative_to(ROOT).as_posix(), "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest(),
                "crc32": row["crc32"], "range": f"{first}-{last}/{stream.size}", "range_received_bytes": len(blob), "range_sha256": hashlib.sha256(blob).hexdigest()}
    records = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(one, row): row for row in missing}
        for future in as_completed(futures):
            result = future.result()
            records.append(result)
            remote.write(OUT/"image-acquisition.json", {"records": records, "previously_acquired": len(names)-len(missing),
                "whole_archive_checksum_verified": False, "downloaded_range_bytes": sum(row["range_received_bytes"] for row in records)})
            print("ACQUIRED", len(records), "/", len(missing), result["member"], result["bytes"], flush=True)
    merge_receipts()


def merge_receipts():
    """Count parallel requests in the same persistent strict download budget."""
    receipt_path = OUT/"image-acquisition.json"
    if not receipt_path.exists():
        return
    receipt = json.loads(receipt_path.read_text())
    ledger = json.loads((OUT/"range-ledger.json").read_text())
    known = {row["range"] for row in ledger["requests"]}
    for row in receipt["records"]:
        if row["range"] not in known:
            ledger["requests"].append({"range": row["range"], "bytes": row["range_received_bytes"], "sha256": row["range_sha256"],
                                       "member": row["member"], "source": remote.URL, "method": "four_worker_crc_verified_image_range"})
            ledger["received_bytes"] += row["range_received_bytes"]
            known.add(row["range"])
    if ledger["received_bytes"] > remote.MAX_BYTES:
        raise ValueError("Combined persistent range budget exceeded")
    remote.write(OUT/"range-ledger.json", ledger)
    print("TOTAL_RANGE_BYTES", ledger["received_bytes"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["inventory", "members", "annotations", "images_parallel", "merge_receipts"])
    parser.add_argument("--selection", type=Path)
    args = parser.parse_args()
    if args.stage == "inventory":
        inventory()
    elif args.stage == "merge_receipts":
        merge_receipts()
    elif args.stage == "images_parallel":
        images_parallel()
    elif args.stage == "annotations":
        rows = json.loads((OUT / "zip-inventory.json").read_text())
        acquire([row["name"] for row in rows if row["name"] == "PHELE/classes.txt" or row["name"].startswith("PHELE/") and "/labels/" in row["name"]])
    else:
        if args.selection is None:
            parser.error("--selection JSON exact member list required")
        acquire(json.loads(args.selection.read_text()))
