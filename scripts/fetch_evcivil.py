"""Download three preselected licensed raw field sequences using bounded ZIP byte ranges."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen
import zipfile
import zlib
from fetch_construction_bridge_video import RangeFile

ROOT = Path(__file__).resolve().parents[1]
SEQUENCES = ["Field_dataset/field_data_day/dset_1/crack_10/",
             "Field_dataset/field_data_day/dset_2/spalling_35/",
             "Field_dataset/day_time_challenge/crack_89/"]


def main():
    # Use the OS certificate store when running inside an environment with a managed CA.
    try:
        from pip._vendor import truststore
        truststore.inject_into_ssl()
    except ImportError:
        pass
    with urlopen("https://api.figshare.com/v2/articles/25924228", timeout=30) as response:
        metadata = json.load(response)
    if metadata["license"]["name"] != "CC BY 4.0":
        raise RuntimeError("Publisher license changed; review before downloading")
    archive = next(f for f in metadata["files"] if f["name"] == "evCIVIL_Field_dataset.zip")
    root = ROOT / "datasets/construction/evcivil"; root.mkdir(parents=True, exist_ok=True)
    (root / "publisher-record.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    with zipfile.ZipFile(RangeFile(archive["download_url"], archive["size"])) as source:
        for prefix in SEQUENCES:
            for entry in source.infolist():
                if not entry.filename.startswith(prefix) or not entry.filename.endswith(("_frames.h5", "_label.npy", "_frame_label.npy")):
                    continue
                target = root / Path(prefix.rstrip("/")).name / Path(entry.filename).name
                target.parent.mkdir(exist_ok=True)
                if not target.exists():
                    temporary = target.with_suffix(target.suffix + ".part")
                    with source.open(entry) as incoming, temporary.open("wb") as outgoing:
                        while block := incoming.read(1024 * 1024): outgoing.write(block)
                    if temporary.stat().st_size != entry.file_size: raise RuntimeError("Truncated sequence member")
                    temporary.replace(target)  # ZipExtFile checks CRC on full member read.
                crc=0
                with target.open("rb") as local:
                    while block:=local.read(1024*1024):crc=zlib.crc32(block,crc)
                if target.stat().st_size!=entry.file_size or crc!=entry.CRC:
                    raise RuntimeError(f"Existing member length/CRC mismatch: {target}")
                provenance = {"source_doi": "https://doi.org/10.11583/DTU.25924228", "license": "CC BY 4.0",
                              "authors": "Kashita Niranjan Udayanga Gangoda Withana Gamage; Silvia Tolu; Matteo Fumagalli; Tobi Delbruck; Cesar Cadena. Annotation contributor: Xuanni Huo.",
                              "zip_member": entry.filename, "zip_crc32": f"{entry.CRC:08x}",
                              "size": target.stat().st_size, "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
                target.with_suffix(target.suffix + ".provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
                print(target)


if __name__ == "__main__": main()
