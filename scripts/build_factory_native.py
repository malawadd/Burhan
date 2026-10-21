"""Build the local CPU forest cascade from reviewable C source; no downloads."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", default="clang")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = root / "edge_runtime/native/factory_cascade.c"
    folder = root / "models/runtime"
    folder.mkdir(parents=True, exist_ok=True)
    compiler = shutil.which(args.compiler)
    if not compiler:
        raise SystemExit("Clang not found. The Python/C scikit-learn threshold cascade remains available.")
    if platform.system() == "Windows":
        artifact = folder / "factory_cascade.dll"
        command = [compiler, "--target=x86_64-pc-windows-msvc", "-O3", "-shared", "-nostdlib",
                   "-fno-stack-protector", "-fuse-ld=lld", "-Wl,/noentry", str(source), "-o", str(artifact)]
    else:
        artifact = folder / "factory_cascade.so"
        command = [compiler, "-O3", "-shared", "-fPIC", str(source), "-o", str(artifact)]
    subprocess.run(command, check=True)
    metadata = {"abi": "factory_cascade_v1", "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                "platform": platform.platform(), "compiler": compiler, "command": command,
                "scope": "Locally built trusted code. No model parameters, training, downloads or inference network calls."}
    artifact.with_suffix(artifact.suffix + ".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"artifact": str(artifact), **metadata}, indent=2))


if __name__ == "__main__":
    main()
