"""Create an exact printable ArUco reference; this is not a validation sample."""
from pathlib import Path
import hashlib
import json

import cv2


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets/construction-reference"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    # Six cells: four payload cells and one black border cell on each side.
    cells = cv2.aruco.generateImageMarker(dictionary, 0, 6)
    cell_mm, quiet_mm = 50/6, 10
    paths = [f'<svg xmlns="http://www.w3.org/2000/svg" width="70mm" height="70mm" viewBox="0 0 70 70">',
             '<rect width="70" height="70" fill="white"/>']
    for y in range(6):
        for x in range(6):
            if cells[y, x] == 0:
                paths.append(f'<rect x="{quiet_mm+x*cell_mm:.9f}" y="{quiet_mm+y*cell_mm:.9f}" width="{cell_mm:.9f}" height="{cell_mm:.9f}" fill="black"/>')
    paths.append('</svg>')
    svg = OUT / "aruco-4x4-50-id0-50mm.svg"
    svg.write_text("\n".join(paths)+"\n", encoding="utf-8")
    preview = cv2.copyMakeBorder(cv2.aruco.generateImageMarker(dictionary, 0, 600), 120, 120, 120, 120,
                                 cv2.BORDER_CONSTANT, value=255)
    png = OUT / "aruco-id0-preview.png"
    if not cv2.imwrite(str(png), preview):
        raise RuntimeError("Cannot save reference preview")
    _, ids, _ = cv2.aruco.ArucoDetector(dictionary).detectMarkers(preview)
    if ids is None or ids.flatten().tolist() != [0]:
        raise RuntimeError("Reference did not decode to its specified ID")
    receipt = {"dictionary": "DICT_4X4_50", "id": 0, "black_outer_edge_mm": 50, "quiet_border_mm": 10,
               "svg_page_mm": [70, 70], "software_decode": [0],
               "scope": "Generated calibration target only; no real installation, camera calibration or measurement accuracy evidence.",
               "print": "Print SVG at actual size / 100%, no fit-to-page. Independently measure the black outer square and enter its actual size.",
               "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (svg, png)},
               "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT / "reference.json").write_text(json.dumps(receipt, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
