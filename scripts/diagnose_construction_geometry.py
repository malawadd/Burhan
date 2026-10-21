"""Real installed-floor photo geometry diagnostics; no training or fault accuracy."""
import argparse
import hashlib
import json
import shutil
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'output/evaluation/construction-resumption-geometry'
DATA = ROOT / 'datasets/construction/installed-floor-diagnostics-v1'
TITLES = ['File:Cracked tile flooring.jpg', 'File:Tiled Floor.jpg']


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf8')


def request(url, limit):
    req = urllib.request.Request(url, headers={'User-Agent': 'InspectFlow-public-research/1.0'})
    with urllib.request.urlopen(req, timeout=50) as response:
        data = response.read(limit + 1)
    if len(data) > limit: raise ValueError('Bounded public response exceeded')
    return data


def acquire():
    DATA.mkdir(parents=True, exist_ok=True)
    metadata = DATA / 'publisher-imageinfo.json'
    if not metadata.exists():
        url = 'https://commons.wikimedia.org/w/api.php?' + urllib.parse.urlencode({
            'action': 'query', 'titles': '|'.join(TITLES), 'prop': 'imageinfo',
            'iiprop': 'url|size|sha1|extmetadata', 'format': 'json'})
        metadata.write_bytes(request(url, 500_000))
    rows = []
    for page in json.loads(metadata.read_text())['query']['pages'].values():
        item = page['imageinfo'][0]
        license = item['extmetadata']['LicenseShortName']['value']
        if license not in ['CC BY-SA 3.0', 'CC BY-SA 4.0']:
            raise ValueError('Re-audit publisher rights')
        filename = page['title'].removeprefix('File:').replace(' ', '_')
        target = DATA / filename
        if not target.exists(): target.write_bytes(request(item['url'], 8_000_000))
        raw = target.read_bytes()
        if len(raw) != item['size'] or hashlib.sha1(raw).hexdigest() != item['sha1']:
            raise ValueError('Publisher image size/SHA1 mismatch')
        rows.append({'title': page['title'], 'path': target.relative_to(ROOT).as_posix(),
                     'source': item['descriptionurl'], 'license': license,
                     'license_url': item['extmetadata']['LicenseUrl']['value'],
                     'author_html': item['extmetadata']['Artist']['value'],
                     'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw),
                     'ground_truth': 'No pixel/box labels or measured joint widths; diagnostic photograph only'})
    save(DATA / 'asset-manifest.json', rows)
    print(json.dumps(rows, indent=2))


def diagnostic(stage):
    import cv2
    import numpy as np
    from edge_runtime.profiles.construction_tasks import GuidedConstructionProfile
    from edge_runtime.contracts import FrameContext
    cv2.setNumThreads(1)
    destination = OUT / stage
    if destination.exists(): raise ValueError('Diagnostic stage already frozen; use a new version name')
    destination.mkdir(parents=True)
    source = ROOT / 'edge_runtime/profiles/construction_tasks.py'
    shutil.copyfile(source, destination / 'executed-construction_tasks.py')
    save(destination / 'protocol-before-pixels.json', {
        'stage': stage, 'code_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'analysis_max_dimensions': [640, 1280], 'threads': 1,
        'no_owner_footage': True, 'no_training': True, 'no_threshold_selection': True,
        'scope': 'Actual public photographs, repeated for geometry timing only; not video or detection accuracy'})
    results = []
    for asset in json.loads((DATA / 'asset-manifest.json').read_text()):
        original = cv2.imread(str(ROOT / asset['path']))
        for maximum in [640, 1280]:
            factor = maximum / max(original.shape[:2])
            image = cv2.resize(original, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
            durations = []
            failure = None
            for _ in range(12):
                started = time.perf_counter()
                try:
                    observations = GuidedConstructionProfile._joint_edges(image)
                except Exception as error:
                    failure = {'type': type(error).__name__, 'message': str(error)}
                    observations = []
                    break
                durations.append((time.perf_counter() - started) * 1000)
            annotated = image.copy()
            for index, row in enumerate(observations):
                poly = np.array(row['polygon'], np.int32)
                cv2.polylines(annotated, [poly], True, (0, 220, 255), 1)
                center = tuple(np.rint(row['center']).astype(int))
                cv2.putText(annotated, f"{index+1}: {row['width_px']:.1f}px", center,
                            cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 220, 255), 1)
            header = np.zeros((55, image.shape[1], 3), np.uint8)
            cv2.putText(header, 'PUBLIC PHOTO - edge-band research; no fault/scale ground truth', (5, 17), cv2.FONT_HERSHEY_SIMPLEX, .42, (255,255,255), 1)
            credit = 'Malcolm Koo (Mk2010) CC BY-SA 4.0' if 'Cracked' in asset['title'] else 'Deavmi CC BY-SA 3.0'
            cv2.putText(header, credit+' | resized, geometry overlay added', (5, 37), cv2.FONT_HERSHEY_SIMPLEX, .42, (255,255,255), 1)
            name = Path(asset['path']).stem + f'-{maximum}.jpg'
            cv2.imwrite(str(destination / name), np.vstack([header, annotated]), [cv2.IMWRITE_JPEG_QUALITY,92])
            profile = GuidedConstructionProfile({'inspection_task': 'tile_installation'})
            processed = profile.process(image, FrameContext(0, time.perf_counter(), asset['path'], 'public-photo-diagnostic', None))
            cv2.imwrite(str(destination / (Path(name).stem+'-actual-profile.jpg')), np.vstack([header, processed.annotated_frame]), [cv2.IMWRITE_JPEG_QUALITY,92])
            results.append({'asset': asset, 'analysis_dimensions': [image.shape[1],image.shape[0]],
                            'observations': observations, 'failure': failure,
                            'p50_geometry_ms': float(np.median(durations[2:])) if len(durations)>2 else None,
                            'p95_geometry_ms': float(np.quantile(durations[2:], .95)) if len(durations)>2 else None,
                            'timing_scope': 'same real still repeated, geometry only, first two warmups omitted',
                            'actual_profile': {'status': processed.status, 'metrics': processed.metrics,
                                               'findings': [finding.to_dict() if hasattr(finding,'to_dict') else vars(finding) for finding in processed.findings],
                                               'domain_state': processed.domain_state},
                            'overlay': (destination / name).relative_to(ROOT).as_posix()})
    save(destination / 'results.json', results)
    print(json.dumps([{'photo': r['asset']['title'], 'dimensions': r['analysis_dimensions'],
                      'pairs':len(r['observations']), 'failure':r['failure'], 'p50_ms':r['p50_geometry_ms']} for r in results], indent=2))


def defect_probe(stage):
    import cv2
    import numpy as np
    from edge_runtime.profiles.construction_model import ConstructionModel
    from edge_runtime.profiles.construction_round4_cpu_asset import VARIANT, SPEC
    cv2.setNumThreads(1)
    destination = OUT / stage
    if destination.exists(): raise ValueError('Probe stage already frozen')
    destination.mkdir(parents=True)
    save(destination / 'protocol-before-inference.json', {
        'variant': VARIANT, 'spec': SPEC, 'options': {'mode':'damage','damage_variant':VARIANT,'engine':'onnxruntime','openvino_device':'CPU','cpu_threads':1},
        'scope':'One fixed inference per real installed-floor photograph; original crack/spalling source classes only',
        'threshold_policy':'Existing source class thresholds unchanged; no fitting or selection on these photographs',
        'no_owner_footage':True,'accuracy_metrics':'unavailable: no masks/boxes or independent scene evaluation'})
    model = ConstructionModel({'mode':'damage','damage_variant':VARIANT,'engine':'onnxruntime','openvino_device':'CPU','cpu_threads':1})
    records = []
    for asset in json.loads((DATA / 'asset-manifest.json').read_text()):
        image = cv2.imread(str(ROOT / asset['path']))
        factor = 640 / max(image.shape[:2])
        image = cv2.resize(image, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
        annotated = image.copy()
        started = time.perf_counter()
        detections = model.infer(image)
        elapsed = (time.perf_counter()-started)*1000
        rows = []
        for index, detection in enumerate(detections):
            row = {}
            for name, value in detection.items():
                if isinstance(value, np.ndarray) and value.dtype == np.bool_:
                    path = destination / (Path(asset['path']).stem+f'-mask-{index}.png')
                    cv2.imwrite(str(path), value.astype(np.uint8)*255)
                    row[name+'_area_px'] = int(value.sum())
                    row[name+'_file'] = path.relative_to(ROOT).as_posix()
                    if value.shape == image.shape[:2]:
                        annotated[value] = (.6*annotated[value]+.4*np.array([40,80,250])).astype(np.uint8)
                else:
                    row[name] = value.tolist() if isinstance(value,np.ndarray) else value.item() if isinstance(value,np.generic) else value
            x,y,w,h = map(int,detection['bbox'])
            cv2.rectangle(annotated,(x,y),(x+w,y+h),(40,80,250),1)
            cv2.putText(annotated,str(detection['label']), (max(2,x),max(30,y)),cv2.FONT_HERSHEY_SIMPLEX,.45,(40,80,250),1)
            rows.append(row)
        header = np.zeros((60,image.shape[1],3),np.uint8)
        cv2.putText(header,'FIXED crack/spalling model - unscored installed-floor transfer', (5,17),cv2.FONT_HERSHEY_SIMPLEX,.44,(255,255,255),1)
        credit = 'Malcolm Koo (Mk2010) CC BY-SA 4.0' if 'Cracked' in asset['title'] else 'Deavmi CC BY-SA 3.0'
        cv2.putText(header,credit+' | resized; predictions added', (5,38),cv2.FONT_HERSHEY_SIMPLEX,.42,(255,255,255),1)
        path = destination / (Path(asset['path']).stem+'-fixed-defect-probe.jpg')
        cv2.imwrite(str(path),np.vstack([header,annotated]),[cv2.IMWRITE_JPEG_QUALITY,92])
        records.append({'asset':asset,'detections':rows,'single_call_inference_ms_including_lazy_initialization':elapsed,
                        'overlay':path.relative_to(ROOT).as_posix()})
    save(destination/'results.json',{'model_provenance':model.describe(),'records':records})
    print(json.dumps([{'photo':r['asset']['title'],'detections':len(r['detections'])} for r in records]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', help='acquire or a unique frozen diagnostic stage name')
    args = parser.parse_args()
    if args.stage == 'acquire': acquire()
    elif args.stage.startswith('defect-probe-'): defect_probe(args.stage)
    else: diagnostic(args.stage)
