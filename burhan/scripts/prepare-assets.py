"""Copy pinned local models and generate an independent, auditable app catalogue."""
import hashlib, json, shutil, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / 'burhan'
sys.path.insert(0, str(ROOT))
from edge_runtime.profiles.construction_model import MODELS, CRACK_VARIANTS, DAMAGE_VARIANTS
from edge_runtime.construction_capabilities import load_capabilities

def main():
    destination = APP / 'assets/models'
    destination.mkdir(parents=True, exist_ok=True)
    models = []
    for name, mode, spec, size, kind in [
        ('ppe','ppe',MODELS['ppe'],640,'yolo_detection'),
        ('crack','crack',CRACK_VARIANTS['joint_aware_candidate'],640,'yolo_segmentation'),
        ('surface320','surface',DAMAGE_VARIANTS['dacl_segformer_b1_320'],320,'dacl'),
        ('surface512','surface',DAMAGE_VARIANTS['dacl_segformer_b1'],512,'dacl')]:
        source=ROOT/'models/construction'/spec['file']
        digest=hashlib.sha256(source.read_bytes()).hexdigest()
        if digest != spec['sha256']: raise ValueError(f'Checksum mismatch: {source}')
        output=destination/f'{name}.onnx'
        shutil.copyfile(source, output)
        models.append(dict(id=name,mode=mode,file=output.name,sha256=digest,bytes=source.stat().st_size,
            inputSize=size,kind=kind,classes=spec['classes'],preprocessing='bgr_uint8_stretch' if kind=='dacl' else 'rgb_float_letterbox',
            confidence=spec.get('confidence',.25),nms=.45,source=spec.get('source',''),
            scope=spec.get('validation_scope','Experimental observations; misses and false alerts documented. No site qualification.'),
            license=spec.get('license','Ultralytics AGPL-3.0 lineage. Joint model merged-source clearance unresolved.' if name=='crack' else 'Ultralytics AGPL-3.0 lineage; author declares CC BY 4.0 weights.')))
    (destination/'manifest.json').write_text(json.dumps(models,indent=2),encoding='utf-8')
    catalog=json.loads((ROOT/'edge_runtime/data/construction-inspections.json').read_text(encoding='utf-8'))
    capabilities=load_capabilities()
    categories=[dict(id=r['id'],label=r['label'],status=r['effective_status'],scope=r['validation_scope']) for r in capabilities['categories']]
    by_id={r['id']:r for r in categories}
    tasks=[]
    for r in catalog['inspections']:
        cap=r.get('capability_id'); mode='manual'
        if cap=='surface_crack': mode='crack'
        elif cap and cap in by_id and by_id[cap]['status'] in ('supported_experimental','measured_weak') and cap.startswith('surface_'): mode='surface'
        elif cap and cap.startswith('ppe_'): mode='ppe'
        tasks.append(dict(id=r['id'],title=r['title'],stage=r['stage'],mode=mode,capabilityId=cap,automatic=mode!='manual',requiredEvidence=r.get('required_evidence',[])))
    # Explicit camera modes remain accessible alongside the 89 construction topics.
    for name,title,stage,mode,cap in [
        ('visual-cracks','Visible cracks / joints','structure','crack','surface_crack'),
        ('visual-surfaces','Concrete and visible surface conditions','structure','surface',None),
        ('visual-ppe','PPE and site observations','site_safety','ppe','ppe_hardhat')]:
        tasks.insert(0,dict(id=name,title=title,stage=stage,mode=mode,capabilityId=cap,automatic=True,requiredEvidence=['Original camera evidence','Inspector review and notes']))
    (APP/'assets/catalogue.json').write_text(json.dumps(dict(stages=catalog['stages'],tasks=tasks,categories=categories),ensure_ascii=False,indent=2),encoding='utf-8')
    (APP/'assets/source-catalogue.json').write_text(json.dumps(catalog,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'Prepared {len(models)} pinned models, {len(tasks)} tasks including 89 original inspection topics.')
if __name__=='__main__': main()
