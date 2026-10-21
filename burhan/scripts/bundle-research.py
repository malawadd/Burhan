"""Pin the selected local research weights and build the shared 30-check catalogue.

Run from any directory; no downloads, training or threshold selection occur.
"""
import ast
import hashlib
import json
from pathlib import Path
import shutil

APP = Path(__file__).resolve().parents[1]
ROOT = APP.parent
ASSETS = APP / 'assets'
source = ROOT / 'edge_runtime/profiles/construction_tasks.py'
constants = {}
for node in ast.parse(source.read_text(encoding='utf-8')).body:
    if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in ('TASK_CATALOG', '_CATEGORY_REASONS', '_LABELS'):
        constants[node.targets[0].id] = ast.literal_eval(node.value)

selected = {
    'research-phele': 'expansion-phele-v1/ssdlite320',
    'research-fasteners': 'expansion-sdnet-v2-performance/yolov8n',
    'research-stain': 'expansion-wall-stain-v1/ssdlite320',
}
manifest_path = ASSETS / 'models/manifest.json'
models = [m for m in json.loads(manifest_path.read_text(encoding='utf-8')) if not m['id'].startswith('research-')]
binding = {}
for model_id, directory in selected.items():
    src = ROOT / 'models/construction' / directory / 'manifest.json'
    spec = json.loads(src.read_text(encoding='utf-8'))
    original = src.parent / spec['model_file']
    raw = original.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == spec['model_sha256'], model_id
    dest = ASSETS / 'models' / (model_id + '.onnx')
    if not dest.exists() or dest.read_bytes() != raw:
        shutil.copyfile(original, dest)
    thresholds = {label: spec.get('class_confidence', {}).get(label, spec['confidence']) for label in spec['classes']}
    models.append(dict(id=model_id, mode='research', file=dest.name, sha256=spec['model_sha256'], bytes=len(raw),
                       inputSize=spec['input_size'], kind='ssdlite_detection' if spec['architecture']=='ssdlite320' else 'yolo_detection',
                       classes=spec['classes'], preprocessing='rgb_float_letterbox', confidence=spec['confidence'], classConfidence=thresholds,
                       nms=.45, researchOnly=True, family=spec['inspection_task'], source=spec['source'], license=spec['license'],
                       sourceManifest=src.relative_to(ROOT).as_posix(), sourceManifestSha256=hashlib.sha256(src.read_bytes()).hexdigest(),
                       scope=spec.get('validation_summary',spec.get('release_blocker','Experimental appearance candidates; independent site/video qualification has not passed.'))))
    for label in spec['classes']:
        binding[label] = model_id

# Display copy describes observations, never a new trained capability.
labels = {
 'damp_stained_appearance': ('Damp / stained appearance','مظهر بقع أو رطوبة محتملة'),
 'mold_appearance': ('Mold-like appearance','مظهر شبيه بالعفن'),
 'peeling_paint': ('Peeling paint','تقشر الطلاء'),
 'paint_blister': ('Paint blister','فقاعات الطلاء'),
 'efflorescence': ('Salt deposits / efflorescence','ترسبات الأملاح'),
 'crack': ('Surface cracks','شقوق السطح'),
 'spalling': ('Surface material loss','فقدان مادة السطح'),
 'paint_run': ('Paint runs','سيلان الطلاء'),
 'patchy_coverage': ('Patchy coating','تفاوت تغطية الطلاء'),
 'broken_sealant': ('Broken sealant','تلف مادة سد الفواصل'),
 'missing_sealant': ('Missing sealant','غياب مادة سد الفواصل'),
 'uneven_door_window_reveal': ('Door / window reveal','انتظام حواف الأبواب والنوافذ'),
 'missing_bolt': ('Expected bolt','وجود المسمار المطلوب'),
 'missing_nut': ('Expected nut','وجود الصامولة المطلوبة'),
 'fastener_absence_appearance': ('Fastener-absence appearance','مظهر غياب مثبت'),
 'visibly_unseated_connection': ('Visibly unseated bolt / nut','مظهر مسمار أو صامولة غير مستقر'),
 'corrosion_appearance': ('Connection corrosion appearance','مظهر تآكل الوصلة'),
 'joint_width': ('Joint width','عرض الفاصل'),
 'joint_too_wide': ('Joint wider than specified','فاصل أعرض من المحدد'),
 'joint_too_narrow': ('Joint narrower than specified','فاصل أضيق من المحدد'),
 'inconsistent_spacing': ('Joint spacing consistency','اتساق تباعد الفواصل'),
 'joint_alignment': ('Joint line alignment','استقامة خط الفاصل'),
 'row_misalignment': ('Tile row alignment','محاذاة صفوف البلاط'),
 'missing_grout': ('Grout presence','وجود حشو الفواصل'),
 'cracked_tile': ('Installed tile cracks','شقوق البلاط المركب'),
 'chipped_edge': ('Tile edge damage','تلف حافة البلاط'),
 'broken_corner': ('Tile corner damage','تلف زاوية البلاط'),
 'missing_tile': ('Expected tile presence','وجود البلاطة المطلوبة'),
 'open_perimeter_joint': ('Perimeter joint detail','تفصيل الفاصل المحيطي'),
 'misaligned_skirting': ('Skirting alignment','محاذاة الوزرة'),
}
families = [dict(id=f,title=d['title'],titleAr={'walls_finishes':'الجدران والتشطيبات','brackets_railings':'الحوامل والدرابزين','tile_installation':'تركيب البلاط'}[f]) for f,d in constants['TASK_CATALOG'].items()]
tasks=[]
for family, definition in constants['TASK_CATALOG'].items():
    for category in definition['categories']:
        title, title_ar = labels[category]
        model_id = binding.get(category)
        if family=='tile_installation':
            guidance='Photograph the joint and surrounding tiles. Record the specified width/tolerance and a physical measurement; judge against the approved installation detail.'
            guidance_ar='صوّر الفاصل والبلاط المحيط. سجّل العرض والسماحية المحددين وقياساً فعلياً، ثم راجع التفصيل المعتمد.'
            required=['overview photo','close-up evidence','approved specification / layout','physical measurements and instrument']
        elif family=='brackets_railings':
            guidance='Show the whole connection, then its visible bolt and nut seats. Compare with the approved expected layout; appearance does not measure torque or anchorage.'
            guidance_ar='أظهر الوصلة كاملة ثم مواضع المسامير والصواميل المرئية. قارن بالمخطط المعتمد؛ المظهر لا يقيس العزم أو كفاءة التثبيت.'
            required=['connection overview','close-up evidence','expected fastening layout','inspector assessment']
        else:
            guidance='Show a sharp, evenly lit close-up and an intact neighbouring surface. Record the visible appearance and confirm it by inspection.'
            guidance_ar='أظهر لقطة قريبة واضحة بإضاءة متجانسة مع جزء سليم مجاور. وثّق المظهر المرئي وتحقق منه بالفحص.'
            required=['surface overview','close-up evidence','inspector assessment']
        task=dict(id=f'research:{family}:{category}',family=family,categoryId=category,title=title,titleAr=title_ar,
                  stage=f'research:{family}',mode='research' if model_id else 'manual',automatic=bool(model_id),
                  capabilityId=category,researchOnly=True,scope=constants['_CATEGORY_REASONS'][category],
                  guidance=guidance,guidanceAr=guidance_ar,requiredEvidence=required)
        if model_id:task['modelId']=model_id
        tasks.append(task)
assert len(tasks)==30 and len({t['id'] for t in tasks})==30
manifest_path.write_text(json.dumps(models,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
(ASSETS/'research-catalogue.json').write_text(json.dumps(dict(schemaVersion=1,families=families,tasks=tasks,releasedCategories=[]),indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(json.dumps(dict(models=len(models),expandedChecks=len(tasks),modelBacked=sum(t['automatic'] for t in tasks),addedBytes=sum(m['bytes'] for m in models if m.get('researchOnly')))))
