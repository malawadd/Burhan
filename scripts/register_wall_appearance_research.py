"""Expose the measured failed wall-photo model and explicit appearance-only scope."""
import hashlib
import json
from pathlib import Path
import pprint
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'output/evaluation/construction-wall-five-class-v3'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    from edge_runtime.profiles.construction_model import DAMAGE_VARIANTS
    proposal = read(OUT / 'integration-proposal.json')
    if proposal['all_minimum_review_guards_pass'] or not proposal['all_actual_serving_parity']:
        raise ValueError('This registration describes the completed failed candidate only')
    if sha(OUT / 'executed-source-archive/manifest.json') != proposal['source_archive_manifest_sha256']:
        raise ValueError('Exact executed-source archive required')
    variant = 'mbdd_wall_total16_failed'
    spec = {**proposal['model_variants'][variant], 'experimental': True,
            'diagnostic_label': 'FAILED WALL PHOTO RESEARCH'}
    if sha(ROOT / 'models/construction' / spec['file']) != spec['sha256']:
        raise ValueError('Model changed')
    source_path = ROOT / 'edge_runtime/profiles/construction_model.py'
    catalog_path, presets_path = ROOT / 'config/construction-capabilities.json', ROOT / 'config/public-presets.json'
    source = source_path.read_text(encoding='utf-8')
    if variant in DAMAGE_VARIANTS:
        raise ValueError('Preserve the first wall-model registration')
    marker = 'DAMAGE_VARIANTS = {\n'
    if source.count(marker) != 1:
        raise ValueError('Unexpected registry structure')
    source = source.replace(marker, 'DAMAGE_VARIANTS = {\n    ' + repr(variant) + ': '
                            + pprint.pformat(spec, width=110).replace('\n', '\n    ') + ',\n', 1)
    compile(source, str(source_path), 'exec')
    catalog, presets = read(catalog_path), read(presets_path)
    manifest_base = ROOT / 'datasets/construction/finishes-expansion/mbdd2025-wall-photos'
    manifest = read(manifest_base / 'asset-manifest.json')
    names = {'crack': 'Wall surface crack', 'leakage': 'Wall damp/stain appearance',
             'abscission': 'Detached surface material', 'corrosion': 'Wall corrosion appearance',
             'bulge': 'Wall bulge appearance'}
    details = {
        'leakage': ('Visible dark, damp or water-stained wall appearance.',
                    'Use a suitable moisture instrument and inspect plumbing, rain exposure and the actual water source.'),
        'abscission': ('Visible detached or missing surface material; paint, plaster and concrete are not distinguished.',
                       'Confirm the actual coating/substrate and inspect loose edges and adhesion before choosing repair.'),
        'corrosion': ('Visible corrosion-like appearance; the material and mechanism remain unconfirmed.',
                      'Identify the affected material, inspect the source and measure actual section loss separately.'),
        'bulge': ('Visible bulge-like surface appearance; no depth or flatness measurement.',
                  'Inspect the actual surface and measure geometry with a straightedge or appropriate depth instrument.')}
    for label in spec['classes']:
        record = next(v for v in manifest['records'] if any(b['label'] == label for b in v['boxes']))
        image = manifest_base / record['image']
        if sha(image) != record['image_sha256']:
            raise ValueError('Fixed source photo changed')
        comparison = OUT / (Path(record['image']).stem + '-raw-publisher-model.jpg')
        if not comparison.is_file():
            raise ValueError('Measured comparison photo missing')
        preset_label = 'Wall PHOTO research - ' + names[label] + ' - first acquired source'
        preset = {'label': preset_label, 'profile': 'construction', 'path': str(image.relative_to(ROOT)),
            'input_fps': 2, 'deadline_ms': 1500, 'preview_fps': 4, 'purpose': 'photo_evidence',
            'media_type': 'still_photo', 'source_sha256': sha(image),
            'comparison_image': str(comparison.relative_to(ROOT)), 'independent_accuracy_credit': False,
            'profile_options': {'mode': 'damage', 'damage_variant': variant, 'damage_classes': [label],
                'engine': 'openvino', 'openvino_device': 'GPU', 'cpu_threads': 1, 'stage': 'finishes',
                'exposure_policy': 'off', 'show_context_objects': False, 'confirm_frames': 1},
            'note': 'Real still PHOTO, first acquired class-positive source, all18 measured photos retained. '
                    'All five accuracy guards fail. No humidity measurement, paint-only, installed-tile or video qualification.'}
        if any(v['label'] == preset_label for v in presets):
            raise ValueError('Refusing to replace a photo preset')
        presets.append(preset)
        if label == 'crack':
            continue  # Existing crack category keeps its own model and measured identity.
        metric, diagnostic = proposal['calibration'][label], proposal['fixed18_diagnostics'][label]
        category = {'id': spec['class_capabilities'][label], 'label': names[label],
            'model_status': 'research_only', 'model_ref': 'damage:' + variant,
            'condition': details[label][0], 'next_check': details[label][1],
            'why_it_matters': 'A model appearance candidate needs direct inspection; it does not establish the cause or severity.',
            'validation_scope': (f"FAILED photo research:48 CAL, {metric['tp']}/{metric['annotations']} box matches "
                f"and{metric['fp']} unmatched predictions;18 previously exposed photos, "
                f"{diagnostic['tp']}/{diagnostic['annotations']} matches and{diagnostic['fp']} unmatched. "
                'IoU>=.50; all class minimum guards fail. No certified healthy negatives, independent site or video qualification.'),
            'metrics': {'calibration': metric, 'exposed18': diagnostic}, 'video_evidence': [],
            'photo_evidence': [{'case_id': preset_label, 'preset_label': preset_label,
                                'comparison_image': str(comparison.relative_to(ROOT))}]}
        if any(v['id'] == category['id'] for v in catalog['categories']):
            raise ValueError('Refusing to overwrite an existing capability')
        catalog['categories'].append(category)
    receipt = OUT / 'registration.json'
    if receipt.exists():
        raise ValueError('Preserve prior registration')
    for path in [source_path, catalog_path, presets_path]:
        with (OUT / ('before-registration-' + path.name)).open('xb') as stream:
            stream.write(path.read_bytes())
    source_path.write_text(source, encoding='utf-8')
    for path, value in [(catalog_path, catalog), (presets_path, presets)]:
        path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    with receipt.open('x', encoding='utf-8') as stream:
        json.dump({'registered_variant': variant, 'research_only': True, 'new_appearance_categories': 4,
                   'fixed_photo_presets': 5, 'defaults_changed': False, 'source_sha256': sha(source_path),
                   'proposal_sha256': sha(OUT / 'integration-proposal.json')}, stream, indent=2)
        stream.write('\n')
    print('Registered failed five-class wall research, four appearance-only categories and five real-photo presets.')


if __name__ == '__main__':
    main()
