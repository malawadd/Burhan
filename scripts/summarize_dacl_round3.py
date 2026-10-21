"""Refresh Markdown and capability evidence from completed DACL reports; no inference."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'output/evaluation/construction-dacl19-round3'
MARKER = '\n## Measured local results: fixed 512 versus 320\n'


def diagnostic(report, name):
    row = dict(report['per_class'][name])
    row.update(model_ref='damage:dacl_segformer_b1', input_size=512,
               matching_iou_threshold=.5,
               scope='55 fixed public-validation photos; checkpoint training/selection overlap unknown; not video or blind test')
    return row


def main():
    reports = {size: json.loads((BASE / str(size) / 'report.json').read_text(encoding='utf-8')) for size in [512, 320]}
    speeds = {size: json.loads((BASE / f'{size}-speed.json').read_text(encoding='utf-8')) for size in [512, 320]}
    ledger_path = ROOT / 'config/construction-capabilities.json'
    ledger = json.loads(ledger_path.read_text(encoding='utf-8'))
    for category in ledger['categories']:
        name = category['id'].removeprefix('surface_')
        if name not in reports[512]['per_class']:
            continue
        row = reports[512]['per_class'][name]
        gt = row['derived_region_tp_iou_0_5'] + row['derived_region_fn_iou_0_5']
        summary = (f"DACL512 photo diagnostic: pixel precision {row['pixel_precision']:.1%}, recall {row['pixel_recall']:.1%}, "
                   f"IoU {row['pixel_iou']:.1%}; derived-region matches {row['derived_region_tp_iou_0_5']}/{gt} at IoU0.50, "
                   f"{row['derived_region_fp_iou_0_5']} unmatched model regions; "
                   f"class-absent photo flags {row['negative_images_flagged']}/{row['negative_images']}. "
                   "55 fixed public-validation stills; checkpoint overlap unknown. No category-specific video validation.")
        category['diagnostic_models'] = [diagnostic(reports[512], name)]
        if name in {'crack', 'spalling'}:
            category['evidence_summary'] = ("Separate DACL broad-model diagnostic, not credited to this category's video-model identity. " + summary)
            continue
        category['model_status'] = 'supported_experimental' if name == 'hollowareas' else 'measured_weak'
        category['metrics'] = diagnostic(reports[512], name)
        category['validation_scope'] = summary
        category['evidence_summary'] = summary + (
            ' Detects inspector chalk boundaries after sounding, not hidden hollow concrete.' if name == 'hollowareas' else
            ' Weak localization: available for human-reviewed experiments, not dependable inspection.')
    ledger_path.write_text(json.dumps(ledger, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    assets = json.loads((ROOT / 'datasets/construction/dacl10k/round3-asset-manifest.json').read_text(encoding='utf-8'))
    assets_by_name = {row['source_name']: row for row in assets['records'] if row['partition'] == 'damage_diagnostic'}
    gallery = []
    for name in reports[512]['per_class']:
        first = next(row for row in reports[512]['records'] if row['classes'][name]['pixel_target'] > 0)
        asset = assets_by_name[first['source_name']]
        gallery.append({'case_id': f'dacl-first-{name}', 'class': name, 'capability_id': 'surface_' + name,
                        'source_name': first['source_name'],
                        'image': 'datasets/construction/dacl10k/' + asset['image'],
                        'annotation': 'datasets/construction/dacl10k/' + asset['annotation'],
                        'source_sha256': first['source_sha256'],
                        'comparison_image': f'output/evaluation/construction-dacl19-round3/512/{name}-raw-annotation-model.jpg',
                        'model_ref': 'damage:dacl_segformer_b1', 'input_size': 512,
                        'selection': 'First GT-positive image for this class in frozen55-photo ordering; not selected for model performance.',
                        'media_type': 'still_photo', 'localization': first['classes'][name],
                        'scope': 'Original real public-validation photo; author checkpoint overlap unknown; no video evidence.'})
    (BASE / 'gallery-manifest.json').write_text(json.dumps({'cases': gallery}, indent=2) + '\n', encoding='utf-8')
    gallery_lines = ['# Real construction conditions: original photo, annotation and local model', '',
                     '20 October 2026. These are 14 real DACL10k inspection photographs, not videos. Each image has separate **RAW CAMERA PHOTO / PUBLISHER ANNOTATION / ACTUAL LOCAL MODEL MASK** panels. Orange in the annotation panel is the publisher label; orange in the model panel comes from actual local DACL512 inference. Their shapes can differ or be absent. No damage was generated or added to the source photos.', '',
                     'For each label, the example is the first GT-positive photo in the previously frozen55-photo inventory. It was not selected for a favorable model result. Failures and partial masks remain visible. Pixel precision/recall/IoU below concern that named class on that single original-resolution photo. Region matches use one-to-one connected components at IoU0.50, not native instance labels. The model uses fixed probability>0.5 and at least four native-grid pixels per component.', '',
                     'The public-validation photos may overlap checkpoint training/selection; bridge-disjoint evaluation is not established. These examples are not construction acceptance, calibrated severity, hidden-defect detection or video-validation evidence. Full55-photo results and source/model licensing are in [the measured report](<' + (ROOT / 'docs/construction-category-expansion-round3.md').as_posix() + '>). Data: original-author CC BY-NC4.0; checkpoint license separately undeclared.', '']
    for case in gallery:
        name, row = case['class'], case['localization']
        title = {'hollowareas': 'Inspector chalk boundary after sounding', 'washout': 'Concrete surface loss / erosion appearance'}.get(name, name.replace('_', ' ').capitalize())
        iou = 'undefined' if row['iou'] is None else f"{row['iou']:.1%}"
        pr = 'undefined' if row['pixel_precision'] is None else f"{row['pixel_precision']:.1%}"
        re = 'undefined' if row['pixel_recall'] is None else f"{row['pixel_recall']:.1%}"
        gallery_lines += [f'## {title}', '',
                          f"Original `{case['source_name']}`. Pixel precision **{pr}**, recall **{re}**, IoU **{iou}**; derived-region matches **{row['region_matches_at_iou']['0.5']}/{row['gt_regions']}** at IoU0.50; predicted regions **{row['predicted_regions']}**. " + ('At least half the labeled pixels were covered.' if row['localized_50pct_coverage'] else '**Less than half the labeled pixels were covered.**'), '']
        if name == 'hollowareas':
            gallery_lines += ['The target is an inspector\'s visible chalk/crayon outline. The model does not see or prove an internal void.', '']
        if name == 'washout':
            gallery_lines += ['The label groups several surface-loss appearances. It does not determine the physical cause or prove chemical corrosion.', '']
        gallery_lines += [f"[Original photo](<{(ROOT / case['image']).as_posix()}>) · [Publisher polygons](<{(ROOT / case['annotation']).as_posix()}>)", '',
                          f"![{title}: raw camera photo, publisher annotation, actual local model mask](<{(ROOT / case['comparison_image']).as_posix()}>)", '']
    (ROOT / 'docs/construction-dacl-gallery-round3.md').write_text('\n'.join(gallery_lines), encoding='utf-8')

    lines = [MARKER,
             'Both ONNX exports and all 110 photo evaluations completed locally at the predeclared strict probability `>0.5`, four-native-grid-pixel component minimum, 20 components per class and 100 observations total. No detector threshold, weight, class or example selection was changed after seeing these results. The 320 graph uses the same learned state at a smaller geometry; it was not retrained for that geometry. The four-grid-pixel minimum covers a larger source area at 320, so this is the complete configured-pipeline tradeoff, not a resolution-only ablation.\n',
             '**The broad model remains weak on most categories.** Some real masks overlap meaningful defects, but a tiny overlap is not a successful localization. Results below include pixel precision/recall/IoU, one-to-one connected-region matches at IoU0.50 and class-absent-photo false flags. Chalk-boundary segmentation is the strongest measured label; that is an inspector marking rather than a diagnosis of internal damage. The weaker 320 model misses every derived crack region at IoU0.50 despite touching crack pixels in some images.\n']
    for size in [512, 320]:
        lines += [f'### {size}×{size} actual masks on the same 55 photos\n',
                  '| Label | Pixel precision | Pixel recall | Pixel IoU | Region TP / FP / FN at IoU0.50 | Positive photos ≥50% GT coverage | Class-absent photos flagged |',
                  '|---|---:|---:|---:|---:|---:|---:|']
        for name, row in reports[size]['per_class'].items():
            label = {'hollowareas': 'Inspector chalk boundaries', 'washout': 'Surface loss / erosion appearance'}.get(name, name.replace('_', ' '))
            lines.append(f"| {label} | {row['pixel_precision']:.1%} | {row['pixel_recall']:.1%} | {row['pixel_iou']:.1%} | "
                         f"{row['derived_region_tp_iou_0_5']} / {row['derived_region_fp_iou_0_5']} / {row['derived_region_fn_iou_0_5']} | "
                         f"{row['positive_images_50pct_coverage']}/{row['positive_images']} | {row['negative_images_flagged']}/{row['negative_images']} |")
        lines.append('')
    lines += ['A class-absent image can contain other genuine damage; these false flags are therefore class-confusion/false-localization diagnostics, not claims that all flagged photos are healthy. Connected semantic regions can fragment or merge, so their TP/FP/FN are explicitly derived diagnostics; pixel counts in the JSON remain the primary spatial measure. Public validation overlap with the checkpoint is unknown, and correlated bridge views cannot be ruled out. No result is an independent all-stage or Saudi construction validation.\n',
              '### Dedicated local warm adapter capacity\n',
              'Actual Intel UHD620 OpenVINO FP32, one stream, six fixed varied real photos, two passes per rendering mode after warmup; source longest edge at most 1920 pixels. Twelve observations per mode. Original JPEG reading, model compilation, camera/video decode, dashboard transport and shared-runtime scheduling are excluded. Full masks include nearest-neighbor per-component output masks at that analysis geometry. This is bounded serial adapter capacity, **not moving-video or end-to-end FPS**. Accuracy-audit timings are not used for this table.\n',
              '| Input | Output mode | Mean adapter ms | p50 ms | p95 ms | Serial adapter FPS |',
              '|---|---|---:|---:|---:|---:|']
    for size in [512, 320]:
        for mode, row in speeds[size]['modes'].items():
            values = row['adapter']
            lines.append(f"| {size} | {mode} | {values['mean']:.1f} | {values['p50']:.1f} | {values['p95']:.1f} | {row['serial_adapter_fps']:.2f} |")
    lines += ['',
              '**Neither broad-model geometry meets 15 FPS on this laptop.** The 512 model is the quality/reference photo-review variant; 320 is an explicitly weaker, faster diagnostic option. It is not substituted into the 512 accuracy claim. Factory15-FPS qualification is measured separately with the factory model. Applying this broad classifier to a video is executable, but these photos do not validate temporal stability or real-video localization for rust, moisture or other added conditions.\n',
              'Ledger statuses describe the measured evidence: newly measured labels below 0.50 pixel IoU are marked `measured_weak`; the chalk-boundary label is `supported_experimental` with its restricted photo scope. That 0.50 status boundary was chosen after measurement for descriptive review, not tuned into the model or represented as an industry acceptance rule. Existing crack and new crack/spall video-model identities retain their own evidence and metrics; supplemental DACL results are recorded separately.\n',
              '### Files and integrity\n',
              '- Primary `models/construction/dacl19/dacl-segformer-b1-512.onnx`: SHA-256 `080fb07297bbb2841fbd5f49b9d8888da53ee1c2adc1ea33bdd2addb8cc9ea7f`.',
              '- Alternate `models/construction/dacl19/dacl-segformer-b1-320.onnx`: SHA-256 `c89aa3002de737cf19ea3043704af36837ad92267466f1234ac2bcecee60ea56`.',
              '- `output/evaluation/construction-dacl19-round3/protocol-before-any-export.json` freezes method before any model forward.',
              '- `output/evaluation/construction-dacl19-round3/{512,320}/report.json` contains full pixel TP/FP/FN, image outcomes and derived matches. Each folder contains 14 first-inventory raw/annotation/model triptychs; these are photographs, not encoded video.',
              '- `output/evaluation/construction-dacl19-round3/{512,320}-speed.json` contains independent dedicated capacity observations.',
              '- `output/evaluation/construction-dacl19-round3/gallery-manifest.json` maps each class to the first annotated original photo and raw/annotation/model comparison, including that case\'s actual spatial metrics.',
              '- `scripts/summarize_dacl_round3.py` refreshes this measured section and capability evidence from completed reports without running inference.', '']
    path = ROOT / 'docs/construction-category-expansion-round3.md'
    original = path.read_text(encoding='utf-8').split(MARKER)[0]
    path.write_text(original + '\n'.join(lines), encoding='utf-8')
    print('Updated measured Markdown and capability evidence; no inference.')


if __name__ == '__main__':
    main()
