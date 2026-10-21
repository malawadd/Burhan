"""Read-only measured-result checks and MD/JSON reporting. Never performs NN."""
from pathlib import Path
import hashlib
import json
import math
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/optimization/factory-round5-root-benchmark'


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path): return json.loads(path.read_text(encoding='utf8'))
def link(path, text): return f'[{text}]({path.as_posix()})'


def main():
    plan_path = OUT / 'plan-before-any-benchmark.json'; plan = read(plan_path)
    original_path = OUT / 'winner-original.json'; original = read(original_path)
    audit_paths = {'v10h': ROOT / 'output/optimization/factory-round3e/before-localization-audit.json',
                   'C200': ROOT / 'output/optimization/factory-round3e/after-localization-audit.json', 'Round5': original_path}
    audits = {name: read(path) for name, path in audit_paths.items()}
    sources = [(row['path'], row['sha256'], row['mask_sha256']) for row in original['records']]
    for name, report in audits.items():
        if [(row['path'], row['sha256'], row['mask_sha256']) for row in report['records']] != sources:
            raise ValueError('Accuracy comparison input/GT identity differs')
        key = 'winner' if name == 'Round5' else name
        if report['model']['model_sha256'] != plan['binding']['models'][key]['sha256']:
            raise ValueError('Actual compared model differs from benchmark binding')
    for relative, digest in plan['source_pins'].items():
        if sha(ROOT / relative) != digest: raise ValueError('Measured source changed')
    pair_path = OUT / 'paired-CAL80.json'; pair = read(pair_path)
    if len(pair['records']) != 240 or pair['plan_sha256'] != sha(plan_path): raise ValueError('Exact paired240 observations required')
    for name, summary in pair['timings'].items():
        values = [row['models'][name]['profile_ms'] for row in pair['records']]
        if not math.isclose(float(np.mean(values)), summary['mean_ms'], rel_tol=1e-12) or not all(row['models'][name]['neural_stage_ms'] for row in pair['records']):
            raise ValueError('Paired timing/actual NN evidence mismatch')
    capacity_path = OUT / 'winner-first64-capacity.json'; capacity = read(capacity_path)
    paths = [Path(__file__), plan_path, original_path, pair_path, capacity_path, OUT / 'host-before-first-benchmark.json', *audit_paths.values()]
    runtimes = {}
    for name in ['v10h', 'C200', 'winner']:
        path, gate_path = OUT / f'{name}-runtime.json', OUT / f'{name}-runtime-gate.json'
        raw, gate = read(path), read(gate_path); metrics = raw['state']['metrics']; trace = raw['diagnostic_trace']
        if (gate['raw_report_sha256'] != sha(path) or gate['plan_sha256'] != sha(plan_path) or raw['requested_fps'] != 16 or raw['requested_seconds'] != 60
            or len(trace) != metrics['processed'] or len({row['sequence'] for row in trace}) != len(trace)
            or raw['input_sha256'] != [row['image_sha256'] for row in plan['binding']['calibration_sources']]
            or not all(isinstance(row.get('neural_stage_ms'), dict) and row['neural_stage_ms']['infer_readback'] > 0 for row in trace)):
            raise ValueError('Actual runtime source/count/NN/plan binding differs')
        runtimes[name] = {'metrics': metrics, 'processed_over_full_uptime_FPS': metrics['processed'] / metrics['uptime_s'],
                          'host_to_return_tails': gate['host_to_profile_return_tails'], 'profile_tails': gate['profile_tails'],
                          'official_late_counter': metrics['late'], 'traced_host_over100ms': len(gate['all_over100ms_trace_rows']),
                          'checks': gate['checks'], 'strict_runtime_gate_passed': gate['runtime_gate_passed'], 'raw_report': str(path.relative_to(ROOT))}
        paths.extend([path, gate_path])
    selected_rows = [row for row in original['records'] if row['defective'] or row['flagged']]
    if sum(row['defective'] for row in selected_rows) != 11 or sum(not row['defective'] for row in selected_rows) != original['metrics']['normal_false_alarm_views']:
        raise ValueError('All defective views and all flagged normals required')
    evidence = []
    for row in selected_rows:
        files = {name: Path(value) for name, value in row['visual_evidence'].items() if name != 'selection'}
        predicted = (cv2.imread(str(files['predicted_mask']), 0) > 0).astype(np.uint8)
        truth = cv2.imread(str(files['publisher_mask']), 0) > 0
        if (list(predicted.shape) != row['mask_shape_hw'] or hashlib.sha256(predicted.tobytes()).hexdigest() != row['prediction_mask_sha256']
            or int(predicted.sum()) != row['pixel_predicted'] or int(truth.sum()) != row['pixel_target']
            or int(np.count_nonzero(predicted.astype(bool) & truth)) != row['pixel_tp']):
            raise ValueError('Saved original-resolution masks differ from measured counts/hashes')
        evidence.append({'path': row['path'], 'defective': row['defective'], 'flagged': row['flagged'], 'pixel_recall': row['pixel_recall'],
                         'pixel_precision': row['pixel_precision'], 'iou': row['iou'], 'pixel_predicted': row['pixel_predicted'],
                         'pixel_target': row['pixel_target'], 'files': {name: str(path.relative_to(ROOT)) for name, path in files.items()}})
        paths.extend(files.values())
    summary = {'scope': 'Actual fixed Round5 result; no NN/reselection/retry performed by this summary. Historical accuracy comparators use byte-identical original80 inputs/GT; paired/runtime timings are fresh.',
               'accuracy': {name: {'report': str(audit_paths[name].relative_to(ROOT)), 'metrics': report['metrics']} for name, report in audits.items()},
               'first64': capacity['timing'], 'paired_CAL80': pair['timings'], 'runtimes': runtimes, 'evidence': evidence,
               'target_achieved': False, 'promotion': False, 'production_qualified': False,
               'limitations': ['Original80 is repeatedly exposed development, not blind', '69 normal views belong to defective products; good-product false-rejection unmeasured',
                               'Clocked varied real-still replay is not moving-line footage', 'Project jobs quiet; ordinary host services uncontrolled, no causal attribution to a background process'],
               'pins': {str(path.relative_to(ROOT)): sha(path) for path in sorted(set(paths))}}
    target = OUT / 'measured-summary-and-evidence-pins.json'
    with target.open('x', encoding='utf8') as stream: json.dump(summary, stream, indent=2, allow_nan=False)
    evidence_path = ROOT / 'docs/factory-round5-localization-evidence.md'
    lines = ['# Round5 complete localization evidence', '', 'All11 defect views and the one flagged normal view are included. Each panel shows unchanged source, external GT, actual predicted pixels and error overlay. No favorable-case filtering. The saved mask PNGs were independently checked against the audit pixel counts and SHA256.', '',
             '| Source | Kind | Pixel recall | Pixel precision | IoU | Four-panel evidence | Actual predicted mask | Publisher mask |', '|---|---|---:|---:|---:|---|---|---|']
    def percent(value): return '—' if value is None else f'{100 * value:.2f}%'
    for row in evidence:
        files = row['files']; source = ROOT / 'datasets/factory/KolektorSDD' / Path(row['path'])
        kind = 'DEFECT — COMPLETE MISS' if row['defective'] and row['pixel_predicted'] == 0 else 'Defect' if row['defective'] else 'NORMAL — FALSE ALARM'
        lines.append('| ' + ' | '.join([link(source, row['path'].replace('\\', '/')), kind, percent(row['pixel_recall']), percent(row['pixel_precision']), percent(row['iou']),
                     link(ROOT / files['visualization'], 'panel'), link(ROOT / files['predicted_mask'], 'prediction'), link(ROOT / files['publisher_mask'], 'GT')]) + ' |')
    lines.extend(['', 'The complete miss remains kos12/Part5: actual predicted foreground0, true foreground retained in the publisher mask, with zero pixel recall/IoU. The normal-view alarm is also shown rather than suppressed.', '',
                  'Box overlap is supplementary. A box can overlap while its actual mask has poor pixel precision; the primary audit uses original-resolution mask pixels. These findings do not establish physical crack width, structural severity or code compliance.', '', link(target, 'Raw summary and SHA256 pins')])
    evidence_path.write_text('\n'.join(lines) + '\n', encoding='utf8')
    doc = ROOT / 'docs/factory-round5-measured-results.md'
    lines = ['# Round5 measured results', '', '**The requested joint improvement was not achieved.** R5 still meaningfully localized10/11 defect views and raised one alarm among69 unaffected views, matching C200 on those counts. It increased foreground pixel recall but reduced precision and strict region overlap. It passed average complete-profile capacity; the fresh full-runtime trial dropped five frames and recorded25 late events. No promotion or production qualification.', '',
             '## Accuracy on the same original80 views', '', 'The original80 comparison is repeatedly exposed development data. Baseline accuracy below comes from retained audits; their exact ordered source-image and GT hashes match the new R5 audit. Baseline weights/metadata are the same hashes used in the fresh timing trials.', '',
             '| Model | Meaningful view recall (IoU≥.1) | Normal-view alarms | ≥50% GT coverage | Pixel recall | Pixel precision | Micro IoU | Mask-region matches IoU≥.5 |', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name, report in audits.items():
        m = report['metrics']; lines.append(f"| {name} | {round(m['localized_view_recall_iou_0_1'] * 11)}/11 | {m['normal_false_alarm_views']}/69 | {round(m['localized_view_recall_50pct_coverage'] * 11)}/11 | {100*m['foreground_pixel_recall']:.3f}% | {100*m['foreground_pixel_precision']:.3f}% | {m['micro_pixel_iou']:.6f} | {round(m['region_recall']['0.5']*11)}/11 |")
    lines.extend(['', 'R5 has0 abstentions and still completely misses kos12/Part5. Against C200, pixel recall rises75.404%→79.448%, precision falls61.536%→58.754%, micro IoU falls.512494→.510021 and strict mask-region matches fall4/11→3/11. Supplementary actual runtime-box matches stay9/11 at IoU≥.5; boxes do not reverse the pixel result.', '',
                  'No completely normal products occur in this comparison:69 unaffected views belong to ten defective items. Good-product false-rejection rate is unmeasured. The zero-alarm CAL80 result did not transfer to the exposed original80.', '',
                  '## Fresh speed and full runtime', '', f"The first64 complete-profile normal-TRAIN calls passed at{capacity['timing']['serial_full_profile_FPS']:.3f}FPS, mean{capacity['timing']['mean_ms']:.3f}ms/P95{capacity['timing']['p95_ms']:.3f}ms/max{capacity['timing']['max_ms']:.3f}ms. This excludes capture, queueing, preview and storage and is not the sustained runtime result.", '',
                  'Paired CAL80 used three balanced repetitions:240 timings per model, not240 new accuracy samples.', '',
                  '| Model | Serial full-profile FPS | Mean ms | P95 ms | P99 ms | Max ms |', '|---|---:|---:|---:|---:|---:|'])
    for name in ['v10h', 'C200', 'winner']:
        m = pair['timings'][name]; lines.append(f"| {'R5' if name=='winner' else name} | {m['serial_full_profile_FPS']:.3f} | {m['mean_ms']:.3f} | {m['p95_ms']:.3f} | {m['p99_ms']:.3f} | {m['max_ms']:.3f} |")
    lines.extend(['', 'Fresh runtime order was v10h→C200→R5. Each replay requested16FPS for60 seconds on the same varied CAL80 sequence, preview6FPS, deadline100ms. All first results/tails were retained.', '',
                  '| Model | Captured / processed | Dropped | Official late events | Processed / full uptime FPS | Host P95 ms | Host P99 ms | Host max ms | Strict gate |', '|---|---:|---:|---:|---:|---:|---:|---:|---|'])
    for name, record in runtimes.items():
        m = record['metrics']; t = record['host_to_return_tails']; lines.append(f"| {'R5' if name=='winner' else name} | {m['captured']} / {m['processed']} | {m['dropped']} | {m['late']} | {record['processed_over_full_uptime_FPS']:.3f} | {t['p95_ms']:.3f} | {t['p99_ms']:.3f} | {t['max_ms']:.3f} | FAIL |")
    lines.extend(['', 'All three fresh runtime trials failed the predeclared no-drop/no-late gate. Average processed rate exceeds15FPS, but this does not guarantee inspection of every captured exposure or a production deadline. All three report0 source-skipped/inspection-error/missed-unit/coverage-gap counters; those counters do not erase the explicit dropped-frame counts. Official late counters and traced over100ms rows are both retained in the raw summary; they are not silently substituted for each other.', '',
                  'Project fits, inference, encoding and tests were stopped for these measurements; ordinary host services were uncontrolled. The host snapshot is pinned. Queue/profile/GPU-stage traces are descriptive and do not prove a low-level cause. This is clocked real-still replay, not physical moving-line validation.', '',
                  '## Reproducible artifacts', '', link(ROOT / 'docs/factory-round5-cldice.md', 'Training, sources, all three candidates and calibration'), '', link(evidence_path, 'All11 defects and the normal false alarm'), '', link(target, 'Measured summary and raw-report/source/model SHA256 pins'), '',
                  link(OUT / 'winner-original.json', 'Actual mask audit'), '', link(pair_path, 'Fresh paired timing'), '', link(OUT / 'winner-runtime.json', 'Actual R5 runtime and every trace row'), '',
                  f'Summary SHA256: `{sha(target)}`. No new neural forward, policy selection, retry or model edit was performed by this summary.'])
    doc.write_text('\n'.join(lines) + '\n', encoding='utf8')
    print(json.dumps({'doc': str(doc), 'evidence': str(evidence_path), 'summary_sha256': sha(target), 'evidence_views': len(evidence)}, indent=2))


if __name__ == '__main__': main()
