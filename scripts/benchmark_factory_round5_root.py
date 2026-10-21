"""Separate root-operated Round5 benchmarks; source preparation performs no NN.

Reuse immutable Round4 measurement functions in this standalone process, with
new output/plan/binding only. No Round4 files, policies or outcomes are changed.
"""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import benchmark_factory_round4_root as common

R5 = ROOT / 'output/optimization/factory-round5-cldice'
OUT = ROOT / 'output/optimization/factory-round5-root-benchmark'
PLAN = OUT / 'plan-before-any-benchmark.json'
FIXTURE = OUT / 'source-preparation-fixtures-v1.json'
PROTOCOL = R5 / 'protocol-before-fit.json'
PROPOSAL_SHA = '7153f2e58df0b37e6230438faaee23e2eb7257e5b2b8b562e6f5e724e58431b7'
NAME = 'supervised_round5_thin_cldice'
STEPS = [200, 400, 600]
MODEL_NAMES = ['v10h', 'C200', 'winner']
PINS = ['scripts/benchmark_factory_round5_root.py', 'tests/test_factory_round5_root_benchmark.py',
        *[name for name in common.PINS if not name.startswith('tests/')],
        'scripts/evaluate_factory_round4.py', 'scripts/factory_mask_metrics.py', 'scripts/factory_stfpm_common.py']
sha, read, write_new = common.sha, common.read, common.write_new


def global_winner(candidates):
    """Fixed earlier-step tie break matches stable sort of 200/400/600 outcomes."""
    from factory_stfpm_common import rank
    ordered = sorted(candidates, key=lambda row: row['step'])
    ordered.sort(key=lambda row: rank(row['point']), reverse=True)
    return (ordered[0] if ordered else None), ordered


def qualifies(point):
    return point is not None and point['meaningful_iou_0_1_recall'] == 1 and point['normal_fp'] == 0 and point['foreground_pixel_precision'] >= .65


def previous_runtime_models(name):
    if name not in MODEL_NAMES:
        raise ValueError('Only fixed three runtime model names allowed')
    return MODEL_NAMES[:MODEL_NAMES.index(name)]


def binding():
    from evaluate_factory_round4 import numeric_passes, recipe_passes, validate_search
    if sha(PROTOCOL) != PROPOSAL_SHA:
        raise ValueError('Exact independently reviewed Round5 protocol required')
    p = read(PROTOCOL)
    decision = R5 / 'parent-approved-fit-decision.json'
    if read(decision).get('approved_exact_round5_fit') is not True or read(decision)['proposal_sha256'] != sha(PROTOCOL):
        raise ValueError('Root exact Round5 fit decision differs')
    done_path, frozen_path = R5 / 'training-complete.json', R5 / 'winner-frozen-before-root-capacity.json'
    done, frozen = read(done_path), read(frozen_path)
    if (done['protocol_sha256'] != sha(PROTOCOL) or done['steps'] != 600 or done['saved_all_three'] != STEPS
        or done['sources_unchanged'] is not True or done['no_CAL_original_feedback'] is not True
        or frozen['protocol_sha256'] != sha(PROTOCOL) or frozen['all_three_complete'] is not True
        or frozen['original_audit_not_authorized_yet'] is not True or frozen['production_qualified'] is not False
        or set(frozen['outcomes']) != {'200', '400', '600'}):
        raise ValueError('Complete fixed600/all3 globally frozen outcomes required')
    paths = [PROTOCOL, decision, done_path, frozen_path]
    manifest = R5 / 'executed-sources-before-optimizer/manifest.json'
    archived = read(manifest)
    if archived['protocol_sha256'] != sha(PROTOCOL) or archived['source_pins'] != p['source_pins']:
        raise ValueError('Executed Round5 source archive differs')
    paths.append(manifest)
    for name, digest in archived['source_pins'].items():
        path = manifest.parent / name
        if sha(path) != digest:
            raise ValueError('Executed Round5 source bytes changed')
        paths.append(path)
    for name, digest in p['artifact_pins'].items():
        path = ROOT / name
        if sha(path) != digest:
            raise ValueError('Protected Round5 preparation artifact changed')
        paths.append(path)
    sources = p['protected_CAL_source_rows']
    if len(sources) != 80 or sum(r['defective'] for r in sources) != 10 or common.historical_cal_identity(sources) != common.CAL_SHA or p['CAL_source_identity_sha256'] != common.CAL_SHA:
        raise ValueError('Exact protected historical CAL80 content required')
    models = {}
    for label, path, model_sha, meta_sha in [
        ('v10h', ROOT / 'models/factory/spatial_unet_v10h_sdd2-step300.onnx',
         'dd5019cbcbc859c4eeb994a7639fb0c520ee3e3175e47a3f1544ec4c5dd60753', '25d8d250c3abc0deab82994c3cc48f6f5d356058bbd74ab4903b9cd497f1ba78'),
        ('C200', ROOT / 'models/factory/spatial_round3e_phase_max_refine-step200.onnx',
         'ca9b90e33f1cacc8cd83ccc8e406b32faa8f45947fd2d672dc48e678b20d7318', '6d727b95adef756399dd72a9d0a6d814c4ce2a4b72c7cf4ee15abd4a3b702bdb')]:
        meta = read(path.with_suffix('.onnx.json'))
        if sha(path) != model_sha or sha(path.with_suffix('.onnx.json')) != meta_sha or meta['split_items'] != p['split_items']:
            raise ValueError('Retained historical model/metadata/split changed')
        models[label] = dict(path=str(path), sha256=model_sha, metadata_sha256=meta_sha,
            policy=common.policy(meta), split_items=meta['split_items'], role='Retained historical comparator')
        paths.extend([path, path.with_suffix('.onnx.json')])
    candidates = []
    for step in STEPS:
        model = ROOT / f'models/factory/{NAME}-step{step}.onnx'
        training_path, numeric_path, terminal_path = [R5 / f'checkpoint{step}-{part}.json' for part in ['training', 'numeric', 'terminal']]
        reference = R5 / f'checkpoint{step}-torch-reference.npz'
        training, numeric, terminal = map(read, [training_path, numeric_path, terminal_path])
        if (training['step'] != step or training['protocol_sha256'] != sha(PROTOCOL) or training['model_sha256'] != sha(model)
            or training['initial_metadata_sha256'] != sha(model.with_suffix('.onnx.json'))
            or training['reference_sha256'] != sha(reference) or training['state_sha256'] != sha(model.with_suffix('.pt'))
            or training['optimizer_sha256'] != sha(R5 / f'checkpoint{step}-optimizer.pt')
            or training['all_state_finite'] is not True or training['frozen_BN_normalization_unchanged'] is not True
            or numeric['step'] != step or numeric['protocol_sha256'] != sha(PROTOCOL) or numeric['model_sha256'] != sha(model)
            or numeric['reference_sha256'] != sha(reference) or numeric['initial_metadata_sha256'] != training['initial_metadata_sha256']
            or numeric['training_receipt_sha256'] != sha(training_path) or numeric['all_checks_passed'] != numeric_passes(numeric)
            or terminal['protocol_sha256'] != sha(PROTOCOL) or terminal['numeric_sha256'] != sha(numeric_path)
            or frozen['outcomes'][str(step)] != terminal):
            raise ValueError('Actual all3 training/numeric/terminal lineage differs')
        paths.extend([model, model.with_suffix('.onnx.json'), model.with_suffix('.pt'), training_path, numeric_path, terminal_path,
                      reference, R5 / f'checkpoint{step}-optimizer.pt'])
        if not numeric_passes(numeric):
            if terminal['status'] != 'numeric_failed_no_CAL' or terminal['CAL_inputs_opened'] != 0 or any((R5 / f'checkpoint{step}-{part}').exists() for part in ['CAL-started.json', 'calibration.json', 'scorecache.npz']):
                raise ValueError('Numeric-failed checkpoint must never open CAL')
            continue
        cal_path, cache = R5 / f'checkpoint{step}-calibration.json', R5 / f'checkpoint{step}-scorecache.npz'
        cal = read(cal_path)
        if (terminal['status'] != 'calibration_complete' or terminal['CAL_inputs_opened'] != 80 or terminal['calibration_sha256'] != sha(cal_path)
            or cal['step'] != step or cal['protocol_sha256'] != sha(PROTOCOL) or cal['model_sha256'] != sha(model)
            or cal['numeric_sha256'] != sha(numeric_path) or cal['scorecache_sha256'] != sha(cache)
            or not recipe_passes(cal['configuration']) or cal['ordered_CAL_sources'] != sources or cal['CAL_identity'] != common.CAL_SHA):
            raise ValueError('Fixed complete80 CAL/cache/source/configuration differs')
        validate_search(cal['raw_grid'], cal['zero_alarm_search'], p, 0)
        point = cal['zero_alarm_search']['selected']
        eligible = qualifies(point)
        if cal['eligible_all10_zero_alarm'] is not eligible:
            raise ValueError('Strict all10/zero-normal-alarms/precision guard differs')
        paths.extend([cal_path, cache])
        if eligible:
            deployed = cal['deployment']; path = Path(deployed['model_path']); meta = read(path.with_suffix('.onnx.json'))
            if (sha(path) != sha(model) or deployed['model_sha256'] != sha(path) or deployed['metadata_sha256'] != sha(path.with_suffix('.onnx.json'))
                or meta['model_sha256'] != sha(path) or meta['round5_training_protocol_sha256'] != sha(PROTOCOL)
                or meta['calibration_selected_point'] != point or meta['split_items'] != p['split_items']
                or meta['calibration_source_identity_sha256'] != common.CAL_SHA
                or common.policy(meta) != {'pixel_threshold': point['high'], 'minimum_component_pixels': point['area'], 'hysteresis_growth_threshold': point['growth']}):
                raise ValueError('Selected byte-identical fitted model/metadata/policy differs')
            candidates.append(dict(step=step, point=point, **deployed))
            paths.extend([path, path.with_suffix('.onnx.json')])
    winner, ordered = global_winner(candidates)
    if frozen['winner'] != winner or frozen['ranked_candidates'] != ordered:
        raise ValueError('Global winner must follow all3 fixed outcomes and declared rank')
    if winner:
        meta = read(Path(winner['model_path']).with_suffix('.onnx.json'))
        models['winner'] = dict(path=winner['model_path'], sha256=winner['model_sha256'], metadata_sha256=winner['metadata_sha256'],
            policy=common.policy(meta), split_items=meta['split_items'], step=winner['step'], role='Round5 globally frozen zero-CAL-alarm winner')
    return {'models': models, 'eligible_round4_arms': ['winner'] if winner else [],
        'compatibility_key_scope': 'eligible_round4_arms is only the immutable common measurement API key; these artifacts/results are Round5.',
        'calibration_sources': sources, 'round5_global_freeze_sha256': sha(frozen_path),
        'artifact_pins': {str(path.relative_to(ROOT)): sha(path) for path in paths}}


def configure_common():
    common.OUT, common.PLAN, common.FIXTURE = OUT, PLAN, FIXTURE
    common.binding, common.PINS = binding, PINS
    return common


def prepare():
    fixture = read(FIXTURE)
    if fixture['passed'] is not True or fixture['tests'] != 4:
        raise ValueError('Four pure Round5 benchmark preparation fixtures required')
    for name, digest in fixture['source_pins'].items():
        if sha(ROOT / name) != digest:
            raise ValueError('Stale benchmark source preparation fixture')
    if sha(common.NORMAL_PROTOCOL) != common.NORMAL_SHA:
        raise ValueError('Fixed16 TRAIN normal source identity changed')
    b = binding(); normal = read(common.NORMAL_PROTOCOL)['capacity']['real_sources']
    train = set(b['models']['C200']['split_items']['train'])
    if len(normal) != 16 or any(row['defective'] is not False or Path(row['image'].replace('\\', '/')).parts[0] not in train for row in normal):
        raise ValueError('Exactly16 real normal TRAIN sources required')
    write_new(PLAN, {'binding': b, 'normal16_sources': normal, 'normal_source_protocol_sha256': common.NORMAL_SHA,
        'preparation_fixture_sha256': sha(FIXTURE), 'source_pins': {name: sha(ROOT / name) for name in PINS},
        'status': 'READY_FOR_ROOT_QUIET_RELEASE' if b['eligible_round4_arms'] else 'NO_ELIGIBLE_ROUND5_WINNER_NO_NN',
        'capacity': {'init_zero_calls': 3, 'real_warmups': 4, 'first_measured_calls': 64, 'normal_source_cycle': 16,
                     'minimum_mean_full_profile_FPS': 15, 'all64must_infer': True},
        'paired': {'partition': 'calibration', 'views': 80, 'repeats': 3, 'order': 'Rotate model order by(view+repetition)%3; same frame each model, all240 calls retained'},
        'runtime': {'fps': 16, 'seconds': 60, 'expected_submitted_and_processed': 960, 'deadline_ms': 100, 'preview_fps': 6,
                    'zero_drop_skip_error_gap_late_required': True, 'fixed_order': MODEL_NAMES},
        'accuracy': 'Root only: one selected winner exposed80 audit after first64>=15; save every11 defective view and every normal alarm with actual masks. No checkpoint/policy fallback or original-informed selection.',
        'scope': 'New separate Round5 results, never replacement/retry of Round4. Ordinary complete profile plus separate existing capture/queue/evidence runtime; clocked real stills, not physical moving-line footage.',
        'quiet_rule': 'All other project fit/export/NN/data/media/test jobs exited; station stopped and browser paused; ordinary host services uncontrolled, no process suspension.',
        'first_result_rule': 'Preserve all first attempts/raw tails/failures. No eligible frozen winner means no NN. No replay/capacity retry or tuning.'})
    print('ROUND5_ROOT_PLAN', sha(PLAN), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['prepare', 'capacity', 'original', 'paired', 'runtime'], required=True)
    parser.add_argument('--name', choices=MODEL_NAMES)
    parser.add_argument('--root-quiet-release', action='store_true')
    parser.add_argument('--root-original-release', action='store_true')
    args = parser.parse_args()
    if args.stage == 'prepare':
        prepare()
    else:
        if not args.root_quiet_release or args.stage == 'original' and not args.root_original_release:
            parser.error('Root quiet release and separate original release are mandatory')
        import cv2
        cv2.setNumThreads(2)
        helper = configure_common(); plan = helper.verify_plan()
        if args.stage == 'paired':
            helper.paired(plan)
        else:
            if args.name not in plan['binding']['models'] or args.stage in ['capacity', 'original'] and args.name != 'winner':
                parser.error('Only the frozen winner can run first64 or new original scoring')
            if args.stage == 'runtime':
                for previous in previous_runtime_models(args.name):
                    gate = read(OUT / (previous + '-runtime-gate.json'))
                    if gate['plan_sha256'] != sha(PLAN) or gate['model'] != plan['binding']['models'][previous] or gate['raw_report_sha256'] != sha(OUT / (previous + '-runtime.json')):
                        raise ValueError('Runtime order requires completed unchanged earlier-model result')
            (helper.capacity if args.stage == 'capacity' else lambda p, n: helper.legacy(p, n, args.stage))(plan, args.name)
