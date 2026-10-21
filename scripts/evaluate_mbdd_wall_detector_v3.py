"""Versioned target/identity wrapper over the immutable executed wall evaluator.

The fixed grid, matcher, parity, failure receipts and18-photo method are reused
without editing the v2 source. The wrapper is itself bound before any CAL NN.
"""
from pathlib import Path
import hashlib,json,sys,traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    old_protocol=ROOT/'output/evaluation/construction-wall-five-class-v2/evaluation-protocol-before-calibration.json'
    base_path=ROOT/'scripts/evaluate_mbdd_wall_detector.py'
    if sha(base_path)!=json.loads(old_protocol.read_text())['script_sha256']:raise RuntimeError('Immutable executed base evaluator changed')
    import evaluate_mbdd_wall_detector as base
    base.MODEL=ROOT/'models/construction/mbdd-wall-five-class-v3/wall-five-class-total16-last12-640.onnx'
    base.OUT=ROOT/'output/evaluation/construction-wall-five-class-v3'
    stage=sys.argv[sys.argv.index('--stage')+1]
    target=base.OUT/'evaluation-protocol-before-calibration.json'
    plan_path=ROOT/'output/training/construction-wall-five-class-v3/approved-fit-plan.json'
    plan=json.loads(plan_path.read_text())
    if not plan.get('parent_approved') or plan['future_evaluator_sha256']!=sha(Path(__file__)) or plan['base_evaluator_sha256']!=sha(base_path):raise RuntimeError('Approved continuation evaluator identity differs')
    binding={'wrapper_path':Path(__file__).relative_to(ROOT).as_posix(),'wrapper_sha256':sha(Path(__file__)),
             'immutable_base_evaluator_sha256':sha(base_path),'approved_continuation_plan_sha256':sha(plan_path),
             'prior_v2_protocol_sha256':sha(old_protocol),'prior18_scope':'All18 already exposed v2 development photos. No independent accuracy credit or diagnostic-driven checkpoint/threshold selection.'}
    if stage=='freeze':
        if target.exists():raise RuntimeError('Preserve frozen v3 evaluation protocol; no silent replacement')
        base.main()
        protocol=json.loads(target.read_text());protocol['versioned_wrapper']=binding
        target.write_text(json.dumps(protocol,indent=2)+'\n',encoding='utf-8')
        print('FINAL_WRAPPER_BOUND_PROTOCOL',sha(target),flush=True);return
    if json.loads(target.read_text()).get('versioned_wrapper')!=binding:raise RuntimeError('Frozen wrapper/base/approved plan differs')
    try:base.main()
    except Exception:
        if base.ACTIVE_RECEIPT is not None:
            failed=json.loads(base.ACTIVE_RECEIPT.read_text());failed.update(state='failed',traceback=traceback.format_exc())
            base.write(base.ACTIVE_RECEIPT,failed)
        raise


if __name__=='__main__':main()
