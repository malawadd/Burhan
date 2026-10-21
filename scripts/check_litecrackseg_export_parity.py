"""Compare actual-camera author Torch output with exported ONNX, calibration only."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import cv2
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from export_litecrackseg_contingency import reviewed_model, Export
from evaluate_construction_round3 import sampled


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', type=int, default=384)
    args = parser.parse_args()
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    rows = sampled('calibration')
    # Fixed first actual calibration exposure: not selected by prediction.
    frame = cv2.imread(str(rows[0]['image']))
    image = cv2.resize(frame, (args.size,args.size), interpolation=cv2.INTER_LINEAR)
    tensor = np.ascontiguousarray(image.transpose(2,0,1)[None])
    model,_,_ = reviewed_model()
    with torch.inference_mode():
        expected = Export(model)(torch.from_numpy(tensor)).numpy()
    import onnxruntime as ort
    settings = ort.SessionOptions(); settings.intra_op_num_threads=1;settings.inter_op_num_threads=1
    settings.add_session_config_entry('session.intra_op.allow_spinning','0')
    path = ROOT/f'models/construction/thin-contingency/litecrackseg-tut-{args.size}.onnx'
    session = ort.InferenceSession(str(path), sess_options=settings, providers=['CPUExecutionProvider'])
    observed = session.run(None,{session.get_inputs()[0].name:tensor})[0]
    absolute = np.abs(expected-observed)
    result = {'model_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
              'source_image':str(rows[0]['image'].relative_to(ROOT)),
              'source_image_sha256':hashlib.sha256(rows[0]['image'].read_bytes()).hexdigest(),
              'max_abs_probability_error':float(absolute.max()),'mean_abs_probability_error':float(absolute.mean()),
              'p99_abs_probability_error':float(np.percentile(absolute,99)),
              'expected_probability_range':[float(expected.min()),float(expected.max())],
              'observed_probability_range':[float(observed.min()),float(observed.max())],
              'scope':'Fixed actual calibration frame, TorchCPU vs ONNXRuntimeCPU; not pixel accuracy or speed',
              'passed':bool(np.allclose(expected,observed,atol=1e-4,rtol=1e-3))}
    target = ROOT/f'output/evaluation/construction-round3/litecrackseg-calibration/export-parity-{args.size}.json'
    target.write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result,indent=2),flush=True)
    if not result['passed']:
        raise RuntimeError('ONNX export differs materially from original author model')


if __name__=='__main__': main()
