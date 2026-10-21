"""Record C-density protocol or explicitly export NEW graphs. Preparation never implies NN."""
import argparse
import json
import shutil
from factory_c_density_common import ROOT, OUT, NAME, SOURCE, SOURCE_ONNX, SOURCE_FREEZE, sha, record_protocol, verify_protocol


def write_new(path,value):
    if path.exists():
        raise ValueError(f'Preserve existing/partial artifact: {path}')
    path.write_text(json.dumps(value,indent=2),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--record-protocol',action='store_true')
    group.add_argument('--export-probe',action='store_true')
    group.add_argument('--export-fitted',action='store_true')
    args=parser.parse_args()
    if args.record_protocol:
        path,_=record_protocol()
        print(json.dumps({'protocol':str(path),'sha256':sha(path),'nn_started':False}));return
    path,value=verify_protocol()
    modeldir=ROOT/'models/factory'
    if args.export_probe:
        targets=[modeldir/(NAME+'-'+suffix+'.onnx') for suffix in ['capacity-probe','normal-feature-extractor','pre-fit-control','pre-fit-diagnostic']]
        receipts=[OUT/(NAME+'-export-parity.json'),OUT/(NAME+'-export-manifest.json')]
    else:
        targets=[modeldir/f'{NAME}-step{row["id"]}.onnx' for row in value['candidates']]+[modeldir/(NAME+'-offline-diagnostic.onnx')]
        receipts=[OUT/(NAME+'-fitted-export-manifest.json')]
        preflight=json.loads((OUT/(NAME+'-fullprofile-preflight.json')).read_text())
        sourceparity=json.loads((OUT/(NAME+'-pre-fit-gpu-parity.json')).read_text())
        if preflight['protocol_sha256']!=sha(path) or preflight['profile_ms']['serial_calls_per_second']<15 or not sourceparity['all_checks_passed'] or sourceparity['protocol_sha256']!=sha(path):
            raise ValueError('Fresh C-density preflight/source parity prerequisites failed')
        fit=json.loads((OUT/(NAME+'-normal-fit.json')).read_text())
        bankpath=OUT/(NAME+'-normal-bank.npz')
        initialpath=OUT/(NAME+'-export-manifest.json')
        if fit['protocol_sha256']!=sha(path) or fit['bank_sha256']!=sha(bankpath) or fit['export_manifest_sha256']!=sha(initialpath) or fit['source_pt_sha256']!=sha(SOURCE):
            raise ValueError('C-specific normal bank/source lineage changed')
    if any(p.exists() for p in targets+receipts):
        raise ValueError('Do not replace completed/failed/partial exports; use separately authorized version')
    # No Torch/model import or constructor occurs in protocol-only mode.
    import numpy as np
    import torch
    from torch import nn
    from factory_c_density_graph import CNormalDensityGraph,CDiagnosticDensityGraph
    from train_factory_spatial_unet import export
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    if args.export_probe:
        model=CNormalDensityGraph()
        torch.manual_seed(value['seed'])
        sample=torch.randint(0,256,(1,3,1280,512),dtype=torch.uint8)
        with torch.inference_mode():
            original=nn.functional.max_pool2d(model.base(sample),2)
            copied,fine=model.parts(sample)
            if not torch.equal(original,copied) or fine.shape!=(1,16,640,256):
                raise ValueError('Exact C phase/context/maxpool semantic path or fine geometry changed')
            model.train()
            for layer in model.modules():
                if isinstance(layer,nn.BatchNorm2d):layer.eval()
            frozen,_=model.parts(sample)
            if not torch.equal(copied,frozen):raise ValueError('Frozen-BN source path changed')
            model.eval();fused=model(sample)
            if fused.shape!=(1,1,640,256) or not torch.isfinite(fused).all():raise ValueError('Invalid nonconstant probe output')
        parity={'raw_logit_torch_identity':True,'frozen_bn_train_eval_identity':True,
                'fullres_phase_context_then_maxpool_preserved':True,'one_shared_encoder':True,
                'probe_fused_raw_logit_min_max':[float(fused.min()),float(fused.max())],
                'scope':'Nonconstant initialization tensor; no performance/accuracy claim','protocol_sha256':sha(path)}
        export(model,targets[0]);export(CNormalDensityGraph(alpha=1.,beta=0.),targets[2])
        diagnostic=CDiagnosticDensityGraph().eval()
        with torch.inference_mode():
            torch.onnx.export(diagnostic,torch.zeros((1,3,1280,512),dtype=torch.uint8),str(targets[3]),
                input_names=['bgr'],output_names=['fused_logits','evidence','fine_features'],opset_version=17,dynamo=False,do_constant_folding=True)
        # Retain ALL diagnostic outputs in the actual fitted-feature context.
        # Exporting a fine-only wrapper would prune semantic/context operations.
        shutil.copyfile(targets[3],targets[1])
        write_new(receipts[0],parity)
        manifest={'protocol_sha256':sha(path),'probe_sha256':sha(targets[0]),'feature_extractor_sha256':sha(targets[1]),
                  'pre_fit_control_sha256':sha(targets[2]),'pre_fit_diagnostic_sha256':sha(targets[3]),
                  'source_pt_sha256':sha(SOURCE),'source_onnx_sha256':sha(SOURCE_ONNX),
                  'source_metadata_sha256':sha(SOURCE_ONNX.with_suffix('.onnx.json')),'source_freeze_sha256':sha(SOURCE_FREEZE),
                  'scope':'NEW C-specific graphs; full all-output feature context; no F2 graph/bank/speed transfer'}
        write_new(receipts[1],manifest)
        print(json.dumps(manifest,indent=2))
    else:
        with np.load(bankpath) as bank:
            candidates=[]
            for row,target in zip(value['candidates'],targets[:-1]):
                export(CNormalDensityGraph(bank,row['alpha'],row['beta']),target)
                candidates.append({**row,'model_path':str(target),'model_sha256':sha(target)})
            diagnostic=CDiagnosticDensityGraph(bank).eval()
            with torch.inference_mode():
                torch.onnx.export(diagnostic,torch.zeros((1,3,1280,512),dtype=torch.uint8),str(targets[-1]),
                    input_names=['bgr'],output_names=['fused_logits','evidence','fine_features'],opset_version=17,dynamo=False,do_constant_folding=True)
        manifest={'protocol_sha256':sha(path),'normal_bank_sha256':sha(bankpath),'normal_fit_report_sha256':sha(OUT/(NAME+'-normal-fit.json')),
                  'source_pt_sha256':sha(SOURCE),'source_onnx_sha256':sha(SOURCE_ONNX),'source_metadata_sha256':sha(SOURCE_ONNX.with_suffix('.onnx.json')),
                  'initial_export_manifest_sha256':sha(initialpath),'diagnostic_sha256':sha(targets[-1]),'candidates':candidates}
        write_new(receipts[0],manifest)
        print(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
