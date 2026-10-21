"""Create an additive, pinned gate-only copy of a future frozen D winner.

No model execution, pixel loading, calibration, threshold selection or test IO.
Use only after the coordinator authorizes this distinct admission experiment.
"""
from __future__ import annotations
import argparse,hashlib,json,math,os
from pathlib import Path,PurePosixPath

GATE=[2.4206,3.4757608695652173]
PRESERVED_GATE=[2.3,2.7]
VARIANT='orientation-multiscaleD-fit-admission-v1'

def digest(raw):return hashlib.sha256(raw).hexdigest()
def encode(value):return (json.dumps(value,indent=2,sort_keys=True)+'\n').encode('utf-8')
def pinned_read(path,expected,label):
    raw=Path(path).read_bytes()
    if digest(raw)!=expected:raise ValueError(label+' differs from required SHA256 identity')
    return raw

def validate_proposal(proposal,parent_metadata):
    if proposal.get('passed_source_identity') is not True or proposal.get('errors'):raise ValueError('FIT proposal source identity did not pass')
    if proposal.get('preserved_height_width_ratio_gate')!=PRESERVED_GATE:raise ValueError('FIT proposal preserved gate changed')
    training=parent_metadata['training_protocol']
    if proposal['sdd2_inner_split_sha256']!=training['sdd2_inner_split_sha256']:raise ValueError('Parent training and FIT proposal use different SDD2 splits')
    images=proposal['images'];counts={'SDD1_TRAIN':0,'SDD2_inner_FIT':0};paths=set();ratios=[]
    fitting=set(training['split_items']['train']);reserved=set(training['split_items']['calibration'])|set(training['split_items']['test'])
    if fitting&reserved:raise ValueError('Parent train items overlap reserved items')
    for row in images:
        name=row['image'];parts=PurePosixPath(name).parts;dataset=row['dataset']
        if name in paths:raise ValueError('Duplicate source path in FIT proposal')
        paths.add(name)
        if dataset=='SDD1_TRAIN':
            if len(parts)!=5 or parts[:3]!=('datasets','factory','KolektorSDD') or parts[3] not in fitting or parts[3] in reserved:raise ValueError('FIT proposal SDD1 source is outside parent fitting items')
        elif dataset=='SDD2_inner_FIT':
            if len(parts)!=5 or parts[:4]!=('datasets','factory','KolektorSDD2','train'):raise ValueError('FIT proposal SDD2 source is outside official TRAIN')
            if row.get('source_hash_matches') is not True or row['sha256']!=row['frozen_fit_image_sha256']:raise ValueError('FIT proposal contains a failed frozen source hash')
        else:raise ValueError('FIT proposal contains an unknown fitting domain')
        counts[dataset]+=1
        width,height=row['width'],row['height']
        if not all(isinstance(v,int) and not isinstance(v,bool) and v>0 for v in [width,height]):raise ValueError('FIT proposal dimensions invalid')
        ratio=height/width
        if not math.isfinite(row['height_width_ratio']) or abs(ratio-row['height_width_ratio'])>1e-12:raise ValueError('FIT proposal ratio differs from recorded dimensions')
        ratios.append(ratio)
    if counts!={'SDD1_TRAIN':239,'SDD2_inner_FIT':1865}:raise ValueError('FIT proposal source membership counts changed')
    envelope=[min(ratios)*.98,max(ratios)*1.02]
    declared=proposal['proposal_only']['candidate_ratio_envelope_with_2pct_tolerance']
    if len(declared)!=2 or any(abs(a-b)>1e-12 for a,b in zip(envelope,GATE)) or any(abs(a-b)>1e-12 for a,b in zip(declared,GATE)):
        raise ValueError('FIT proposal does not reproduce the fixed min/max plus2pct gate')

def create_variant(freeze_path,expected_freeze_sha,proposal_path,expected_proposal_sha,output_model,output_freeze):
    freeze_path=Path(freeze_path).resolve();proposal_path=Path(proposal_path).resolve()
    freeze=json.loads(pinned_read(freeze_path,expected_freeze_sha,'Parent freeze'))
    proposal=json.loads(pinned_read(proposal_path,expected_proposal_sha,'FIT proposal'))
    if freeze.get('attempt')!=4 or freeze.get('variant')!='orientation-multiscaleD':raise ValueError('Only a frozen attempt4 orientation-multiscaleD parent is accepted')
    winner=freeze['winner'];source=Path(winner['model_path']).resolve();source_metadata=source.with_suffix('.onnx.json')
    model_bytes=pinned_read(source,winner['model_sha256'],'Frozen parent NN')
    metadata_bytes=pinned_read(source_metadata,winner['metadata_sha256'],'Frozen parent metadata')
    metadata=json.loads(metadata_bytes)
    if metadata['model_sha256']!=winner['model_sha256']:raise ValueError('Parent metadata NN identity differs from winner')
    training=metadata.get('training_protocol',{})
    if training.get('name')!='spatial_round3f_orientation_multiscale' or training.get('fine_steps')!=900 or training.get('save_at')!=[300,600,900] or winner.get('step') not in [300,600,900]:
        raise ValueError('Parent is not a planned frozen D checkpoint')
    policy=(metadata['pixel_threshold'],metadata['minimum_component_pixels'],metadata.get('hysteresis_growth_threshold'))
    point=winner['point']
    if policy!=(point['high'],point['area'],point['growth']):raise ValueError('Parent seed/area/growth differs from frozen selected policy')
    if metadata.get('supported_height_width_ratio')!=PRESERVED_GATE:raise ValueError('Parent admission gate is not the preserved2.3–2.7 policy')
    if 'admission_variant' in metadata:raise ValueError('Refusing to derive recursively from an admission variant')
    validate_proposal(proposal,metadata)
    output_model=Path(output_model).resolve();output_freeze=Path(output_freeze).resolve();output_metadata=output_model.with_suffix('.onnx.json')
    if output_model.suffix!='.onnx':raise ValueError('Destination model must have .onnx suffix')
    outputs=[output_model,output_metadata,output_freeze]
    inputs={freeze_path,proposal_path,source,source_metadata}
    if len(set(outputs))!=3 or any(path in inputs for path in outputs):raise ValueError('Additive outputs must be distinct and cannot overwrite source identities')
    if any(path.exists() or path.is_symlink() for path in outputs):raise FileExistsError('Refusing to overwrite any existing admission output')
    provenance={'variant':VARIANT,'parent_freeze_sha256':expected_freeze_sha,'parent_model_sha256':winner['model_sha256'],
                'parent_metadata_sha256':winner['metadata_sha256'],'fit_proposal_sha256':expected_proposal_sha,
                'fit_proposal_path':str(proposal_path),'sdd2_inner_split_sha256':proposal['sdd2_inner_split_sha256'],
                'rule':'FIT-only combined minH/W*0.98,maxH/W*1.02; fixed gate2.4206–3.4757608695652173',
                'changed_runtime_fields':['supported_height_width_ratio'],
                'scope':'Separate geometric admission variant only. Identical NN bytes and selected seed/growth/area. No accuracy/speed/physical-domain validation credit; original parent remains preserved.'}
    child={**metadata,'supported_height_width_ratio':list(GATE),'admission_variant':provenance}
    if {key for key in set(child)|set(metadata) if child.get(key)!=metadata.get(key)}!={'supported_height_width_ratio','admission_variant'}:raise ValueError('Unexpected metadata transformation')
    child_bytes=encode(child);child_sha=digest(child_bytes)
    frozen={'schema_version':1,'attempt':4,'variant':VARIANT,'parent_freeze_path':str(freeze_path),
            'parent_freeze_sha256':expected_freeze_sha,'fit_proposal_sha256':expected_proposal_sha,
            'winner':{**winner,'model_path':str(output_model),'model_sha256':digest(model_bytes),'metadata_sha256':child_sha},
            'preserved_parent_winner':winner,'transformation':provenance,
            'preserved_gate':PRESERVED_GATE,'admission_gate':GATE,
            'no_model_execution':True,'no_calibration_or_test_opening':True,
            'inherited_point_metrics_scope':'Parent calibration values copied only as frozen-policy provenance; not new admission-variant calibration or validation.',
            'selection':'No new NN/threshold selection. This copy inherits the already frozen D winner; only explicit geometric admission differs.'}
    payloads=[(output_model,model_bytes),(output_metadata,child_bytes),(output_freeze,encode(frozen))]
    created=[]
    try:
        for path,raw in payloads:
            path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('xb') as stream:
                created.append(path);stream.write(raw);stream.flush();os.fsync(stream.fileno())
    except BaseException:
        for path in reversed(created):path.unlink(missing_ok=True)
        raise
    return {'model_path':str(output_model),'model_sha256':digest(model_bytes),'metadata_path':str(output_metadata),
            'metadata_sha256':child_sha,'freeze_path':str(output_freeze),'freeze_sha256':digest(encode(frozen)),
            'parent_freeze_sha256':expected_freeze_sha,'admission_gate':GATE}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze',type=Path,required=True);parser.add_argument('--freeze-sha256',required=True)
    parser.add_argument('--fit-proposal',type=Path,required=True);parser.add_argument('--fit-proposal-sha256',required=True)
    parser.add_argument('--output-model',type=Path,required=True);parser.add_argument('--output-freeze',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(create_variant(args.freeze,args.freeze_sha256,args.fit_proposal,args.fit_proposal_sha256,args.output_model,args.output_freeze),indent=2))

if __name__=='__main__':main()
