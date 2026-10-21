"""TRAIN/FIT-only header and source-hash audit; no pixel decode or NN."""
from pathlib import Path
import argparse,hashlib,json,struct

ROOT=Path(__file__).resolve().parents[1]
SPLIT_SHA='ecee83db4becdac3133a2d4c0950ef383c511d143b5509aa8c1fdf61d445a47c'

def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()

def header_size(path):
    with path.open('rb') as stream:
        signature=stream.read(24)
        if signature[:8]==b'\x89PNG\r\n\x1a\n':
            if len(signature)!=24 or signature[12:16]!=b'IHDR':raise ValueError('Invalid PNG header')
            width,height=struct.unpack('>II',signature[16:24]);return width,height,'PNG IHDR'
        stream.seek(0)
        if stream.read(2)!=b'\xff\xd8':raise ValueError('Unsupported image header')
        for _ in range(1000):
            if stream.read(1)!=b'\xff':raise ValueError('Malformed JPEG marker')
            marker=stream.read(1)
            while marker==b'\xff':marker=stream.read(1)
            if not marker:raise ValueError('Truncated JPEG header')
            code=marker[0]
            if code in {0xd9,0xda}:raise ValueError('JPEG has no pre-scan frame header')
            if code==0x01 or 0xd0<=code<=0xd8:continue
            raw=stream.read(2)
            if len(raw)!=2:raise ValueError('Truncated JPEG segment')
            length=struct.unpack('>H',raw)[0]
            if length<2:raise ValueError('Invalid JPEG segment length')
            if code in {0xc0,0xc1,0xc2,0xc3,0xc5,0xc6,0xc7,0xc9,0xca,0xcb,0xcd,0xce,0xcf}:
                raw=stream.read(5)
                if len(raw)!=5:raise ValueError('Truncated JPEG frame header')
                height,width=struct.unpack('>HH',raw[1:5]);return width,height,'JPEG SOF'
            stream.seek(length-2,1)
    raise ValueError('Bounded header scan exhausted')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata',type=Path,default=ROOT/'models/factory/spatial_unet_v10h_sdd2-step300.onnx.json')
    parser.add_argument('--output',type=Path,default=ROOT/'output/optimization/factory-round3/fit-admission-geometry-proposal.json')
    parser.add_argument('--markdown',type=Path,default=ROOT/'docs/factory-fit-admission-geometry-proposal.md')
    args=parser.parse_args();metadata=json.loads(args.metadata.read_text(encoding='utf-8'))
    splitpath=ROOT/'datasets/factory/KolektorSDD2/train-inner-split-before-fit.json'
    if sha(splitpath)!=SPLIT_SHA:raise ValueError('Protected SDD2 FIT split identity changed')
    split=json.loads(splitpath.read_text(encoding='utf-8'));gate=metadata['supported_height_width_ratio']
    sdd1=ROOT/'datasets/factory/KolektorSDD';sdd2=ROOT/'datasets/factory/KolektorSDD2'
    paths=[]
    for item in metadata['split_items']['train']:
        if item in metadata['split_items']['calibration'] or item in metadata['split_items']['test']:raise ValueError('SDD1 train metadata overlap')
        for path in sorted((sdd1/item).glob('*.jpg')):paths.append(('SDD1_TRAIN',path,None))
    if len(paths)!=239:raise ValueError('Expected239 original SDD1 TRAIN images only')
    if len(split['fit'])!=1865:raise ValueError('Expected1865 protected SDD2 inner-FIT images')
    for record in split['fit']:
        relative=Path(record['image']);path=(sdd2/relative).resolve()
        if relative.parts[0]!='train' or not path.is_relative_to((sdd2/'train').resolve()):raise ValueError('FIT path escapes official TRAIN')
        paths.append(('SDD2_inner_FIT',path,record))
    identities=[];errors=[]
    for dataset,path,record in paths:
        width,height,method=header_size(path)
        if min(width,height)<=0:raise ValueError('Invalid source dimensions')
        identity=sha(path);ratio=height/width
        matched=None if record is None else identity==record['image_sha256']
        if matched is False:errors.append(str(path.relative_to(ROOT))+' differs from frozen FIT image hash')
        if record is not None and [height,width]!=record['annotation_shape']:errors.append(str(path.relative_to(ROOT))+' header differs from saved annotation geometry')
        identities.append({'dataset':dataset,'image':path.relative_to(ROOT).as_posix(),'sha256':identity,'width':width,'height':height,
                           'height_width_ratio':ratio,'header_reader':method,'outside_preserved_gate':not gate[0]<=ratio<=gate[1],
                           **({'frozen_fit_image_sha256':record['image_sha256'],'source_hash_matches':matched,'defective_from_frozen_metadata':record['defective']} if record is not None else {})})
    summaries={}
    for dataset in ['SDD1_TRAIN','SDD2_inner_FIT','combined_fitting_sources']:
        rows=[r for r in identities if dataset=='combined_fitting_sources' or r['dataset']==dataset]
        outside=sum(r['outside_preserved_gate'] for r in rows)
        summaries[dataset]={'images':len(rows),'outside_gate_images':outside,'outside_gate_fraction':outside/len(rows),
            'min_ratio':min(r['height_width_ratio'] for r in rows),'max_ratio':max(r['height_width_ratio'] for r in rows),
            'min_width':min(r['width'] for r in rows),'max_width':max(r['width'] for r in rows),
            'min_height':min(r['height'] for r in rows),'max_height':max(r['height'] for r in rows)}
    combined=summaries['combined_fitting_sources'];observed=[combined['min_ratio'],combined['max_ratio']]
    proposed=[observed[0]*.98,observed[1]*1.02]
    result={'scope':'FIT/TRAIN-only source image header+byte hash audit; no image decode, mask read, calibration pixels, official ZIP/test contents or NN.',
        'passed_source_identity':not errors,'errors':errors,'model_metadata_path':str(args.metadata.resolve()),'model_metadata_sha256':sha(args.metadata),
        'preserved_height_width_ratio_gate':gate,'sdd2_inner_split_sha256':SPLIT_SHA,
        'source_selection':'Exactly239 original SDD1 TRAIN JPGs from protected30items and1865 exact frozen SDD2 inner-FIT paths; calibration and official-test files excluded.',
        'summaries':summaries,'proposal_only':{'observed_fit_ratio_envelope':observed,'candidate_ratio_envelope_with_2pct_tolerance':proposed,
            'rule':'Use combined observed fitting-image min ratio*0.98 and max ratio*1.02. Fixed bounded geometric allowance; no test-driven quantile/threshold selection.',
            'status':'PROPOSAL ONLY; no active metadata/profile/protocol/serving change',
            'justification':'All included appearances were already eligible fitting sources before this header audit. A2% multiplicative margin accommodates small acquisition/crop dimension changes; it is an engineering proposal, not an empirically validated accuracy tolerance.',
            'limits':['Training-dimension coverage is not physical domain, material, imaging-quality or defect-localization validation.',
                      'The continuous interval includes ratios between observed sources; data do not establish performance throughout that interval or under arbitrary stretching.',
                      'Keep original frozen policy/reports unchanged; any admission variant requires distinct pinned metadata and a pre-opening protocol, with abstention accounting reported separately.',
                      'Header/source identity audit cannot qualify15FPS, blind-test generalization, product rejection or acceptance.']},
        'images':identities}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    lines=['# Fitting-source aspect-ratio admission proposal','',result['scope'],'',
       f"Preserved ratio gate: **{gate}**. Source identity checks: **{'PASS' if not errors else 'FAIL'}**. No serving policy changed.",'',
       '| Fitting domain | Images | Width range | Height range | Observed H/W | Outside preserved gate |','|---|---:|---|---|---|---|']
    for name,s in summaries.items():lines.append(f"| {name} | {s['images']} | {s['min_width']}–{s['max_width']} | {s['min_height']}–{s['max_height']} | {s['min_ratio']:.8f}–{s['max_ratio']:.8f} | {s['outside_gate_images']} ({100*s['outside_gate_fraction']:.2f}%) |")
    lines.extend(['',f"Proposed separate TRAIN-derived geometric envelope: **[{proposed[0]:.10f}, {proposed[1]:.10f}]**, using observed min×0.98/max×1.02. This rule uses fitting data only; the2% margin is a bounded engineering allowance, not a measured accuracy result.",'',
       result['proposal_only']['justification'],'',*['- '+text for text in result['proposal_only']['limits']],'',
       'No official test archive, calibration images, annotations or image pixels were read. PNG IHDR/JPEG SOF headers supplied dimensions; complete source-file bytes were hashed. SDD2 hashes agree with the protected fit manifest when the identity check passes. The raw proposal retains every source path/hash/dimension and its current-gate result.','',
       f"[Machine-readable identities and proposal]({args.output.resolve().as_posix()})",'',
       'Any later admission variant must retain the same frozen NN/thresholds while naming its changed gate separately. The preserved policy remains the baseline; a broader geometric gate cannot be credited as improved learned defect localization without actual spatial evaluation.'])
    if errors:lines.extend(['','## Identity failures','',*['- '+e for e in errors]])
    args.markdown.parent.mkdir(parents=True,exist_ok=True);args.markdown.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'source_identity_passed':not errors,'summaries':summaries,'proposal':proposed,'json':str(args.output),'markdown':str(args.markdown)},indent=2))

if __name__=='__main__':main()
