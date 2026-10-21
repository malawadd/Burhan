"""Record coordinator-approved source acceptance only; no copying, NN or fitting."""
from pathlib import Path
from collections import Counter
import hashlib,json,statistics
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'datasets/construction/evcivil-dec31-future'
MANIFEST_SHA='b549ed181a57ce847298614e5e83463743f9931a1881d70dd34a02fd0ad31bdc'
PAIR_SHA='50d9a1d8b7f857882572b54d9fd2f4fa4b853b72a86d159565a05af34e2fe021'
ACCEPTED={16,17,18,20,24,27,35,37,41,42,46,48,51}
UNCERTAIN={30,34,40,45,49,50}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def extent(values):return {'min':min(values),'median':statistics.median(values),'max':max(values)}
def main():
    manifest_path=OUT/'asset-manifest.json';pair_path=OUT/'qa/pair-review-v2.json'
    if sha(manifest_path)!=MANIFEST_SHA or sha(pair_path)!=PAIR_SHA:raise ValueError('Frozen acquisition/review differs')
    manifest=json.loads(manifest_path.read_text());review=json.loads(pair_path.read_text())
    if set(review['quarantined_sequence_ids'])!=UNCERTAIN:raise ValueError('Conservative unresolved endpoints differ')
    target=OUT/'acceptance-v1.json'
    if target.exists():raise ValueError('Preserve immutable acceptance history')
    protocol_path=OUT/'protocol-before-acquisition.json';protocol=json.loads(protocol_path.read_text())
    if sha(protocol_path)!=manifest['protocol_sha256']:raise ValueError('Original protocol identity changed')
    assets={s['sequence']:s for s in manifest['source_assets']};decisions=[];root_sheets=[]
    for s in manifest['sequence_summary']:
        number=int(s['canonical_id'].split('_')[-1]);accepted=number in ACCEPTED
        if not accepted and number not in UNCERTAIN|{52}:raise ValueError('Undeclared source disposition')
        sheet=OUT/'qa'/(s['sequence']+'-fixed-raw-input-gt.jpg')
        root_sheets.append({'sequence':s['sequence'],'path':sheet.relative_to(OUT).as_posix(),'sha256':sha(sheet),
                            'review_scope':'Coordinator visually reviewed fixed first/middle/last sampled exposure sheet; not every saved exposure'})
        originals=[]
        for member in assets[s['sequence']]['original_members']:
            if sha(OUT/member['path'])!=member['sha256']:raise ValueError('Original raw source identity changed')
            originals.append({'path':member['path'],'zip_member':member['name'],'sha256':member['sha256'],'crc32':member['crc32'],'bytes':member['bytes']})
        decisions.append({'sequence':s['sequence'],'canonical_id':s['canonical_id'],'status':'accepted_future_train_only'if accepted else'quarantined',
            'reason':('Explicit coordinator acceptance after all20 source sheets and completed pair-review recommendation; faint real traces, noisy/partial box scope retained.'if accepted else
                      'Unresolved specific cross-candidate identity/layout flags in pair-review-v2; do not include without new explicit disposition.'if number in UNCERTAIN else
                      'Coordinator quality rejection: largely occluded/black areas, high noise and insufficient independently usable visible damage in fixed samples despite publisher boxes.'),
            'selected_views':s['selected_exposures'],'all_qualified_exposures':s['qualified_crack_exposures'],'original_sources':originals,
            'physical_site_independence':'unknown','severity_or_structural_acceptance':False})
    rows=[r for r in manifest['exposures']if int(r['canonical_id'].split('_')[-1])in ACCEPTED]
    counts=Counter(b['label']for r in rows for b in r['boxes'])
    if len(rows)!=250 or counts!={'crack':405} or any(r['gain']<=1 for r in rows):raise ValueError('Expected250 real dark views/405 crack boxes changed')
    if {r['canonical_id']for r in rows}&set(protocol['reserved_canonical_ids']):raise ValueError('Protected canonical alias overlap')
    accepted_views=[]
    for row in rows:
        if sha(OUT/row['image'])!=row['image_sha256']:raise ValueError('Proposed saved exposure changed')
        h,w=row['source_shape'];source=assets[row['sequence']]['original_members']
        for box in row['boxes']:
            x,y,bw,bh=box['bbox']
            if box['class_id']!=1 or box['label']!='crack' or min(x,y)<0 or min(bw,bh)<2 or x+bw>w+1e-6 or y+bh>h+1e-6:raise ValueError('Publisher class/native bbox contract differs')
        accepted_views.append({**row,'camera_source_sha256':next(s['sha256']for s in source if s['path'].endswith('_frames.h5')),
                               'annotation_source_sha256':next(s['sha256']for s in source if s['path'].endswith('_label.npy'))})
    value={'schema_version':1,'experiment':'evcivil-dec31-future','status':'ACCEPTANCE ARTIFACT ONLY; no copy/YAML/model/fit authorized',
        'authorized_review':'Coordinator explicitly reviewed all20 fixed RAW/native gain/publisher-GT sequence sheets and selected exactly13 source sequences for future TRAIN only.',
        'source':'https://doi.org/10.11583/DTU.25924228','license':'CC BY4.0','acquisition_protocol_sha256':sha(protocol_path),
        'source_manifest_sha256':MANIFEST_SHA,'independent_pair_review_v2_sha256':PAIR_SHA,'root_reviewed_sequence_sheets':root_sheets,
        'source_decisions':decisions,'accepted_source_ids':sorted(ACCEPTED),'quarantined_source_ids':sorted(UNCERTAIN|{52}),
        'counts':{'accepted_sequences':13,'accepted_views':250,'publisher_boxes_by_class':dict(counts),'gain_applied_views':250,
                  'all_qualified_exposures_in_accepted_sources':sum(d['all_qualified_exposures']for d in decisions if d['status']=='accepted_future_train_only'),
                  'quarantined_sequences':7,'quarantined_proposed_views':sum(d['selected_views']for d in decisions if d['status']=='quarantined')},
        'accepted_native_p99':extent([r['p99']for r in rows]),'accepted_actual_gain':extent([r['gain']for r in rows]),
        'training_class_map':{'0':'spalling','1':'crack'},'accepted_views':accepted_views,
        'preserved':{'all_original_protocols_assets_and_source_selection':True,'current832_and_approved880_datasets_unchanged':True,
                     'calibration_and_heldout_sources_and_aliases_excluded':True,'no_nn_or_fit_or_copy':True},
        'limitations':['No physical-site or independent-defect guarantee from different visible layouts/global IDs.',
            'Publisher timestamped boxes are not mask GT or independently confirmed structural diagnosis. Faint/noisy camera traces remain challenging.',
            'All accepted exposures are correlated TRAIN-only data, with no new evaluation/accuracy credit. Unannotated exposures are not negatives.',
            'No confidence/ROI/dilation/exposure-rule tuning or synthetic darkness/damage. Original grayscale PNGs and raw H5/NPY untouched.',
            'This authorization ends at an acceptance artifact; a combined data version requires a separate pinned preparation/protocol and coordinator authorization before fitting.']}
    target.write_text(json.dumps(value,indent=2)+'\n',encoding='utf8')
    doc=ROOT/'docs/construction-evcivil-dec31-acceptance-v1.md'
    text=['# Dec31 source acceptance v1: future training artifact only','',
      'The coordinator reviewed all20 fixed RAW / actual shared-gain input / publisher-GT sequence sheets and explicitly accepted13 source sequences for a future TRAIN data version. This artifact authorizes **no copying, training-YAML changes, model changes or fit**. Original acquisition/protocol/raw assets and current832/approved880 datasets are unchanged.','',
      f'[Acceptance JSON]({target.as_posix()}), SHA256 `{sha(target)}`. It pins acquisition manifest `{MANIFEST_SHA}`, independent pair-review-v2 `{PAIR_SHA}`, the original acquisition protocol, all20 reviewed sheet hashes and original member SHA/CRC identities.','',
      '**Accepted:250 actual dark camera views,405 publisher crack boxes, zero spalling boxes,250 actual gain-applied views.** They come from398 total qualified exposures in accepted sources before the fixed uniform sampling. All existing CAL/heldout/exposed/current-TRAIN global-ID exclusions remain intact. Box geometry/class/file identity checks passed without NN or new CAL/heldout pixel reads.','',
      'The seven quarantined sources retain136 views from the six unresolved similarity endpoints30/34/40/45/49/50 plus24 source52 views rejected for quality:160 proposed views total. Source52 fixed samples contain largely black/occluded regions, noise and insufficient independently usable damage visibility despite publisher annotations.','',
      'Different visible arrangements do not prove different physical defects/sites. Faint real traces and incomplete publisher box scope remain; structural severity, physical crack width and Saudi-code compliance are not inferred. These are correlated TRAIN-only examples with no independent evaluation credit. A separate combined-data preparation and pinned future-fit protocol is still required.','',
      'For coordinator pair spot review: [gain page000]('+str((OUT/'qa/gain-pair-review/fixed-pairs-000.jpg').as_posix())+') rows1–2 are known-TRAIN flags0/1, row5 is uncertain pair4. [Gain page006]('+str((OUT/'qa/gain-pair-review/fixed-pairs-006.jpg').as_posix())+') row1 is uncertain pair36. All original raw/gain pair pages and judgments remain preserved.','',
      '| Source | Disposition | Selected views | Publisher crack boxes |','|---|---|---:|---:|']
    for d in decisions:
        n=sum(len(r['boxes'])for r in manifest['exposures']if r['sequence']==d['sequence'])
        text.append(f"| {d['sequence']} | {d['status']} | {d['selected_views']} | {n} |")
    doc.write_text('\n'.join(text)+'\n',encoding='utf8');print(json.dumps({'acceptance_sha256':sha(target),'counts':value['counts'],'p99':value['accepted_native_p99'],'gain':value['accepted_actual_gain']},indent=2))
if __name__=='__main__':main()
