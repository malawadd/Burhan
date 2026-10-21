"""Additive agent visual-review ledger; never mutates acquired or active datasets."""
from pathlib import Path
import hashlib,json

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'datasets/construction/evcivil-spall-future-r4'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    original=BASE/'asset-manifest.json';protocol=BASE/'protocol-before-acquisition.json'
    audit=BASE/'qa/duplicate-audit.json';index=BASE/'qa/all-similarity-pairs-index.json'
    expected='e98ecea5e66da8cc4ef3c09db0ff5f46320d110b09893d870ac9eeca250f8e26'
    if sha(original)!=expected:raise RuntimeError('Original acquisition identity changed')
    data=json.loads(original.read_text());plan=json.loads(protocol.read_text());pairs=json.loads(index.read_text())
    if len(pairs)!=121 or [p['pair_id'] for p in pairs]!=list(range(121)):raise RuntimeError('Visual review membership changed')
    if any(s['canonical_id'] in plan['reserved_canonical_ids'] for s in data['sequence_summary']):raise RuntimeError('Reserved canonical source present')
    notes={'spalling_11':'Concrete edge loss, uneven lighting and partial frame-boundary targets.',
           'spalling_6':'Small concrete corner/surface-loss patches with moving-camera blur and partial views.',
           'spalling_16':'Large detached concrete surface region and smaller patches; crack annotations retained.',
           'spalling_8':'Concrete narrow horizontal bands and vertical loss; strong flashlight gradient.',
           'spalling_9':'Low-contrast concrete vertical surface loss; only28 qualified exposures,24 sampled.',
           'spalling_13':'Narrow curved concrete surface-loss region under variable lighting.',
           'spalling_43':'Broad aggregate-exposing concrete surface loss; several regions across camera motion.',
           'spalling_38':'Shallow coating/bulge-like patch with cracks; visible concrete-cover loss not established.',
           'spalling_46':'Multiple small concrete surface-loss patches and motion blur.',
           'spalling_24':'Missing render over visible brickwork; original spalling label retained, different substrate.',
           'spalling_40':'Large corner/pillar concrete loss beside a dark vertical object.',
           'spalling_27':'Paving and lower-wall damage; original spalling label broader than concrete-cover loss.'}
    entries=[]
    for seq in data['sequence_summary']:
        name=seq['canonical_id'];material='concrete_surface' if name not in {'spalling_38','spalling_24','spalling_27'} else {'spalling_38':'ambiguous_coating_or_shallow_surface','spalling_24':'render_over_masonry','spalling_27':'paving_and_lower_wall_mixed'}[name]
        status='eligible_for_future_train_only' if material=='concrete_surface' else 'quarantine_ambiguous_damage' if name=='spalling_38' else 'separate_material_scope_not_concrete_spall'
        exposures=[r for r in data['exposures'] if r['canonical_id']==name and r['source_group']==seq['group']]
        for item in exposures:
            for field in ['image','label']:
                if sha(BASE/item[field])!=item[field+'_sha256']:raise RuntimeError('Preserved exposure changed')
        entries.append({'canonical_id':name,'source_capture_folder':seq['group'],'material_scope':material,'acceptance':status,
                        'selected_exposures':len(exposures),'qualified_original_exposures':seq['qualified_spall_exposures'],
                        'visual_note':notes[name],'publisher_label':'spalling','future_training_class':'spalling' if status=='eligible_for_future_train_only' else None,
                        'physical_project_identity':'UNKNOWN; capture folder and canonical ID are not independent-project guarantees',
                        'capture_split_rule':'Every original frame/renamed alias of this source stays in fitting only; no random frame split.',
                        'images':[{'path':r['image'],'sha256':r['image_sha256'],'label':r['label'],'label_sha256':r['label_sha256'],
                                   'original_frame_index':r['original_frame_index'],'timestamp_us':r['timestamp_us'],'source_pts_us':r['source_pts_us']} for r in exposures]})
    result={'version':'r4-additive-visual-acceptance-v1','created_date':'2026-10-20','status':'PROPOSED FUTURE DATA ONLY; no active import or training authorization',
            'links':{'acquisition_manifest_sha256':sha(original),'protocol_sha256':sha(protocol),'non_neural_audit_sha256':sha(audit),'review_pair_index_sha256':sha(index)},
            'source':'https://doi.org/10.11583/DTU.25924228','license':'CC BY4.0; credit ev-CIVIL authors and preserve publisher source metadata',
            'review_method':'Independent agent inspected fixed first/middle/last raw/publisher-box sheets for12 sources and every121 flagged raw/raw pair on16 pages. No NN, scores or heldout accuracy used.',
            'pair_review':[{'pair_id':p['pair_id'],'page':p['page'],'decision':'not_visually_same_damage_layout_in_reviewed_frames','caveat':'Does not establish different physical sites or unobserved frame independence.'} for p in pairs],
            'review_sheets':[{'path':f'qa/all-similarity-pairs-{p:02d}.jpg','sha256':sha(BASE/f'qa/all-similarity-pairs-{p:02d}.jpg')} for p in range(1,17)],
            'summary':{'reviewed_known_reference_pairs':86,'reviewed_cross_proposed_pairs':35,'confirmed_duplicate_pairs':0,
                       'future_concrete_train_sequences':9,'future_concrete_train_selected_exposures':216,'separate_material_sequences':2,'ambiguous_excluded_sequences':1},
            'provenance_limits':['No physical bridge/project IDs verified; source-folder/visual uniqueness does not prove independence.',
                'Pixel/exposure screening covers288 proposed exposures and966 known references, not all original footage or undisclosed publisher duplicates.',
                'No clear-sequence claim for exposures outside fixed reviewed frames. Original publisher labels are boxes, not pixel masks or damage mechanisms.',
                'Old exposed crack10/crack89/spall35 are excluded; capture-domain overlap makes them development-only for any future model trained on folders1/2.',
                'All current calibration6/heldout5/nightJan2 and their known canonical aliases remain reserved; existing failed models are not retroactively revalidated.'],
            'reserved_canonical_ids':plan['reserved_canonical_ids'],'whole_eval_capture_groups_excluded':plan['whole_eval_capture_groups_excluded'],
            'source_acceptance':entries}
    target=BASE/'acceptance-review-v1.json'
    if target.exists():raise RuntimeError('Version additive acceptance instead of overwriting')
    target.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(target);print(sha(target));print(json.dumps(result['summary']))

if __name__=='__main__':main()
