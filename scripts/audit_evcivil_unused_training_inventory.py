"""Read-only publisher inventory proposal; no media downloads, decoding or NN."""
from pathlib import Path
from collections import defaultdict,Counter
import hashlib
import json
import re

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'datasets/construction'


def main():
    inventory_path=DATA/'evcivil/field-member-inventory.json'
    inventory=json.loads(inventory_path.read_text(encoding='utf-8'))
    protocol=json.loads((DATA/'evcivil-round3/protocol.json').read_text(encoding='utf-8'))
    continuation=json.loads((DATA/'evcivil-round3-continuation/protocol.json').read_text(encoding='utf-8'))
    reserved=set(protocol['global_identity_split'])|set(continuation['reserved_canonical_ids'])|{'crack_10','crack_89','spalling_35','crack_121'}
    reserved|={f'crack_{number}' for number in continuation['night_ids']}
    groups=defaultdict(list)
    for member in inventory:
        match=re.search(r'field_data_day/(dset_\d+)/((?:crack|spalling)_\d+)/',member['name'])
        if match:groups[(match[1],match[2])].append(member)
    rows=[]
    for (group,identity),members in sorted(groups.items()):
        camera=[row for row in members if row['name'].endswith('_frames.h5')]
        labels=[row for row in members if row['name'].endswith('_frame_label.npy')]
        if not labels:labels=[row for row in members if row['name'].endswith('_label.npy')]
        if group in {'dset_5','dset_6'}:status='forbidden_entire_eval_group'
        elif identity in reserved:status='excluded_existing_reserved_or_known_duplicate_id'
        elif not camera or not labels:status='incomplete_camera_label_pair'
        elif group in {'dset_3','dset_4'}:status='preferred_metadata_candidate'
        else:status='conditional_exposed_capture_group_not_allowed_current_protocol'
        chosen=camera+labels
        rows.append({'group':group,'canonical_id':identity,'class':identity.split('_')[0],
                     'status':status,'members':chosen,'compressed_bytes':sum(row['compressed_bytes'] for row in chosen),
                     'uncompressed_bytes':sum(row['bytes'] for row in chosen),
                     'annotation_scope':'Publisher label NPY present, but actual timestamp-qualified camera exposures and physical/near-duplicate independence remain unverified from directory metadata.'})
    summary={}
    for status in sorted({row['status'] for row in rows}):
        subset=[row for row in rows if row['status']==status]
        summary[status]={'sequences':len(subset),'classes':dict(Counter(row['class'] for row in subset)),
                         'compressed_bytes':sum(row['compressed_bytes'] for row in subset),
                         'uncompressed_bytes':sum(row['uncompressed_bytes'] for row in subset)}
    preferred=[row for row in rows if row['status']=='preferred_metadata_candidate']
    report={'source':'https://doi.org/10.11583/DTU.25924228','license':'CC BY4.0',
            'inventory_sha256':hashlib.sha256(inventory_path.read_bytes()).hexdigest(),
            'scope':'Read-only central-directory candidate proposal. No acquisition, active protocol change, model inference, new evaluation or near-duplicate clearance.',
            'reserved_canonical_ids':sorted(reserved),'summary':summary,'preferred_candidates':preferred,
            'all_day_sequence_inventory':rows,
            'important_limit':'Additional sequence IDs are not proven independent physical defects. Entire dset5/dset6 and Jan2/Nov18 eval sources stay excluded. dset1/dset2 are prohibited by the current protocol due exposed capture-group correlation; using their other IDs requires a distinct future experiment, untouched fresh eval groups and independent duplicate screening. No metadata-only promise of zero near duplicates.'}
    target=DATA/'evcivil/unused-train-day-inventory-proposal-round3.json'
    target.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'path':str(target),'summary':summary,'preferred':[(r['group'],r['canonical_id'],r['compressed_bytes']) for r in preferred]},indent=2))


if __name__=='__main__':main()
