"""Record human-agent visual judgments after inspecting all19 fixed pair sheets."""
from pathlib import Path
import hashlib,json,re
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'datasets/construction/evcivil-dec31-future'
UNCERTAIN={4,17,36,40,41,42,43,44,45,48,51,52,55,56,63,64,65,66,67,68,69,70,75,76,77,79,85,87,88,89,90,91}
FEATURES={27:'oblique branching discontinuities and scattered bright points',30:'broad upper-left diagonal border/illumination band and branching lines',
34:'very weak lines and an isolated bright point; low recoverable detail',35:'brighter scattered scratch-like/branching surface marks',
40:'a narrow roughly vertical central groove/line',41:'a thin transverse/horizontal line and different surrounding texture',
42:'oblique branches with bright clustered points toward the upper/left area',45:'weak sparse near-vertical features and isolated points',
46:'oblique branching marks with broader illuminated texture',49:'a weak vertical line with bright upper/right point',
50:'multiple long roughly vertical/branching furrows',52:'a largely black region with a narrow textured upper strip in the flagged frame46'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def seq(path):
    m=re.search(r'31_crack_(\d+)',path)
    return int(m[1]) if m else None
def main():
    p=OUT/'qa/gain-pair-review/index.json';index=json.loads(p.read_text());pairs=index['pairs']
    if len(pairs)!=109 or len(index['pages'])!=19:raise ValueError('Reviewed page/pair count changed')
    target=OUT/'qa/pair-review-v2.json'
    if target.exists():raise ValueError('Preserve prior visual review')
    reviewed=[];quarantine=set()
    for i,row in enumerate(pairs):
        a,b=seq(row['left']),seq(row['right']);uncertain=i in UNCERTAIN
        if uncertain:quarantine.update(x for x in (a,b)if x is not None)
        if row['kind']=='known_train':
            note='Dec31 narrow dark vertical feature versus brighter TRAIN143 transverse/oblique crack and different texture; no matching landmark layout confirmed. The raw and gained TRAIN references are two representations of the same known frame.'
        elif uncertain:
            note=f'Low contrast/noise and limited anchors prevent confident identity resolution: left {FEATURES[a]}; right {FEATURES[b]}. Possible changed framing/illumination cannot be ruled out.'
        else:
            note=f'Visible arrangements differ: left {FEATURES[a]}; right {FEATURES[b]}. No matching branch/landmark layout observed in this pair; this is not a physical-site independence finding.'
        page=OUT/index['pages'][i//6]
        reviewed.append({'pair_index':i,**row,'judgment':'uncertain'if uncertain else'different_visible_layout','note':note,
                         'review_sheet':str(page.relative_to(OUT)),'review_sheet_sha256':sha(page),'review_row':i%6})
    if quarantine!={30,34,40,45,49,50} or len(UNCERTAIN)!=32:raise ValueError('Conservative recommendation differs from inspected set')
    manifest_path=OUT/'asset-manifest.json';manifest=json.loads(manifest_path.read_text());sources=[]
    for r in manifest['sequence_summary']:
        ident=int(r['canonical_id'].split('_')[-1]);blocked=ident in quarantine
        sources.append({'sequence':r['sequence'],'canonical_id':r['canonical_id'],'selected_views':r['selected_exposures'],
            'recommendation':'quarantine_unresolved_similarity'if blocked else'eligible_for_coordinator_source_quality_review',
            'reason':'Unresolved low-detail cross-candidate identity flags; no automatic fitting acceptance.'if blocked else'No remaining specific unresolved pair-identity flag after review; real defect visibility, source quality and final acceptance still need coordinator review.',
            'physical_site_independence':'unknown','fit_included':False})
    value={'schema_version':1,'status':'INDEPENDENT RECOMMENDATION ONLY; no source accepted or fit modified','manifest_sha256':sha(manifest_path),'review_index_sha256':sha(p),'supersedes_count_only_v1_sha256':sha(OUT/'qa/pair-review-v1.json'),
        'source_audit_sha256':index['source_audit_sha256'],'all_fixed_pages_visually_inspected':19,'reviewed_pairs':len(reviewed),
        'judgment_counts':{'different_visible_layout':77,'uncertain':32,'same_damage_layout':0},'pairs':reviewed,'sources':sources,
        'quarantined_sequence_ids':sorted(quarantine),'recommended_quality_review_sequences':14,'recommended_quality_review_views':sum(r['selected_views'] for r in sources if r['recommendation']=='eligible_for_coordinator_source_quality_review'),
        'quarantined_sequences':6,'quarantined_views':sum(r['selected_views'] for r in sources if r['recommendation']=='quarantine_unresolved_similarity'),'no_neural_inference':True,'heldout_or_calibration_pixels_opened':False,
        'limitations':'No confirmed duplicate is not proof of independent physical defects/sites. Publisher boxes in very dark camera images may have weak/no recoverable visible cue. Source52 flagged frame46 is largely black/occluded and specifically needs quality review. All repeated views remain correlated TRAIN-only; no new evaluation credit.'}
    target.write_text(json.dumps(value,indent=2)+'\n',encoding='utf8')
    doc=ROOT/'docs/construction-evcivil-dec31-pair-review.md'
    lines=['# Dec31 pair review: conservative future-data recommendation','',
      'All19 fixed shared-gain pair sheets were visually inspected:109 entries (107 cross-candidate pairs and two raw/gained representations of one known-TRAIN comparison). No NN ran and no CAL/heldout pixels were opened. This is a separate additive review; acquisition/source/view manifests remain unchanged. Version2 corrects view totals from the original per-sequence manifest (source45 has16 selected views, not24); all109 judgments and the six-source quarantine set are unchanged. Version1 remains preserved.','',
      '**77 entries show different visible layouts;32 remain uncertain; zero confirmed matching damage layouts.** Keep Dec31 IDs30/34/40/45/49/50 quarantined pending closer coordinator review (six sequences/136 views). The other14 sequences/274 proposed views can proceed to coordinator raw/GT quality review; none is automatically accepted into fitting.','',
      'Uncertain pairs have weak similar vertical features/glints and insufficient stable anchors; altered crop/pose/illumination cannot be excluded. This is an unresolved specific identity flag, not proof of duplication. A different visible arrangement does not prove different physical defects/sites. Night images remain noisy after the exact shared gain; source52 frame46 has a large black/occluding region and a thin textured upper strip, so useful damage visibility requires separate quality judgment. No threshold or sample choice was adjusted from these results.','',
      f'[Full immutable review JSON]({target.as_posix()}), SHA256 `{sha(target)}`. [Source acquisition report]({(ROOT/"docs/construction-evcivil-dec31-future-data.md").as_posix()}).','',
      '| Pair | Judgment | Left / right | Fixed reviewed page |','|---:|---|---|---|']
    for r in reviewed:lines.append(f"| {r['pair_index']} | {r['judgment']} | {Path(r['left']).name} / {Path(r['right']).name} | [Page, row{r['review_row']+1}]({(OUT/r['review_sheet']).as_posix()}) |")
    doc.write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(json.dumps({'review_sha256':sha(target),'judgments':value['judgment_counts'],'quarantine':sorted(quarantine)},indent=2))
if __name__=='__main__':main()
