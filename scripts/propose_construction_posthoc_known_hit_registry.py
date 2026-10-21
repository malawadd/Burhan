"""Read complete post-hoc artifacts and write a proposal only; no app mutation."""
from pathlib import Path
import copy,hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
OUT=ROOT/'output/evaluation/construction-posthoc-known-hit-videos'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    from edge_runtime.profiles.construction_model import DAMAGE_VARIANTS
    completed=OUT/'complete-posthoc-paired-summary.json'
    pairs=json.loads(completed.read_text());protocol_path=OUT/'two-posthoc-known-hit-protocol-before-forward.json';protocol=json.loads(protocol_path.read_text())
    execution=json.loads((OUT/'export-started.json').read_text())
    if execution['state']!='completed' or execution['protocol_sha256']!=sha(protocol_path) or execution['summary_sha256']!=sha(completed):raise RuntimeError('Require complete source-bound post-hoc exports')
    known=json.loads((ROOT/'output/evaluation/construction-last4-candidate/known-calibration-hits-metadata.json').read_text())
    wanted={row['annotation_id'] for row in known['matches']}
    specs={};rows=[]
    variant={'BEFORE':'evcivil_r4_real_lowlight_failed','AFTER':'evcivil_r4_last4_failed'}
    after=copy.deepcopy(DAMAGE_VARIANTS['evcivil_r4_real_lowlight_failed'])
    after.update(file=protocol['models']['AFTER']['path'].split('models/construction/')[-1],sha256=protocol['models']['AFTER']['sha256'],
                 display_name='Final epoch4 · failed camera calibration research',diagnostic_only=True,
                 validation_scope='All122 reused CAL: crack8/125 matches+75unmatched; spall5/38+5unmatched; thin1/62;5/10normalflags. Post-hoc known-hit videos are illustrations without independent accuracy credit. No promotion.')
    specs[variant['AFTER']]=after
    for tag,reports in pairs.items():
        for item in reports:
            path=ROOT/item['report'];video=ROOT/item['video']
            if sha(path)!=item['report_sha256'] or sha(video)!=item['video_sha256']:raise RuntimeError('Completed video/report bytes differ')
            report=json.loads(path.read_text());identity=protocol['models'][tag]
            if report['model']['sha256']!=identity['sha256'] or report['class_confidence']!=identity['class_confidence'] or report['selection_protocol_sha256']!=sha(protocol_path):raise RuntimeError('Report source/model/point differs')
            matched=[]
            for frame in report['frames']:
                expected={e['id']:e for e in frame['expected']}
                for prediction in frame['predictions']:
                    annotation=prediction.get('matched_annotation_id')
                    if annotation:
                        matched.append({'source_frame_index':frame['source_frame_index'],'video_time_s':frame['video_time_s'],'source_timestamp_us':frame['source_timestamp_us'],
                                        'annotation_id':annotation,'label':prediction['label'],'publisher_bbox':expected[annotation]['bbox'],
                                        'model_bbox':prediction['bbox'],'score':prediction['score'],'iou':prediction['iou'],'was_known_calibration_hit_before_video_export':annotation in wanted})
            name=report['sequence'];material='surface material loss' if name.startswith('spalling') else 'surface crack'
            title=f'POST-HOC known-hit {material} · '+('best epoch1' if tag=='BEFORE' else 'final epoch4')
            rows.append({'id':f'posthoc-{tag.lower()}-{name.replace("_", "-")}', 'title':title,'purpose':'failure_diagnostic',
                         'video':item['video'],'evidence_path':item['report'],'video_sha256':sha(video),'evidence_sha256':sha(path),
                         'source_preset_label':title+' · original lossless recording','source_input':report['source']['lossless_replay_input']['path'],
                         'options':{'mode':'damage','damage_variant':variant[tag],'exposure_policy':'low_light_gain','mask_rendering':'boxes','show_context_objects':False},
                         'issue_summary':f'Actual publisher-labelled {material}; source chosen POST-HOC for known CAL hits. Full recording retains every miss and unmatched prediction.',
                         'why_it_matters':'Visible surface damage needs material-specific close inspection; these failed research candidates do not qualify an installation or structure.',
                         'next_check':'Inspect the actual substrate and measure dimensions; use appropriate instruments to establish depth, moisture or structural severity.',
                         'selection_status':'POST-HOC KNOWN-HIT ILLUSTRATION; FAILED CALIBRATION; NO INDEPENDENT ACCURACY CREDIT',
                         'independent_accuracy_credit':False,'summary':report['summary'],'all_matched_source_frames':matched,
                         'known_calibration_match_examples':[row for row in matched if row['was_known_calibration_hit_before_video_export']],
                         'cached_predictions_reused_without_nn':report['cached_predictions_reused_without_nn'],
                         'source_frames':len(report['frames']),'source_pts_scope':report['source']['timebase'],
                         'fixed_preview':report['fixed_preview']})
    result={'proposal_only':True,'no_shared_config_modified':True,'selection_protocol_sha256':sha(protocol_path),
            'foreground_scope':'These sources were selected AFTER CAL hits were known. They illustrate real localized damage and failures; they cannot establish independent improvement or override failed all122/old4case gates.',
            'new_damage_variants':specs,'reuse_existing_damage_variant':variant['BEFORE'],'recordings_and_source_presets':rows,
            'prior_eight_videos_and_metrics_unchanged':True,'no_new_wall_capability_claim':True}
    target=OUT/'integration-proposal.json'
    if target.exists():raise RuntimeError('Preserve completed post-hoc proposal')
    target.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(target,sha(target),flush=True)


if __name__=='__main__':main()
