"""Create an additive reviewed-source proposal; never mutate app configuration."""
from pathlib import Path,PurePosixPath
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    report_path=ROOT/'output/evaluation/construction-finishes-tile/report.json'
    report=json.loads(report_path.read_text(encoding='utf-8'))
    serving_path=ROOT/'output/evaluation/construction-finishes-tile/serving-audit/report.json'
    serving=json.loads(serving_path.read_text(encoding='utf-8'))
    metadata=ROOT/'models/construction/finishes-tile/model.json'
    if sha(metadata)!=report['model_metadata_sha256']:raise RuntimeError('Actual measured tile model differs')
    data=ROOT/'datasets/construction/finishes-expansion/mvtec-tile'
    manifest=json.loads((data/'asset-manifest.json').read_text(encoding='utf-8'))
    rows=[]
    for label,metrics in report['by_defect_type'].items():
        first=next(r for r in manifest['records'] if r['partition']=='test' and r['defect_type']==label)
        source=data/'original'/str(PurePosixPath(first['source_path']).relative_to('MVTec-AD/tile'))
        if sha(source)!=first['sha256']:raise RuntimeError('Original tile evidence differs')
        comparison=ROOT/'output/evaluation/construction-finishes-tile'/(label+'-raw-gt-model.jpg')
        rows.append({'id':'tile-photo-first-'+label,'purpose':'photo_evidence','media_type':'still_photo',
                     'label':'Tile PHOTO research · '+label+' · first fixed source case',
                     'source':source.relative_to(ROOT).as_posix(),'source_sha256':sha(source),
                     'publisher_mask':first['mask_source_path'],'publisher_mask_sha256':first['mask_sha256'],
                     'comparison_image':comparison.relative_to(ROOT).as_posix(),'comparison_image_sha256':sha(comparison),
                     'model_ref':'damage:tile_surface_anomaly_research','score_scope':'Generic tile surface anomaly, not semantic '+label+' classification',
                     'aggregate_defect_type_metrics':metrics,
                     'actual_serving_defect_type_metrics':serving['by_defect_type'][label],
                     'profile_options':{'mode':'damage','damage_variant':'tile_surface_anomaly_research','stage':'finishes','engine':'onnxruntime',
                                        'openvino_device':'CPU','cpu_threads':1,'mask_rendering':'masks','exposure_policy':'off',
                                        'damage_classes':['tile_surface_anomaly'],'confirm_frames':1}})
    spec={'file':'finishes-tile/model.json','sha256':sha(metadata),'decoder':'tile_patch_bank','input_size':256,
          'classes':['tile_surface_anomaly'],'capabilities':['tile_surface_anomaly'],
          'source':'https://www.mvtec.com/research-teaching/datasets/mvtec-ad','display_name':'Tile surface anomaly · manufacturing PHOTO research',
          'license':report['model']['license'],'diagnostic_only':True,'exposure_policy':'off',
          'validation_scope':'All 117 original tile test photos after 48 normal fitting and 10 normal calibration photos. Crack pixel precision31.14%, recall59.71%, IoU25.73%; 0/17 derived crack regions at IoU.50; 3/33 normal photos flagged. Installed flooring and video are unvalidated.',
          'wrapper':'edge_runtime.profiles.construction_tile_adapter.TileConstructionModel',
          'wrapper_sha256':sha(ROOT/'edge_runtime/profiles/construction_tile_adapter.py'),
          'fitted_source_sha256':sha(ROOT/'edge_runtime/profiles/construction_tile.py')}
    pending=[
        ('finish_wall_moisture_content','Wall material moisture content','Requires calibrated material-appropriate moisture measurement. An RGB wet-spot candidate is not percent water or hidden-leak ground truth.'),
        ('finish_room_relative_humidity','Room relative humidity','Requires a temperature/RH sensor with units, calibration, time and location. It is not inferred from wall color.'),
        ('finish_paint_peeling','Peeling or flaking wall paint','Material-loss detection does not identify the coating or substrate. Real mask-data leads were found; no paint-specific local model is fitted yet.'),
        ('finish_paint_bubbles','Paint blister or bubble appearance','Separate paint-mask supervision is required. A generic publisher bulge label is not verified paint blistering.'),
        ('finish_paint_coverage','Paint coverage/run/color mismatch','Requires an approved reference and controlled illumination. No verified trained detector or suitable dataset was acquired.'),
        ('finish_ceramic_crack','Cracked installed ceramic','The tile anomaly specialist is related research only:0/17 derived manufacturing crack regions at IoU.50. Decorative veins and installed grout are unvalidated.'),
        ('finish_joint_alignment','Tile joint gap, skew or shift','Requires calibrated geometry on the same plane and an approved grid/nominal gap. No installed-video ground truth or model is verified.'),
        ('finish_surface_flatness','Surface flatness or tile lippage','Requires measured depth, a straightedge or a scan reference. A two-dimensional image homography does not measure height.'),
        ('finish_installation_fault','Incorrect or missing installation detail','Requires a project-specific design, required component states and real incorrect examples. Object presence alone is not installation acceptance.')]
    proposal={'schema_version':1,'status':'PROPOSAL ONLY; parent owns dispatch/config/presets; no appdefault changed',
              'document':'docs/construction-finishes-expansion.md','preserves':'All eight existing whole-video outputs/proposals and old model policies remain unchanged',
              'model_variants':{'tile_surface_anomaly_research':spec},
              'category_proposals':[{'id':'tile_surface_anomaly','label':'Visible tile surface texture anomaly','model_status':'measured_weak',
                                      'model_ref':'damage:tile_surface_anomaly_research','condition':'An unusual surface pattern relative to normal manufacturing tiles; defect subtype is unclassified',
                                      'why_it_matters':'The candidate area needs close inspection. It may be material loss, a crack, contamination or an unusual pattern.',
                                      'next_check':'Confirm the actual tile material and defect. This model cannot check the installed grid, lippage, adhesion or installation cause.',
                                      'metrics':report['by_defect_type'],'image_auroc':report['image_auroc'],'video_evidence':[],
                                      'photo_evidence':[{'case_id':r['id'],'comparison_image':r['comparison_image']} for r in rows]}]
                                   +[{'id':i,'label':label,'model_status':'unsupported','validation_scope':note,'video_evidence':[]} for i,label,note in pending],
              'existing_capability_notes':{'surface_wetspot':'Retain existing weak photo metrics. Visible appearance only; no humidity quantity or active-leak determination.',
                                           'surface_efflorescence':'Retain existing weak photo metrics. Salt-deposit appearance; current moisture is not measured.',
                                           'surface_spalling':'Broad surface material loss only. Do not rename it paint peeling or concrete cover loss without substrate evidence.'},
              'photo_presets':rows,'actual_report':report_path.relative_to(ROOT).as_posix(),'actual_report_sha256':sha(report_path),
              'actual_serving_audit':{'path':serving_path.relative_to(ROOT).as_posix(),'sha256':sha(serving_path),
                                      'changed_pixels_vs_original_raw_threshold':serving['changed_pixels'],
                                      'scope':'All 117 actual wrapper inferences verified. Two gray-stroke GT pixels removed by the original backend native component<4 filter; crack/normal decisions and all derived-region counts unchanged. Original raw-mask report remains immutable.',
                                      'by_defect_type':serving['by_defect_type']},
              'integration_requirements':['Root must add the tile_patch_bank dispatcher to the separate wrapper before exposing this variant',
                                          'ConstructionProfile needs an issue-only review label and plain-language guidance for tile_surface_anomaly',
                                          'No confidence/class-cut override. The frozen raw-distance operating point stays unchanged. Score is uncalibrated distance/(distance+threshold), not probability.',
                                          'CPU ONNX Runtime only; no GPU or FPS claim. Masks and boxes preserve the same fitted decision.',
                                          'The first normal control photo and all failed crack metrics remain visible; no automatic promotion.']}
    output=ROOT/'output/evaluation/construction-finishes-integration-proposal.json'
    output.write_text(json.dumps(proposal,indent=2)+'\n',encoding='utf-8')
    print(str(output),sha(output),flush=True)


if __name__=='__main__':main()
