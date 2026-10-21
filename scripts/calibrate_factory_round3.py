"""Calibrate round-three checkpoint/policy on original calibration items only."""
from pathlib import Path
import argparse,hashlib,json,sys,time
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2,numpy as np
from edge_runtime.profiles.factory_encoder import FactorySegmentation,seeded_growth_mask
from train_factory_surface import samples
from factory_mask_metrics import component_mask_statistics,at_area,aggregate,original_pixel_weights


def summarize(rows):
    value=aggregate(rows);positive=[row for row in rows if row['defective']]
    value['meaningful_iou_0_1_recall']=sum((row['pixel_iou'] or 0)>=.1 for row in positive)/max(1,len(positive))
    flagged=sum(row['flagged'] for row in rows)
    value['flagged_view_precision']=sum(row['flagged'] for row in positive)/max(1,flagged)
    value['meaningful_view_precision']=sum((row['pixel_iou'] or 0)>=.1 for row in positive)/max(1,flagged)
    value['coverage_50pct_recall']=sum((row['foreground_pixel_recall'] or 0)>=.5 for row in positive)/max(1,len(positive))
    value['worst_positive_iou']=min((row['pixel_iou'] or 0 for row in positive),default=0.)
    value['worst_positive_coverage']=min((row['foreground_pixel_recall'] or 0 for row in positive),default=0.)
    return value


def point_rank(row,attempt2=False):
    if attempt2:
        return (row['meaningful_iou_0_1_recall'],row['worst_positive_iou'],row['coverage_50pct_recall'],
                row['worst_positive_coverage'],row['micro_pixel_iou'],-row['normal_fp'],row['foreground_pixel_recall'])
    return row['meaningful_iou_0_1_recall'],-row['normal_fp'],row['micro_pixel_iou'],row['foreground_pixel_recall']


def eligible_final(row,attempt2=False):
    return row['normal_fp']<=1 and (not attempt2 or row['foreground_pixel_precision']>=.65)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant',choices=['shallow','context'],required=True)
    parser.add_argument('--attempt2',action='store_true',help='Use predeclared SDD2-rehearsal/FN-aware attempt2 protocol and weakest-positive rank')
    parser.add_argument('--attempt3',action='store_true',help='Full-resolution learned refinement; same fixed weakest-positive rank')
    parser.add_argument('--packed-refinement',action='store_true',help='Attempt3 variantB, packed2x2 pixel phases with half-grid learned operations')
    parser.add_argument('--phase-max-refinement',action='store_true',help='Attempt3 variantC: full-res phase training, explicit half-grid max-logit serving')
    parser.add_argument('--orientation-balanced',action='store_true',help='Attempt4 samephase-maxgraph with orientation/scale-balanced square training')
    parser.add_argument('--native-width-balanced',action='store_true',help='Attempt5 unchanged phase-max serving with native640x512 full-width crop training')
    parser.add_argument('--curvature-augmentation',action='store_true',help='Separate native full-width C200 continuation with matched TRAIN-only smooth curvature augmentation')
    parser.add_argument('--normal-density-complement',action='store_true',help='F frozen v10h plus TRAIN-normal shared-feature density fusion; nine candidate IDs, not SGD steps')
    parser.add_argument('--precision-corrected-density',action='store_true',help='F2 exact-half constants/fullgraph feature-stat context, separate immutable prefix')
    parser.add_argument('--steps',type=int,nargs='+',default=None,help='Defaults200/400/600; orientation-balancedD defaults to its protected300/600/900 save_at')
    parser.add_argument('--resume-cache',action='store_true',help='Reuse existing calibration-only scores and preserve completed reports')
    parser.add_argument('--contended-calibration',action='store_true',help='Accuracy only during a coordinated CPU fit; raw latency is not capacity evidence')
    parser.add_argument('--contention-note',default='Concurrent construction CPU fitting',help='Actual coordinated workload; provenance only, no selection effect')
    parser.add_argument('--cpu-threads',type=int,choices=[1,2],default=2)
    args=parser.parse_args();cv2.setNumThreads(args.cpu_threads)
    if args.attempt2 and args.attempt3:raise ValueError('Choose only one experiment')
    if args.packed_refinement and not args.attempt3:raise ValueError('Packed refinement requires attempt3')
    if args.phase_max_refinement and (not args.attempt3 or args.packed_refinement):raise ValueError('Phase-max requires attempt3 and no other refinement flag')
    if args.orientation_balanced and not args.phase_max_refinement:raise ValueError('Orientation-balanced uses phase-max serving')
    if args.native_width_balanced and (not args.phase_max_refinement or args.orientation_balanced):raise ValueError('Native-width uses phase-max serving and no orientation experiment flag')
    if args.curvature_augmentation and (not args.phase_max_refinement or args.orientation_balanced or args.native_width_balanced):raise ValueError('Curvature uses phase-max serving and its own separate native-training experiment flag')
    if args.normal_density_complement and (args.attempt2 or args.attempt3 or args.orientation_balanced or args.native_width_balanced or args.curvature_augmentation):raise ValueError('Density complement is a separate single-output experiment')
    if args.precision_corrected_density and not args.normal_density_complement:raise ValueError('Precision correction requires density complement')
    pin_sources=args.orientation_balanced or args.native_width_balanced or args.curvature_augmentation or args.normal_density_complement
    weakest=args.attempt2 or args.attempt3 or args.normal_density_complement
    if weakest and args.variant!='context':raise ValueError('Attempts2/3 use the predeclared context graph')
    directory=ROOT/'output/optimization/factory-round3';name='spatial_round3j_native_curvature' if args.curvature_augmentation else 'spatial_round3i_normal_density_f16' if args.precision_corrected_density else 'spatial_round3h_normal_density' if args.normal_density_complement else 'spatial_round3g_native_width' if args.native_width_balanced else 'spatial_round3f_orientation_multiscale' if args.orientation_balanced else 'spatial_round3e_phase_max_refine' if args.phase_max_refinement else 'spatial_round3d_packed_refine' if args.packed_refinement else 'spatial_round3c_fullres_refine' if args.attempt3 else 'spatial_round3b_context_sdd2_fn' if args.attempt2 else f'spatial_round3_{args.variant}'
    protocol=json.loads((directory/(name+'-protocol-before-fit.json')).read_text(encoding='utf-8'))
    args.steps=args.steps if args.steps is not None else protocol['save_at'] if pin_sources else [200,400,600]
    if pin_sources and any(step not in protocol['save_at'] for step in args.steps):raise ValueError('Calibration only supports this experiment protected save_at checkpoints')
    if args.curvature_augmentation:
        for relative,field in [('scripts/factory_curvature_sampling.py','sampler_sha256'),('scripts/train_factory_round3j.py','wrapper_sha256'),('scripts/train_factory_round3c.py','trainer_helper_sha256'),('scripts/train_factory_round3d.py','source_model_implementation_sha256')]:
            if hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()!=protocol[field]:raise ValueError('Curvature training implementation changed after protected protocol')
        fixture=directory/(name+'-pairing-fixtures.json')
        if hashlib.sha256(fixture.read_bytes()).hexdigest()!=protocol['pairing_fixtures_sha256'] or not json.loads(fixture.read_text(encoding='utf-8'))['all_checks_passed']:raise ValueError('Curvature pairing receipt changed/failed')
    if args.normal_density_complement:
        from factory_density_identity import validate_density_artifacts
        validate_density_artifacts(name=name)
        parity=json.loads((directory/(name+'-fitted-gpu-parity.json')).read_text(encoding='utf-8'))
        fit=json.loads((directory/(name+'-normal-fit.json')).read_text(encoding='utf-8'))
        bank_path=directory/(name+'-normal-bank.npz')
        if not parity['all_checks_passed'] or parity['protocol_sha256']!=hashlib.sha256((directory/(name+'-protocol-before-fit.json')).read_bytes()).hexdigest():raise ValueError('Fitted density/source GPU parity failed or changed')
        if hashlib.sha256(bank_path.read_bytes()).hexdigest()!=fit['bank_sha256']:raise ValueError('Normal bank changed before calibration')
        exports_path=directory/(name+'-fitted-export-manifest.json')
        exports=json.loads(exports_path.read_text(encoding='utf-8'))
        if parity['normal_bank_sha256']!=fit['bank_sha256'] or parity['fitted_export_manifest_sha256']!=hashlib.sha256(exports_path.read_bytes()).hexdigest():raise ValueError('Parity was not bound to this exact bank/candidate export')
        for candidate_export in exports['candidates']:
            if hashlib.sha256(Path(candidate_export['model_path']).read_bytes()).hexdigest()!=candidate_export['model_sha256']:raise ValueError('An actual GPU candidate changed after export/parity')
        guard_path=directory/(name+'-guard-clipping-audit.json')
        guard_audit=json.loads(guard_path.read_text(encoding='utf-8'))
        if guard_audit['bank_sha256']!=fit['bank_sha256']:raise ValueError('Guard diagnostics use a different fitted bank')
    original=json.loads((ROOT/'models/factory/spatial_unet_v10h_sdd2-step300.onnx.json').read_text(encoding='utf-8'))
    # The same rule applies across checkpoints and architectures. No selected
    # point or subsequent comparison is allowed to rewrite this protocol.
    selection={'written_before_scoring':True,'false_alarm_budget':'<=1/70 original normal calibration views',
               'seed_search':'For each high threshold choose eligible area maximizing any-mask localized recall, foreground recall, microIoU; keep top4 distinct high thresholds',
               'final_rank':'Meaningful per-view pixelIoU>=.1 recall, fewer normal FP, micro pixelIoU, foreground recall',
               'high_grid':protocol['raw_high_grid'],'area_grid':protocol['area_grid'],'growth_grid':protocol['growth_grid'],
               'comparison_images_opened':0,'quality_abstentions':'Raw calibration scores; complete-profile quality/speed gate follows before comparison'}
    selection['execution_scope']={'contended_calibration':args.contended_calibration,'opencv_threads':args.cpu_threads,
                                 'concurrent_workload':args.contention_note if args.contended_calibration else None,
                                 'raw_latency_capacity_evidence':not args.contended_calibration,
                                 'note':'Accuracy-only calibration during the recorded coordinated workload; exclusive GPU. Raw timing cannot qualify speed/capacity.' if args.contended_calibration else 'Project-quiescent calibration window; raw timing is still not a complete-runtime benchmark and ordinary host services remain active'}
    if weakest:
        selection.update(false_alarm_budget='<=1/70 original normal calibration views AND foreground precision>=.65',
                         seed_search=protocol['seed_search'],final_rank=protocol['selection_rank'],
                         seed_stage_eligibility='NormalFP<=1 only; precision>=.65 applies to final grown/no-growth candidates',
                         search_scope='Heuristic best-area-per-high then top6distincthighs; not an exhaustive optimum over all area/high/growth combinations',
                         official_sdd2_test_still_sealed=True)
    (directory/(name+'-calibration-protocol-before-scoring.json')).write_text(json.dumps(selection,indent=2),encoding='utf-8')
    report_path=directory/(name+'-calibration.json')
    reports=json.loads(report_path.read_text(encoding='utf-8'))['reports'] if args.resume_cache and report_path.exists() else []
    for step in args.steps:
        path=ROOT/'models/factory'/f'{name}-step{step}.onnx';started=time.perf_counter()
        checkpoint_sha=hashlib.sha256(path.read_bytes()).hexdigest()
        cache=[];times=[];cache_path=directory/f'{name}-step{step}-calibration-scoremaps.npz'
        saved=np.load(cache_path) if args.resume_cache and cache_path.exists() else None
        if saved is not None:
            cached_sha=str(saved['model_sha256']) if 'model_sha256' in saved.files else next((r['model_sha256'] for r in reports if r['step']==step),None)
            if cached_sha!=checkpoint_sha:
                raise ValueError('Calibration cache is not tied to this exact checkpoint hash')
        saved_scores=saved['scores'] if saved is not None else None
        saved_source_identities=None
        if pin_sources and saved is not None:
            if 'source_identities_json' not in saved.files:raise ValueError('Attempt4 cache lacks source-pixel/mask identity')
            saved_source_identities=json.loads(str(saved['source_identities_json']))
            if saved_scores.shape!=(80,*protocol['output_shape'][-2:]) or not np.isfinite(saved_scores).all():raise ValueError('Attempt4 cache has invalid score geometry/values')
        model=None if saved is not None else FactorySegmentation(path,output_logits=True)
        source_identities=[]
        for index,(image,truth,defective) in enumerate(samples(ROOT/'datasets/factory/KolektorSDD',original['split_items']['calibration'])):
            identity=None
            if pin_sources:
                mask_path=image.with_name(image.stem+'_label.bmp')
                identity={'image':str(image.relative_to(ROOT/'datasets/factory/KolektorSDD')),
                          'defective':bool(defective),
                          'image_sha256':hashlib.sha256(image.read_bytes()).hexdigest(),
                          'mask_sha256':hashlib.sha256(mask_path.read_bytes()).hexdigest(),
                          'native_image_shape':list(cv2.imread(str(image)).shape),'native_mask_shape':list(truth.shape)}
                source_identities.append(identity)
            if saved is not None:
                assert str(image.relative_to(ROOT/'datasets/factory/KolektorSDD'))==str(saved['images'][index]),'Cached calibration order mismatch'
                assert bool(defective)==bool(saved['defective'][index]),'Cached calibration target mismatch'
                if pin_sources:
                    if saved_source_identities[index]!=identity:raise ValueError('Calibration source pixels/mask/geometry changed since cache')
                scores=saved_scores[index]
            else:
                tick=time.perf_counter();scores,_,_=model.predict(cv2.imread(str(image)),.999,16);times.append((time.perf_counter()-tick)*1000)
            cache.append({'image':str(image.relative_to(ROOT/'datasets/factory/KolektorSDD')),'truth':truth,'defective':bool(defective),'scores':scores,
                          'projection_weights':original_pixel_weights(truth,scores.shape)})
        # Preserve every calibration score map for reproducible offline policy
        # audits without recompiling/rerunning a neural model. This cache never
        # includes comparison views and cannot serve as fitting data.
        if args.normal_density_complement and hashlib.sha256(json.dumps(source_identities,sort_keys=True).encode('utf-8')).hexdigest()!=guard_audit['calibration_source_identity_sha256']:
            raise ValueError('Calibration source changed since fixed guard diagnostics')
        if saved is None:
            temporary=cache_path.with_name(cache_path.stem+'.writing.npz')
            np.savez_compressed(temporary,
                            model_sha256=np.asarray(checkpoint_sha),
                            scores=np.stack([row['scores'] for row in cache]),
                            images=np.asarray([row['image'] for row in cache]),
                            defective=np.asarray([row['defective'] for row in cache]),
                            **({'source_identities_json':np.asarray(json.dumps(source_identities))} if pin_sources else {}))
            temporary.replace(cache_path)
        raw=[];raw_records={}
        for high in protocol['raw_high_grid']:
            per_area={area:[] for area in protocol['area_grid']}
            for row in cache:
                statistics=component_mask_statistics(row['scores'],high,row['truth'],row['projection_weights'])
                for area in protocol['area_grid']:
                    per_area[area].append({'defective':row['defective'],**at_area(statistics,area)})
            for area,rows in per_area.items():
                value={'high':high,'area':area,**summarize(rows)};raw.append(value);raw_records[(high,area)]=rows
        seed_choices=[]
        for high in protocol['raw_high_grid']:
            eligible=[r for r in raw if r['high']==high and r['normal_fp']<=1]
            if eligible:seed_choices.append(max(eligible,key=(lambda r:point_rank(r,True)) if weakest else (lambda r:(r['localized_view_recall'],r['foreground_pixel_recall'],r['micro_pixel_iou'],-r['area']))))
        seed_choices=sorted(seed_choices,key=(lambda r:point_rank(r,True)) if weakest else (lambda r:(r['localized_view_recall'],r['foreground_pixel_recall'],r['micro_pixel_iou'])),reverse=True)[:6 if weakest else 4]
        grown=[];grown_records={}
        for seed in seed_choices:
            high,area=seed['high'],seed['area']
            for low in protocol['growth_grid']:
                if low is not None and low>high:continue
                rows=[]
                for row in cache:
                    binary=seeded_growth_mask(row['scores'],high,area,low)
                    values=at_area(component_mask_statistics(binary.astype(np.float32),.5,row['truth'],row['projection_weights']),1)
                    rows.append({'image':row['image'],'defective':row['defective'],**values})
                value={'high':high,'area':area,'growth':low,**summarize(rows)}
                assert value['normal_fp']==seed['normal_fp'],'Growth changed flagged normal view count'
                grown.append(value);grown_records[(high,area,low)]=rows
        eligible=[point for point in grown if eligible_final(point,weakest)]
        point=max(eligible,key=lambda r:point_rank(r,weakest)) if eligible else None
        report={'checkpoint':str(path),'step':step,'selected':point,'raw_grid':raw,'grown_grid':grown,
                'execution_scope':selection['execution_scope'],
                'selected_records':grown_records.get((point['high'],point['area'],point['growth']),[]) if point else [],
                'elapsed_seconds':time.perf_counter()-started,'score_cache_reused':saved is not None,
                'raw_predict_ms':{'p50':float(np.median(times)),'p95':float(np.percentile(times,95))} if times else None,
                'model_sha256':checkpoint_sha}
        if args.normal_density_complement:validate_density_artifacts(report,name=name)
        if pin_sources:
            report['calibration_source_identity']={'ordered_sources':source_identities,
                'sha256':hashlib.sha256(json.dumps(source_identities,sort_keys=True).encode('utf-8')).hexdigest(),
                'note':'Ordered original image/mask fileSHA256 and nativegeometry pinned; any sourcechange invalidates cache'}
        reports=[value for value in reports if value['step']!=step]+[report]
        reports.sort(key=lambda value:value['step'])
        if point:
            metadata={**original,'profile_version':f'{name}_step{step}','model_name':f'Round3 {args.variant} learned spatial defect segmentation',
                      'trained_utc':datetime.fromtimestamp(path.stat().st_mtime,timezone.utc).isoformat(),
                      'model_sha256':report['model_sha256'],'training_protocol':protocol,
                      'pixel_threshold':point['high'],'minimum_component_pixels':point['area'],
                      'hysteresis_growth_threshold':point['growth'],'calibration_selected_point':point,
                      'round3_calibration_protocol':selection,'threshold_selection':selection['final_rank'],
                      'architecture':protocol['architecture'],
                      'localization_validation':'Original-resolution calibration masks only; independent frozen comparison audit pending',
                      'manual_review_only':True,'automatic_acceptance':False,'automatic_rejection':False}
            if args.attempt2:
                pretrain=json.loads((directory/(name+'-pretrain.json')).read_text(encoding='utf-8'))
                metadata.update(input_width=512,input_height=1280,input_shape=[1,3,1280,512],output_shape=[1,1,640,256],
                                model_grid_width=256,model_grid_height=640,
                                model_name='SDD2-rehearsed FN-aware compact context defect segmentation',
                                architecture='MobileNetV3Small blocks0:7 encoder plus trained blocks7:12 context,96-to-40 residual fusion,depthwise spatial U-Net skip decoder,stride2 logits',
                                attempt2_sdd2_context_pretrain_weights_sha256=pretrain['weights_sha256'],
                                official_sdd2_test_still_sealed=True)
            if args.attempt3:
                output_shape=protocol['output_shape']
                metadata.update(input_width=512,input_height=1280,input_shape=[1,3,1280,512],output_shape=output_shape,
                                model_grid_width=output_shape[-1],model_grid_height=output_shape[-2],
                                model_name='Full-resolution learned surface-detail refinement',architecture=protocol['architecture'],
                                full_resolution_warmstart_sha256=protocol['warmstart_sha256'],official_sdd2_test_still_sealed=True,
                                minimum_component_area_units='Native512x1280 output pixels; four times as many cells as prior256x640 grid')
                if args.phase_max_refinement:
                    metadata.update(model_name='Phase-supervised detail refinement withhalfgridmax-logitserving',
                                    minimum_component_area_units='Native256x640outputcells, sameareaunitsasv10h',
                                    serving_score_reduction=protocol['serving_score_reduction'],
                                    score_meaning='Each256x640cell is sigmoid(maximum2x2full-resolution learnedlogits); notcalibratedprobability; retainedhighseeds andconnectedlower-scoregrowth usefrozen calibratedthresholds')
                if args.orientation_balanced:
                    metadata.update(model_name='Orientation andscale-balanced phase-supervised surface refinement',
                                    calibration_source_identity_sha256=report['calibration_source_identity']['sha256'])
                if args.native_width_balanced:
                    metadata.update(model_name='Native full-width phase-supervised surface refinement',
                                    calibration_source_identity_sha256=report['calibration_source_identity']['sha256'])
                if args.curvature_augmentation:
                    metadata.update(model_name='Native full-width curvature-augmented phase-supervised surface refinement',
                                    calibration_source_identity_sha256=report['calibration_source_identity']['sha256'],
                                    training_only_curvature_augmentation=True,
                                    evaluation_inputs_untouched_real=True,
                                    curvature_pairing_fixtures_sha256=protocol['pairing_fixtures_sha256'])
            if args.normal_density_complement:
                candidate=next(row for row in protocol['candidates'] if row['id']==step)
                metadata.update(model_name='Shared-feature semantic and TRAIN-normal density segmentation',
                    input_width=512,input_height=1280,input_shape=[1,3,1280,512],output_shape=[1,1,640,256],
                    model_grid_width=256,model_grid_height=640,output_logits=True,
                    minimum_component_area_units='Native256x640 output cells, unchanged source-coordinate mapping',
                    score_meaning='Sigmoid(alpha*original semantic raw logit + beta*signed TRAIN-normal log-density evidence); not calibrated defect probability',
                    calibration_source_identity_sha256=report['calibration_source_identity']['sha256'],
                    normal_density_candidate=candidate,normal_density_bank_sha256=fit['bank_sha256'],
                    normal_density_fit_report_sha256=hashlib.sha256((directory/(name+'-normal-fit.json')).read_bytes()).hexdigest(),
                    normal_density_source_pt_sha256=protocol['warmstart_sha256'],
                    normal_density_source_onnx_sha256=protocol['source_onnx_sha256'],
                    normal_density_normal_views=fit['normal_views'],normal_density_numeric_guards=protocol['numeric_guards'],
                    normal_density_q50=fit['q50'],normal_density_q95=fit['q95'],
                    normal_density_qspan=fit['qspan_with_floor'],
                    normal_density_covariance=fit['covariance'],official_sdd2_test_still_sealed=True)
                metadata['normal_density_guard_diagnostics_sha256']=hashlib.sha256(guard_path.read_bytes()).hexdigest()
            # Remove v10's old selected-point fields, retaining explicit original
            # SDD2 pretraining lineage and frozen warmstart hashes in the new fit.
            metadata.pop('hysteresis_calibration_point',None);metadata.pop('hysteresis_calibration_protocol',None)
            path.with_suffix('.onnx.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
            report['metadata_sha256']=hashlib.sha256(path.with_suffix('.onnx.json').read_bytes()).hexdigest()
        (directory/(name+'-calibration.json')).write_text(json.dumps({'selection':selection,'reports':reports},indent=2),encoding='utf-8')
        printed={k:v for k,v in report.items() if k not in {'raw_grid','grown_grid','selected_records','calibration_source_identity'}}
        if 'calibration_source_identity' in report:
            printed['calibration_source_identity_sha256']=report['calibration_source_identity']['sha256']
            printed['calibration_source_records']=len(report['calibration_source_identity']['ordered_sources'])
        print(json.dumps(printed,indent=2),flush=True)
    print('Calibration complete; global architecture/checkpoint winner must be frozen before any comparison scoring.',flush=True)


if __name__=='__main__':main()
