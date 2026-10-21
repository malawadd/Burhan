"""Clocked real-image replay stress; no camera or independent accuracy claim."""
from pathlib import Path
import argparse,hashlib,json,re,sys,tempfile,time

ROOT=Path(__file__).resolve().parents[1]


def select_replay_inputs(root, model, *, varied=False, partition='test', decode_input=False):
    """Resolve sources without decoding images, importing runtime or loading a model."""
    root,model=Path(root),Path(model)
    if partition not in {'test','calibration'}:
        raise ValueError('Unknown source partition')
    if partition!='test' and not varied:
        raise ValueError('--partition calibration requires --varied')
    if varied and decode_input:
        raise ValueError('--varied and --decode-input are separate experiments')
    dataset=root/'datasets/factory/KolektorSDD'
    if not varied:
        return {'paths':[dataset/'kos04/Part3.jpg'],'partition':'single',
                'description':'Historical single real-image replay; no independent accuracy claim',
                'metadata_path':None,'metadata_sha256':None}
    if partition=='test':
        metadata=root/'output/evaluation/factory-speed-comparison.json'
        items=json.loads(metadata.read_text(encoding='utf-8'))['split_items']
        description='Exposed development comparison: historical 80-view source order; not blind validation'
    else:
        metadata=model.with_suffix(model.suffix+'.json')
        splits=json.loads(metadata.read_text(encoding='utf-8'))['split_items']
        items=splits['calibration']
        if set(items)&(set(splits['train'])|set(splits['test'])):
            raise ValueError('Calibration item identities overlap train or exposed comparison')
        description='Frozen selected-model calibration items used for model/operating-point selection; not independent validation'
    if not isinstance(items,list) or len(items)!=10 or len(set(items))!=10 or any(not isinstance(item,str) or not re.fullmatch(r'kos[0-9]{2}',item) for item in items):
        raise ValueError('Expected ten unique canonical physical item IDs')
    # This is exactly the original test path ordering, independent of item-list order.
    paths=sorted(path for item in items for path in (dataset/item).glob('Part*.jpg'))
    if len(paths)!=80:
        raise ValueError(f'Expected 80 {partition} replay views, found {len(paths)}')
    return {'paths':paths,'partition':partition,'description':description,
            'metadata_path':str(metadata.resolve()),'metadata_sha256':hashlib.sha256(metadata.read_bytes()).hexdigest()}


def argument_parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--code-root',type=Path,default=ROOT)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seconds',type=float,default=8)
    p.add_argument('--fps',type=float,default=30)
    p.add_argument('--deadline',type=float,default=100)
    p.add_argument('--backend',default='auto')
    p.add_argument('--model',type=Path,default=ROOT/'models/factory/kolektor_surface_rf.joblib')
    p.add_argument('--current-profile',action='store_true',help='Use current profile code with baseline runtime to isolate runtime overhead')
    p.add_argument('--preview-fps',type=float,default=6)
    p.add_argument('--local-source','--decode-input',dest='decode_input',action='store_true',help='Use the real LocalSource still path: initial decode, then cached-frame copies')
    p.add_argument('--varied',action='store_true',help='Cycle 80 frozen-partition images, predecoded before timing')
    p.add_argument('--trace',action='store_true',help='Diagnostic per-profile-call timing, separate from official runtime decision latency')
    p.add_argument('--partition',choices=['test','calibration'],default='test',help='For --varied only: historical exposed test order or selected model calibration items')
    return p


def main(argv=None):
    p=argument_parser()
    a=p.parse_args(argv)
    if a.partition!='test' and not a.varied:p.error('--partition calibration requires --varied')
    if a.varied and a.decode_input:p.error('--varied and --decode-input are separate experiments')
    sys.path.insert(0,str(a.code_root.resolve()))
    import cv2
    import numpy as np
    import edge_runtime.runtime as rt
    trace_rows=[]
    if a.trace:
        original_create_profile=rt.create_profile
        def traced_profile(*args,**kwargs):
            profile=original_create_profile(*args,**kwargs)
            original_process=profile.process
            def process(frame,context):
                started=time.perf_counter()
                result=original_process(frame,context)
                finished=time.perf_counter()
                trace_rows.append({'sequence':context.sequence,'captured_monotonic':context.captured_monotonic,
                                   'started_monotonic':started,'profile_ms':(finished-started)*1000,
                                   'queue_before_profile_ms':(started-context.captured_monotonic)*1000,
                                   'host_to_profile_return_ms':(finished-context.captured_monotonic)*1000})
                if result.metrics.get('factory_neural_stage_ms') is not None:
                    trace_rows[-1]['neural_stage_ms']=dict(result.metrics['factory_neural_stage_ms'])
                return result
            profile.process=process
            return profile
        rt.create_profile=traced_profile
    if a.current_profile:
        import edge_runtime.profiles as profile_package
        profile_package.__path__=[str(ROOT/'edge_runtime/profiles')]

    selection=select_replay_inputs(ROOT,a.model,varied=a.varied,partition=a.partition,decode_input=a.decode_input)
    input_paths=selection['paths']
    sample=input_paths[0]
    # Select the entire partition before decoding anything: CAL never touches the old sample.
    frames=[cv2.imdecode(np.fromfile(path,np.uint8),cv2.IMREAD_COLOR) for path in input_paths]
    assert all(value is not None for value in frames)
    OriginalSource=rt.LocalSource
    class ClockedReplay:
        backend='clocked_public_still_benchmark'
        def __init__(self,*args):self.start=None;self.sequence=0
        def open(self):
            self.source=OriginalSource('image:'+str(sample),'factory',a.fps,960,540) if a.decode_input else None
            if self.source:self.source.open()
            self.start=time.perf_counter()
        def read(self):
            if time.perf_counter()-self.start>=a.seconds:return None,None
            self.sequence+=1
            if self.source:return self.source.read()
            return frames[(self.sequence-1)%len(frames)],self.sequence*1000/a.fps
        def close(self):
            if self.source:self.source.close()
    rt.LocalSource=ClockedReplay
    with tempfile.TemporaryDirectory(prefix='inspectflow-bench-') as directory:
        edge=rt.EdgeRuntime(directory)
        try:
            fields={'profile':'factory','source':'file:clocked-public-still','input_fps':a.fps,
                'deadline_ms':a.deadline,'profile_options':{'defect_model_path':str(a.model.resolve()),
                'defect_backend':a.backend,'confirmation_frames':1}}
            if 'preview_fps' in rt.RuntimeConfig.__dataclass_fields__:fields['preview_fps']=a.preview_fps
            start_requested=time.perf_counter()
            edge.start(rt.RuntimeConfig(**fields))
            station_start_ms=(time.perf_counter()-start_requested)*1000
            deadline=time.monotonic()+a.seconds+30
            while time.monotonic()<deadline:
                state=edge.snapshot()
                if state['error']:raise RuntimeError(state['error'])
                if state['state']=='eof':break
                time.sleep(.02)
            else:raise RuntimeError('Runtime did not drain within timeout')
            report={'note':'Real public still images replayed at a fixed software clock, cycling when requested. Physical line coverage and independent accuracy are not measured; LocalSource acquisition is included only when selected.',
                'code_root':str(a.code_root.resolve()),'current_profile_override':a.current_profile,
                'runtime_module_file':str(Path(rt.__file__).resolve()),
                'profile_module_file':str(Path(sys.modules[type(edge.profile).__module__].__file__).resolve()),
                'input_images':[str(path.relative_to(ROOT)) for path in input_paths],
                'input_sha256':[hashlib.sha256(path.read_bytes()).hexdigest() for path in input_paths],
                'requested_fps':a.fps,'requested_seconds':a.seconds,
                'station_start_ms':station_start_ms,
                'input_acquisition':'LocalSource cached still: initial decode, then frame copies' if a.decode_input else f"predecoded arrays cycling 80 {selection['partition']} images" if a.varied else 'predecoded single array',
                'source_partition':selection['partition'],
                'source_partition_description':selection['description'],
                'source_selection_metadata':selection['metadata_path'],
                'source_selection_metadata_sha256':selection['metadata_sha256'],
                'state':state}
            if a.trace:
                report['diagnostic_trace_scope']='Per-profile-call diagnostic; official runtime latency additionally includes result publication. Trace instrumentation is opt-in.'
                report['diagnostic_trace']=trace_rows
        finally:edge.close()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'counts':state['metrics'],'status':state['last_result']['status'] if state['last_result'] else None},indent=2))


if __name__=='__main__':
    main()
