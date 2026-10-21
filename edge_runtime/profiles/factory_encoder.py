"""Local MobileNet models; GPU default or explicit CPU4, no network or Torch."""
import hashlib
import time
from pathlib import Path
import cv2
import numpy as np

ENCODER_FEATURE_VERSION='surface_mobilenet_early_v1'
ENCODER_COLUMNS=51


def seeded_growth_mask(scores, threshold, minimum_area, growth_threshold=None):
    """Retain high-score seeds; optionally grow only their connected low-score regions.

    Scores are never edited. Growth cannot create a flagged observation because
    every grown region must contain an already-retained high-threshold seed.
    """
    if growth_threshold is not None and not 0<=growth_threshold<=threshold:
        raise ValueError('Factory growth threshold must be between zero and the seed threshold')
    count,labels,stats,_=cv2.connectedComponentsWithStats((scores>=threshold).astype(np.uint8),connectivity=8)
    indices=np.flatnonzero(stats[1:,cv2.CC_STAT_AREA]>=minimum_area)+1
    indices=indices[np.argsort(-stats[indices,cv2.CC_STAT_AREA],kind='stable')[:20]]
    keep=np.zeros(count,dtype=np.uint8);keep[indices]=1
    seed=keep[labels]
    if growth_threshold is None or not len(indices):return seed
    count,low_labels=cv2.connectedComponents((scores>=growth_threshold).astype(np.uint8),connectivity=8)
    touched=np.unique(low_labels[seed>0]);touched=touched[touched>0]
    low_keep=np.zeros(count,dtype=np.uint8);low_keep[touched]=1
    return low_keep[low_labels]

class FactoryEncoder:
    def __init__(self,path,expected_sha256):
        path=Path(path)
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected_sha256:
            raise ValueError('Factory encoder missing or checksum mismatched')
        import openvino as ov
        self.core=ov.Core()
        if not any(d.startswith('GPU') for d in self.core.available_devices):
            raise ValueError('Factory neural encoder explicitly requires the installed Intel GPU')
        graph=self.core.read_model(str(path))
        if list(graph.input().shape)!=[1,3,1280,512]:
            raise ValueError('Unsupported factory encoder input shape')
        self.compiled=self.core.compile_model(graph,'GPU',{'PERFORMANCE_HINT':'LATENCY','NUM_STREAMS':'1','INFERENCE_PRECISION_HINT':'f16'})
        self.request=self.compiled.create_infer_request()
        self.input=self.compiled.input();self.outputs=list(self.compiled.outputs)
        self.sha256=expected_sha256
        self.execution_devices=list(self.compiled.get_property('EXECUTION_DEVICES'))

    def extract(self,frame):
        from .factory_defect import extract_features_fast
        tick=time.perf_counter()
        resized=cv2.resize(frame,(512,1280),interpolation=cv2.INTER_AREA)
        value=np.ascontiguousarray(resized.transpose(2,0,1)[None])
        pre=time.perf_counter()
        result=self.request.infer({self.input:value})
        inferred=time.perf_counter()
        layers=[np.asarray(result[out])[0].transpose(1,2,0) for out in self.outputs]
        early=next((layer for layer in layers if layer.shape==(640,256,16)),None)
        mid=next((layer for layer in layers if layer.shape==(160,64,24)),None)
        if early is None or mid is None:raise ValueError('Unsupported factory encoder output inventory')
        handcrafted,shape=extract_features_fast(resized,256)
        hand=time.perf_counter()
        learned=cv2.resize(mid,(256,640),interpolation=cv2.INTER_LINEAR)
        merged=np.concatenate([handcrafted.reshape(640,256,11),early,learned],axis=2)
        done=time.perf_counter()
        self.last_timing_ms={'preprocess':(pre-tick)*1000,'gpu':(inferred-pre)*1000,
                             'handcrafted':(hand-inferred)*1000,'upsample_fuse':(done-hand)*1000}
        return np.ascontiguousarray(merged.reshape(-1,ENCODER_COLUMNS),dtype=np.float32),shape


class FactorySegmentation:
    """Fused local neural segmentation: transfer one score map to the CPU."""
    def __init__(self,path,output_logits=False,growth_threshold=None,precision_hint='f16',disable_winograd_convolution=False,device='GPU',cpu_threads=None):
        initialization_started=time.perf_counter()
        if not isinstance(precision_hint,str) or precision_hint not in {'f16','f32'}:
            raise ValueError('Factory inference precision hint must be f16 or f32')
        self.precision_hint=precision_hint
        if type(disable_winograd_convolution) is not bool:
            raise ValueError('Factory disable_winograd_convolution must be a bool')
        self.disable_winograd_convolution=disable_winograd_convolution
        if not isinstance(device,str) or device not in {'GPU','CPU'}:raise ValueError('Factory inference device must be GPU or CPU')
        if device=='CPU':
            if precision_hint!='f32' or type(cpu_threads) is not int or cpu_threads!=4 or disable_winograd_convolution:
                raise ValueError('Explicit factory CPU recipe requires f32, exactly4threads and no GPU-only flags')
        elif cpu_threads is not None:raise ValueError('CPU thread metadata cannot configure the existing GPU backend')
        self.device=device;self.cpu_threads=cpu_threads
        self.backend_name='openvino_cpu_neural' if device=='CPU' else 'openvino_gpu_neural'
        import openvino as ov
        self.core=ov.Core();graph=self.core.read_model(str(path))
        output_shape=list(graph.output().shape)
        if list(graph.input().shape)!=[1,3,1280,512] or output_shape not in ([1,1,640,256],[1,1,1280,512]):
            raise ValueError('Unsupported factory segmentation model shape')
        self.grid_height,self.grid_width=output_shape[-2:]
        if device=='GPU' and not any(device.startswith('GPU') for device in self.core.available_devices):
            raise ValueError('Factory neural model explicitly requires the installed Intel GPU')
        if device=='CPU' and not any(value.split('.')[0]=='CPU' for value in self.core.available_devices):raise ValueError('Explicit CPU neural device unavailable')
        config={'PERFORMANCE_HINT':'LATENCY','NUM_STREAMS':'1','INFERENCE_PRECISION_HINT':precision_hint}
        if device=='GPU' and disable_winograd_convolution:config['GPU_DISABLE_WINOGRAD_CONVOLUTION']=True
        if device=='CPU':config['INFERENCE_NUM_THREADS']=cpu_threads
        self.compiled=self.core.compile_model(graph,device,config)
        self.request=self.compiled.create_infer_request();self.input=self.compiled.input();self.output=self.compiled.output()
        self.execution_devices=list(self.compiled.get_property('EXECUTION_DEVICES'))
        self.output_logits=bool(output_logits)
        self.growth_threshold=growth_threshold
        self.initialization_report={'compile_and_request_ms':round((time.perf_counter()-initialization_started)*1000,2),
                                    'warmup_completed':False}
        if device=='CPU':self.initialization_report.update(device='CPU',precision_hint='f32',inference_threads=cpu_threads,num_streams=1,model_backend=self.backend_name)

    def warmup(self,threshold,minimum_area,iterations=3):
        """Initialize real kernels/readback before capture; no inspection is emitted."""
        import threading
        values=[]
        # Initialization tensors do not contribute to accuracy, throughput,
        # source counters, inspection state or any acceptance decision.
        shapes=[(1280,500,3),(1281,500,3),(1267,500,3)]
        for index in range(iterations):
            started=time.perf_counter()
            frame=np.zeros(shapes[index%len(shapes)],dtype=np.uint8)
            self.predict(frame,threshold,minimum_area)
            values.append(round((time.perf_counter()-started)*1000,3))
        self.initialization_report.update(warmup_completed=True,warmup_iterations=iterations,
            warmup_predict_ms=values,warmup_total_ms=round(sum(values),3),
            warmup_thread=threading.current_thread().name,
            warmup_source_shapes=[list(shapes[index%len(shapes)]) for index in range(iterations)],
            warmup_input='Fixed-shape zero initialization tensor; never an inspected observation or accuracy example',
            warmup_scope='Neural input preparation, inference, output readback, sigmoid and component policy before capture; no source/quality/event processing',
            residual_latency_caveat='Initialization does not guarantee absence of later OS/CPU scheduling stalls' if getattr(self,'device','GPU')=='CPU' else 'Initialization does not guarantee absence of later OS/GPU scheduling stalls')
        return dict(self.initialization_report)

    def predict(self,frame,threshold,minimum_area):
        timing_started=time.perf_counter()
        resized=cv2.resize(frame,(512,1280),interpolation=cv2.INTER_AREA)
        value=np.ascontiguousarray(resized.transpose(2,0,1)[None])
        preprocessed=time.perf_counter()
        result=self.request.infer({self.input:value})
        scores=np.array(result[self.output][0,0],dtype=np.float32,copy=True)
        readback=time.perf_counter()
        if not np.isfinite(scores).all():raise ValueError('Factory neural model produced invalid scores')
        if self.output_logits:
            scores=1./(1.+np.exp(-np.clip(scores,-40.,40.)))
        growth=getattr(self,'growth_threshold',None)
        selected=seeded_growth_mask(scores,threshold,minimum_area,growth) if growth is not None else (scores>=threshold).astype(np.uint8)
        h,w=frame.shape[:2];grid_height,grid_width=scores.shape;candidates=[]
        if np.any(selected):
            count,labels,stats,_=cv2.connectedComponentsWithStats(selected,connectivity=8)
            sums=np.bincount(labels.ravel(),weights=scores.ravel(),minlength=count)
            for region in range(1,count):
                x,y,rw,rh,area=map(int,stats[region])
                if area<minimum_area:continue
                left,top=int(x*w/grid_width),int(y*h/grid_height)
                right=min(w,max(left+1,int(np.ceil((x+rw)*w/grid_width))))
                bottom=min(h,max(top+1,int(np.ceil((y+rh)*h/grid_height))))
                candidates.append({'bbox':(left,top,right-left,bottom-top),'area':area,'score':float(sums[region]/area)})
        candidates.sort(key=lambda candidate:candidate['area'],reverse=True)
        postprocessed=time.perf_counter()
        return scores,candidates[:20],{'model_backend':getattr(self,'backend_name','openvino_gpu_neural'),'encoder_execution_devices':self.execution_devices,
                                      'model_score_max':round(float(scores.max()),5),'model_score_map_exact':True,
                                      'model_score_semantics':'Learned_neural_sigmoid_score_not_calibrated_probability',
                                      'model_vote_threshold':threshold,'model_minimum_component_pixels':minimum_area,
                                      'model_hysteresis_growth_threshold':growth,
                                      'factory_neural_stage_ms':{'preprocess':round((preprocessed-timing_started)*1000,3),
                                          'infer_readback':round((readback-preprocessed)*1000,3),
                                          'postprocess':round((postprocessed-readback)*1000,3),
                                          'total':round((postprocessed-timing_started)*1000,3)},
                                      'model_mask_semantics':'High-score seed retained first, then connected lower-score foreground growth' if growth is not None else 'Thresholded neural foreground',
                                      'model_grid_width':grid_width,'model_grid_height':grid_height,'candidate_regions':len(candidates)}
