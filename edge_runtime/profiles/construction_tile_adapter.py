"""Construction-facing wrapper; fitted tile source/assets stay immutable."""
from pathlib import Path
import hashlib
import math
from .construction_tile import TileSurfaceModel

ROOT=Path(__file__).resolve().parents[2]
LABEL='tile_surface_anomaly'


class TileConstructionModel:
    def __init__(self,options,spec):
        self.spec=spec;self.mode='damage';self.variant=options.get('damage_variant')
        if spec.get('classes')!=[LABEL]:raise ValueError('Tile specialist has one anomaly class only')
        self.path=(ROOT/'models/construction'/spec['file']).resolve()
        if not self.path.is_relative_to((ROOT/'models/construction').resolve()):raise ValueError('Tile asset path outside localmodel root')
        if hashlib.sha256(self.path.read_bytes()).hexdigest()!=spec['sha256']:raise ValueError('Tile metadata identity differs')
        self.selected=options.get('damage_classes',[LABEL])
        if self.selected!=[LABEL]:raise ValueError('Tile task selection must contain only tile_surface_anomaly')
        if options.get('class_confidence') or 'confidence' in options:raise ValueError('Tile distance threshold is frozen; confidence override is not supported')
        self.engine=options.get('engine','onnxruntime');self.device=options.get('openvino_device','CPU')
        if self.engine!='onnxruntime' or self.device!='CPU':raise ValueError('Measured tile specialist supports ONNX Runtime CPU only')
        if options.get('exposure_policy','off')!='off':raise ValueError('Tile photo specialist has frozen exposure policy off')
        self.rendering=options.get('mask_rendering','masks')
        if self.rendering not in {'boxes','masks'}:raise ValueError('Tile mask_rendering must be boxes or masks')
        self.threads=options.get('cpu_threads',1)
        if type(self.threads) is not int or not 1<=self.threads<=8:raise ValueError('Tile cpu_threads must be1..8')
        self.backend=TileSurfaceModel(self.path,self.threads)
        self.sha256=spec['sha256'];self.input_size=256;self.exposure_policy='off'

    def infer(self,frame):
        output=[]
        for original in self.backend.infer(frame):
            row=dict(original);distance=row['anomaly_distance'];threshold=self.backend.threshold
            if not math.isfinite(distance) or not math.isfinite(threshold) or threshold<=0:raise ValueError('Tile anomaly score is invalid')
            row['score']=distance/(distance+threshold)
            if self.rendering=='boxes':row.pop('mask',None)
            output.append(row)
        return output

    def describe(self):
        return {**self.backend.describe(),'mode':'damage','damage_variant':self.variant,'model_path':str(self.path),
                'sha256':self.sha256,'model_source':self.spec['source'],'classes':[LABEL],
                'selected_damage_classes':[LABEL],'supported_capabilities':[LABEL],
                'input_shape':[1,3,256,256],'mask_rendering':self.rendering,'engine':'onnxruntime','device':'CPU',
                'execution_devices':['CPUExecutionProvider'],'cpu_threads':self.threads,'exposure_policy':'off',
                'diagnostic_only':True,'validation_scope':self.spec.get('validation_scope','Manufacturing tile stillphotos, weakcracklocalization'),
                'score_meaning':'Uncalibrated positive anomaly-distance/(distance+frozen threshold), bounded0..1; NOT defect probability. Actual raw distance retained per finding.',
                'raw_anomaly_distance_threshold':self.backend.threshold,
                'encoder':self.backend.spec['encoder'],'normal_bank':self.backend.spec['bank'],
                'unsupported_tasks':['semantic_crack_classification','installed_tile_alignment','lippage','adhesion','humidity','code_compliance']}
