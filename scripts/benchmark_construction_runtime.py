"""Real-file, real-time construction runtime benchmark; no generated frames."""
from pathlib import Path
import argparse
import hashlib
import json
import socket
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from edge_runtime import runtime as rt


def file_hash(path):
    digest=hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda:source.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video',type=Path,required=True)
    parser.add_argument('--mode',choices=['ppe','crack','damage'],default='ppe')
    parser.add_argument('--crack-variant')
    parser.add_argument('--damage-variant')
    parser.add_argument('--confidence',type=float)
    parser.add_argument('--video-mode',choices=['realtime','every_frame'],default='realtime')
    parser.add_argument('--engine',choices=['onnxruntime','openvino'],default='openvino')
    parser.add_argument('--device',choices=['CPU','GPU'])
    parser.add_argument('--mask-rendering',choices=['masks','boxes'],default='boxes')
    parser.add_argument('--roi',type=float,nargs=4)
    parser.add_argument('--seconds',type=float,default=30)
    parser.add_argument('--fps',type=float,default=15)
    parser.add_argument('--deadline',type=float,default=200)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    video = args.video.resolve()
    if not video.is_file(): parser.error('Real local video is required')
    if args.seconds <= 0: parser.error('--seconds must be positive')
    base_source = rt.LocalSource

    class TimedVideo(base_source):
        def open(self):
            super().open()
            self.limit_started = time.perf_counter()
        def read(self):
            if time.perf_counter()-self.limit_started >= args.seconds:
                return None,None
            return super().read()

    def deny(*unused,**kwargs):
        raise AssertionError('Outbound Python socket connections are disabled during installed-model inference')

    rt.LocalSource = TimedVideo
    socket.socket.connect = deny
    device = args.device or ('GPU' if args.engine=='openvino' else 'CPU')
    options = {'mode':args.mode,'engine':args.engine,'openvino_device':device,
               'cpu_threads':4,'mask_rendering':args.mask_rendering,'stage':'site_safety' if args.mode=='ppe' else 'structure'}
    for key in ('crack_variant','damage_variant','confidence'):
        value = getattr(args,key)
        if value is not None: options[key] = value
    if args.roi is not None: options['roi']=args.roi
    with tempfile.TemporaryDirectory(prefix='inspectflow-real-video-') as directory:
        edge = rt.EdgeRuntime(directory)
        try:
            requested = time.perf_counter()
            edge.start(rt.RuntimeConfig(profile='construction',source='file:'+str(video),input_fps=args.fps,
                                       deadline_ms=args.deadline,preview_fps=6,video_mode=args.video_mode,profile_options=options))
            startup_ms = (time.perf_counter()-requested)*1000
            until = time.perf_counter()+args.seconds+30
            while time.perf_counter()<until:
                state = edge.snapshot()
                if state['error']: raise RuntimeError(state['error'])
                if state['state']=='eof': break
                time.sleep(.03)
            else: raise RuntimeError('Video runtime did not finish within bounded duration')
            report = {'scope':'Actual local video frames with the declared runtime replay scheduler. Optional wall-time excerpt; no generated, looped or repeated still frames. Sequential mode does not deliberately skip source frames, but the inference queue may still drop frames. Runtime file pacing uses nominal FPS, not exact variable-frame-rate presentation times. This tests throughput, not Saudi code or independent field accuracy.',
                      'video':str(video),'video_sha256':file_hash(video),
                      'video_mode':args.video_mode,'profile_options':options,
                      'maximum_excerpt_seconds':args.seconds,'requested_fps':args.fps,'deadline_ms':args.deadline,
                      'station_start_ms':startup_ms,'network':'Python socket.connect denied','state':state}
        finally:
            edge.close()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'metrics':state['metrics'],'video':state['video'],'startup_ms':startup_ms},indent=2))


if __name__=='__main__': main()
