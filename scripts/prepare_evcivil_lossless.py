"""Build genuine lossless FFV1/NUT replay inputs and verify every pixel and camera timestamp.

The browser's annotated H264 videos are unchanged. NUT is used because FFmpeg's
Matroska muxer uses millisecond timecodes; these camera arrays have microsecond PTS.
Requires PyAV in the isolated export environment. No model is loaded or executed.
"""
from __future__ import annotations
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import av
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def encode_and_verify(folder):
    frames=np.load(folder/'frames.npy',allow_pickle=False)
    times=np.load(folder/'timestamps-us.npy',allow_pickle=False).astype(np.int64)
    expected_times=times-int(times[0]);period=int(np.median(np.diff(times)))
    target=folder/f'{folder.name}-lossless-frames.nut'
    timebase=Fraction(1,1_000_000)
    with av.open(str(target),'w',format='nut') as container:
        stream=container.add_stream('ffv1',rate=Fraction(1_000_000,period))
        stream.width=int(frames.shape[2]);stream.height=int(frames.shape[1]);stream.pix_fmt='gray'
        stream.time_base=timebase;stream.codec_context.time_base=timebase
        stream.codec_context.thread_count=1;stream.codec_context.gop_size=1
        stream.options={'level':'3','coder':'1','context':'1','slicecrc':'1'}
        for array,timestamp in zip(frames,expected_times):
            frame=av.VideoFrame.from_ndarray(array,format='gray')
            frame.pts=int(timestamp);frame.time_base=timebase
            for packet in stream.encode(frame):container.mux(packet)
        for packet in stream.encode():container.mux(packet)
    actual_times=[]
    with av.open(str(target)) as source:
        codec=source.streams.video[0].codec_context
        codec_name=codec.name;pixel_format=codec.format.name if codec.format else None
        for index,frame in enumerate(source.decode(video=0)):
            actual_times.append(Fraction(frame.pts)*frame.time_base*1_000_000)
            if not np.array_equal(frame.to_ndarray(format='gray'),frames[index]):
                raise RuntimeError(f'PyAV pixel mismatch at frame {index}')
    if len(actual_times)!=len(frames):raise RuntimeError('PyAV frame count differs from source exposures')
    errors=[abs(actual-Fraction(int(expected))) for actual,expected in zip(actual_times,expected_times)]
    if any(errors):raise RuntimeError(f'NUT changed original camera PTS: max {max(errors)} microseconds')
    capture=cv2.VideoCapture(str(target));decoded=0
    if not capture.isOpened():raise RuntimeError('OpenCV cannot open the lossless source')
    decoder_metadata={'backend':capture.getBackendName(),'fps':capture.get(cv2.CAP_PROP_FPS)}
    try:
        while True:
            okay,bgr=capture.read()
            if not okay:break
            if decoded>=len(frames):raise RuntimeError('OpenCV produced extra frames')
            expected=cv2.cvtColor(frames[decoded],cv2.COLOR_GRAY2BGR)
            if bgr.dtype!=np.uint8 or not np.array_equal(bgr,expected):
                raise RuntimeError(f'OpenCV BGR pixel mismatch at frame {decoded}')
            decoded+=1
    finally:capture.release()
    if decoded!=len(frames):raise RuntimeError(f'OpenCV frame count {decoded} != source count {len(frames)}')
    annotations=json.loads((folder/'annotations.json').read_text(encoding='utf8'))
    report={'source_sequence':folder.name,'path':str(target.relative_to(ROOT)),
            'source_dataset':annotations['source'],'authors':annotations['authors'],'license':annotations['license'],
            'format':'NUT','codec':codec_name,'pixel_format':pixel_format,'encoder_threads':1,
            'original_camera_exposures':len(frames),'opencv_decoded_frames':decoded,
            'all_opencv_bgr_pixels_equal_raw_gray_to_bgr':True,'all_pyav_gray_pixels_equal_raw_gray':True,
            'max_presentation_timestamp_error_us':float(max(errors)),
            'timebase_scope':'All original microsecond camera PTS shifted to zero, one lossless frame per actual exposure; no duplicates, interpolation or generated images.',
            'decoder':decoder_metadata,'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
            'bytes':target.stat().st_size,
            'model_scope':'No model execution here. This verifies replay input identity with offline arrays, not deployment accuracy or real-time scheduling.',
            'runtime_scope':'LocalSource currently schedules using nominal FPS; verified container PTS do not make that scheduler fully VFR-aware.'}
    target.with_suffix('.provenance.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sequences',nargs='+',default=['crack_10','crack_89','spalling_35']);args=parser.parse_args()
    cv2.setNumThreads(1)
    reports=[]
    for name in args.sequences:
        report=encode_and_verify(ROOT/'datasets/construction/evcivil'/name);reports.append(report)
        print(json.dumps(report),flush=True)
    (ROOT/'output/evaluation/construction-lossless-source-verification.json').write_text(json.dumps(reports,indent=2),encoding='utf8')
