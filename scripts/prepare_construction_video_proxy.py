"""Downscale original real footage into a one-for-one H.264 source proxy."""
from __future__ import annotations
import argparse,hashlib,json,math,os,subprocess,time
from pathlib import Path
import cv2
import imageio_ffmpeg
from transcode_construction_videos import decoded_video_info

ROOT=Path(__file__).resolve().parents[1]
if __name__=="__main__":
    a=argparse.ArgumentParser();a.add_argument("--source",required=True);a.add_argument("--output",required=True);a.add_argument("--max-edge",type=int,default=1280);args=a.parse_args()
    source=(ROOT/args.source).resolve();target=(ROOT/args.output).resolve()
    if not source.is_file() or source==target or not 320<=args.max_edge<=1920:raise ValueError("Use an existing source, separate output, and max-edge 320..1920")
    target.parent.mkdir(parents=True,exist_ok=True);temporary=target.with_name(target.stem+'.tmp.mp4')
    before=decoded_video_info(source);h,w=before['shape_hw'];ratio=min(1,args.max_edge/max(h,w));width=max(2,int(w*ratio)//2*2);height=max(2,int(h*ratio)//2*2)
    exe=imageio_ffmpeg.get_ffmpeg_exe();began=time.perf_counter()
    # The original drone file has irregular PTS relative to its reported mean
    # frame rate. Map each original frame ordinal to that nominal CFR timeline.
    # Do not combine CFR with a demuxer encoder time base: that would duplicate
    # frames. Explicit zero duplicate/drop counters below are mandatory.
    fps=before['fps']
    command=[exe,'-hide_banner','-loglevel','error','-nostdin','-y','-i',str(source),'-an','-vf',f'scale={width}:{height}:flags=area,setpts=N/({fps:.15g}*TB)','-c:v','libx264','-preset','veryfast','-crf','18','-pix_fmt','yuv420p','-threads','2','-r',f'{fps:.15g}','-fps_mode','cfr','-progress','pipe:1','-movflags','+faststart',str(temporary)]
    process=subprocess.run(command,capture_output=True,text=True,timeout=180,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    if process.returncode:raise RuntimeError(process.stderr[-2000:])
    progress={}
    for line in process.stdout.splitlines():
        if '=' in line:
            key,value=line.split('=',1);progress[key]=value.strip()
    if progress.get('progress')!='end' or int(progress.get('dup_frames','-1'))!=0 or int(progress.get('drop_frames','-1'))!=0 or int(progress.get('frame','-1'))!=before['decoded_frames']:
        raise ValueError('Rejected proxy: FFmpeg generated, dropped or failed to encode original frame ordinals')
    after=decoded_video_info(temporary)
    if (before['decoded_frames']!=after['decoded_frames'] or not math.isclose(before['fps'],after['fps'],rel_tol=1e-5,abs_tol=.001)
            or abs(before['duration_seconds_from_decoded_frames']-after['duration_seconds_from_decoded_frames'])>.002 or after['shape_hw']!=[height,width]):
        raise ValueError('Rejected proxy: changed count/FPS/duration or unexpected dimensions')
    temporary.replace(target)
    report={'source':str(source),'proxy':str(target),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'proxy_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
            'source_video':before,'proxy_video':after,'verified_one_for_one_frames':True,'max_edge':args.max_edge,'downscale':{'x':width/w,'y':height/h,'filter':'area'},
            'proxy_bytes':target.stat().st_size,'conversion_and_proxy_verification_seconds':time.perf_counter()-began,'ffmpeg':exe,'imageio_ffmpeg':imageio_ffmpeg.__version__,'opencv':cv2.__version__,
            'ffmpeg_progress':progress,'timestamp_policy':'One-to-one decoded frame ordinals mapped to original reported nominal FPS; CFR timestamps normalized, original irregular PTS are not claimed preserved',
            'frame_mapping':'Proxy frame index i corresponds to original decoded frame index i. Consult original file for original capture PTS.',
            'scope':'Original actual recorded footage downscaled and lossily H.264 encoded; one source frame per proxy frame with verified zero duplicate/drop counters, no generated/interpolated images, no labels or model overlays. Audio omitted. Proxy timestamps are normalized to source nominal FPS; resolution/detail changes, so model accuracy on this proxy is not assumed equal to raw 4K. This is offline preprocessing, not a real-time live 4K conversion benchmark.'}
    provenance_path=source.with_suffix('.provenance.json')
    if provenance_path.is_file():
        provenance=json.loads(provenance_path.read_text(encoding='utf-8'))
        report['source_attribution']={key:provenance.get(key) for key in ('creator','record','license','title')}
        target.with_suffix('.LICENSE.md').write_text(f"# Source attribution\n\n{provenance.get('creator','Source creator')}. {provenance.get('title','Original recording')}. {provenance.get('record','')}\n\nSource licence: {provenance.get('license','See original provenance')}. Original raw video is unmodified. This proxy adds area downscaling, lossy H.264 encoding and explicit nominal-FPS timestamp normalization; audio is omitted. One original frame per proxy frame; no synthetic/interpolated frames. See the adjacent .proxy.json for hashes and verification.\n",encoding='utf-8')
    target.with_suffix('.proxy.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
