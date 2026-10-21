"""Create browser-playable H.264 copies of actual inference exports, offline."""
from __future__ import annotations
import argparse,hashlib,json,math,os,subprocess,time
from pathlib import Path
import imageio_ffmpeg
import cv2

ROOT=Path(__file__).resolve().parents[1]


def decoded_video_info(path):
    cap=cv2.VideoCapture(str(path))
    if not cap.isOpened():raise ValueError(f"Cannot decode video {path}")
    fps=float(cap.get(cv2.CAP_PROP_FPS));reported=int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)));count=0;shape=None;first_ms=None;last_ms=None
    try:
        if not math.isfinite(fps) or fps<=0:raise ValueError(f"Invalid FPS: {path}")
        while True:
            ok,frame=cap.read()
            if not ok:break
            current=list(frame.shape[:2])
            if shape is not None and current!=shape:raise ValueError(f"Video resolution changed: {path}")
            shape=current;position=float(cap.get(cv2.CAP_PROP_POS_MSEC));first_ms=position if first_ms is None else first_ms;last_ms=position;count+=1
    finally:cap.release()
    if not count or count!=reported:raise ValueError(f"Incomplete decode: {path}: decoded {count}, reported {reported}")
    return {"decoded_frames":count,"reported_frames":reported,"fps":fps,"duration_seconds_from_decoded_frames":count/fps,"shape_hw":shape,"first_decoded_pts_ms":first_ms,"last_decoded_pts_ms":last_ms}


if __name__=="__main__":
    a=argparse.ArgumentParser();a.add_argument("--input",action="append",help="Actual MP4 export, repeatable; default is the four measured exports")
    args=a.parse_args();sources=args.input or ["output/construction-observed-real-video-baseline.mp4","output/construction-observed-real-video-optimized-letterbox.mp4","output/construction-bridge-real-crack.mp4","output/construction-niosh-real-ppe.mp4"]
    exe=imageio_ffmpeg.get_ffmpeg_exe();rows=[]
    for filename in sources:
        source=(ROOT/filename).resolve()
        if not source.is_file():raise FileNotFoundError(source)
        target=source.with_name(source.stem+"-h264.mp4");temporary=target.with_name(target.stem+".tmp.mp4")
        before=decoded_video_info(source)
        command=[exe,"-hide_banner","-loglevel","error","-nostdin","-y","-i",str(source),"-an","-c:v","libx264","-preset","veryfast","-crf","24","-pix_fmt","yuv420p","-threads","2","-movflags","+faststart","-fps_mode","passthrough",str(temporary)]
        started=time.perf_counter();subprocess.run(command,check=True,creationflags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0)
        after=decoded_video_info(temporary)
        same_frames=before["decoded_frames"]==after["decoded_frames"]
        same_fps=math.isclose(before["fps"],after["fps"],rel_tol=1e-5,abs_tol=.001)
        same_duration=abs(before["duration_seconds_from_decoded_frames"]-after["duration_seconds_from_decoded_frames"])<=.002
        if not (same_frames and same_fps and same_duration and before["shape_hw"]==after["shape_hw"]):
            raise ValueError(f"Rejected H.264 export with changed frame count, FPS, duration or shape: {source}")
        temporary.replace(target)
        row={"source":str(source),"target":str(target),"source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),"target_sha256":hashlib.sha256(target.read_bytes()).hexdigest(),"bytes":target.stat().st_size,"seconds":time.perf_counter()-started,"codec":"H.264 libx264","pixel_format":"yuv420p","verification":{"source":before,"target":after,"frame_count_equal":same_frames,"fps_equal_within_tolerance":same_fps,"duration_equal_within_2ms":same_duration},"scope":"Lossy viewing copy of actual recorded-video inference export. Source and target fully decoded; count/FPS/duration/shape preserved. No generated or repeated frames, no added inference or ground-truth assertions. Audio absent from the OpenCV annotation exports."}
        rows.append(row);print(target,flush=True)
    (ROOT/"output/construction-h264-manifest.json").write_text(json.dumps({"ffmpeg":exe,"imageio_ffmpeg":imageio_ffmpeg.__version__,"opencv":cv2.__version__,"exports":rows},indent=2),encoding="utf-8")
