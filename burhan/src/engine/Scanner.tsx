import {useEffect,useRef,useState} from 'react';
import {Capacitor} from '@capacitor/core';
import {BurhanNative as nativeEngine} from '../native';
import type {Evidence,Inspection,Language,ModelInfo,Observation} from '../types';
import {paintObservations} from './paint';
import {CandidateTracker} from './stability';
import './Scanner.css';

export interface ScannerProps {inspection:Inspection;onCapture:(evidence:Evidence,observations:Observation[])=>void;onClose:()=>void;autoSave:boolean;language:Language;followUp?:boolean;findingId?:string}
interface AnalysedFrame {canvas:HTMLCanvasElement;observations:Observation[];capturedAt:string;inferenceMs:number}
export function Scanner({inspection,onCapture,onClose,autoSave,language,followUp=false,findingId}:ScannerProps){
  const ar=language==='ar',native=Capacitor.isNativePlatform(),manual=inspection.mode==='manual'||followUp;
  const text=(en:string,arabic:string)=>ar?arabic:en;
  const video=useRef<HTMLVideoElement>(null),overlay=useRef<HTMLCanvasElement>(null),stream=useRef<MediaStream|null>(null),worker=useRef<Worker|null>(null),frame=useRef<AnalysedFrame|null>(null),tracker=useRef(new CandidateTracker()),running=useRef(false),busy=useRef(false),mounted=useRef(true),requestId=useRef(0),callbacks=useRef(new Map<number,{resolve:(result:{observations:Observation[];inferenceMs:number})=>void;reject:(error:Error)=>void}>());
  const captureCallback=useRef(onCapture);captureCallback.current=onCapture;
  const [error,setError]=useState(''),[status,setStatus]=useState(text('Ready','جاهز')),[active,setActive]=useState(false),[saved,setSaved]=useState(0),[observations,setObservations]=useState<Observation[]>([]),[fps,setFps]=useState(0),[age,setAge]=useState<number|null>(null),[detailed,setDetailed]=useState(false),[torch,setTorch]=useState(false),[zoom,setZoom]=useState(1),[zoomMax,setZoomMax]=useState(1),[hasTorch,setHasTorch]=useState(false),[processing,setProcessing]=useState(false);
  const lastResult=useRef(0),models=useRef<ModelInfo[]|null>(null),nativeLaunching=useRef(false);
  const cameraGeneration=useRef(0),lifecycleGeneration=useRef(0),startingCamera=useRef(false);
  async function modelFor(detail=false){
    if(!models.current){const response=await fetch('/models/manifest.json');if(!response.ok)throw Error('Model manifest unavailable');models.current=await response.json();}
    const id=inspection.mode==='surface'?(detail?'surface512':'surface320'):inspection.mode;
    const model=models.current!.find(m=>m.id===id);if(!model)throw Error(`Model unavailable: ${id}`);return model;
  }
  function stop(){cameraGeneration.current++;startingCamera.current=false;running.current=false;stream.current?.getTracks().forEach(track=>track.stop());stream.current=null;setActive(false);setTorch(false);setHasTorch(false);}
  function save(captured:AnalysedFrame,selected=captured.observations){
    try{const canvas=document.createElement('canvas');canvas.width=captured.canvas.width;canvas.height=captured.canvas.height;const ctx=canvas.getContext('2d')!;ctx.drawImage(captured.canvas,0,0);paintObservations(ctx,selected,canvas.width,canvas.height);
      const evidence:Evidence={id:crypto.randomUUID(),capturedAt:captured.capturedAt,width:canvas.width,height:canvas.height,original:captured.canvas.toDataURL('image/jpeg',.9),annotated:canvas.toDataURL('image/jpeg',.9)};
      captureCallback.current(evidence,selected);setSaved(count=>count+1);setStatus(text('Evidence saved','تم حفظ الدليل'));
    }catch(e){setError(e instanceof Error?e.message:String(e));}
  }
  function getWorker(){if(!worker.current){const created=new Worker(new URL('./inference.worker.ts',import.meta.url),{type:'module'});created.onmessage=(event)=>{if(event.data.status){setStatus(text('Loading and verifying model…','جار تحميل النموذج والتحقق منه…'));return;}const callback=callbacks.current.get(event.data.id);if(!callback)return;callbacks.current.delete(event.data.id);if(event.data.error)callback.reject(Error(event.data.error));else callback.resolve(event.data);};created.onerror=()=>{for(const callback of callbacks.current.values())callback.reject(Error('Inference worker failed. Stop and reopen the scanner.'));callbacks.current.clear();};worker.current=created;}return worker.current;}
  async function analyse(canvas:HTMLCanvasElement,detail=false):Promise<AnalysedFrame>{
    const lifecycle=lifecycleGeneration.current,capturedAt=new Date().toISOString(),model=await modelFor(detail);
    if(!mounted.current||lifecycle!==lifecycleGeneration.current)throw Error('Scanner closed');
    const rgba=canvas.getContext('2d')!.getImageData(0,0,canvas.width,canvas.height).data,id=++requestId.current;
    const result=await new Promise<{observations:Observation[];inferenceMs:number}>((resolve,reject)=>{callbacks.current.set(id,{resolve,reject});getWorker().postMessage({id,model,width:canvas.width,height:canvas.height,rgba},[rgba.buffer]);});
    const frameId=crypto.randomUUID();return {canvas,capturedAt,inferenceMs:result.inferenceMs,observations:result.observations.filter(o=>inspection.mode!=='crack'||o.label==='crack').map(o=>({...o,frameId,capturedAt,modelId:model.id,inferenceMs:result.inferenceMs,model:{sha256:model.sha256,inputSize:model.inputSize,preprocessing:model.preprocessing,confidence:model.confidence,nms:model.nms,runtime:'onnxruntime-web@1.30.0/wasm'}}))};
  }
  function snapshot(){const canvas=document.createElement('canvas'),element=video.current!;canvas.width=element.videoWidth;canvas.height=element.videoHeight;if(!canvas.width||!canvas.height)throw Error(text('Camera is not ready','الكاميرا غير جاهزة'));canvas.getContext('2d')!.drawImage(element,0,0);return canvas;}
  function showFrame(result:AnalysedFrame){frame.current=result;lastResult.current=performance.now();setAge(Math.max(0,Date.now()-Date.parse(result.capturedAt)));setObservations(result.observations);if(overlay.current){overlay.current.width=result.canvas.width;overlay.current.height=result.canvas.height;const ctx=overlay.current.getContext('2d')!;ctx.clearRect(0,0,overlay.current.width,overlay.current.height);paintObservations(ctx,result.observations,overlay.current.width,overlay.current.height);}}
  async function tick(generation:number){
    if(!running.current||!mounted.current||generation!==cameraGeneration.current)return;
    if(busy.current){setTimeout(()=>void tick(generation),100);return;}busy.current=true;const started=performance.now();
    try{if(video.current?.readyState&&video.current.readyState>=2){const canvas=snapshot();if(manual){frame.current={canvas,observations:[],capturedAt:new Date().toISOString(),inferenceMs:0};}else{const result=await analyse(canvas);if(!running.current||!mounted.current||generation!==cameraGeneration.current)return;showFrame(result);setFps(1000/(performance.now()-started));setStatus(result.observations.length?text('Candidate observations','ملاحظات مرشحة'):text('No candidate detected — review required','لم تُرصد ملاحظة — يلزم التقييم'));if(autoSave){const ready=tracker.current.update(result.observations,performance.now());if(ready.length)save(result,ready);}}}}
    catch(e){if(mounted.current&&generation===cameraGeneration.current){setError(e instanceof Error?e.message:String(e));stop();}}
    finally{busy.current=false;if(running.current&&mounted.current&&generation===cameraGeneration.current)setTimeout(()=>void tick(generation),manual?150:80);}
  }
  async function startBrowser(){
    if(running.current||startingCamera.current)return;startingCamera.current=true;const generation=++cameraGeneration.current;
    setError('');tracker.current.reset();frame.current=null;setObservations([]);setStatus(text('Opening camera…','جار فتح الكاميرا…'));
    try{if(!window.isSecureContext||!navigator.mediaDevices?.getUserMedia)throw Error(text('Camera requires localhost or HTTPS. You can still import an image.','تتطلب الكاميرا localhost أو HTTPS. يمكنك استيراد صورة.'));
      const camera=await navigator.mediaDevices.getUserMedia({video:{facingMode:{ideal:'environment'},width:{ideal:1280},height:{ideal:720}},audio:false});if(!mounted.current||generation!==cameraGeneration.current){camera.getTracks().forEach(t=>t.stop());return;}stream.current=camera;
      const element=video.current!;element.srcObject=camera;await element.play();
      if(!mounted.current||generation!==cameraGeneration.current){camera.getTracks().forEach(t=>t.stop());return;}
      const capabilities=camera.getVideoTracks()[0].getCapabilities() as MediaTrackCapabilities&{torch?:boolean;zoom?:{max:number}};setHasTorch(!!capabilities.torch);setZoomMax(capabilities.zoom?.max||1);setZoom(1);
      camera.getVideoTracks()[0].addEventListener('ended',()=>{if(mounted.current&&generation===cameraGeneration.current){stop();setStatus(text('Camera interrupted. Resume to reconnect.','انقطع اتصال الكاميرا. استأنف لإعادة الاتصال.'));}},{once:true});
      setActive(true);running.current=true;setStatus(manual?text('Manual evidence camera','كاميرا الأدلة اليدوية'):text('Starting analysis…','جار بدء التحليل…'));void tick(generation);
    }catch(e){if(mounted.current&&generation===cameraGeneration.current){setError(e instanceof Error?e.message:String(e));stop();}}
    finally{if(generation===cameraGeneration.current)startingCamera.current=false;}
  }
  async function startNative(){if(nativeLaunching.current)return;nativeLaunching.current=true;setError('');setProcessing(true);
    try{const result=await nativeEngine.scan({inspectionId:inspection.id,mode:manual?'manual':inspection.mode,autoSave:manual?false:autoSave,language,followUp,findingId});for(const capture of result.captures){captureCallback.current(capture.evidence,capture.observations);setSaved(count=>count+1);}setStatus(text('Native inspection saved. Continue scanning or review findings.','تم حفظ الفحص. تابع التصوير أو راجع الملاحظات.'));}
    catch(e){setError(e instanceof Error?e.message:String(e));}finally{nativeLaunching.current=false;if(mounted.current)setProcessing(false);}
  }
  useEffect(()=>{mounted.current=true;lifecycleGeneration.current++;if(native)void startNative();else void startBrowser();const visibility=()=>{if(document.hidden){stop();setStatus(text('Paused in background. Resume camera to continue.','توقف في الخلفية. استأنف الكاميرا للمتابعة.'));}};document.addEventListener('visibilitychange',visibility);const timer=setInterval(()=>{if(lastResult.current){const milliseconds=performance.now()-lastResult.current;setAge(frame.current?Math.max(0,Date.now()-Date.parse(frame.current.capturedAt)):milliseconds);if(milliseconds>2000&&overlay.current)overlay.current.getContext('2d')?.clearRect(0,0,overlay.current.width,overlay.current.height);}},500);
    return()=>{mounted.current=false;lifecycleGeneration.current++;cameraGeneration.current++;startingCamera.current=false;running.current=false;stream.current?.getTracks().forEach(t=>t.stop());stream.current=null;worker.current?.terminate();worker.current=null;for(const cb of callbacks.current.values())cb.reject(Error('Scanner closed'));callbacks.current.clear();document.removeEventListener('visibilitychange',visibility);clearInterval(timer);};
  },[]);
  async function importImage(file:File){stop();setError('');setProcessing(true);setStatus(text('Analysing selected image…','جار تحليل الصورة المختارة…'));
    try{const bitmap=await createImageBitmap(file,{imageOrientation:'from-image'});const canvas=document.createElement('canvas');const ratio=Math.min(1,2048/Math.max(bitmap.width,bitmap.height));canvas.width=Math.round(bitmap.width*ratio);canvas.height=Math.round(bitmap.height*ratio);canvas.getContext('2d')!.drawImage(bitmap,0,0,canvas.width,canvas.height);bitmap.close();const result=manual?{canvas,observations:[],capturedAt:new Date().toISOString(),inferenceMs:0}:await analyse(canvas,detailed);frame.current=result;setObservations(result.observations);save(result);setStatus(text('Image evidence saved. Resume camera for live inspection.','تم حفظ الصورة. استأنف الكاميرا للفحص المباشر.'));}
    catch(e){setError(e instanceof Error?e.message:String(e));}finally{setProcessing(false);}
  }
  async function changeTrackConstraint(kind:'torch'|'zoom',value:boolean|number){try{const track=stream.current?.getVideoTracks()[0];if(track)await track.applyConstraints({advanced:[{[kind]:value} as MediaTrackConstraintSet]});if(kind==='torch')setTorch(value as boolean);else setZoom(value as number);}catch(e){setError(e instanceof Error?e.message:String(e));}}
  function manualCapture(){try{if(!manual&&frame.current){if(!active||performance.now()-lastResult.current<2500)save(frame.current);else setError(text('Wait for a fresh analysed frame before saving.','انتظر إطار تحليل حديث قبل الحفظ.'));}else if(active)save({canvas:snapshot(),observations:[],capturedAt:new Date().toISOString(),inferenceMs:0});else setError(text('Resume camera or import an image.','استأنف الكاميرا أو استورد صورة.'));}catch(e){setError(e instanceof Error?e.message:String(e));}}
  return <section className="burhan-scanner" role="dialog" aria-modal="true" aria-label={text('Live inspection','الفحص المباشر')} dir={ar?'rtl':'ltr'}>
    <header className="scan-header"><div><strong>BURHAN / {text('LIVE INSPECTION','فحص مباشر')}</strong><p>{inspection.title} · {inspection.location}</p></div><button onClick={()=>{stop();onClose();}}>{text('Done · review','إنهاء · مراجعة')}</button></header>
    <div className="scan-view">{native?<div className="scan-native"><strong>{text('Native camera engine','محرك الكاميرا الأصلي')}</strong><p>{text('CameraX · ONNX Runtime · offline','CameraX · ONNX Runtime · دون اتصال')}</p><button disabled={processing} onClick={()=>void startNative()}>{processing?text('Camera is open…','الكاميرا مفتوحة…'):text('Open live camera','فتح الكاميرا المباشرة')}</button></div>:<><video ref={video} autoPlay muted playsInline aria-label={text('Live camera preview','معاينة الكاميرا المباشرة')}/><canvas ref={overlay} className="scan-overlay" aria-hidden="true"/>{!active&&<div className="scan-paused">{text('Camera paused','الكاميرا متوقفة')}</div>}</>}</div>
    <div className="scan-status" role="status"><span>{status}</span><span>{saved} {text('saved','محفوظ')} {!manual&&!native&&<>· {fps.toFixed(1)} {text('analysis/s','تحليل/ث')} {age!==null&&`· ${(age/1000).toFixed(1)}s`}</>}</span></div>
    {error&&<p className="scan-error" role="alert">{error}</p>}
    {!manual&&<p className="scan-note">{text('Experimental visual observations. Review every finding. No detection does not mean a pass.','ملاحظات بصرية تجريبية. راجع كل ملاحظة. عدم الرصد لا يعني اجتياز الفحص.')}{autoSave&&' '+text('Stable candidates save automatically; review duplicates.','تُحفظ الملاحظات المستقرة تلقائياً؛ راجع التكرارات.')}</p>}
    <div className="scan-controls">{!native&&<><button disabled={processing} onClick={()=>active?stop():void startBrowser()}>{active?text('Pause camera','إيقاف الكاميرا'):text('Resume camera','استئناف الكاميرا')}</button><button className="scan-primary" disabled={processing||!active} onClick={manualCapture}>{text('Save evidence','حفظ الدليل')}</button>{hasTorch&&<button aria-pressed={torch} onClick={()=>void changeTrackConstraint('torch',!torch)}>{text('Torch','المصباح')} {torch?'●':'○'}</button>}{zoomMax>1&&<label>{text('Zoom','تقريب')}<input aria-label={text('Camera zoom','تقريب الكاميرا')} type="range" min={1} max={zoomMax} step={.1} value={zoom} onChange={e=>void changeTrackConstraint('zoom',Number(e.target.value))}/></label>}</>}
      <label className="scan-import">{text('Import image','استيراد صورة')}<input type="file" accept="image/*" disabled={processing} onChange={event=>{const file=event.target.files?.[0];if(file)void importImage(file);event.target.value='';}}/></label>
      {inspection.mode==='surface'&&!manual&&<label><input type="checkbox" checked={detailed} onChange={event=>setDetailed(event.target.checked)}/>{text('Detailed imported image (512; slower)','صورة مستوردة تفصيلية (512؛ أبطأ)')}</label>}
    </div>
    {observations.length>0&&<div className="scan-labels">{observations.slice(0,12).map((observation,index)=><span key={`${observation.frameId}-${index}`}>{observation.label.replaceAll('_',' ')} {Math.round(observation.score*100)}%</span>)}</div>}
  </section>;
}
export default Scanner;
