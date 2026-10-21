// Run through the local Vite dev server in an isolated browser, never shipped.
import * as ort from 'onnxruntime-web/wasm';
import {preparePixels,decodeOutputs} from '../src/engine/decoder.ts';
const assets=location.port==='4191'?'/assets/':'/';
ort.env.wasm.wasmPaths=assets+'ort/';ort.env.wasm.numThreads=1;ort.env.logLevel='error';
const base='/tests/fixtures/research/';
async function binary(file,type){const r=await fetch(base+file);if(!r.ok)throw Error('Fixture unavailable');const b=await new Response(r.body.pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();return type==='int64'?new BigInt64Array(b):type==='uint8'?new Uint8ClampedArray(b):new Float32Array(b);}
const overlap=(a,b)=>{const w=Math.max(0,Math.min(a[0]+a[2],b[0]+b[2])-Math.max(a[0],b[0])),h=Math.max(0,Math.min(a[1]+a[3],b[1]+b[3])-Math.max(a[1],b[1]));return w*h/(a[2]*a[3]+b[2]*b[3]-w*h)};
export async function run(){
 const fixtures=await(await fetch(base+'reference.json')).json(),models=await(await fetch(assets+'models/manifest.json')).json(),results=[];
 let session,current='',ticks=0,last=performance.now(),maxGap=0,running=true;
 const responsive=()=>{const now=performance.now();maxGap=Math.max(maxGap,now-last);last=now;ticks++;if(running)requestAnimationFrame(responsive)};requestAnimationFrame(responsive);
 try{
  for(const f of fixtures.cases){
   const model=models.find(m=>m.id===f.modelId);
   if(current!==model.id){await session?.release();session=undefined;const bytes=await(await fetch(assets+'models/'+model.file)).arrayBuffer(),digest=await crypto.subtle.digest('SHA-256',bytes);if([...new Uint8Array(digest)].map(v=>v.toString(16).padStart(2,'0')).join('')!==model.sha256)throw Error('Model checksum');session=await ort.InferenceSession.create(bytes,{executionProviders:['wasm']});current=model.id;}
   const rgba=await binary(f.rgba,'uint8'),pythonInput=await binary(f.input,'float32'),prepared=preparePixels(rgba,f.width,f.height,model);
   const measurements=[];
   for(const [source,data] of [['python-input',pythonInput],['browser-input',prepared.data]]){
    const input=new ort.Tensor('float32',data,[1,3,model.inputSize,model.inputSize]);const started=performance.now(),output=await session.run({[session.inputNames[0]]:input});const inferenceMs=performance.now()-started;
    const decoded=decodeOutputs(session.outputNames.map(name=>({name,data:output[name].data,dims:output[name].dims})),model,prepared.geometry);
    const matches=f.observations.map(ref=>({label:ref.label,bestIoU:Math.max(0,...decoded.filter(d=>d.label===ref.label).map(d=>overlap(d.bbox,ref.bbox)))}));
    let maxRawDelta=0;
    if(source==='python-input')for(const tensor of f.outputs){const reference=await binary(tensor.file,tensor.dtype),actual=output[tensor.name].data;if(reference.length!==actual.length)continue;for(let i=0;i<reference.length;i++)maxRawDelta=Math.max(maxRawDelta,Math.abs(Number(reference[i])-Number(actual[i])));}
    measurements.push({source,inferenceMs,expectedRows:f.observations.length,actualRows:decoded.length,minReferenceIoU:matches.length?Math.min(...matches.map(m=>m.bestIoU)):null,maxRawDelta:source==='python-input'?maxRawDelta:undefined});
    for(const t of Object.values(output))t.dispose();input.dispose();
   }
   results.push({id:f.id,model:model.id,measurements});
  }
 }finally{running=false;await session?.release()}
 return {runtime:ort.env.versions,scope:'Desktop browser real WASM image diagnostics, not physical camera or independent accuracy',results,mainThreadAnimationFrames:ticks,maxAnimationFrameGapMs:maxGap,memory:performance.memory?{usedJSHeapSize:performance.memory.usedJSHeapSize,totalJSHeapSize:performance.memory.totalJSHeapSize}:null};
}
