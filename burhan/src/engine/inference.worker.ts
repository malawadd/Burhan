/// <reference lib="webworker" />
import * as ort from 'onnxruntime-web/wasm';
import { decodeOutputs, preparePixels } from './decoder';
import type { ModelInfo } from '../types';

let session:ort.InferenceSession|undefined,loadedId='';
ort.env.wasm.wasmPaths='/ort/';
ort.env.wasm.numThreads=1;
ort.env.logLevel='error';
// An imported detailed image may arrive while a paused live frame is finishing.
// Serialize requests so model switching cannot release an active session.
let queue=Promise.resolve();
self.onmessage=(event:MessageEvent)=>{queue=queue.then(()=>processMessage(event));};
async function processMessage(event:MessageEvent){
  const {id,model,width,height,rgba}=event.data as {id:number;model:ModelInfo;width:number;height:number;rgba:Uint8ClampedArray};
  try{
    if(!session||loadedId!==model.id){
      self.postMessage({id,status:'loading'});await session?.release();session=undefined;loadedId='';
      const response=await fetch(`/models/${model.file}`);if(!response.ok)throw Error(`Model unavailable (${response.status})`);
      const bytes=await response.arrayBuffer();
      const digest=await crypto.subtle.digest('SHA-256',bytes),hash=Array.from(new Uint8Array(digest),b=>b.toString(16).padStart(2,'0')).join('');
      if(hash!==model.sha256)throw Error('Model integrity check failed');
      session=await ort.InferenceSession.create(bytes,{executionProviders:['wasm'],graphOptimizationLevel:'all'});loadedId=model.id;
    }
    const start=performance.now(),prepared=preparePixels(rgba,width,height,model);
    const input=new ort.Tensor(model.kind==='dacl'?'uint8':'float32',prepared.data,[1,3,model.inputSize,model.inputSize]);
    const output=await session.run({[session.inputNames[0]]:input});
    const observations=decodeOutputs(session.outputNames.map(name=>({data:output[name].data as Float32Array,dims:output[name].dims})),model,prepared.geometry);
    const inferenceMs=performance.now()-start;
    for(const tensor of Object.values(output))tensor.dispose();input.dispose();
    self.postMessage({id,observations,inferenceMs});
  }catch(error){self.postMessage({id,error:error instanceof Error?error.message:String(error)});}
}
