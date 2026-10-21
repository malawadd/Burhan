/// <reference lib="webworker" />
import * as ort from 'onnxruntime-web/wasm';
import { decodeOutputs, preparePixels } from './decoder';
import type { ModelInfo } from '../types';

let session:ort.InferenceSession|undefined,loadedId='';
ort.env.wasm.wasmPaths='/ort/';
ort.env.wasm.numThreads=1;
ort.env.logLevel='error';
// One active frame plus only the newest pending frame; never grow a video backlog.
let active=false,pending:MessageEvent|undefined;
self.onmessage=(event:MessageEvent)=>{if(pending)self.postMessage({id:pending.data.id,error:'Frame superseded'});pending=event;if(!active)void drain();};
async function drain(){active=true;try{while(pending){const event=pending;pending=undefined;await processMessage(event);}}finally{active=false;}}
async function processMessage(event:MessageEvent){
  const {id,model,width,height,rgba}=event.data as {id:number;model:ModelInfo;width:number;height:number;rgba:Uint8ClampedArray};
  try{
    if(!session||loadedId!==`${model.id}:${model.sha256}`){
      self.postMessage({id,status:'loading'});await session?.release();session=undefined;loadedId='';
      const response=await fetch(`/models/${model.file}`);if(!response.ok)throw Error(`Model unavailable (${response.status})`);
      const bytes=await response.arrayBuffer();
      const digest=await crypto.subtle.digest('SHA-256',bytes),hash=Array.from(new Uint8Array(digest),b=>b.toString(16).padStart(2,'0')).join('');
      if(hash!==model.sha256)throw Error('Model integrity check failed');
      session=await ort.InferenceSession.create(bytes,{executionProviders:['wasm'],graphOptimizationLevel:'all'});loadedId=`${model.id}:${model.sha256}`;
    }
    const start=performance.now(),prepared=preparePixels(rgba,width,height,model);
    const input=new ort.Tensor(model.kind==='dacl'?'uint8':'float32',prepared.data,[1,3,model.inputSize,model.inputSize]);
    let output:Record<string,ort.Tensor>|undefined;
    try{output=await session.run({[session.inputNames[0]]:input});const observations=decodeOutputs(session.outputNames.map(name=>({name,data:output![name].data as Float32Array|BigInt64Array,dims:output![name].dims})),model,prepared.geometry);
    const inferenceMs=performance.now()-start;
    self.postMessage({id,observations,inferenceMs});
    }finally{for(const tensor of Object.values(output??{}))tensor.dispose();input.dispose();}
  }catch(error){self.postMessage({id,error:error instanceof Error?error.message:String(error)});}
}
