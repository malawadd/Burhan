import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {gunzipSync} from 'node:zlib';
import {preparePixels,geometry,decodeOutputs,iou} from '../src/engine/decoder.ts';
const fixtures=new URL('./fixtures/',import.meta.url);
const reference=JSON.parse(await readFile(new URL('reference.json',fixtures),'utf8'));
const models=JSON.parse(await readFile(new URL('../assets/models/manifest.json',import.meta.url),'utf8'));
const sha=bytes=>createHash('sha256').update(bytes).digest('hex');
const rgba=await readFile(new URL(reference.input.file,fixtures));assert.equal(sha(rgba),reference.input.sha256,'Fixture RGBA identity');
async function unpack(spec){const bytes=gunzipSync(await readFile(new URL(spec.file,fixtures)));assert.equal(sha(bytes),spec.sha256,`${spec.file} fixture hash`);const copy=bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength);return spec.dtype==='float32'?new Float32Array(copy):new Uint8Array(copy);}
function maskBytes(mask){const pixels=new Uint8Array(mask.width*mask.height);let offset=0;mask.runs.forEach((run,index)=>{if(index%2)pixels.fill(1,offset,offset+run);offset+=run;});assert.equal(offset,pixels.length);return pixels;}
const metrics=[];
for(const entry of reference.models){
  const model=models.find(m=>m.id===entry.id);assert.ok(model);assert.equal(model.sha256,entry.modelSha256,'Model identity matches reference');
  const modelBytes=await readFile(new URL('../assets/models/'+model.file,import.meta.url));assert.equal(sha(modelBytes),entry.modelSha256,'Bundled model integrity');
  const prepared=preparePixels(new Uint8ClampedArray(rgba),reference.input.width,reference.input.height,model),expected=await unpack(entry.input);
  assert.equal(prepared.data.length,expected.length);const byteScale=model.kind==='dacl'?1:255;
  let changed=0,maxError=0,sumError=0;for(let p=0;p<expected.length;p++){const error=Math.abs(prepared.data[p]-expected[p])*byteScale;if(error>1e-5)changed++;maxError=Math.max(maxError,error);sumError+=error;}
  // OpenCV's optimized uint8 resize rounds some interpolation values down by
  // one integer compared with JS bilinear rounding. Quantify that difference.
  assert.ok(maxError<=1.0001,`${model.id} preprocessing maximum <= one intensity level: ${maxError}`);
  assert.ok(sumError/expected.length<=.15,`${model.id} preprocessing mean <= .15 intensity levels: ${sumError/expected.length}`);
  const outputs=await Promise.all(entry.outputs.map(async spec=>({data:await unpack(spec),dims:spec.dims})));
  const actual=decodeOutputs(outputs,model,geometry(reference.input.width,reference.input.height,model.inputSize,model.kind==='dacl'));
  assert.equal(actual.length,entry.observations.length,`${model.id} observation count`);let minimumMaskIoU=1,maxScoreError=0,maskPixelsDifferent=0;
  for(let index=0;index<actual.length;index++){
    const a=actual[index],b=entry.observations[index];assert.equal(a.classId,b.classId);assert.equal(a.label,b.label);assert.deepEqual(a.bbox,b.bbox,`${model.id} box ${index}`);const scoreError=Math.abs(a.score-b.score);maxScoreError=Math.max(maxScoreError,scoreError);assert.ok(scoreError<1e-5);
    if(b.mask){assert.ok(a.mask);const am=maskBytes(a.mask),bm=maskBytes(b.mask);let intersection=0,union=0;for(let p=0;p<am.length;p++){if(am[p]&&bm[p])intersection++;if(am[p]||bm[p])union++;if(am[p]!==bm[p])maskPixelsDifferent++;}const maskIoU=union?intersection/union:1;minimumMaskIoU=Math.min(minimumMaskIoU,maskIoU);assert.ok(maskIoU>=.999,`${model.id} mask ${index} IoU ${maskIoU}`);}
  }
  metrics.push({model:model.id,inputMaximumErrorIntensity:maxError,inputMeanErrorIntensity:sumError/expected.length,inputChangedFraction:changed/expected.length,observations:actual.length,maximumScoreError:maxScoreError,minimumMaskIoU,maskPixelsDifferent});
}
console.log(JSON.stringify({description:'Numerical preprocessing and decoder parity against real Python ONNX CPU outputs; no accuracy or device-performance claim.',pythonReference:reference.python,fixtureBytes:'~8 MB compressed; full outputs without quantization',metrics},null,2));

// Optional full JS/WASM inference. No onnxruntime-node dependency is required:
// onnxruntime-web itself supports its WASM backend under Node.
if(process.argv.includes('--wasm')){
  const ort=await import('onnxruntime-web');ort.env.wasm.numThreads=1;ort.env.wasm.wasmPaths=new URL('../node_modules/onnxruntime-web/dist/',import.meta.url).href;
  for(const entry of reference.models){const model=models.find(m=>m.id===entry.id),input=await unpack(entry.input),session=await ort.InferenceSession.create(new Uint8Array(await readFile(new URL('../assets/models/'+model.file,import.meta.url))),{executionProviders:['wasm']});
    const result=await session.run({[session.inputNames[0]]:new ort.Tensor(model.kind==='dacl'?'uint8':'float32',input,entry.input.dims)});let maximum=0,sum=0,count=0;
    for(let i=0;i<entry.outputs.length;i++){const expected=await unpack(entry.outputs[i]),actual=result[session.outputNames[i]].data;assert.equal(expected.length,actual.length);for(let p=0;p<expected.length;p++){const difference=Math.abs(expected[p]-actual[p]);maximum=Math.max(maximum,difference);sum+=difference;count++;}}
    assert.ok(maximum<(model.kind==='dacl'?.0001:.005),`${model.id} raw inference numerical tolerance`);
    const g=geometry(reference.input.width,reference.input.height,model.inputSize,model.kind==='dacl');
    const pureWasm=decodeOutputs(session.outputNames.map(name=>({data:result[name].data,dims:result[name].dims})),model,g);assert.equal(pureWasm.length,entry.observations.length,'WASM CPU golden-input candidate count');
    for(const output of Object.values(result))output.dispose();
    const jsInput=preparePixels(new Uint8ClampedArray(rgba),reference.input.width,reference.input.height,model);
    const jsResult=await session.run({[session.inputNames[0]]:new ort.Tensor(model.kind==='dacl'?'uint8':'float32',jsInput.data,entry.input.dims)});
    const jsObservations=decodeOutputs(session.outputNames.map(name=>({data:jsResult[name].data,dims:jsResult[name].dims})),model,g);
    let minimumBoxIoU=1,minimumMaskIoU=1,maximumScoreDifference=0,maximumBoxEdgeDrift=0,maximumMaskBoundaryDrift=0,maskPixelsDifferent=0;
    const spatialTolerance=model.kind==='dacl'?Math.ceil(Math.max(reference.input.width,reference.input.height)/(model.inputSize/4)):1;
    assert.equal(jsObservations.length,entry.observations.length,'Full JS preprocessing+WASM candidate count for fixture');
    for(const golden of entry.observations){const match=jsObservations.filter(row=>row.classId===golden.classId).sort((a,b)=>iou(b.bbox,golden.bbox)-iou(a.bbox,golden.bbox))[0];assert.ok(match,`Full JS candidate class ${golden.classId}`);minimumBoxIoU=Math.min(minimumBoxIoU,iou(match.bbox,golden.bbox));maximumScoreDifference=Math.max(maximumScoreDifference,Math.abs(match.score-golden.score));
      const ae=[match.bbox[0],match.bbox[1],match.bbox[0]+match.bbox[2],match.bbox[1]+match.bbox[3]],be=[golden.bbox[0],golden.bbox[1],golden.bbox[0]+golden.bbox[2],golden.bbox[1]+golden.bbox[3]];for(let edge=0;edge<4;edge++)maximumBoxEdgeDrift=Math.max(maximumBoxEdgeDrift,Math.abs(ae[edge]-be[edge]));
      if(golden.mask){const am=maskBytes(match.mask),bm=maskBytes(golden.mask);let intersection=0,union=0,different=0;for(let p=0;p<am.length;p++){if(am[p]&&bm[p])intersection++;if(am[p]||bm[p])union++;
        if(am[p]!==bm[p]){different++;const other=am[p]?bm:am,x=p%golden.mask.width,y=Math.floor(p/golden.mask.width);let distance=Infinity;for(let radius=1;radius<=Math.max(golden.mask.width,golden.mask.height)&&!Number.isFinite(distance);radius++){for(let dy=-radius;dy<=radius;dy++)for(let dx=-radius;dx<=radius;dx++){if(Math.max(Math.abs(dx),Math.abs(dy))!==radius)continue;const nx=x+dx,ny=y+dy;if(nx>=0&&nx<golden.mask.width&&ny>=0&&ny<golden.mask.height&&other[ny*golden.mask.width+nx])distance=radius;}}maximumMaskBoundaryDrift=Math.max(maximumMaskBoundaryDrift,distance);}
      }minimumMaskIoU=Math.min(minimumMaskIoU,union?intersection/union:1);maskPixelsDifferent+=different;
        const grid=model.kind==='dacl'?model.inputSize/4:160,pixelBudget=Math.max(2*Math.ceil(reference.input.width/grid)*Math.ceil(reference.input.height/grid),Math.ceil(union*.05));assert.ok(different<=pixelBudget,`Full JS mask disagreement ${different} <= budget ${pixelBudget} (two output-cell footprints or 5% union)`);
      }
    }
    console.log(JSON.stringify({model:model.id,wasmRawOutputMaximumAbsoluteError:maximum,wasmRawOutputMeanAbsoluteError:sum/count,fullJs:{observations:jsObservations.length,minimumBoxIoU,minimumMaskIoU,maximumBoxEdgeDrift,maximumMaskBoundaryDrift,maskPixelsDifferent,boxSpatialToleranceSourcePixels:spatialTolerance,maximumScoreDifference}}));
    // A one-pixel change in a tiny retained component can have low IoU. Keep the
    // measured IoU visible; bound drift to one native output-grid cell instead.
    assert.ok(maximumBoxEdgeDrift<=spatialTolerance,'Full JS box drift <= one output-grid cell');assert.ok(maximumScoreDifference<=.01,'Full JS score drift <= .01 on numerical fixture');
    for(const output of Object.values(jsResult))output.dispose();await session.release();
  }
}


