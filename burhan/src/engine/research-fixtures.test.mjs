import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {gunzipSync} from 'node:zlib';
import {decodeOutputs,geometry,preparePixels} from './decoder.ts';
const base=new URL('../../tests/fixtures/research/',import.meta.url);
const fixtures=JSON.parse(readFileSync(new URL('reference.json',base)));
const models=JSON.parse(readFileSync(new URL('../../assets/models/manifest.json',import.meta.url)));
function load(file,type){const data=gunzipSync(readFileSync(new URL(file,base))),buffer=data.buffer.slice(data.byteOffset,data.byteOffset+data.byteLength);return type==='int64'?new BigInt64Array(buffer):type==='uint8'?new Uint8ClampedArray(buffer):new Float32Array(buffer);}
const canonical=rows=>rows.map(r=>({label:r.label,bbox:r.bbox,score:r.score})).sort((a,b)=>a.label.localeCompare(b.label)||a.bbox[0]-b.bbox[0]||a.bbox[1]-b.bbox[1]||b.score-a.score);
for(const fixture of fixtures.cases){
  test(`Real reused TRAIN photograph decoder parity: ${fixture.id}`,()=>{
    const model=models.find(m=>m.id===fixture.modelId),outputs=fixture.outputs.map(o=>({name:o.name,dims:o.dims,data:load(o.file,o.dtype)}));
    const actual=canonical(decodeOutputs(outputs,model,geometry(fixture.width,fixture.height,model.inputSize))),expected=canonical(fixture.observations);
    assert.equal(actual.length,expected.length);
    for(let i=0;i<actual.length;i++){assert.equal(actual[i].label,expected[i].label);assert.deepEqual(actual[i].bbox,expected[i].bbox);assert.ok(Math.abs(actual[i].score-expected[i].score)<1e-6);}
  });
  test(`Real reused TRAIN photograph preprocessing diagnostic: ${fixture.id}`,()=>{
    const model=models.find(m=>m.id===fixture.modelId),rgba=load(fixture.rgba,'uint8'),reference=load(fixture.input,'float32'),prepared=preparePixels(rgba,fixture.width,fixture.height,model).data;
    assert.equal(prepared.length,reference.length);let max=0,total=0,differing=0;
    for(let i=0;i<prepared.length;i++){const delta=Math.abs(prepared[i]-reference[i]);max=Math.max(max,delta);total+=delta;if(delta>1e-6)differing++;}
    // cv::resize's integer SIMD rounding differs from mathematical bilinear by at most one level.
    assert.ok(max<=1/255+1e-7,`max RGB delta ${max}`);assert.ok(total/prepared.length<.001,`mean RGB delta ${total/prepared.length}`);
    console.log(JSON.stringify({fixture:fixture.id,maxRgbDelta:max,meanRgbDelta:total/prepared.length,differingSamples:differing}));
  });
}
