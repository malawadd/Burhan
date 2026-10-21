import test from 'node:test';
import assert from 'node:assert/strict';
import {preparePixels,decodeOutputs,geometry,roundEven,encodeMask,iou} from './decoder.ts';
const model={id:'test',file:'test.onnx',sha256:'',bytes:0,inputSize:640,classes:['crack','joint'],kind:'yolo_segmentation',mode:'crack',preprocessing:'',confidence:.25,nms:.45,scope:'',license:''};
test('Python ties-to-even and letterbox geometry',()=>{assert.equal(roundEven(2.5),2);assert.equal(roundEven(3.5),4);assert.deepEqual(geometry(1280,720,640),{width:1280,height:720,size:640,scale:.5,padX:0,padY:140,resizedWidth:640,resizedHeight:360});});
test('RGB float YOLO, BGR uint8 DACL, and 114 padding',()=>{
  const pixels=new Uint8ClampedArray([255,30,4,255,255,30,4,255]);
  const yolo=preparePixels(pixels,2,1,{...model,inputSize:4});assert.equal(yolo.data[0],new Float32Array([114/255])[0]);assert.equal(yolo.data[4],1);assert.equal(yolo.data[20],new Float32Array([30/255])[0]);
  const dacl=preparePixels(pixels,2,1,{...model,kind:'dacl',inputSize:4});assert.ok(dacl.data instanceof Uint8Array);assert.equal(dacl.data[0],4);assert.equal(dacl.data[16],30);assert.equal(dacl.data[32],255);
});
test('YOLO per-class NMS, source coordinates and prototype masks',()=>{
  const data=new Float32Array(38*8400),proto=new Float32Array(32*160*160);proto.fill(1,0,160*160);
  for(let i=0;i<2;i++){data[i]=320;data[8400+i]=320;data[2*8400+i]=320;data[3*8400+i]=320;data[4*8400+i]=i===0?.9:.8;data[6*8400+i]=1;}
  const rows=decodeOutputs([{dims:[1,38,8400],data},{dims:[1,32,160,160],data:proto}],model,geometry(640,320,640));assert.equal(rows.length,1);assert.deepEqual(rows[0].bbox,[160,0,320,320]);assert.equal(rows[0].label,'crack');assert.equal(rows[0].mask.width,640);assert.equal(rows[0].mask.runs.reduce((a,b)=>a+b,0),640*320);assert.equal(rows[0].mask.runs.filter((_,i)=>i%2===1).reduce((a,b)=>a+b,0),320*320);
});
test('DACL overlapping sigmoid classes, eight connectivity and min component area',()=>{
  const m={...model,kind:'dacl',inputSize:16,classes:Array.from({length:19},(_,i)=>`class${i}`),confidence:.5};const data=new Float32Array(19*16).fill(-10);
  for(const c of [0,1,13])for(const p of [0,1,4,5])data[c*16+p]=10;data[2*16+15]=10;
  const rows=decodeOutputs([{dims:[1,19,4,4],data}],m,geometry(8,8,16,true));assert.equal(rows.length,2);assert.deepEqual(rows.map(r=>r.classId),[0,1]);assert.deepEqual(rows[0].bbox,[0,0,4,4]);assert.deepEqual(rows[0].mask,rows[1].mask);
});
test('RLE begins with background, covers every pixel and IoU is correct',()=>{assert.deepEqual(encodeMask(2,2,(x,y)=>x===y).runs,[0,1,2,1]);assert.equal(iou([0,0,10,10],[5,0,10,10]),1/3);});
