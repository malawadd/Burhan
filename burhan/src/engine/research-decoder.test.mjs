import test from 'node:test';
import assert from 'node:assert/strict';
import {decodeOutputs,geometry} from './decoder.ts';
import {validatePlan,normalizeSmartHistory} from '../lib/smart-contract.mjs';

const model={id:'research-phele',file:'phele.onnx',sha256:'a'.repeat(64),bytes:1,inputSize:320,classes:['peeling_paint','mold_appearance'],kind:'ssdlite_detection',mode:'research',preprocessing:'rgb_float_letterbox',confidence:.25,classConfidence:{peeling_paint:.35,mold_appearance:.7},nms:.45,scope:'Research only',license:'test',researchOnly:true};
const ssd=(boxes,labels,scores)=>[
  {name:'scores',dims:[scores.length],data:new Float32Array(scores)},
  {name:'boxes',dims:[labels.length,4],data:new Float32Array(boxes)},
  {name:'labels',dims:[labels.length],data:new BigInt64Array(labels.map(BigInt))},
];

test('SSDLite uses named outputs, one-based int64 labels, per-class thresholds and retained fractional extents',()=>{
  const rows=decodeOutputs(ssd([1.1,81.1,100.1,150.1,10,90,100,150,10,90,100,150],[1,2,2],[.9,.6,.8]),model,geometry(641,320,320));
  assert.equal(rows.length,2);assert.deepEqual(rows.map(r=>r.label),['peeling_paint','mold_appearance']);
  assert.deepEqual(rows[0].bbox,[2,2,199,139]);
});
test('SSDLite preserves overlapping graph results without a second NMS and rejects ambiguous output contracts',()=>{
  const rows=decodeOutputs(ssd([0,0,100,100,0,0,100,100],[1,1],[.9,.8]),model,geometry(320,320,320));
  assert.equal(rows.length,2);assert.throws(()=>decodeOutputs(ssd([0,0,100,100],[1],[.9]).map(o=>({...o,name:'wrong'})),model,geometry(320,320,320)),/SSDLite/);
});
test('512px YOLO accepts 5376 anchors, thresholds each class and uses explicit kind instead of model ID',()=>{
  const m={...model,id:'crack',kind:'yolo_detection',inputSize:512,classes:['unseated','absent'],classConfidence:{unseated:.025,absent:.9}};
  const anchors=5376,data=new Float32Array(6*anchors);
  for(let i=0;i<3;i++){data[i]=100+i*120;data[anchors+i]=100;data[2*anchors+i]=40;data[3*anchors+i]=40;}
  data[4*anchors]=.03;data[5*anchors+1]=.89;data[5*anchors+2]=.95;
  const rows=decodeOutputs([{dims:[1,6,anchors],data}],m,geometry(512,512,512));
  assert.equal(rows.length,2);assert.deepEqual(rows.map(r=>r.label),['absent','unseated']);assert.ok(rows.every(r=>r.mask===undefined));
});
test('Remote contracts retain research guidance and history while dropping unknown analysis modes',()=>{
  const plan=validatePlan({scene:{environment:'indoor',description:'Wall',confidence:.8},guidance:'Review wall',reason:'Visible area',suggestions:[{taskId:'research:walls_finishes:peeling_paint',mode:'research',title:'Paint'},{taskId:'arbitrary',mode:'shell'}],actions:[]});
  assert.equal(plan.suggestions.length,1);assert.equal(plan.suggestions[0].mode,'research');
  const history=normalizeSmartHistory({sessionId:'s',provider:'codex',steps:[{at:new Date().toISOString(),instruction:'Review',reason:'Evidence',mode:'research',scene:'wall'}]});assert.equal(history.steps[0].mode,'research');
});
