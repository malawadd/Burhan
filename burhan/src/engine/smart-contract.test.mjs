import test from 'node:test';
import assert from 'node:assert/strict';
import {validatePlan,normalizeBackendUrl,sceneDifference,normalizeSmartHistory} from '../lib/smart-contract.mjs';
test('untrusted agent output cannot introduce shell actions or unknown detectors',()=>{
 const result=validatePlan({scene:{environment:'indoor',description:'Room',confidence:2},guidance:'Look at the wall',suggestions:[{taskId:'wall',mode:'surface',title:'Wall',automatic:true},{taskId:'x',mode:'shell'}],actions:[{id:'1',type:'exec',instruction:'rm *'},{id:'2',type:'select_detector',mode:'shell'},{id:'3',type:'select_detector',mode:'surface',taskId:'wall',instruction:'Inspect wall'}]});
 assert.equal(result.actions.length,1);assert.equal(result.actions[0].mode,'surface');assert.equal(result.suggestions.length,1);assert.equal(result.scene.confidence,1);assert.equal(result.needsHumanReview,true);
});
test('backend origins require secure remote URLs and reject embedded credentials',()=>{
 assert.equal(normalizeBackendUrl('https://merhaba-api.kiyan.finance/'),'https://merhaba-api.kiyan.finance');assert.equal(normalizeBackendUrl('http://127.0.0.1:3210'),'http://127.0.0.1:3210');
 for(const bad of ['http://example.com','https://user:secret@example.com','https://example.com/api','https://example.com?token=secret','javascript:alert(1)'])assert.throws(()=>normalizeBackendUrl(bad));
});
test('malformed guidance fails closed and scene metric detects changed frames',()=>{
 assert.throws(()=>validatePlan({scene:{},guidance:'',actions:[],suggestions:[]}));assert.equal(sceneDifference([0,0],[1,1]),1);assert.equal(sceneDifference([.1,.1],[.1,.1]),0);
});
test('native guidance provenance is bounded and malformed steps cannot corrupt saved state',()=>{
 const safe=normalizeSmartHistory({sessionId:'s',provider:'codex',scene:'Room',steps:[{at:'invalid',instruction:'bad',mode:'surface'},{at:'2026-10-21T10:00:00Z',instruction:'Move closer',mode:'surface',scene:'Room',reason:'More detail'}]});
 assert.equal(safe.steps.length,1);assert.equal(safe.steps[0].mode,'surface');assert.throws(()=>normalizeSmartHistory({sessionId:'s',provider:{},steps:[]}));
});
