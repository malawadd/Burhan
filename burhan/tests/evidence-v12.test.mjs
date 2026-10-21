import test from 'node:test';import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {validateState,snapshotReport} from '../src/lib/core.mjs';
import {evidenceMetadataHtml} from '../src/lib/evidence-report.mjs';
const stamp='2026-10-21T00:00:00Z';
const image='data:image/png;base64,AAAA';
const make=()=>({schemaVersion:1,projects:[{id:'p',name:'Site',address:'',createdAt:stamp}],inspections:[{id:'i',projectId:'p',title:'Wall',location:'Room',status:'in_progress',mode:'manual',checklist:{},acknowledged:true,inspector:'QA',stage:'structure',taskId:'',createdAt:stamp,updatedAt:stamp}],findings:[{id:'f',inspectionId:'i',label:'Wall',notes:'',action:'',review:'confirmed',source:'manual',status:'open',priority:'medium',evidence:[{id:'e',original:image,annotated:image,capturedAt:stamp,width:2048,height:1536,captureSource:'photo',detectorBaseline:image,annotations:{version:1,shapes:[{kind:'arrow',points:[.1,.2,.7,.8]},{kind:'ellipse',points:[.2,.2,.4,.6]}]}}],followUp:[],createdAt:stamp,updatedAt:stamp}],reports:[],settings:{language:'en',theme:'dark',inspector:'QA',autoSave:true}});

test('camera preference catalogue has all independent bilingual switches and intended defaults',()=>{
  const definitions=JSON.parse(readFileSync(new URL('../public/camera-settings.json',import.meta.url),'utf8'));
  assert.equal(definitions.schemaVersion,1);assert.equal(definitions.settings.length,16);assert.equal(new Set(definitions.settings.map(d=>d.key)).size,16);
  const off=['focusLock','framingGrid','cameraLevel','higherDetailPhotos','photoMarkup','individualPhotoSharing'];
  for(const d of definitions.settings){assert.equal(d.default,!off.includes(d.key));assert.ok(d.label.en&&d.label.ar&&d.description.en&&d.description.ar);assert.equal(typeof d.nativeOnly,'boolean');}
});
test('new evidence metadata validates and reports preserve immutable original, detector baseline and markup',()=>{
  const state=make();validateState(state);const report=snapshotReport(state,'i','en','r',stamp);state.reports.push(report);
  state.findings[0].evidence[0].annotated='data:image/png;base64,NEW';state.findings[0].evidence[0].annotations.shapes[0].points[0]=.3;
  assert.equal(report.snapshot.findings[0].evidence[0].annotated,image);assert.equal(report.snapshot.findings[0].evidence[0].annotations.shapes[0].points[0],.1);validateState(state);
  const e=state.findings[0].evidence[0];assert.match(evidenceMetadataHtml(e),/2048 × 1536 px.*Photo capture.*Inspector markup: 2/);assert.match(evidenceMetadataHtml(e,'ar'),/التقاط صور/);
  delete e.captureSource;delete e.detectorBaseline;delete e.annotations;assert.equal(validateState(state),state);
});
test('malformed annotations, unsupported capture sources and missing baselines fail before restore',()=>{
  const variants=[e=>e.captureSource='live-cloud',e=>e.annotations.version=2,e=>e.annotations.shapes[0].points=[0,0,2,1],e=>e.annotations.shapes[0].kind='script',e=>e.annotations.shapes[0].points=[0,0,0,0],e=>delete e.detectorBaseline];
  for(const mutate of variants){const state=make();mutate(state.findings[0].evidence[0]);assert.throws(()=>validateState(state));}
});
