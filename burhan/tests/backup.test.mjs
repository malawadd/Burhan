import test from 'node:test';import assert from 'node:assert/strict';import {mkdirSync} from 'node:fs';import {buildSync} from 'esbuild';import {zipSync,unzipSync,strToU8,strFromU8} from 'fflate';
mkdirSync(new URL('../.tools/tests/',import.meta.url),{recursive:true});
const output=new URL('../.tools/tests/backup.mjs',import.meta.url);
buildSync({entryPoints:[new URL('../src/lib/backup.ts',import.meta.url).pathname.replace(/^\/(\w:)/,'$1')],outfile:output.pathname.replace(/^\/(\w:)/,'$1'),platform:'node',format:'esm',bundle:true,packages:'external'});
const {exportBackup,importBackup}=await import(output.href);
const png='data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jR3sAAAAASUVORK5CYII=';
const time='2026-10-21T00:00:00Z';
function make(){const finding={id:'f',inspectionId:'i',label:'ملاحظة',source:'manual',review:'confirmed',status:'open',priority:'medium',notes:'صورة اختبار',action:'Review',createdAt:time,updatedAt:time,evidence:[{id:'e',original:png,annotated:png,capturedAt:time,width:1,height:1}],followUp:[]};return {schemaVersion:1,projects:[{id:'p',name:'Test',address:'Site',createdAt:time}],inspections:[{id:'i',projectId:'p',title:'Test',location:'Room',inspector:'QA',stage:'structure',taskId:'visual-cracks',mode:'manual',status:'completed',createdAt:time,updatedAt:time,acknowledged:true,checklist:{}}],findings:[finding],reports:[],settings:{language:'ar',theme:'dark',inspector:'QA',autoSave:true}};}
let downloaded;
test('expanded evidence and entered measurements survive portable backup with a frozen report',async()=>{
  const state=make(),e=state.findings[0].evidence[0];state.inspections[0].mode='research';state.inspections[0].researchInputs={nominalJointMm:3,toleranceMm:.5};
  e.observations=[];e.provenance={taskId:'research:tile_installation:joint_width',categoryId:'joint_width',family:'tile_installation',experimental:true,measurements:{nominalJointMm:3,measuredJointMm:3.2,notes:'Gauge <3mm> مقياس'}};
  state.reports=[{id:'r',inspectionId:'i',version:1,language:'bilingual',createdAt:time,summary:'Saved',snapshot:structuredClone({project:state.projects[0],inspection:state.inspections[0],findings:state.findings})}];
  const originalCreate=URL.createObjectURL,originalDocument=globalThis.document;globalThis.document={createElement(){return {click(){}}}};URL.createObjectURL=blob=>{downloaded=blob;return 'blob:test'};
  try{await exportBackup(state)}finally{URL.createObjectURL=originalCreate;globalThis.document=originalDocument}
  const restored=await importBackup(new File([await downloaded.arrayBuffer()],'expanded.zip'));assert.deepEqual(restored.findings[0].evidence[0].provenance,e.provenance);assert.deepEqual(restored.inspections[0].researchInputs,state.inspections[0].researchInputs);
  e.provenance.measurements.measuredJointMm=10;assert.equal(restored.reports[0].snapshot.findings[0].evidence[0].provenance.measurements.measuredJointMm,3.2);
});
test('portable ZIP retains Arabic records and deduplicated evidence; altered content rejected',async()=>{
  const originalCreate=URL.createObjectURL;const originalDocument=globalThis.document;
  globalThis.document={createElement(){return {click(){}};}};URL.createObjectURL=blob=>{downloaded=blob;return 'blob:test';};
  try{await exportBackup(make());}finally{URL.createObjectURL=originalCreate;globalThis.document=originalDocument;}
  const bytes=new Uint8Array(await downloaded.arrayBuffer());const restored=await importBackup(new File([bytes],'backup.zip'));
  assert.equal(restored.findings[0].notes,'صورة اختبار');assert.equal(restored.findings[0].evidence[0].original,png);assert.equal(restored.settings.language,'ar');
  const files=unzipSync(bytes);assert.equal(Object.keys(files).filter(k=>k.startsWith('media/')).length,1);
  const image=Object.keys(files).find(k=>k.startsWith('media/'));files[image]=strToU8(png+'tampered');
  await assert.rejects(()=>importBackup(new File([zipSync(files)],'broken.zip')),/integrity/);
});
test('restore rejects nested malformed metadata and traversal archive names',async()=>{
  const state=make();delete state.inspections[0].updatedAt;
  const bytes=zipSync({'burhan.json':strToU8(JSON.stringify({format:'burhan-backup',version:1,state}))});
  await assert.rejects(()=>importBackup(new File([bytes],'bad.zip')),/updatedAt/);
  const bad=zipSync({'../burhan.json':strToU8('{}')});await assert.rejects(()=>importBackup(new File([bad],'bad.zip')),/Unexpected/);
});


test('new capture source, baseline and normalized markup survive ZIP round trip with immutable report media',async()=>{
  const state=make();const e=state.findings[0].evidence[0];e.captureSource='photo';e.detectorBaseline=png;e.annotations={version:1,shapes:[{kind:'ellipse',points:[.1,.1,.9,.9]}]};
  state.reports=[{id:'r',inspectionId:'i',version:1,language:'en',createdAt:time,summary:'Saved report',snapshot:structuredClone({project:state.projects[0],inspection:state.inspections[0],findings:state.findings})}];
  e.annotated='data:image/png;base64,TkVX';
  const originalCreate=URL.createObjectURL;const originalDocument=globalThis.document;globalThis.document={createElement(){return {click(){}};}};URL.createObjectURL=blob=>{downloaded=blob;return 'blob:test';};
  try{await exportBackup(state);}finally{URL.createObjectURL=originalCreate;globalThis.document=originalDocument;}
  const bytes=new Uint8Array(await downloaded.arrayBuffer());const restored=await importBackup(new File([bytes],'backup.zip'));
  assert.deepEqual(restored.findings[0].evidence[0],e);assert.equal(restored.reports[0].snapshot.findings[0].evidence[0].annotated,png);
  assert.equal(Object.keys(unzipSync(bytes)).filter(k=>k.startsWith('media/')).length,2);
});
