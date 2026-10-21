import test from 'node:test';
import assert from 'node:assert/strict';
import {validateState,snapshotReport} from '../src/lib/core.mjs';
import {evidenceMetadataHtml} from '../src/lib/evidence-report.mjs';
const at='2026-10-21T00:00:00Z';
const make=()=>({schemaVersion:1,projects:[{id:'p',name:'Site',address:'',createdAt:at}],inspections:[{id:'i',projectId:'p',title:'Connection',location:'Room',inspector:'QA',stage:'research:brackets_railings',taskId:'research:brackets_railings:visibly_unseated_connection',mode:'research',createdAt:at,updatedAt:at,status:'in_progress',acknowledged:true,checklist:{}}],findings:[{id:'f',inspectionId:'i',label:'Evidence',source:'manual',review:'confirmed',status:'open',priority:'medium',notes:'',action:'',createdAt:at,updatedAt:at,evidence:[{id:'e',original:'data:image/png;base64,AAAA',annotated:'data:image/png;base64,AAAA',capturedAt:at,width:320,height:240,observations:[],provenance:{taskId:'research:brackets_railings:visibly_unseated_connection',family:'brackets_railings',categoryId:'visibly_unseated_connection',experimental:true,modelId:'research-fasteners',sha256:'a'.repeat(64),classConfidence:{visibly_unseated_connection:.025,fastener_absence_appearance:.9}}}],followUp:[]}],reports:[],settings:{language:'en',theme:'dark',inspector:'QA',autoSave:true}});
test('zero detections preserve selected model provenance in frozen report and escape external text',()=>{
 const state=make();validateState(state);const report=snapshotReport(state,'i','en','r',at);state.findings[0].evidence[0].provenance.classConfidence.visibly_unseated_connection=.99;
 assert.equal(report.snapshot.findings[0].evidence[0].provenance.classConfidence.visibly_unseated_connection,.025);
 const evidence=report.snapshot.findings[0].evidence[0];evidence.provenance.measurements={measuredJointMm:3,notes:'<script>أداة</script>'};
 const html=evidenceMetadataHtml(evidence,'bilingual');assert.match(html,/Experimental inspection/);assert.match(html,/research-fasteners/);assert.match(html,/0.025/);assert.match(html,/3 mm/);assert.ok(!html.includes('<script>'));assert.ok(html.includes('&lt;script&gt;'));
});
test('invalid provenance, thresholds and physical inputs cannot enter restored workspace or report snapshots',()=>{
 const mutations=[p=>p.sha256='../model',p=>p.classConfidence.visibly_unseated_connection=2,p=>p.experimental='yes',p=>p.measurements={measuredJointMm:-1},p=>p.measurements={nominalJointMm:Infinity},p=>p.measurements={cameraMoisture:95}];
 for(const mutate of mutations){const s=make();mutate(s.findings[0].evidence[0].provenance);assert.throws(()=>validateState(s));}
 const state=make();state.reports.push(snapshotReport(state,'i','ar','r',at));state.reports[0].snapshot.inspection.researchInputs={nominalJointMm:-3};assert.throws(()=>validateState(state));
});
