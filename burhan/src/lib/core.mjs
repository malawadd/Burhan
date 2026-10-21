const text=(value,name)=>{if(typeof value!=='string')throw new Error(`Invalid ${name}`);};
const timestamp=(value,name)=>{text(value,name);if(!Number.isFinite(Date.parse(value)))throw new Error(`Invalid ${name}`);};
const enumeration=(value,values,name)=>{if(!values.includes(value))throw new Error(`Invalid ${name}`);};
const boundedText=(value,name,max=4000)=>{text(value,name);if(value.length>max)throw new Error(`Invalid ${name} length`);};
function smartRow(s){
  if(!s||typeof s!=='object'||Array.isArray(s))throw new Error('Invalid smart inspection history');
  boundedText(s.sessionId,'smart session',256);boundedText(s.provider,'smart provider',80);boundedText(s.scene,'smart scene',2048);
  if(!Array.isArray(s.steps)||s.steps.length>100)throw new Error('Invalid smart guidance steps');
  for(const step of s.steps){if(!step||typeof step!=='object')throw new Error('Invalid smart step');timestamp(step.at,'smart step timestamp');boundedText(step.instruction,'smart instruction');boundedText(step.reason,'smart reason');boundedText(step.scene,'smart step scene',2048);enumeration(step.mode,['manual','ppe','crack','surface'],'smart detector');}
}
function projectRow(p){if(!p||typeof p.id!=='string')throw new Error('Invalid project');text(p.name,'project name');text(p.address,'project address');timestamp(p.createdAt,'project createdAt');}
function inspectionRow(i){
  if(!i)throw new Error('Invalid inspection');
  for(const key of ['id','projectId','title','location','inspector','stage','taskId'])text(i[key],`inspection ${key}`);
  timestamp(i.createdAt,'inspection createdAt');timestamp(i.updatedAt,'inspection updatedAt');
  enumeration(i.status,['draft','in_progress','completed'],'inspection status');enumeration(i.mode,['manual','ppe','crack','surface'],'analysis mode');
  if(typeof i.acknowledged!=='boolean'||!i.checklist||Array.isArray(i.checklist)||typeof i.checklist!=='object')throw new Error('Invalid checklist');
  for(const v of Object.values(i.checklist))enumeration(v,['not_checked','acceptable','issue','not_applicable'],'checklist decision');
  if(i.smart!==undefined)smartRow(i.smart);
}
function observationRow(o){
  if(!o||!Number.isInteger(o.classId)||o.classId<0||!Number.isFinite(o.score)||o.score<0||o.score>1||!Number.isFinite(o.inferenceMs)||o.inferenceMs<0)throw new Error('Invalid observation');
  for(const k of ['label','modelId','frameId'])text(o[k],`observation ${k}`);timestamp(o.capturedAt,'observation capturedAt');
  if(!Array.isArray(o.bbox)||o.bbox.length!==4||o.bbox.some(v=>!Number.isFinite(v))||o.bbox[2]<=0||o.bbox[3]<=0)throw new Error('Invalid observation box');
  if(o.model){if(!/^[a-f0-9]{64}$/.test(o.model.sha256)||!Number.isInteger(o.model.inputSize)||o.model.inputSize<1||!Number.isFinite(o.model.confidence)||!Number.isFinite(o.model.nms))throw new Error('Invalid model identity');text(o.model.preprocessing,'preprocessing');text(o.model.runtime,'runtime');}
}
function evidenceRow(e){
  if(!e||typeof e.id!=='string'||typeof e.original!=='string'||typeof e.annotated!=='string'||!Number.isFinite(e.width)||!Number.isFinite(e.height)||e.width<1||e.height<1)throw new Error('Invalid evidence');
  timestamp(e.capturedAt,'evidence capturedAt');
  if(e.captureSource!==undefined)enumeration(e.captureSource,['analysis','photo','import'],'capture source');
  if(e.detectorBaseline!==undefined){boundedText(e.detectorBaseline,'detector baseline',32*1024*1024);if(!e.detectorBaseline)throw new Error('Invalid detector baseline');}
  if(e.annotations!==undefined){
    const a=e.annotations;if(!a||a.version!==1||!Array.isArray(a.shapes)||a.shapes.length>200)throw new Error('Invalid photo annotations');
    for(const shape of a.shapes){if(!shape||!['arrow','ellipse'].includes(shape.kind)||!Array.isArray(shape.points)||shape.points.length!==4||shape.points.some(v=>!Number.isFinite(v)||v<0||v>1))throw new Error('Invalid annotation shape');
      if(shape.points[0]===shape.points[2]&&shape.points[1]===shape.points[3])throw new Error('Empty annotation shape');
    }
    if(!e.detectorBaseline)throw new Error('Annotated evidence requires detector baseline');
  }
  if(e.observations!==undefined){if(!Array.isArray(e.observations))throw new Error('Invalid evidence observations');e.observations.forEach(observationRow);}
  if(e.smartContext!==undefined){const c=e.smartContext;if(!c||typeof c!=='object'||!Number.isSafeInteger(c.sceneVersion)||c.sceneVersion<0)throw new Error('Invalid evidence smart context');for(const k of ['sessionId','provider','guidance','activeTaskId'])boundedText(c[k],`evidence smart ${k}`);enumeration(c.mode,['manual','ppe','crack','surface'],'evidence smart mode');}
}
function findingRow(f){
  if(!f)throw new Error('Invalid finding');
  for(const k of ['id','inspectionId','label','notes','action'])text(f[k],`finding ${k}`);
  timestamp(f.createdAt,'finding createdAt');timestamp(f.updatedAt,'finding updatedAt');
  enumeration(f.source,['ai','manual'],'finding source');enumeration(f.review,['unreviewed','confirmed','dismissed'],'finding review');enumeration(f.status,['open','in_progress','resolved'],'finding status');enumeration(f.priority,['low','medium','high'],'priority');
  if(!Array.isArray(f.evidence)||!Array.isArray(f.followUp))throw new Error('Invalid finding evidence');
  [...f.evidence,...f.followUp].forEach(evidenceRow);if(f.observation)observationRow(f.observation);
}
export function validateState(input) {
  if (!input || input.schemaVersion !== 1) throw new Error('Unsupported backup/data version');
  for (const key of ['projects','inspections','findings','reports']) {
    if (!Array.isArray(input[key]) || input[key].length > 50000) throw new Error(`Invalid ${key}`);
    const ids = new Set();
    for (const row of input[key]) {
      if (!row || typeof row.id !== 'string' || !row.id || ids.has(row.id)) throw new Error(`Invalid or duplicate ${key} ID`);
      ids.add(row.id);
    }
  }
  if (!input.settings || !['en','ar'].includes(input.settings.language) || !['dark','light','system'].includes(input.settings.theme)||typeof input.settings.inspector!=='string'||typeof input.settings.autoSave!=='boolean') throw new Error('Invalid settings');
  const projects = new Set(input.projects.map(p=>p.id));
  const inspections = new Set(input.inspections.map(i=>i.id));
  input.projects.forEach(projectRow);
  for (const i of input.inspections) {
    inspectionRow(i);if (!projects.has(i.projectId)) throw new Error('Invalid inspection project reference');
  }
  for (const f of input.findings) {
    findingRow(f);if (!inspections.has(f.inspectionId)) throw new Error('Invalid finding inspection reference');
  }
  for (const r of input.reports) {
    if (!inspections.has(r.inspectionId) || !['en','ar','bilingual'].includes(r.language)||!Number.isInteger(r.version)||r.version<1||!r.snapshot?.project||!r.snapshot?.inspection||!Array.isArray(r.snapshot?.findings)) throw new Error('Invalid report');
    timestamp(r.createdAt,'report createdAt');text(r.summary,'report summary');
    projectRow(r.snapshot.project);inspectionRow(r.snapshot.inspection);r.snapshot.findings.forEach(findingRow);
    if(r.snapshot.inspection.id!==r.inspectionId||r.snapshot.inspection.projectId!==r.snapshot.project.id||r.snapshot.findings.some(f=>f.inspectionId!==r.inspectionId))throw new Error('Invalid report snapshot references');
  }
  return input;
}
export const escapeHtml = value => String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
export function safeImage(uri) {
  if(typeof uri!=='string')return '';
  return /^(data:image\/(jpeg|jpg|png|webp);base64,|https?:\/\/|capacitor:\/\/|file:\/\/)/i.test(uri)?uri:'';
}
export function snapshotReport(state, inspectionId, language, id, now) {
  const inspection=state.inspections.find(i=>i.id===inspectionId);
  if(!inspection)throw new Error('Inspection not found');
  const project=state.projects.find(p=>p.id===inspection.projectId);
  if(!project)throw new Error('Project not found');
  const findings=state.findings.filter(f=>f.inspectionId===inspectionId);
  if(findings.some(f=>f.review==='unreviewed')&&!inspection.acknowledged)throw new Error('Review candidates or acknowledge outstanding reviews before generating a report.');
  return {id,inspectionId,version:1+Math.max(0,...state.reports.filter(r=>r.inspectionId===inspectionId).map(r=>r.version)),language,createdAt:now,snapshot:structuredClone({project,inspection,findings}),summary:`${findings.filter(f=>f.review==='confirmed').length} confirmed / ${findings.filter(f=>f.review==='unreviewed').length} pending review`};
}
