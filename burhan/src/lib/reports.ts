import {Capacitor} from '@capacitor/core';
import {BurhanNative} from '../native';
import type {AppState,Language,Report} from '../types';
import {escapeHtml as esc,safeImage,snapshotReport} from './core.mjs';
import {smartReportHtml} from './smart-report.mjs';
import {evidenceMetadataHtml} from './evidence-report.mjs';
import {imageSrc,uid} from './store';
import arabicCatalogue from '../../assets/catalogue-ar.json';

export function createReport(state:AppState,inspectionId:string,language:Language|'bilingual'):Report{return snapshotReport(state,inspectionId,language,uid(),new Date().toISOString());}
const labels={en:{report:'Construction inspection report',project:'Project',location:'Location',inspector:'Inspector',date:'Issued',version:'Version',scope:'Inspection scope',findings:'Findings',none:'No saved findings. This does not establish a defect-free inspection.',confirmed:'Confirmed findings',pending:'Unreviewed candidates',dismissed:'Dismissed candidates',notes:'Inspector notes',action:'Corrective action',status:'Issue status',priority:'Inspector priority',before:'Original evidence',overlay:'Detection evidence',after:'Follow-up evidence',checks:'Inspection checks',limitations:'Method and limitations',limit:'Local experimental visual models assist an inspector. Misses and false alerts are documented. Model confidence is not physical severity. An empty detection is not a pass. Strength, hidden defects, moisture content, dimensions and code compliance require appropriate evidence and qualified review.',review:'Review status',model:'Model',score:'Model score',signed:'Inspector record; not an engineering certification',unknown:'Not recorded'},ar:{report:'تقرير فحص أعمال البناء',project:'المشروع',location:'الموقع',inspector:'المفتش',date:'تاريخ الإصدار',version:'الإصدار',scope:'نطاق الفحص',findings:'الملاحظات',none:'لا توجد ملاحظات محفوظة. لا يعني ذلك خلو العنصر من العيوب.',confirmed:'ملاحظات مؤكدة',pending:'مرشحات لم تراجع',dismissed:'مرشحات مستبعدة',notes:'ملاحظات المفتش',action:'الإجراء التصحيحي',status:'حالة الملاحظة',priority:'أولوية المفتش',before:'الدليل الأصلي',overlay:'دليل الكشف',after:'دليل المتابعة',checks:'بنود الفحص',limitations:'المنهج والحدود',limit:'تساعد النماذج البصرية التجريبية المحلية المفتش، وقد تفوت عيوباً أو تنتج إنذارات خاطئة. درجة النموذج لا تحدد الخطورة الفعلية، وغياب الكشف لا يعني القبول. تتطلب المقاومة والعيوب المخفية والرطوبة والأبعاد ومطابقة الكود أدلة مناسبة ومراجعة مختصة.',review:'حالة المراجعة',model:'النموذج',score:'درجة النموذج',signed:'سجل المفتش؛ ليس شهادة هندسية',unknown:'غير مسجل'}};
const values:Record<string,string>={confirmed:'مؤكد',unreviewed:'لم يراجع',dismissed:'مستبعد',open:'مفتوحة',in_progress:'قيد المعالجة',resolved:'تمت المعالجة',low:'منخفضة',medium:'متوسطة',high:'عالية',not_checked:'لم يفحص',acceptable:'مقبول',issue:'ملاحظة',not_applicable:'لا ينطبق',draft:'مسودة',completed:'مكتمل',manual:'يدوي',crack:'شقوق',surface:'أسطح',ppe:'معدات الوقاية'};
export function reportHtml(report:Report):string{
  const lang=report.language==='bilingual'?'en':report.language;const l=labels[lang];
  const text=(key:keyof typeof labels.en)=>report.language==='bilingual'?`${labels.en[key]} / ${labels.ar[key]}`:l[key];
  const value=(v:string)=>report.language==='ar'?values[v]??v:report.language==='bilingual'&&values[v]?`${v.replaceAll('_',' ')} / ${values[v]}`:v.replaceAll('_',' ');
  const {project,inspection,findings}=report.snapshot;
  const stageArabic=(arabicCatalogue.stages as Record<string,string>)[inspection.stage];
  const stageLabel=report.language==='ar'?(stageArabic??inspection.stage):report.language==='bilingual'&&stageArabic?`${inspection.stage.replaceAll('_',' ')} / ${stageArabic}`:inspection.stage.replaceAll('_',' ');
  const checkLabel=(key:string)=>key==='visual'?text('scope'):key==='evidence'?text('before'):key;
  const recordedModels=[...new Map(findings.flatMap(f=>[...(f.observation?[f.observation]:[]),...f.evidence.flatMap(e=>e.observations??[])]).filter(o=>o.model).map(o=>[o.model!.sha256,{id:o.modelId,...o.model!}])).values()];
  const image=(path:string,caption:string)=>{const uri=safeImage(imageSrc(path));return uri?`<figure><img src="${esc(uri)}" alt="${esc(caption)}"/><figcaption>${esc(caption)}</figcaption></figure>`:'';};
  return `<!doctype html><html lang="${lang}" dir="${lang==='ar'?'rtl':'ltr'}"><head><meta charset="utf-8"><title>Burhan — ${esc(inspection.title)}</title><style>
  @font-face{font-family:BurhanArabic;src:url('${location.origin}/fonts/NotoSansArabic-Regular.ttf')}@font-face{font-family:BurhanLatin;src:url('${location.origin}/fonts/NotoSans-Regular.ttf')}
  *{box-sizing:border-box}body{margin:0;color:#15191c;background:#fff;font:12px/1.65 BurhanLatin,BurhanArabic,sans-serif}main{max-width:850px;margin:auto;padding:32px}header{border-bottom:4px solid #15191c;padding-bottom:18px;display:flex;gap:20px;align-items:start;justify-content:space-between}.brand{font-size:30px;font-weight:900;letter-spacing:-1px}h1{font-size:23px;margin:8px 0}h2{font-size:17px;border-bottom:2px solid #15191c;padding-bottom:6px;margin-top:24px}h3{font-size:15px;margin:0 0 10px}p{white-space:pre-wrap;overflow-wrap:anywhere}.meta{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin:18px 0}.meta div{padding:10px;background:#f3f3ee;border:1px solid #ccc}.meta b{display:block;font-size:10px;color:#555}.stats{display:flex;gap:16px;flex-wrap:wrap}.stats span{padding:8px 12px;background:#ffdf62}.finding{border:1px solid #777;padding:16px;margin:16px 0;break-inside:avoid-page}.images{display:grid;grid-template-columns:1fr 1fr;gap:12px}figure{margin:0}img{width:100%;max-height:260px;object-fit:contain;background:#f4f4ef}figcaption{font-size:10px;color:#555}table{width:100%;border-collapse:collapse}td,th{text-align:start;border-bottom:1px solid #ddd;padding:8px}footer{margin-top:24px;border-top:2px solid #15191c;font-size:10px;padding-top:10px}.muted{color:#555;font-size:10px}@page{size:A4;margin:12mm}@media print{main{padding:0;max-width:none}header{break-after:avoid}h2,h3{break-after:avoid}.finding{break-inside:avoid}.meta{break-inside:avoid}}</style></head><body><main>
  <header><div><div class="brand">BURHAN <span dir="rtl">برهان</span></div><h1>${text('report')}</h1><p>${esc(inspection.title)}</p></div><div>${text('version')} ${report.version}<br><span class="muted">${esc(report.id)}</span></div></header>
  <section class="meta"><div><b>${text('project')}</b>${esc(project.name)}</div><div><b>${text('location')}</b>${esc(inspection.location||l.unknown)}</div><div><b>${text('inspector')}</b>${esc(inspection.inspector||l.unknown)}</div><div><b>${text('date')}</b>${esc(new Date(report.createdAt).toLocaleString(lang==='ar'?'ar-SA':'en-GB',{timeZone:'Asia/Riyadh',calendar:'gregory'}))} (UTC+3)</div></section>
  <h2>${text('scope')}</h2><p>${esc(value(inspection.mode))} · ${esc(stageLabel)} · ${esc(value(inspection.status))}</p>
  <section class="stats"><span>${text('confirmed')}: ${findings.filter(f=>f.review==='confirmed').length}</span><span>${text('pending')}: ${findings.filter(f=>f.review==='unreviewed').length}</span><span>${text('dismissed')}: ${findings.filter(f=>f.review==='dismissed').length}</span></section>
  <h2>${text('findings')}</h2>${findings.length?findings.map((f,index)=>`<article class="finding"><h3>${index+1}. ${esc(f.label)}</h3><p>${text('review')}: <b>${esc(value(f.review))}</b> · ${text('status')}: ${esc(value(f.status))} · ${text('priority')}: ${esc(value(f.priority))}</p><p><b>${text('notes')}</b><br>${esc(f.notes||'—')}</p><p><b>${text('action')}</b><br>${esc(f.action||'—')}</p>${f.observation?`<p class="muted">${text('model')}: ${esc(f.observation.modelId)} · ${text('score')}: ${(f.observation.score*100).toFixed(1)}% · ${esc(f.observation.capturedAt)}</p>`:''}<div class="images">${f.evidence.flatMap(e=>[image(e.original,l.before),image(e.annotated,e.annotations?.shapes.length?(lang==='ar'?'دليل مع تعليقات المفتش':'Evidence with inspector markup'):l.overlay),e.detectorBaseline&&e.detectorBaseline!==e.annotated?image(e.detectorBaseline,lang==='ar'?'خط أساس الكاشف':'Detector baseline'):'',evidenceMetadataHtml(e,report.language)]).join('')}${f.followUp.map(e=>image(e.annotated,l.after)+evidenceMetadataHtml(e,report.language)).join('')}</div></article>`).join(''):`<p>${text('none')}</p>`}
  <h2>${text('checks')}</h2><table>${Object.entries(inspection.checklist).map(([key,v])=>`<tr><td>${esc(checkLabel(key))}</td><td>${esc(value(v))}</td></tr>`).join('')}</table>
  ${smartReportHtml(inspection.smart,report.language)}
  <h2>${text('limitations')}</h2><p>${text('limit')}</p>${recordedModels.map(m=>`<div class="muted" style="overflow-wrap:anywhere"><b>${text('model')}: ${esc(m.id)}</b><br>SHA-256: <bdi dir="ltr">${esc(m.sha256)}</bdi><br>${esc(m.runtime)} · ${m.inputSize}px · ${esc(m.preprocessing)} · confidence ${m.confidence} · NMS ${m.nms}</div>`).join('')}<p class="muted">${esc(report.summary)}</p><footer>Burhan · ${text('signed')} · <bdi dir="ltr">${esc(report.createdAt)}</bdi></footer></main></body></html>`;
}
export async function dataUri(path:string):Promise<string>{
  if(path.startsWith('data:image/'))return path;
  const response=await fetch(imageSrc(path));if(!response.ok)throw new Error('Evidence file unavailable');
  const blob=await response.blob();return new Promise((resolve,reject)=>{const reader=new FileReader();reader.onload=()=>resolve(String(reader.result));reader.onerror=reject;reader.readAsDataURL(blob);});
}
export async function exportReport(report:Report):Promise<void>{
  const copy=structuredClone(report);
  for(const f of copy.snapshot.findings)for(const e of [...f.evidence,...f.followUp]){e.original=await dataUri(e.original);e.annotated=await dataUri(e.annotated);if(e.detectorBaseline)e.detectorBaseline=await dataUri(e.detectorBaseline);}
  let html=reportHtml(copy);
  for(const font of ['NotoSansArabic-Regular.ttf','NotoSans-Regular.ttf']){
    const r=await fetch(`/fonts/${font}`);if(!r.ok)throw new Error('Report font unavailable');
    const b=await r.blob();const uri=await new Promise<string>((resolve,reject)=>{const fr=new FileReader();fr.onload=()=>resolve(String(fr.result));fr.onerror=reject;fr.readAsDataURL(b);});
    html=html.replace(`${location.origin}/fonts/${font}`,uri);
  }
  if(Capacitor.isNativePlatform()){
    await BurhanNative.printReport({html,jobName:`Burhan-${report.id}-v${report.version}`});return;
  }
  const frame=document.createElement('iframe');frame.style.cssText='position:fixed;left:-10000px;width:850px;height:1100px';document.body.append(frame);
  await new Promise<void>((resolve,reject)=>{frame.onload=()=>resolve();frame.onerror=()=>reject(new Error('Cannot prepare report'));frame.srcdoc=html;});
  await frame.contentDocument?.fonts.ready;
  await Promise.all([...frame.contentDocument!.images].map(img=>img.complete?Promise.resolve():new Promise<void>(r=>{img.onload=()=>r();img.onerror=()=>r();})));
  frame.contentWindow?.print();setTimeout(()=>frame.remove(),60000);
}
