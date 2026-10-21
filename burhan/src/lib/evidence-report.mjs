import {escapeHtml} from './core.mjs';
/** Render only captured metadata; never infer capture quality from the requested setting. */
export function evidenceMetadataHtml(e,language='en'){
  const ar=language==='ar';const both=language==='bilingual';
  const sources={analysis:['Analysis frame','إطار التحليل'],photo:['Photo capture','التقاط صور'],import:['Imported image','صورة مستوردة']};
  const label=pair=>both?pair.join(' / '):pair[ar?1:0];
  const parts=[`${Number(e.width)} × ${Number(e.height)} px`];
  if(sources[e.captureSource])parts.push(label(sources[e.captureSource]));
  if(e.annotations?.shapes?.length)parts.push(`${label(['Inspector markup','تعليقات المفتش'])}: ${e.annotations.shapes.length}`);
  const p=e.provenance;
  if(p?.experimental)parts.push(label(['Experimental inspection — human review required','فحص تجريبي — يتطلب مراجعة بشرية']));
  if(p?.taskId)parts.push(p.taskId);
  if(p?.categoryId)parts.push(p.categoryId);
  if(p?.modelId)parts.push(`${label(['Model','النموذج'])}: ${p.modelId}`);
  if(p?.sha256)parts.push(`SHA-256: ${p.sha256}`);
  if(p?.classConfidence)parts.push(`${label(['Class thresholds','عتبات الفئات'])}: ${JSON.stringify(p.classConfidence)}`);
  if(p?.runtime)parts.push(p.runtime);
  let measurement='';
  if(p?.measurements){
    const m=p.measurements, rows=[];
    for(const [key,en,arabic] of [['nominalJointMm','Specified joint width','عرض الفاصل المحدد'],['toleranceMm','Specified tolerance','السماحية المحددة'],['measuredJointMm','Entered measured width','العرض المقاس المدخل']])if(m[key]!==undefined)rows.push(`${label([en,arabic])}: ${m[key]} mm`);
    if(m.notes)rows.push(m.notes);
    if(rows.length)measurement=`<p class="muted">${escapeHtml(label(['Inspector-entered measurements, not camera estimates','قياسات أدخلها المفتش وليست تقديرات بالكاميرا']))}<br>${rows.map(escapeHtml).join('<br>')}</p>`;
  }
  return `<p class="muted">${escapeHtml(parts.join(' · '))}</p>${measurement}`;
}
