import {escapeHtml} from './core.mjs';
/** Render only captured metadata; never infer capture quality from the requested setting. */
export function evidenceMetadataHtml(e,language='en'){
  const ar=language==='ar';const both=language==='bilingual';
  const sources={analysis:['Analysis frame','إطار التحليل'],photo:['Photo capture','التقاط صور'],import:['Imported image','صورة مستوردة']};
  const label=pair=>both?pair.join(' / '):pair[ar?1:0];
  const parts=[`${Number(e.width)} × ${Number(e.height)} px`];
  if(sources[e.captureSource])parts.push(label(sources[e.captureSource]));
  if(e.annotations?.shapes?.length)parts.push(`${label(['Inspector markup','تعليقات المفتش'])}: ${e.annotations.shapes.length}`);
  return `<p class="muted">${escapeHtml(parts.join(' · '))}</p>`;
}
