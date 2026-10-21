import {escapeHtml as esc} from './core.mjs';

/** Guidance is recorded advice, never a finding or an automatic inspection verdict. */
export function smartReportHtml(smart,language='en'){
  if(!smart?.steps?.length)return '';
  const words={en:{heading:'Smart inspection guidance record',scene:'Suggested scene',provider:'Planning provider',session:'Session',note:'Recorded assistance, subject to inspector review. Scene suggestions and guidance are not verified defects or compliance decisions.'},ar:{heading:'سجل إرشادات الفحص الذكي',scene:'المشهد المقترح',provider:'مزود التخطيط',session:'الجلسة',note:'مساعدة مسجلة تخضع لمراجعة المفتش. اقتراحات المشهد والإرشادات ليست عيوباً مؤكدة أو قرارات مطابقة.'}};
  const w=key=>language==='bilingual'?`${words.en[key]} / ${words.ar[key]}`:words[language==='ar'?'ar':'en'][key];
  return `<section class="smart-history"><h2>${w('heading')}</h2><p class="muted">${w('note')}</p><p>${w('scene')}: ${esc(smart.scene)}<br>${w('provider')}: <bdi dir="ltr">${esc(smart.provider)}</bdi> · ${w('session')}: <bdi dir="ltr">${esc(smart.sessionId)}</bdi></p>${smart.steps.slice(-100).map(step=>`<div class="finding"><b>${esc(step.instruction)}</b><p>${esc(step.reason)}</p><small><bdi dir="ltr">${esc(step.at)}</bdi> · ${esc(step.mode)} · ${esc(step.scene)}</small></div>`).join('')}</section>`;
}
