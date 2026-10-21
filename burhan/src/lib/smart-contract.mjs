const modes=new Set(['manual','crack','surface','ppe','research']);
const actionTypes=new Set(['select_detector','request_view','request_manual_check','ask_context','prepare_finding']);
const clean=(v,n)=>typeof v==='string'?v.slice(0,n):'';
/** Treat the remote planner as untrusted input, including during local development. */
export function validatePlan(value){
 if(!value||typeof value!=='object'||!value.scene||!Array.isArray(value.actions)||!Array.isArray(value.suggestions))throw Error('Invalid inspection guidance');
 const env=['indoor','outdoor','transitional','unknown'].includes(value.scene.environment)?value.scene.environment:'unknown';
 if(typeof value.guidance!=='string'||!value.guidance.trim())throw Error('Guidance is empty');
 return {scene:{environment:env,description:clean(value.scene.description,800),confidence:Math.max(0,Math.min(1,Number(value.scene.confidence)||0))},guidance:clean(value.guidance,1200),guidanceTaskId:typeof value.guidanceTaskId==='string'?clean(value.guidanceTaskId,160):null,reason:clean(value.reason,1200),needsHumanReview:true,
 suggestions:value.suggestions.slice(0,3).filter(s=>s&&typeof s.taskId==='string'&&modes.has(s.mode)).map(s=>({taskId:clean(s.taskId,120),title:clean(s.title,200),mode:s.mode,reason:clean(s.reason,600),automatic:s.automatic===true})),
 actions:value.actions.slice(0,8).filter(a=>a&&actionTypes.has(a.type)&&(a.mode==null||modes.has(a.mode))).map(a=>({id:clean(a.id,100),type:a.type,mode:a.mode??null,taskId:typeof a.taskId==='string'?clean(a.taskId,120):null,instruction:clean(a.instruction,800)}))};
}
export function normalizeBackendUrl(value){
 const u=new URL(value);if(u.username||u.password||u.search||u.hash||u.pathname!=='/')throw Error('Enter the backend origin only');
 if(u.protocol!=='https:'&&!(u.protocol==='http:'&&['localhost','127.0.0.1','[::1]'].includes(u.hostname)))throw Error('Backend requires HTTPS');
 return u.origin;
}
export function sceneDifference(a,b){return !a||a.length!==b.length?1:a.reduce((s,v,i)=>s+Math.abs(v-b[i]),0)/a.length;}
export function normalizeSmartHistory(value){
 if(!value||typeof value!=='object'||typeof value.sessionId!=='string'||!value.sessionId||typeof value.provider!=='string'||!Array.isArray(value.steps))throw Error('Invalid guidance history');
 return {sessionId:value.sessionId.slice(0,256),provider:value.provider.slice(0,80),scene:clean(value.scene,2048),steps:value.steps.slice(-100).filter(s=>s&&typeof s.instruction==='string'&&typeof s.at==='string'&&Number.isFinite(Date.parse(s.at))&&modes.has(s.mode)).map(s=>({at:s.at,instruction:clean(s.instruction,4000),reason:clean(s.reason,4000),mode:s.mode,scene:clean(s.scene,2048)}))};
}
