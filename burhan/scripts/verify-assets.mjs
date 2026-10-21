import {readFile,stat} from 'node:fs/promises';import {createHash} from 'node:crypto';
const base=new URL('../assets/',import.meta.url);const models=JSON.parse(await readFile(new URL('models/manifest.json',base),'utf8'));
for(const model of models){const bytes=await readFile(new URL('models/'+model.file,base));if(bytes.length!==model.bytes||createHash('sha256').update(bytes).digest('hex')!==model.sha256)throw Error('Model checksum mismatch: '+model.id);}
const catalogue=JSON.parse(await readFile(new URL('catalogue.json',base),'utf8'));if(catalogue.tasks.length!==92||catalogue.stages.length!==16)throw Error('Incomplete construction catalogue');
const research=JSON.parse(await readFile(new URL('research-catalogue.json',base),'utf8'));
if(research.tasks.length!==30||research.families.length!==3||new Set(research.tasks.map(t=>t.id)).size!==30)throw Error('Incomplete expanded catalogue');
if(research.releasedCategories.length)throw Error('Research categories must not be promoted by packaging');
for(const task of research.tasks){
  if(!task.researchOnly||!task.titleAr||!task.guidanceAr||!task.scope)throw Error('Missing research disclosure or bilingual copy');
  const model=models.find(m=>m.id===task.modelId);
  if(task.automatic&&(!model?.researchOnly||!model.classes.includes(task.categoryId)||task.mode!=='research'))throw Error('Invalid research task/model binding');
  if(!task.automatic&&(task.modelId||task.mode!=='manual'))throw Error('Unsupported check became automatic');
}
const cameraSettings=await readFile(new URL('camera-settings.json',base),'utf8');if(cameraSettings!==await readFile(new URL('../public/camera-settings.json',base),'utf8'))throw Error('Camera preference asset differs from shared definitions; run sync-runtime');
for(const file of ['fonts/NotoSansArabic-Regular.ttf','fonts/NotoSans-Regular.ttf','ort/ort-wasm-simd-threaded.wasm','ort/ort-wasm-simd-threaded.mjs'])if((await stat(new URL(file,base))).size===0)throw Error('Missing '+file);
const definitions=JSON.parse(cameraSettings).settings;
if(definitions.find(d=>d.key==='expandedInspectionModels')?.default!==false)throw Error('Expansion must default off');
console.log(`Verified ${models.length} SHA-256 pinned models, 92 existing +30 optional checks, default-off expansion, fonts and offline runtime.`);
