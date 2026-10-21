import {readFile,stat} from 'node:fs/promises';import {createHash} from 'node:crypto';
const base=new URL('../assets/',import.meta.url);const models=JSON.parse(await readFile(new URL('models/manifest.json',base),'utf8'));
for(const model of models){const bytes=await readFile(new URL('models/'+model.file,base));if(bytes.length!==model.bytes||createHash('sha256').update(bytes).digest('hex')!==model.sha256)throw Error('Model checksum mismatch: '+model.id);}
const catalogue=JSON.parse(await readFile(new URL('catalogue.json',base),'utf8'));if(catalogue.tasks.length!==92||catalogue.stages.length!==16)throw Error('Incomplete construction catalogue');
const cameraSettings=await readFile(new URL('camera-settings.json',base),'utf8');if(cameraSettings!==await readFile(new URL('../public/camera-settings.json',base),'utf8'))throw Error('Camera preference asset differs from shared definitions; run sync-runtime');
for(const file of ['fonts/NotoSansArabic-Regular.ttf','fonts/NotoSans-Regular.ttf','ort/ort-wasm-simd-threaded.wasm','ort/ort-wasm-simd-threaded.mjs'])if((await stat(new URL(file,base))).size===0)throw Error('Missing '+file);
console.log(`Verified ${models.length} SHA-256 pinned models, 92 inspection tasks, fonts and offline runtime.`);
