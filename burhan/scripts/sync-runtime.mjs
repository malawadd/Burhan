import {mkdir,copyFile,readdir,readFile} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
const root=fileURLToPath(new URL('../',import.meta.url));
const source=path.join(root,'node_modules/onnxruntime-web/dist'),target=path.join(root,'assets/ort');
await mkdir(target,{recursive:true});
async function copyChanged(from,to){const bytes=await readFile(from);try{if(bytes.equals(await readFile(to)))return;}catch(error){if(error.code!=='ENOENT')throw error;}await copyFile(from,to);}
// Avoid reloading an active dev camera when a build copies unchanged public assets.
await copyChanged(path.join(root,'public/camera-settings.json'),path.join(root,'assets/camera-settings.json'));
for(const name of ['ort-wasm-simd-threaded.wasm','ort-wasm-simd-threaded.mjs'])await copyChanged(path.join(source,name),path.join(target,name));
const runtime=JSON.parse(await readFile(path.join(root,'node_modules/onnxruntime-web/package.json'),'utf8'));
console.log(`Bundled offline ONNX Runtime Web ${runtime.version}`);
