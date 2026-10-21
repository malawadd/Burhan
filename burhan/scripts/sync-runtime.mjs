import {mkdir,copyFile,readdir,readFile} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
const root=fileURLToPath(new URL('../',import.meta.url));
const source=path.join(root,'node_modules/onnxruntime-web/dist'),target=path.join(root,'assets/ort');
await mkdir(target,{recursive:true});
for(const name of ['ort-wasm-simd-threaded.wasm','ort-wasm-simd-threaded.mjs'])await copyFile(path.join(source,name),path.join(target,name));
const runtime=JSON.parse(await readFile(path.join(root,'node_modules/onnxruntime-web/package.json'),'utf8'));
console.log(`Bundled offline ONNX Runtime Web ${runtime.version}`);
