import {readFile,mkdir,writeFile,readdir,stat} from 'node:fs/promises';import {zipSync} from 'fflate';import {fileURLToPath} from 'node:url';import path from 'node:path';
const root=fileURLToPath(new URL('../',import.meta.url));const files={};
async function add(relative){const full=path.join(root,relative);const s=await stat(full);if(s.isDirectory()){for(const name of await readdir(full))await add(path.join(relative,name));}else files[relative.replaceAll('\\','/')]=new Uint8Array(await readFile(full));}
for(const item of ['README.md','docs','delivery/examples','delivery/Burhan-v1.0.0-debug.apk','delivery/Burhan-v1.0.0-debug.apk.sha256','delivery/build-metadata.json','delivery/verification.json'])await add(item);
await mkdir(path.join(root,'delivery'),{recursive:true});await writeFile(path.join(root,'delivery/Burhan-v1.0.0-field-kit.zip'),zipSync(files,{level:1}));console.log('Packaged APK + documentation field kit. Source stays in the Burhan folder.');
