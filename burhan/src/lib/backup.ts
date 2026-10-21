import {zipSync,unzipSync,strToU8,strFromU8} from 'fflate';
import type {AppState} from '../types';
import {dataUri} from './reports';
import {validateState} from './core.mjs';
import {Capacitor} from '@capacitor/core';
import {BurhanNative} from '../native';
export async function exportBackup(state:AppState):Promise<void>{
  const backupDate=new Intl.DateTimeFormat('en-CA',{timeZone:'Asia/Riyadh',year:'numeric',month:'2-digit',day:'2-digit'}).format(new Date());
  const copy=structuredClone(state);const media:Record<string,Uint8Array>={};
  const convert=async(path:string)=>{
    if(path.startsWith('backup:'))return path;
    const uri=await dataUri(path);const id=await crypto.subtle.digest('SHA-256',strToU8(uri));
    const key='media/'+[...new Uint8Array(id)].map(b=>b.toString(16).padStart(2,'0')).join('')+'.txt';media[key]=strToU8(uri);return 'backup:'+key;
  };
  const evidence=copy.findings.flatMap(f=>[...f.evidence,...f.followUp]).concat(copy.reports.flatMap(r=>r.snapshot.findings.flatMap(f=>[...f.evidence,...f.followUp])));
  for(const e of evidence){e.original=await convert(e.original);e.annotated=await convert(e.annotated);}
  media['burhan.json']=strToU8(JSON.stringify({format:'burhan-backup',version:1,state:copy}));
  const zipped=zipSync(media,{level:1});
  if(zipped.byteLength>200*1024*1024)throw new Error('Backup exceeds 200 MB limit. Export individual reports and reduce evidence before retrying.');
  const blob=new Blob([zipped as Uint8Array<ArrayBuffer>],{type:'application/zip'});
  // Native WebViews do not implement download links; share through native file storage.
  if(Capacitor.isNativePlatform()){
    const {id}=await BurhanNative.beginBackup({filename:`Burhan-backup-${backupDate}.zip`});
    try {
      for(let offset=0;offset<zipped.length;offset+=512*1024){const chunk=zipped.subarray(offset,offset+512*1024);let binary='';for(let i=0;i<chunk.length;i+=32768)binary+=String.fromCharCode(...chunk.subarray(i,i+32768));await BurhanNative.appendBackup({id,base64:btoa(binary)});}
      await BurhanNative.finishBackup({id});
    }catch(error){await BurhanNative.cancelBackup({id}).catch(()=>{});throw error;}
  } else {
    const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download=`Burhan-backup-${backupDate}.zip`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  }
}
export async function importBackup(file:File):Promise<AppState>{
  if(file.size>200*1024*1024)throw new Error('Backup exceeds 200 MB limit');
  let expanded=0;
  const files=unzipSync(new Uint8Array(await file.arrayBuffer()),{filter(entry){
    expanded+=entry.originalSize;
    if(expanded>350*1024*1024||entry.originalSize>100*1024*1024)throw new Error('Expanded backup too large');
    if(!/^(burhan\.json|media\/[a-f0-9]{64}\.txt)$/.test(entry.name))throw new Error('Unexpected backup entry');
    return true;
  }});
  if(!files['burhan.json'])throw new Error('Not a Burhan backup');
  const data=JSON.parse(strFromU8(files['burhan.json']));if(data.format!=='burhan-backup'||data.version!==1)throw new Error('Unsupported backup');
  const state=validateState(data.state);
  const evidence=state.findings.flatMap(f=>[...f.evidence,...f.followUp]).concat(state.reports.flatMap(r=>r.snapshot.findings.flatMap(f=>[...f.evidence,...f.followUp])));
  for(const e of evidence)for(const field of ['original','annotated'] as const){
    const ref=e[field];if(!ref.startsWith('backup:'))throw new Error('Nonportable backup evidence');
    const key=ref.slice(7);const bytes=files[key];if(!bytes)throw new Error('Missing backup evidence');
    const uri=strFromU8(bytes);const digest=await crypto.subtle.digest('SHA-256',strToU8(uri));
    const expected='media/'+[...new Uint8Array(digest)].map(b=>b.toString(16).padStart(2,'0')).join('')+'.txt';
    if(key!==expected||!/^data:image\/(jpeg|png|webp);base64,/.test(uri))throw new Error('Backup integrity check failed');
    e[field]=uri;
  }
  return state;
}
