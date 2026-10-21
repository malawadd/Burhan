import {openDB} from 'idb';
import {Capacitor} from '@capacitor/core';
import {BurhanNative as native} from '../native';
import type {AppState,Catalogue,Evidence,Finding,Observation} from '../types';
import {validateState} from './core.mjs';
export const uid=()=>crypto.randomUUID();
export function initialState():AppState{return {schemaVersion:1,projects:[],inspections:[],findings:[],reports:[],settings:{language:'en',theme:'dark',inspector:'',autoSave:true}};}
interface Capture {evidence:Evidence;observations:Observation[];inspectionId?:string;followUp?:boolean;findingId?:string}
const dbPromise=()=>openDB('burhan-site',1,{upgrade(db){db.createObjectStore('state');}});
let saveQueue:Promise<void>=Promise.resolve();
async function persist(state:AppState){
  const json=JSON.stringify(state);
  if(Capacitor.isNativePlatform()){
    await native.saveState({json});
    const ids=state.findings.flatMap(f=>[...f.evidence,...f.followUp].map(e=>e.id));
    const unique=[...new Set(ids)];
    for(let start=0;start<unique.length;start+=250)await native.acknowledgeCaptures({ids:unique.slice(start,start+250)});
  } else {const db=await dbPromise();await db.put('state',json,'current');db.close();}
}
export function saveState(state:AppState):Promise<void>{
  const snapshot=structuredClone(state);
  const next=saveQueue.catch(()=>{}).then(()=>persist(snapshot));saveQueue=next;return next;
}
export async function loadState():Promise<AppState>{
  let json:string|null|undefined;
  if(Capacitor.isNativePlatform())json=(await native.loadState()).json;
  else {const db=await dbPromise();json=await db.get('state','current');db.close();}
  const state=json?validateState(JSON.parse(json)):initialState();
  if(Capacitor.isNativePlatform()){
    const pending=await native.getPendingCaptures({});
    const existing=new Set(state.findings.flatMap(f=>[...f.evidence,...f.followUp].map(e=>e.id)));
    let changed=false;
    for(const capture of pending.captures){
      if(existing.has(capture.evidence.id))continue;
      const inspectionId=capture.inspectionId;
      if(!inspectionId||!state.inspections.some(i=>i.id===inspectionId))continue;
      capture.evidence.observations=structuredClone(capture.observations);
      const target=capture.followUp&&capture.findingId?state.findings.find(f=>f.id===capture.findingId&&f.inspectionId===inspectionId):undefined;
      if(target){target.followUp.push(capture.evidence);target.updatedAt=new Date().toISOString();existing.add(capture.evidence.id);changed=true;continue;}
      const observations=capture.observations.length?capture.observations:[undefined];
      for(const observation of observations){
        const now=capture.evidence.capturedAt;
        const finding:Finding={id:uid(),inspectionId,label:observation?.label??'Recovered camera evidence',source:observation?'ai':'manual',review:observation?'unreviewed':'confirmed',status:'open',priority:'medium',notes:'Recovered from interrupted camera session',action:'',createdAt:now,updatedAt:now,evidence:[capture.evidence],followUp:[],observation};
        state.findings.push(finding);
      }
      existing.add(capture.evidence.id);changed=true;
    }
    if(changed||pending.captures.some(c=>existing.has(c.evidence.id)))await persist(state);
  }
  return state;
}
export async function loadCatalogue():Promise<Catalogue>{
  const [r,a]=await Promise.all([fetch('/catalogue.json'),fetch('/catalogue-ar.json')]);
  if(!r.ok||!a.ok)throw new Error('Construction catalogue unavailable');
  const cat:Catalogue=await r.json(),ar=await a.json();
  cat.stages.forEach(s=>s.labelAr=ar.stages[s.id]);cat.tasks.forEach(t=>t.titleAr=ar.tasks[t.id]);cat.categories.forEach(c=>c.labelAr=ar.categories[c.id]);return cat;
}
export function imageSrc(path:string){return path.startsWith('file:')?Capacitor.convertFileSrc(path):path;}
