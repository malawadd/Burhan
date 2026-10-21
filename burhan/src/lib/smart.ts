import {Capacitor} from '@capacitor/core';
import {BurhanNative} from '../native';
import type {Mode} from '../types';
import {normalizeBackendUrl,validatePlan} from './smart-contract.mjs';
export const DEFAULT_BACKEND='https://merhaba-api.kiyan.finance';
export interface SmartSettings {url:string;token:string}
export interface SmartPlan {scene:{environment:'indoor'|'outdoor'|'transitional'|'unknown';description:string;confidence:number};guidance:string;guidanceTaskId?:string|null;reason:string;needsHumanReview:boolean;suggestions:{taskId:string;title:string;mode:Mode;reason:string;automatic:boolean}[];actions:{id:string;type:'select_detector'|'request_view'|'request_manual_check'|'ask_context'|'prepare_finding';mode:Mode|null;taskId:string|null;instruction:string}[]}
export interface CheckEvent {eventId:string;type:'proposed'|'selected'|'dismissed'|'stopped'|'completed'|'skipped';taskId:string;sceneVersion:number;sourceTurnId?:string|null}
export interface SmartHistory {sessionId:string;provider:string;scene:string;steps:{at:string;instruction:string;reason:string;mode:Mode;scene:string}[]}
export async function getSmartSettings():Promise<SmartSettings>{
 if(Capacitor.isNativePlatform()){const s=await BurhanNative.getSmartSettings();return {url:s.url||DEFAULT_BACKEND,token:s.token||''};}
 try{const s=JSON.parse(localStorage.getItem('burhan-smart-connection')||'{}');return {url:normalizeBackendUrl(s.url||DEFAULT_BACKEND),token:typeof s.token==='string'?s.token:''};}catch{return {url:DEFAULT_BACKEND,token:''};}
}
export async function setSmartSettings(settings:SmartSettings){const s={url:normalizeBackendUrl(settings.url),token:settings.token};if(Capacitor.isNativePlatform())await BurhanNative.setSmartSettings(s);else localStorage.setItem('burhan-smart-connection',JSON.stringify(s));}
export class SmartApi {
 constructor(readonly settings:SmartSettings){}
 async request<T>(path:string,body?:unknown,signal?:AbortSignal,method=body===undefined?'GET':'POST'):Promise<T>{
  const timeout=AbortSignal.timeout(20000),combined=signal?AbortSignal.any([signal,timeout]):timeout;
  const response=await fetch(`${normalizeBackendUrl(this.settings.url)}/api/v1${path}`,{method,headers:{...(body===undefined?{}:{'Content-Type':'application/json'}),...(this.settings.token?{Authorization:`Bearer ${this.settings.token}`}:{})},body:body===undefined?undefined:JSON.stringify(body),signal:combined,credentials:'omit',cache:'no-store'});
  const data=await response.json().catch(()=>({}));if(!response.ok)throw Error(typeof data.detail==='string'?data.detail:typeof data.error==='string'?data.error:`Backend unavailable (${response.status})`);return data as T;
 }
 checkEvent(sessionId:string,event:CheckEvent){return this.request<{accepted:boolean;duplicate:boolean;eventId:string}>(`/sessions/${encodeURIComponent(sessionId)}/events`,event);}
 health(){return this.request<{status:string;provider:string;ready:boolean}>('/health');}
 async pair(code:string){const result=await this.request<{token:string}>('/pair',{code,deviceName:Capacitor.isNativePlatform()?'Burhan Android':'Burhan browser'});if(!result.token)throw Error('Pairing did not return a token');return result.token;}
 async plan(sessionId:string,body:unknown,signal:AbortSignal){
  const created=await this.request<{id:string}>(`/sessions/${encodeURIComponent(sessionId)}/turns`,body,signal);
  for(let count=0;count<180;count++){
   await new Promise<void>((resolve,reject)=>{const id=setTimeout(()=>{signal.removeEventListener('abort',abort);resolve()},1000);const abort=()=>{clearTimeout(id);reject(new DOMException('Cancelled','AbortError'))};if(signal.aborted)abort();else signal.addEventListener('abort',abort,{once:true})});
   const job=await this.request<{id:string;status:string;sceneVersion:number;provider:string;result?:unknown;error?:string}>(`/sessions/${encodeURIComponent(sessionId)}/turns/${encodeURIComponent(created.id)}`,undefined,signal);
   if(job.status==='completed'){if(typeof job.provider!=='string'||job.provider.length>80)throw Error('Invalid planning provider');return {...job,result:validatePlan(job.result) as SmartPlan};}
   if(['failed','cancelled','superseded'].includes(job.status))throw Error(job.error||job.status);
  }throw Error('Inspection guidance timed out; local camera remains available');
 }
}
