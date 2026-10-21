import {registerPlugin} from '@capacitor/core';
import type {Evidence,Language,Observation} from './types';
export interface NativeCapture {evidence:Evidence;observations:Observation[];inspectionId?:string;followUp?:boolean;findingId?:string}
interface NativeApi {
  scan(options:{inspectionId:string;mode:string;autoSave:boolean;language:Language;followUp:boolean;findingId?:string}):Promise<{captures:NativeCapture[]}>;
  loadState():Promise<{json:string|null}>;saveState(p:{json:string}):Promise<void>;
  getPendingCaptures(p:{inspectionId?:string}):Promise<{captures:NativeCapture[]}>;
  acknowledgeCaptures(p:{ids:string[]}):Promise<void>;
  printReport(p:{html:string;jobName:string}):Promise<void>;
  beginBackup(p:{filename:string}):Promise<{id:string}>;
  appendBackup(p:{id:string;base64:string}):Promise<void>;
  finishBackup(p:{id:string}):Promise<void>;cancelBackup(p:{id:string}):Promise<void>;
}
export const BurhanNative=registerPlugin<NativeApi>('BurhanNative');
