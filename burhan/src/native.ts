import {registerPlugin} from '@capacitor/core';
import type {Evidence,Language,Observation,Mode} from './types';
export interface SmartScanSummary {sessionId:string;provider:string;scene:string;steps:{at:string;instruction:string;reason:string;mode:Mode;scene:string}[]}
export interface NativeCapture {evidence:Evidence;observations:Observation[];inspectionId?:string;followUp?:boolean;findingId?:string;smartContext?:{sessionId:string;provider:string;sceneVersion:number;guidance:string;mode:string;activeTaskId:string}}
interface NativeApi {
  scan(options:{inspectionId:string;mode:string;autoSave:boolean;language:Language;followUp:boolean;findingId?:string;smart?:boolean;backendUrl?:string;backendToken?:string;smartSessionId?:string;stage?:string;taskId?:string;location?:string;modelId?:string;categoryId?:string}):Promise<{captures:NativeCapture[];smart?:SmartScanSummary}>;
  sharePhoto(options:{uri:string;filename?:string}):Promise<void>;
  getCameraPreferences():Promise<{preferences:import('./lib/camera-preferences').CameraPreferences}>;
  setCameraPreferences(options:{preferences:import('./lib/camera-preferences').CameraPreferences}):Promise<{preferences:import('./lib/camera-preferences').CameraPreferences}>;
  getSmartSettings():Promise<{url:string;token:string}>;
  setSmartSettings(options:{url:string;token:string}):Promise<void>;
  loadState():Promise<{json:string|null}>;saveState(p:{json:string}):Promise<void>;
  getPendingCaptures(p:{inspectionId?:string}):Promise<{captures:NativeCapture[]}>;
  acknowledgeCaptures(p:{ids:string[]}):Promise<void>;
  printReport(p:{html:string;jobName:string}):Promise<void>;
  beginBackup(p:{filename:string}):Promise<{id:string}>;
  appendBackup(p:{id:string;base64:string}):Promise<void>;
  finishBackup(p:{id:string}):Promise<void>;cancelBackup(p:{id:string}):Promise<void>;
}
export const BurhanNative=registerPlugin<NativeApi>('BurhanNative');
