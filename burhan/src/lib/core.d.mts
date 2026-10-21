import type {AppState,Language,Report} from '../types';
export function validateState(input:unknown):AppState;
export function escapeHtml(value:unknown):string;
export function safeImage(uri:string):string;
export function snapshotReport(state:AppState,inspectionId:string,language:Language|'bilingual',id:string,now:string):Report;
