import type {SmartPlan} from './smart';
import type {SmartHistory} from './smart';
export function validatePlan(value:unknown):SmartPlan;
export function normalizeBackendUrl(value:string):string;
export function sceneDifference(a:number[]|null,b:number[]):number;
export function normalizeSmartHistory(value:unknown):SmartHistory;
