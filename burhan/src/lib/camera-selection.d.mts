import type {SmartPlan} from './smart';
export interface Dismissal {sceneKey:number;at:number}
export function isSuppressed(taskId:string,sceneKey:number,dismissals:Record<string,Dismissal>,now?:number):boolean;
export function eligibleSuggestions(suggestions:SmartPlan['suggestions'],activeTask:string,completed:string[],sceneKey:number,dismissals:Record<string,Dismissal>,now?:number):SmartPlan['suggestions'];
export function guidanceForSelection(plan:SmartPlan|null,activeTask:string):string;
export function canRetake(capture:{taskId:string}|undefined,activeTask:string):boolean;
export function latestCaptureForTask<T extends {taskId:string}>(captures:T[],activeTask:string):T|undefined;
