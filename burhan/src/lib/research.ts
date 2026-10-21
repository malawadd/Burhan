import type {Catalogue,ModelInfo,Task} from '../types';
import source from '../../assets/research-catalogue.json';

export const researchFamilies=source.families;
export const researchTasks=source.tasks as Task[];
export const RESEARCH_MODEL_IDS=['research-phele','research-fasteners','research-stain'] as const;
export interface ClientCapabilities {expandedInspectionModels:boolean;installedModelIds:string[]}
export async function loadResearchTasks():Promise<Task[]>{return source.tasks as Task[];}
export function mergeResearchCatalogue(catalogue:Catalogue,enabled:boolean):Catalogue{const merged=withResearchTasks(catalogue,source.tasks as Task[],enabled);return enabled?{...merged,stages:[...catalogue.stages,...source.families.map((f,index)=>({id:`research:${f.id}`,label:f.title,labelAr:f.titleAr,order:100+index}))]}:merged;}
/** Disabled clients retain the exact original task catalogue. */
export function withResearchTasks(catalogue:Catalogue,research:Task[],enabled:boolean):Catalogue{
  return enabled?{...catalogue,tasks:[...catalogue.tasks.filter(t=>!t.researchOnly),...research]}:{...catalogue,tasks:catalogue.tasks.filter(t=>!t.researchOnly)};
}
export function researchModel(task:Task|undefined,models:ModelInfo[],enabled:boolean):ModelInfo|undefined{
  if(!task?.researchOnly||task.mode!=='research')return undefined;
  if(!enabled)throw Error('Expanded inspection models are disabled');
  const model=models.find(m=>m.id===task.modelId&&m.researchOnly===true);
  if(!model)throw Error(`Selected experimental model unavailable: ${task.modelId??task.id}`);
  return model;
}
export function taskObservations<T extends {classId:number;label:string}>(rows:T[],task:Task|undefined):T[]{
  if(!task?.researchOnly||!task.modelId)return rows;
  // Match the registered category, never model-wide unrelated classes.
  const ids=task.classIds;return ids?.length?rows.filter(row=>ids.includes(row.classId)):rows.filter(row=>row.label===task.categoryId);
}
