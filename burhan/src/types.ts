export type Language = 'en'|'ar';
export type Mode = 'crack'|'surface'|'ppe'|'manual';
export interface Project {id:string;name:string;address:string;createdAt:string}
export interface Inspection {smart?:import('./lib/smart').SmartHistory;id:string;projectId:string;title:string;location:string;inspector:string;stage:string;taskId:string;mode:Mode;status:'draft'|'in_progress'|'completed';createdAt:string;updatedAt:string;acknowledged:boolean;checklist:Record<string,'not_checked'|'acceptable'|'issue'|'not_applicable'>}
export interface Observation {classId:number;label:string;score:number;bbox:[number,number,number,number];mask?:{width:number;height:number;runs:number[]};modelId:string;frameId:string;capturedAt:string;inferenceMs:number;model?:{sha256:string;inputSize:number;preprocessing:string;confidence:number;nms:number;runtime:string}}
export interface Evidence {captureSource?:'analysis'|'photo'|'import';detectorBaseline?:string;annotations?:{version:1;shapes:{kind:'arrow'|'ellipse';points:[number,number,number,number]}[]};smartContext?:{sessionId:string;provider:string;sceneVersion:number;guidance:string;mode:Mode;activeTaskId:string};id:string;original:string;annotated:string;capturedAt:string;width:number;height:number;observations?:Observation[]}
export interface Finding {id:string;inspectionId:string;label:string;source:'ai'|'manual';review:'unreviewed'|'confirmed'|'dismissed';status:'open'|'in_progress'|'resolved';priority:'low'|'medium'|'high';notes:string;action:string;createdAt:string;updatedAt:string;evidence:Evidence[];followUp:Evidence[];observation?:Observation}
export interface Report {id:string;inspectionId:string;version:number;createdAt:string;language:Language|'bilingual';snapshot:{project:Project;inspection:Inspection;findings:Finding[]};summary:string}
export interface Settings {language:Language;theme:'dark'|'light'|'system';inspector:string;autoSave:boolean}
export interface AppState {schemaVersion:1;projects:Project[];inspections:Inspection[];findings:Finding[];reports:Report[];settings:Settings}
export interface Task {id:string;title:string;titleAr?:string;stage:string;mode:Mode;capabilityId:string|null;automatic:boolean;requiredEvidence:string[]}
export interface Catalogue {stages:{id:string;label:string;labelAr?:string;order:number}[];tasks:Task[];categories:{id:string;label:string;labelAr?:string;status:string;scope:string}[]}
export interface ModelInfo {id:string;file:string;sha256:string;bytes:number;inputSize:number;classes:string[];kind:'yolo_detection'|'yolo_segmentation'|'dacl';mode:Mode;preprocessing:string;confidence:number;nms:number;scope:string;license:string}
export interface EngineResult {observations:Observation[];modelId:string;inferenceMs:number;imageWidth:number;imageHeight:number}
