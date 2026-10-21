export function commitCapture<T>(getCurrent:()=>T,commit:(state:T)=>void,persist:(state:T)=>Promise<void>,mutate:(state:T)=>T):Promise<T>;
export function persistCurrent<T>(snapshot:T,getCurrent:()=>T,persist:(state:T)=>Promise<void>):Promise<void>;
