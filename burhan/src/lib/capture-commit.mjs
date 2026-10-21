/** Rebase a durable capture mutation if remote history changed while storage was pending. */
export async function commitCapture(getCurrent,commit,persist,mutate){
 let base=getCurrent();
 for(;;){const next=mutate(base);await persist(next);if(getCurrent()===base){commit(next);return next;}base=getCurrent();}
}
export async function persistCurrent(snapshot,getCurrent,persist){if(getCurrent()===snapshot)await persist(snapshot);}
