/** Dismissals span scenes for at least a minute, and remain suppressed in their original scene. */
export function isSuppressed(taskId,sceneKey,dismissals,now=Date.now()) {
 const d=dismissals[taskId];return !!d&&(d.sceneKey===sceneKey||now-d.at<60000);
}
export function eligibleSuggestions(suggestions,activeTask,completed,sceneKey,dismissals,now=Date.now()) {
 return suggestions.filter(s=>s.taskId!==activeTask&&!completed.includes(s.taskId)&&!isSuppressed(s.taskId,sceneKey,dismissals,now));
}
export function guidanceForSelection(plan,activeTask){
 if(!plan)return '';return (plan.guidanceTaskId||'')===activeTask?plan.guidance:'';
}
export function canRetake(capture,activeTask){return !!capture&&capture.taskId===activeTask;}
export function latestCaptureForTask(captures,activeTask){for(let i=captures.length-1;i>=0;i--)if(captures[i].taskId===activeTask)return captures[i];}
