import type { Observation } from '../types';
import {iou} from './decoder';
/** Short-term association only; inspector review/merge remains authoritative. */
export class CandidateTracker {
  private tracks:{last:number;first:number;count:number;box:Observation['bbox'];label:string;saved:boolean}[]=[];
  private lastSave=-Infinity;
  reset(){this.tracks=[];this.lastSave=-Infinity;}
  update(observations:Observation[],now:number):Observation[]{
    this.tracks=this.tracks.filter(t=>now-t.last<=2000);const matched=new Set<number>(),ready:Observation[]=[];
    for(const observation of observations){let index=this.tracks.findIndex((t,i)=>!matched.has(i)&&t.label===observation.label&&iou(t.box,observation.bbox)>=.35);
      if(index<0){index=this.tracks.length;this.tracks.push({last:now,first:now,count:0,box:observation.bbox,label:observation.label,saved:false});}
      const track=this.tracks[index];matched.add(index);track.last=now;track.count++;track.box=observation.bbox;
      if(track.count>=3&&now-track.first<=2000&&!track.saved&&now-this.lastSave>=1000)ready.push(observation);
    }
    if(ready.length){this.lastSave=now;for(const track of this.tracks)if(ready.some(o=>o.label===track.label&&iou(o.bbox,track.box)>=.35))track.saved=true;}
    return ready;
  }
}
