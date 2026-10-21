import type { ModelInfo, Observation } from '../types';

export interface TensorOutput {data:Float32Array;dims:readonly number[]}
export interface Geometry {width:number;height:number;size:number;scale:number;padX:number;padY:number;resizedWidth:number;resizedHeight:number}
export type RawObservation = Omit<Observation,'modelId'|'frameId'|'capturedAt'|'inferenceMs'>;
export function geometry(width:number,height:number,size:number,stretch=false):Geometry {
  const scale=stretch?1:Math.min(size/width,size/height), resizedWidth=stretch?size:roundEven(width*scale),resizedHeight=stretch?size:roundEven(height*scale);
  return {width,height,size,scale,padX:Math.floor((size-resizedWidth)/2),padY:Math.floor((size-resizedHeight)/2),resizedWidth,resizedHeight};
}
/** Python/OpenCV use ties-to-even, unlike JavaScript Math.round. */
export function roundEven(value:number) { const floor=Math.floor(value),f=value-floor; return f===.5?(floor%2===0?floor:floor+1):Math.round(value); }
export function iou(a:readonly number[],b:readonly number[]) {const area=Math.max(0,Math.min(a[0]+a[2],b[0]+b[2])-Math.max(a[0],b[0]))*Math.max(0,Math.min(a[1]+a[3],b[1]+b[3])-Math.max(a[1],b[1])); return area/(a[2]*a[3]+b[2]*b[3]-area||1);}
/** Bilinear half-pixel sampling, clamped at borders, as in cv2.resize. */
function sample(data:ArrayLike<number>,width:number,height:number,x:number,y:number,offset=0,stride=1):number {
  const bx=Math.floor(x),by=Math.floor(y),fx=x-bx,fy=y-by;
  const x0=Math.max(0,Math.min(width-1,bx)),x1=Math.max(0,Math.min(width-1,bx+1)),y0=Math.max(0,Math.min(height-1,by)),y1=Math.max(0,Math.min(height-1,by+1));
  return (data[offset+(y0*width+x0)*stride]*(1-fx)+data[offset+(y0*width+x1)*stride]*fx)*(1-fy)+(data[offset+(y1*width+x0)*stride]*(1-fx)+data[offset+(y1*width+x1)*stride]*fx)*fy;
}
export function preparePixels(rgba:Uint8ClampedArray,width:number,height:number,model:ModelInfo) {
  const g=geometry(width,height,model.inputSize,model.kind==='dacl'),plane=g.size*g.size;
  const data=model.kind==='dacl'?new Uint8Array(plane*3):new Float32Array(plane*3);
  if(model.kind!=='dacl') data.fill(114/255);
  for(let y=0;y<g.resizedHeight;y++) for(let x=0;x<g.resizedWidth;x++) {
    const sourceX=(x+.5)*width/g.resizedWidth-.5,sourceY=(y+.5)*height/g.resizedHeight-.5,p=(y+g.padY)*g.size+x+g.padX;
    for(let c=0;c<3;c++) {const channel=model.kind==='dacl'?2-c:c;const value=Math.max(0,Math.min(255,Math.round(sample(rgba,width,height,sourceX,sourceY,channel,4))));data[c*plane+p]=model.kind==='dacl'?value:value/255;}
  }
  return {data,geometry:g};
}
/** Mask format: row-major alternating runs starting with background; source-image dimensions. */
export function encodeMask(width:number,height:number,active:(x:number,y:number)=>boolean):NonNullable<Observation['mask']> {
  const runs:number[]=[];let previous=false,count=0;
  for(let y=0;y<height;y++)for(let x=0;x<width;x++){const next=active(x,y);if(next===previous)count++;else{runs.push(count);count=1;previous=next;}}
  runs.push(count);return {width,height,runs};
}
function decodeYolo(outputs:TensorOutput[],model:ModelInfo,g:Geometry):RawObservation[] {
  const pred=outputs.find(o=>o.dims.length===3);if(!pred)throw Error('Missing detection output');
  const channels=4+model.classes.length+(model.kind==='yolo_segmentation'?32:0),anchors=8400;
  if(pred.dims[1]!==channels||pred.dims[2]!==anchors)throw Error(`Unexpected detection output: ${pred.dims}`);
  const rows:{observation:RawObservation;index:number}[]=[];
  for(let i=0;i<anchors;i++){
    let classId=0,score=-Infinity,finite=true;for(let c=0;c<channels;c++)if(!Number.isFinite(pred.data[c*anchors+i])){finite=false;break;}if(!finite)continue;
    for(let c=0;c<model.classes.length;c++)if(pred.data[(4+c)*anchors+i]>score){score=pred.data[(4+c)*anchors+i];classId=c;}
    if(score<model.confidence)continue;
    const cx=pred.data[i],cy=pred.data[anchors+i],w=pred.data[2*anchors+i],h=pred.data[3*anchors+i];
    const clamp=(v:number,max:number)=>Math.max(0,Math.min(max,roundEven(v)));
    const x1=clamp((cx-w/2-g.padX)/g.scale,g.width),y1=clamp((cy-h/2-g.padY)/g.scale,g.height),x2=clamp((cx+w/2-g.padX)/g.scale,g.width),y2=clamp((cy+h/2-g.padY)/g.scale,g.height);
    if(x2>x1&&y2>y1)rows.push({index:i,observation:{classId,label:model.classes[classId],score,bbox:[x1,y1,x2-x1,y2-y1]}});
  }
  rows.sort((a,b)=>b.observation.score-a.observation.score);const kept:typeof rows=[];
  for(const row of rows){if(!kept.some(k=>k.observation.classId===row.observation.classId&&iou(k.observation.bbox,row.observation.bbox)>model.nms))kept.push(row);if(kept.length>=100)break;}
  if(model.kind==='yolo_segmentation'){
    const proto=outputs.find(o=>o.dims.length===4);if(!proto||proto.dims.join(',')!=='1,32,160,160')throw Error('Unexpected segmentation prototypes');
    for(const row of kept){const logits=new Float32Array(160*160);for(let c=0;c<32;c++){const coefficient=pred.data[(4+model.classes.length+c)*anchors+row.index];for(let p=0;p<logits.length;p++)logits[p]+=coefficient*proto.data[c*logits.length+p];}
      // Match Python's two bilinear resizes: prototype -> 640 -> unpadded source.
      const expanded=new Float32Array(g.size*g.size);for(let y=0;y<g.size;y++)for(let x=0;x<g.size;x++)expanded[y*g.size+x]=sample(logits,160,160,(x+.5)*160/g.size-.5,(y+.5)*160/g.size-.5);
      const [left,top,w,h]=row.observation.bbox;row.observation.mask=encodeMask(g.width,g.height,(x,y)=>{
        if(x<left||y<top||x>=left+w||y>=top+h)return false;
        const sourceX=(x+.5)*g.resizedWidth/g.width-.5,sourceY=(y+.5)*g.resizedHeight/g.height-.5;
        // The second resize clamps to the unpadded crop, not the padded image.
        return sample(expanded,g.size,g.size,Math.max(0,Math.min(g.resizedWidth-1,sourceX))+g.padX,Math.max(0,Math.min(g.resizedHeight-1,sourceY))+g.padY)>0;
      });
    }
  }
  return kept.map(row=>row.observation);
}
function decodeDacl(output:TensorOutput,model:ModelInfo,g:Geometry):RawObservation[]{
  const [,channels,h,w]=output.dims;if(channels!==19||h!==g.size/4||w!==g.size/4)throw Error(`Unexpected DACL maps: ${output.dims}`);
  const result:RawObservation[]=[];const plane=w*h;
  for(let c=0;c<channels;c++){
    if(c>=13&&c!==18)continue; // Fourteen visible-condition channels, no site-equipment classes.
    const probs=new Float32Array(plane),labels=new Int32Array(plane);for(let p=0;p<plane;p++){const logit=output.data[c*plane+p];if(!Number.isFinite(logit))throw Error('Model returned nonfinite values');probs[p]=1/(1+Math.exp(-Math.max(-40,Math.min(40,logit))));}
    const regions:{id:number;area:number;sum:number;left:number;right:number;top:number;bottom:number}[]=[];let id=0;const queue=new Int32Array(plane);
    for(let p=0;p<plane;p++){
      if(labels[p]||probs[p]<=model.confidence)continue;id++;let head=0,tail=1;queue[0]=p;labels[p]=id;let left=w,right=0,top=h,bottom=0,sum=0;
      while(head<tail){const index=queue[head++],x=index%w,y=Math.floor(index/w);sum+=probs[index];left=Math.min(left,x);right=Math.max(right,x);top=Math.min(top,y);bottom=Math.max(bottom,y);
        for(let dy=-1;dy<=1;dy++)for(let dx=-1;dx<=1;dx++){const nx=x+dx,ny=y+dy,np=ny*w+nx;if(nx>=0&&nx<w&&ny>=0&&ny<h&&!labels[np]&&probs[np]>model.confidence){labels[np]=id;queue[tail++]=np;}}
      }
      if(tail>=4)regions.push({id,area:tail,sum,left,right:right+1,top,bottom:bottom+1});
    }
    if(regions.length>20)regions.sort((a,b)=>b.area-a.area);for(const region of regions.slice(0,20)){
      const left=Math.ceil(region.left*g.width/w),right=Math.min(g.width,Math.ceil(region.right*g.width/w)),top=Math.ceil(region.top*g.height/h),bottom=Math.min(g.height,Math.ceil(region.bottom*g.height/h));if(right<=left||bottom<=top)continue;
      result.push({classId:c,label:model.classes[c],score:region.sum/region.area,bbox:[left,top,right-left,bottom-top],mask:encodeMask(g.width,g.height,(x,y)=>labels[Math.floor(y*h/g.height)*w+Math.floor(x*w/g.width)]===region.id)});
    }
  }
  return result.sort((a,b)=>b.score-a.score).slice(0,100);
}
export function decodeOutputs(outputs:TensorOutput[],model:ModelInfo,g:Geometry):RawObservation[]{return model.kind==='dacl'?decodeDacl(outputs[0],model,g):decodeYolo(outputs,model,g);}
