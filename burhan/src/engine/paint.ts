import type { Observation } from '../types';
export function paintObservations(context:CanvasRenderingContext2D,observations:Observation[],width:number,height:number){
  for(const observation of observations){
    const color=['#FFD43B','#5FFFE0','#FF7A81','#A6A0FF'][observation.classId%4];
    if(observation.mask){const mask=observation.mask,image=context.createImageData(mask.width,mask.height);let cursor=0,active=false;const rgb=color.match(/\w\w/g)!.map(hex=>parseInt(hex,16));
      for(const run of mask.runs){const end=Math.min(mask.width*mask.height,cursor+run);if(active)for(let p=cursor;p<end;p++){image.data[p*4]=rgb[0];image.data[p*4+1]=rgb[1];image.data[p*4+2]=rgb[2];image.data[p*4+3]=90;}cursor=end;active=!active;}
      const canvas=document.createElement('canvas');canvas.width=mask.width;canvas.height=mask.height;canvas.getContext('2d')!.putImageData(image,0,0);context.drawImage(canvas,0,0,width,height);
    }
    const [x,y,w,h]=observation.bbox;context.strokeStyle=color;context.lineWidth=Math.max(2,width/400);context.strokeRect(x,y,w,h);
    const font=Math.max(15,width/55);context.font=`bold ${font}px sans-serif`;const label=`${observation.label.replaceAll('_',' ')} ${Math.round(observation.score*100)}%`,labelWidth=context.measureText(label).width+12;
    const labelY=Math.max(font+6,y);context.fillStyle='#111315';context.fillRect(x,labelY-font-6,labelWidth,font+9);context.fillStyle=color;context.fillText(label,x+6,labelY);
  }
}
