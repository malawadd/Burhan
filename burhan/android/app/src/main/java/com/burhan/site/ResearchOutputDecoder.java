package com.burhan.site;

import java.io.IOException;
import java.util.*;

/** Pure decoder matching ConstructionTaskDetector; no image/runtime dependency. */
final class ResearchOutputDecoder {
    static final class Box {
        final int category; final float score; final float[] xyxy;
        Box(int category,float score,float[] xyxy){this.category=category;this.score=score;this.xyxy=xyxy;}
    }
    static List<Box> decode(float[][] boxes,long[] indices,float[] scores,float[] thresholds,
                            int width,int height,int size,boolean ssdlite,float nms)throws IOException {
        if(boxes.length!=indices.length||indices.length!=scores.length)throw new IOException("Inconsistent detection outputs");
        double scale=Math.min(size/(double)width,size/(double)height);
        int nw=Math.max(1,(int)Math.rint(width*scale)),nh=Math.max(1,(int)Math.rint(height*scale));
        int px=(size-nw)/2,py=(size-nh)/2;
        for(int i=0;i<boxes.length;i++){
            if(boxes[i].length!=4||!Float.isFinite(scores[i]))throw new IOException("Non-finite or malformed detection output");
            for(float v:boxes[i])if(!Float.isFinite(v))throw new IOException("Non-finite detection output");
        }
        List<Box> result=new ArrayList<>();
        for(int category=0;category<thresholds.length;category++){
            final int c=category;List<Integer> candidates=new ArrayList<>();
            for(int i=0;i<boxes.length;i++)if(indices[i]-(ssdlite?1:0)==c&&scores[i]>=thresholds[c]&&scores[i]<=1)candidates.add(i);
            candidates.sort((a,b)->Float.compare(scores[b],scores[a]));
            List<float[]> kept=new ArrayList<>();
            for(int j=0;j<Math.min(300,candidates.size());j++){
                int i=candidates.get(j);float[] b=boxes[i];
                float l=clip((b[0]-px)/(nw/(float)width),width),t=clip((b[1]-py)/(nh/(float)height),height);
                float r=clip((b[2]-px)/(nw/(float)width),width),bottom=clip((b[3]-py)/(nh/(float)height),height);
                if(r-l<1||bottom-t<1)continue;float[] nativeBox={l,t,r,bottom};
                boolean suppressed=false;if(!ssdlite)for(float[] prior:kept)if(iou(prior,nativeBox)>nms){suppressed=true;break;}
                if(suppressed)continue;kept.add(nativeBox);
                result.add(new Box(c,scores[i],new float[]{(float)Math.floor(l),(float)Math.floor(t),(float)Math.ceil(r),(float)Math.ceil(bottom)}));
            }
        }
        return result;
    }
    static List<Box> yolo(float[][][] output,float[] thresholds,int width,int height,int size,float nms)throws IOException {
        if(output.length!=1||output[0].length!=4+thresholds.length)throw new IOException("Unexpected YOLO output geometry");
        float[][] values=output[0];int anchors=values[0].length;
        for(float[] channel:values){if(channel.length!=anchors)throw new IOException("Inconsistent YOLO anchors");for(float v:channel)if(!Float.isFinite(v))throw new IOException("Non-finite YOLO output");}
        float[][] boxes=new float[anchors][4];long[] ids=new long[anchors];float[] scores=new float[anchors];
        for(int i=0;i<anchors;i++){
            int c=0;for(int j=1;j<thresholds.length;j++)if(values[4+j][i]>values[4+c][i])c=j;
            ids[i]=c;scores[i]=values[4+c][i];boxes[i]=new float[]{values[0][i]-values[2][i]/2,values[1][i]-values[3][i]/2,values[0][i]+values[2][i]/2,values[1][i]+values[3][i]/2};
        }
        return decode(boxes,ids,scores,thresholds,width,height,size,false,nms);
    }
    private static float clip(float v,int max){return Math.max(0,Math.min(max,v));}
    private static float iou(float[] a,float[] b){float intersection=Math.max(0,Math.min(a[2],b[2])-Math.max(a[0],b[0]))*Math.max(0,Math.min(a[3],b[3])-Math.max(a[1],b[1]));return intersection/Math.max(1e-12f,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection);}
}
