package com.burhan.site;

import android.content.Context;
import android.graphics.*;
import ai.onnxruntime.*;
import org.json.*;
import org.opencv.android.*;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import java.io.*;
import java.nio.*;
import java.security.MessageDigest;
import java.time.Instant;
import java.util.*;

/** CPU baseline. Geometry and decoders mirror the existing Python construction adapters. */
final class OnnxInspector implements AutoCloseable {
    static final class Detection {
        int classId;String label;float score;RectF box,maskDestination;byte[] mask;Bitmap maskBitmap;int maskWidth,maskHeight;
        JSONObject json(String model,String frame,String time,long ms,int width,int height) throws JSONException {
            JSONObject item=new JSONObject().put("classId",classId).put("label",label).put("score",score)
                .put("bbox",new JSONArray(new double[]{box.left,box.top,box.width(),box.height()}))
                .put("modelId",model).put("frameId",frame).put("capturedAt",time).put("inferenceMs",ms);
            // Masks are drawn into annotated evidence. Omit the optional wire mask to keep
            // SQLite rows bounded, especially with many high-resolution components.
            return item;
        }
        void makeMaskBitmap(int width,int height){
            if(mask==null)return;
            if(maskWidth>0)return; // DACL grids render exact nearest-projected pixel runs.
            int left=(int)box.left,top=(int)box.top,w=(int)box.width(),h=(int)box.height();int[] pixels=new int[w*h];
            for(int y=0;y<h;y++)for(int x=0;x<w;x++)if(mask[(top+y)*width+left+x]!=0)pixels[y*w+x]=0x75FFD43B;
            maskBitmap=Bitmap.createBitmap(pixels,w,h,Bitmap.Config.ARGB_8888);maskDestination=new RectF(box);
            mask=null;
        }
        void drawMask(Canvas canvas,int width,int height){
            if(maskBitmap!=null){canvas.drawBitmap(maskBitmap,null,maskDestination,null);return;}
            if(mask==null||maskWidth==0)return;Paint fill=new Paint();fill.setColor(0x75FFD43B);
            // OpenCV INTER_NEAREST maps floor(dst * grid / source). Its inverse is
            // these ceil boundaries; texture-centre bitmap scaling can shift them.
            for(int y=0;y<maskHeight;y++){int top=(int)Math.ceil(y*height/(double)maskHeight),bottom=(int)Math.ceil((y+1)*height/(double)maskHeight);for(int x=0;x<maskWidth;){if(mask[y*maskWidth+x]==0){x++;continue;}int start=x;while(x<maskWidth&&mask[y*maskWidth+x]!=0)x++;int left=(int)Math.ceil(start*width/(double)maskWidth),right=(int)Math.ceil(x*width/(double)maskWidth);canvas.drawRect(left,top,right,bottom,fill);}}
        }
    }
    static final class Frame {
        Bitmap image;List<Detection> detections=new ArrayList<>();String modelId,frameId=UUID.randomUUID().toString(),time=Instant.now().toString();long millis;JSONObject modelInfo;
    }
    private static final OrtEnvironment ENV=OrtEnvironment.getEnvironment();
    private final Context context;
    private OrtSession session;
    private JSONObject spec;
    private String loadedId;
    private int size;
    private String[] labels;
    private float threshold;
    OnnxInspector(Context context) {this.context=context.getApplicationContext();if(!OpenCVLoader.initLocal())throw new IllegalStateException("OpenCV could not load on this phone");}
    private JSONObject model(String id) throws Exception {
        String text;try(InputStream input=context.getAssets().open("public/models/manifest.json");ByteArrayOutputStream output=new ByteArrayOutputStream()){copy(input,output);text=output.toString("UTF-8");}
        JSONArray array=text.trim().startsWith("[")?new JSONArray(text):new JSONObject(text).getJSONArray("models");
        for(int i=0;i<array.length();i++)if(array.getJSONObject(i).getString("id").equals(id))return array.getJSONObject(i);
        throw new IOException("Bundled model not found: "+id);
    }
    synchronized void load(String id) throws Exception {
        if(id.equals(loadedId))return;
        JSONObject next=model(id);String file=next.getString("file");
        if(file.contains("..")||file.startsWith("/"))throw new IOException("Invalid model asset path");
        File directory=new File(context.getNoBackupFilesDir(),"models");if(!directory.exists()&&!directory.mkdirs())throw new IOException("Cannot create model directory");
        File target=new File(directory,next.getString("sha256")+".onnx");
        // Verify once per session load, including cached copies; corruption never silently falls back.
        if(!target.isFile()){
            File temporary=new File(directory,target.getName()+".tmp");
            try(InputStream input=context.getAssets().open("public/models/"+file);OutputStream output=new FileOutputStream(temporary)){copy(input,output);}
            if(!temporary.renameTo(target))throw new IOException("Cannot install bundled model");
        }
        MessageDigest digest=MessageDigest.getInstance("SHA-256");try(InputStream input=new FileInputStream(target)){byte[] buffer=new byte[65536];for(int n;(n=input.read(buffer))!=-1;)digest.update(buffer,0,n);}
        StringBuilder hash=new StringBuilder();for(byte value:digest.digest())hash.append(String.format(Locale.ROOT,"%02x",value&255));
        if(!hash.toString().equals(next.getString("sha256")))throw new IOException("Model integrity check failed: "+id);
        OrtSession.SessionOptions options=new OrtSession.SessionOptions();options.setIntraOpNumThreads(Math.min(4,Math.max(1,Runtime.getRuntime().availableProcessors()/2)));options.setInterOpNumThreads(1);options.addConfigEntry("session.intra_op.allow_spinning","0");
        OrtSession replacement;try{replacement=ENV.createSession(target.getAbsolutePath(),options);}finally{options.close();}
        if(session!=null)session.close();session=replacement;spec=next;loadedId=id;size=spec.getInt("inputSize");threshold=(float)spec.optDouble("confidence",.25);
        JSONArray classes=spec.getJSONArray("classes");labels=new String[classes.length()];for(int i=0;i<labels.length;i++)labels[i]=classes.getString(i);
    }
    synchronized Frame infer(Bitmap bitmap,String id) throws Exception {
        load(id);long start=android.os.SystemClock.elapsedRealtime();Frame frame=new Frame();frame.image=bitmap;frame.modelId=id;
        frame.modelInfo=new JSONObject().put("sha256",spec.getString("sha256")).put("inputSize",size).put("preprocessing",spec.getString("preprocessing")).put("confidence",threshold).put("nms",spec.optDouble("nms",.45)).put("runtime","onnxruntime-android@1.23.2/CPU");
        Mat rgba=new Mat(),bgr=new Mat(),resized=new Mat();Utils.bitmapToMat(bitmap,rgba);Imgproc.cvtColor(rgba,bgr,Imgproc.COLOR_RGBA2BGR);rgba.release();
        boolean dacl=spec.getString("kind").equals("dacl");float sx=(float)size/bitmap.getWidth(),sy=(float)size/bitmap.getHeight();int px=0,py=0;
        try {
            if(dacl){Imgproc.resize(bgr,resized,new Size(size,size),0,0,Imgproc.INTER_LINEAR);}
            else {
                // Both registered YOLO models use the Python runtime's letterbox contract.
                float ratio=Math.min(sx,sy);sx=sy=ratio;int rw=(int)Math.rint(bitmap.getWidth()*ratio),rh=(int)Math.rint(bitmap.getHeight()*ratio);px=(size-rw)/2;py=(size-rh)/2;
                Mat scaled=new Mat();Imgproc.resize(bgr,scaled,new Size(rw,rh),0,0,Imgproc.INTER_LINEAR);resized=new Mat(size,size,CvType.CV_8UC3,new Scalar(114,114,114));Mat roi=resized.submat(py,py+rh,px,px+rw);scaled.copyTo(roi);roi.release();scaled.release();
            }
            byte[] pixels=new byte[size*size*3];resized.get(0,0,pixels);long[] shape={1,3,size,size};
            try(OnnxTensor input=dacl?byteTensor(pixels,shape):floatTensor(pixels,shape);OrtSession.Result result=session.run(Collections.singletonMap(session.getInputNames().iterator().next(),input))){
                if(dacl)decodeDacl((float[][][][])result.get(0).getValue(),frame);
                else decodeYolo((float[][][])result.get(0).getValue(),id.equals("crack")?(float[][][][])result.get(1).getValue():null,sx,sy,px,py,frame);
            }
            for(Detection detection:frame.detections)detection.makeMaskBitmap(bitmap.getWidth(),bitmap.getHeight());
            frame.millis=android.os.SystemClock.elapsedRealtime()-start;return frame;
        } finally {bgr.release();resized.release();}
    }
    private OnnxTensor byteTensor(byte[] interleaved,long[] shape) throws OrtException {
        int area=size*size;ByteBuffer buffer=ByteBuffer.allocateDirect(area*3).order(ByteOrder.nativeOrder());
        for(int channel=0;channel<3;channel++)for(int i=0;i<area;i++)buffer.put(interleaved[i*3+channel]); // BGR uint8, internal ONNX normalization
        buffer.rewind();return OnnxTensor.createTensor(ENV,buffer,shape,OnnxJavaType.UINT8);
    }
    private OnnxTensor floatTensor(byte[] interleaved,long[] shape) throws OrtException {
        int area=size*size;FloatBuffer buffer=ByteBuffer.allocateDirect(area*3*4).order(ByteOrder.nativeOrder()).asFloatBuffer();
        for(int channel=2;channel>=0;channel--)for(int i=0;i<area;i++)buffer.put((interleaved[i*3+channel]&255)/255f); // RGB float NCHW
        buffer.rewind();return OnnxTensor.createTensor(ENV,buffer,shape);
    }
    private void decodeYolo(float[][][] outputs,float[][][][] prototypes,float sx,float sy,int px,int py,Frame frame) throws Exception {
        float[][] values=outputs[0];int expected=4+labels.length+(prototypes==null?0:32);
        if(values.length!=expected||values[0].length!=8400)throw new IOException("Unexpected YOLO output geometry");
        List<Integer> anchors=new ArrayList<>();
        for(int index=0;index<values[0].length;index++){
            int category=0;for(int c=1;c<labels.length;c++)if(values[4+c][index]>values[4+category][index])category=c;
            float score=values[4+category][index];boolean finite=Float.isFinite(score);for(float[] channel:values)finite &= Float.isFinite(channel[index]);if(!finite||score<threshold)continue;
            Detection detection=new Detection();detection.classId=category;detection.label=labels[category];detection.score=score;
            int left=clamp((int)Math.rint((values[0][index]-values[2][index]/2-px)/sx),0,frame.image.getWidth()),top=clamp((int)Math.rint((values[1][index]-values[3][index]/2-py)/sy),0,frame.image.getHeight());
            int right=clamp((int)Math.rint((values[0][index]+values[2][index]/2-px)/sx),0,frame.image.getWidth()),bottom=clamp((int)Math.rint((values[1][index]+values[3][index]/2-py)/sy),0,frame.image.getHeight());
            if(right<=left||bottom<=top)continue;detection.box=new RectF(left,top,right,bottom);frame.detections.add(detection);anchors.add(index);
        }
        List<Integer> order=new ArrayList<>();for(int i=0;i<frame.detections.size();i++)order.add(i);order.sort((a,b)->Float.compare(frame.detections.get(b).score,frame.detections.get(a).score));
        List<Detection> kept=new ArrayList<>();float nms=(float)spec.optDouble("nms",.45);
        for(int index:order){Detection detection=frame.detections.get(index);boolean suppressed=false;for(Detection other:kept)if(other.classId==detection.classId&&iou(other.box,detection.box)>nms){suppressed=true;break;}if(suppressed)continue;
            if(prototypes!=null&&detection.label.equals("crack")){detection.mask=crackMask(values,anchors.get(index),prototypes[0],detection.box,frame.image.getWidth(),frame.image.getHeight(),sx,sy,px,py);detection.makeMaskBitmap(frame.image.getWidth(),frame.image.getHeight());}
            kept.add(detection);if(kept.size()>=100)break;
        }
        frame.detections=new ArrayList<>();for(Detection detection:kept)if(prototypes==null||detection.label.equals("crack"))frame.detections.add(detection);
    }
    private byte[] crackMask(float[][] values,int anchor,float[][][] proto,RectF box,int width,int height,float sx,float sy,int px,int py){
        int ph=proto[0].length,pw=proto[0][0].length;float[] logits=new float[ph*pw];
        for(int c=0;c<32;c++){float weight=values[4+labels.length+c][anchor];for(int y=0;y<ph;y++)for(int x=0;x<pw;x++)logits[y*pw+x]+=weight*proto[c][y][x];}
        Mat small=new Mat(ph,pw,CvType.CV_32F),full=new Mat(),projected=new Mat(),binary=new Mat();small.put(0,0,logits);Imgproc.resize(small,full,new Size(size,size),0,0,Imgproc.INTER_LINEAR);
        int rw=(int)Math.rint(width*sx),rh=(int)Math.rint(height*sy);Mat roi=full.submat(py,py+rh,px,px+rw);Imgproc.resize(roi,projected,new Size(width,height),0,0,Imgproc.INTER_LINEAR);Core.compare(projected,new Scalar(0),binary,Core.CMP_GT);
        byte[] mask=new byte[width*height];binary.get(0,0,mask);for(int y=0;y<height;y++)for(int x=0;x<width;x++)if(x<box.left||x>=box.right||y<box.top||y>=box.bottom)mask[y*width+x]=0;
        roi.release();small.release();full.release();projected.release();binary.release();return mask;
    }
    private void decodeDacl(float[][][][] outputs,Frame frame) throws Exception {
        if(outputs.length!=1)throw new IOException("Unexpected DACL batch");float[][][] values=outputs[0];if(values.length!=19)throw new IOException("Unexpected DACL classes");int height=values[0].length,width=values[0][0].length;
        if(height!=size/4||width!=size/4)throw new IOException("Unexpected DACL output geometry");
        for(float[][] channel:values){if(channel.length!=height)throw new IOException("Invalid DACL channel");for(float[] row:channel){if(row.length!=width)throw new IOException("Invalid DACL row");for(float value:row)if(!Float.isFinite(value))throw new IOException("Non-finite DACL output");}}
        for(int category=0;category<labels.length;category++){
            // Registered damage set excludes functional components 13..17, preserving washout 18.
            if(category>=13&&category<=17)continue;
            byte[] thresholded=new byte[width*height];float[] probability=new float[width*height];
            for(int y=0;y<height;y++)for(int x=0;x<width;x++){float logit=values[category][y][x];if(!Float.isFinite(logit))throw new IOException("Non-finite model output");float p=(float)(1/(1+Math.exp(-Math.max(-40,Math.min(40,logit)))));probability[y*width+x]=p;if(p>threshold)thresholded[y*width+x]=(byte)255;}
            Mat mask=new Mat(height,width,CvType.CV_8U),components=new Mat(),stats=new Mat(),centroids=new Mat();mask.put(0,0,thresholded);
            int count=Imgproc.connectedComponentsWithStats(mask,components,stats,centroids,8,CvType.CV_32S);int[] ids=new int[width*height];components.get(0,0,ids);List<Integer> retained=new ArrayList<>();
            for(int i=1;i<count;i++)if(stats.get(i,Imgproc.CC_STAT_AREA)[0]>=4)retained.add(i);retained.sort((a,b)->Double.compare(stats.get(b,Imgproc.CC_STAT_AREA)[0],stats.get(a,Imgproc.CC_STAT_AREA)[0]));if(retained.size()>20)retained=new ArrayList<>(retained.subList(0,20));
            for(int region:retained){
                int x=(int)stats.get(region,Imgproc.CC_STAT_LEFT)[0],y=(int)stats.get(region,Imgproc.CC_STAT_TOP)[0],rw=(int)stats.get(region,Imgproc.CC_STAT_WIDTH)[0],rh=(int)stats.get(region,Imgproc.CC_STAT_HEIGHT)[0],area=(int)stats.get(region,Imgproc.CC_STAT_AREA)[0];
                Detection detection=new Detection();detection.classId=category;detection.label=labels[category];double sum=0;byte[] component=new byte[width*height];for(int i=0;i<ids.length;i++)if(ids[i]==region){sum+=probability[i];component[i]=(byte)255;}detection.score=(float)(sum/area);
                detection.box=new RectF((int)Math.ceil(x*frame.image.getWidth()/(double)width),(int)Math.ceil(y*frame.image.getHeight()/(double)height),(int)Math.ceil((x+rw)*frame.image.getWidth()/(double)width),(int)Math.ceil((y+rh)*frame.image.getHeight()/(double)height));
                // Retain the native component grid: nearest projection during drawing is
                // exact and avoids 100 full-resolution mask allocations on larger photos.
                detection.mask=component;detection.maskWidth=width;detection.maskHeight=height;frame.detections.add(detection);
            }
            mask.release();components.release();stats.release();centroids.release();
        }
        frame.detections.sort((a,b)->Float.compare(b.score,a.score));if(frame.detections.size()>100)frame.detections=new ArrayList<>(frame.detections.subList(0,100));
    }
    static float iou(RectF a,RectF b){float intersection=Math.max(0,Math.min(a.right,b.right)-Math.max(a.left,b.left))*Math.max(0,Math.min(a.bottom,b.bottom)-Math.max(a.top,b.top));return intersection/Math.max(1,a.width()*a.height()+b.width()*b.height()-intersection);}
    private static int clamp(int value,int low,int high){return Math.max(low,Math.min(high,value));}
    private static void copy(InputStream input,OutputStream output)throws IOException{byte[] bytes=new byte[65536];for(int n;(n=input.read(bytes))!=-1;)output.write(bytes,0,n);}
    @Override public synchronized void close() throws Exception {if(session!=null){session.close();session=null;}loadedId=null;}
}
