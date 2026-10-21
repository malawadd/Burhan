package com.burhan.site;

import android.graphics.Bitmap;
import org.json.JSONObject;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

/** OpenCV quality measurements are capture hints, never structural measurements. */
final class SmartQuality {
    private double[] prior;
    private int changed;
    private double[] candidate;
    int sceneVersion=1;
    SmartQuality(){if(!OpenCVLoader.initLocal())throw new IllegalStateException("OpenCV unavailable");}
    JSONObject measure(Bitmap bitmap) throws Exception {
        Mat rgba=new Mat(),gray=new Mat(),small=new Mat(),lap=new Mat();MatOfDouble mean=new MatOfDouble(),std=new MatOfDouble();
        try{
            Utils.bitmapToMat(bitmap,rgba);Imgproc.cvtColor(rgba,gray,Imgproc.COLOR_RGBA2GRAY);Imgproc.resize(gray,small,new Size(160,120));Imgproc.Laplacian(small,lap,CvType.CV_64F);Core.meanStdDev(lap,mean,std);double variance=std.toArray()[0]*std.toArray()[0];
            byte[] pixels=new byte[160*120];small.get(0,0,pixels);double[] histogram=new double[16];int dark=0,bright=0;double sum=0;
            for(byte value:pixels){int p=value&255;histogram[p/16]+=1d/pixels.length;sum+=p;if(p<16)dark++;if(p>239)bright++;}
            double distance=0;if(prior!=null)for(int i=0;i<16;i++)distance+=Math.abs(histogram[i]-prior[i]);distance/=2;
            // Require a sustained distribution shift, rather than a single moving frame.
            if(prior==null)prior=histogram;else if(distance>.42){changed++;candidate=histogram;if(changed>=3){sceneVersion++;prior=candidate;changed=0;}}else changed=0;
            return new JSONObject().put("opencvVersion",Core.VERSION).put("laplacianVariance",variance).put("brightness",sum/pixels.length).put("darkFraction",dark/(double)pixels.length).put("brightFraction",bright/(double)pixels.length).put("sceneDistance",distance).put("sceneVersion",sceneVersion).put("blurred",variance<45).put("underexposed",sum/pixels.length<35).put("overexposed",bright/(double)pixels.length>.4);
        }finally{rgba.release();gray.release();small.release();lap.release();mean.release();std.release();}
    }
}
