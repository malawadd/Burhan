package com.burhan.site;

import org.junit.Test;
import java.io.IOException;
import java.util.List;
import static org.junit.Assert.*;

public class ResearchOutputDecoderTest {
    @Test public void ssdliteUsesOneBasedLongLabelsClassThresholdsAndNoSecondNms()throws Exception {
        List<ResearchOutputDecoder.Box> result=ResearchOutputDecoder.decode(new float[][]{{10,10,50,50},{10,10,50,50},{10,10,50,50},{10,10,50,50}},new long[]{1,1,2,2},new float[]{.8f,.7f,.4f,.9f},new float[]{.35f,.8f},320,320,320,true,.45f);
        assertEquals(3,result.size());assertEquals(0,result.get(0).category);assertEquals(0,result.get(1).category);assertEquals(1,result.get(2).category);
    }
    @Test public void projectionUsesRoundedResizeRatioAndOutsideExtentRounding()throws Exception {
        List<ResearchOutputDecoder.Box> result=ResearchOutputDecoder.decode(new float[][]{{10.5f,80.5f,100.2f,150.2f}},new long[]{1},new float[]{.8f},new float[]{.35f},641,319,320,true,.45f);
        assertEquals(1,result.size());assertArrayEquals(new float[]{21,1,201,141},result.get(0).xyxy,0);
    }
    @Test public void yoloSupports5376AnchorsAndClassAwareNms()throws Exception {
        float[][][] output=new float[1][6][5376];for(int anchor:new int[]{0,5375}){output[0][0][anchor]=100;output[0][1][anchor]=100;output[0][2][anchor]=40;output[0][3][anchor]=40;output[0][4][anchor]=.8f;}
        assertEquals(1,ResearchOutputDecoder.yolo(output,new float[]{.5f,.5f},512,512,512,.45f).size());
        output[0][4][5375]=.1f;output[0][5][5375]=.9f;assertEquals(2,ResearchOutputDecoder.yolo(output,new float[]{.5f,.5f},512,512,512,.45f).size());
    }
    @Test public void negativeInvalidClassesAndScoresOverOneAreNotCandidates()throws Exception {
        assertTrue(ResearchOutputDecoder.decode(new float[][]{{0,0,10,10},{0,0,10,10},{0,0,.1f,.1f}},new long[]{0,1,1},new float[]{.9f,1.1f,.9f},new float[]{.2f},320,320,320,true,.45f).isEmpty());
    }
    @Test(expected=IOException.class) public void nonFiniteOutputIsAReportedFailure()throws Exception {
        ResearchOutputDecoder.decode(new float[][]{{0,Float.NaN,10,10}},new long[]{1},new float[]{.9f},new float[]{.2f},320,320,320,true,.45f);
    }
    @Test public void ssdliteBoundsDecodedWorkTo300PerClass()throws Exception {
        float[][] boxes=new float[400][4];long[] ids=new long[400];float[] scores=new float[400];for(int i=0;i<400;i++){boxes[i]=new float[]{0,0,10,10};ids[i]=1;scores[i]=.9f;}
        assertEquals(300,ResearchOutputDecoder.decode(boxes,ids,scores,new float[]{.2f},320,320,320,true,.45f).size());
    }
}
