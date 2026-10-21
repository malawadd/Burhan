package com.burhan.site;

import java.io.*;
import java.nio.*;
import java.nio.file.Files;
import java.util.*;
import java.util.zip.GZIPInputStream;
import org.json.*;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.robolectric.RobolectricTestRunner;
import org.robolectric.annotation.Config;
import static org.junit.Assert.*;

/** Real CPU model output replay; numerical decoder parity, not Android JNI or accuracy qualification. */
@RunWith(RobolectricTestRunner.class) @Config(sdk=35)
public class ResearchRealImageParityTest {
    @Test public void nineSourceImageRotatedAndNegativeOutputsMatchPythonAdapter()throws Exception {
        File directory=new File(System.getProperty("user.dir"),"../../tests/fixtures/research").getCanonicalFile();
        JSONObject reference=new JSONObject(new String(Files.readAllBytes(new File(directory,"reference.json").toPath()),java.nio.charset.StandardCharsets.UTF_8));
        JSONArray specs=new JSONArray(new String(Files.readAllBytes(new File(directory,"../../../assets/models/manifest.json").getCanonicalFile().toPath()),java.nio.charset.StandardCharsets.UTF_8));
        JSONArray cases=reference.getJSONArray("cases");assertEquals(9,cases.length());
        for(int i=0;i<cases.length();i++){
            JSONObject row=cases.getJSONObject(i),spec=null;
            for(int j=0;j<specs.length();j++)if(specs.getJSONObject(j).getString("id").equals(row.getString("modelId")))spec=specs.getJSONObject(j);
            assertNotNull(spec);JSONArray classes=spec.getJSONArray("classes");float[] thresholds=new float[classes.length()];JSONObject classConfidence=spec.getJSONObject("classConfidence");
            for(int j=0;j<thresholds.length;j++)thresholds[j]=(float)classConfidence.optDouble(classes.getString(j),spec.getDouble("confidence"));
            JSONArray outputs=row.getJSONArray("outputs");List<ResearchOutputDecoder.Box> actual;
            if(spec.getString("kind").equals("ssdlite_detection")){
                Map<String,ByteBuffer> named=new HashMap<>();for(int j=0;j<outputs.length();j++){JSONObject tensor=outputs.getJSONObject(j);named.put(tensor.getString("name"),bytes(directory,tensor.getString("file")));}
                ByteBuffer b=named.get("boxes"),l=named.get("labels"),s=named.get("scores");int count=l.remaining()/8;float[][] boxes=new float[count][4];long[] labels=new long[count];float[] scores=new float[count];
                for(int j=0;j<count;j++){for(int k=0;k<4;k++)boxes[j][k]=b.getFloat();labels[j]=l.getLong();scores[j]=s.getFloat();}
                actual=ResearchOutputDecoder.decode(boxes,labels,scores,thresholds,row.getInt("width"),row.getInt("height"),spec.getInt("inputSize"),true,(float)spec.getDouble("nms"));
            }else{
                JSONObject tensor=outputs.getJSONObject(0);JSONArray dims=tensor.getJSONArray("dims");float[][][] output=new float[1][dims.getInt(1)][dims.getInt(2)];ByteBuffer b=bytes(directory,tensor.getString("file"));
                for(int j=0;j<output[0].length;j++)for(int k=0;k<output[0][j].length;k++)output[0][j][k]=b.getFloat();
                actual=ResearchOutputDecoder.yolo(output,thresholds,row.getInt("width"),row.getInt("height"),spec.getInt("inputSize"),(float)spec.getDouble("nms"));
            }
            JSONArray expected=row.getJSONArray("observations");assertEquals(row.getString("id"),expected.length(),actual.size());
            List<ResearchOutputDecoder.Box> remaining=new ArrayList<>(actual);
            for(int j=0;j<expected.length();j++){
                JSONObject target=expected.getJSONObject(j);JSONArray bbox=target.getJSONArray("bbox");ResearchOutputDecoder.Box match=null;
                for(ResearchOutputDecoder.Box candidate:remaining)if(classes.getString(candidate.category).equals(target.getString("label"))&&candidate.xyxy[0]==bbox.getInt(0)&&candidate.xyxy[1]==bbox.getInt(1)&&candidate.xyxy[2]-candidate.xyxy[0]==bbox.getInt(2)&&candidate.xyxy[3]-candidate.xyxy[1]==bbox.getInt(3)&&Math.abs(candidate.score-target.getDouble("score"))<1e-5){match=candidate;break;}
                assertNotNull("Mismatch: "+row.getString("id")+" "+target,match);remaining.remove(match);
            }
        }
    }
    private static ByteBuffer bytes(File directory,String name)throws IOException {
        try(InputStream in=new GZIPInputStream(new FileInputStream(new File(directory,name)));ByteArrayOutputStream out=new ByteArrayOutputStream()){byte[] buffer=new byte[65536];for(int n;(n=in.read(buffer))!=-1;)out.write(buffer,0,n);return ByteBuffer.wrap(out.toByteArray()).order(ByteOrder.LITTLE_ENDIAN);}
    }
}
