package com.burhan.site;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Color;
import androidx.room.Room;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.platform.app.InstrumentationRegistry;
import org.junit.Test;
import org.junit.runner.RunWith;
import java.util.*;
import static org.junit.Assert.*;

/** Run with connectedDebugAndroidTest on a physical ARM64 API31+ phone. */
@RunWith(AndroidJUnit4.class)
public class NativeStateTest {
    private Context context(){return InstrumentationRegistry.getInstrumentation().getTargetContext();}
    private String repeated(String value,int count){StringBuilder result=new StringBuilder();for(int i=0;i<count;i++)result.append(value);return result.toString();}
    @Test public void largeStateAndCaptureJournalRoundTrip() {
        BurhanDatabase db=Room.inMemoryDatabaseBuilder(context(),BurhanDatabase.class).build();
        try{
            // More than a CursorWindow, including a surrogate pair across a chunk boundary.
            String json="{\"note\":\""+repeated("x",131062)+"🏗️"+repeated("مرحباً",500000)+"\"}";
            db.store().replaceState(json);assertEquals(json,db.store().state());
            BurhanDatabase.CaptureRow row=new BurhanDatabase.CaptureRow();row.id="capture-1";row.inspectionId="inspection-1";row.createdAt=1;row.json="{\"evidence\":{\"id\":\"capture-1\"},\"observations\":[]}";db.store().capture(row);
            assertEquals(1,db.store().pending("inspection-1").size());assertTrue(db.store().pending("other").isEmpty());
            db.store().replaceState("{\"schemaVersion\":1}");assertEquals(1,db.store().pending().size());
            db.store().acknowledge(Collections.singletonList("capture-1"));assertTrue(db.store().pending().isEmpty());
        }finally{db.close();}
    }
    @Test public void bundledModelsLoadAndProduceFiniteOutputOnDevice() throws Exception {
        // Execution-contract smoke test, not a field accuracy or performance claim.
        Bitmap image=Bitmap.createBitmap(640,480,Bitmap.Config.ARGB_8888);image.eraseColor(Color.rgb(128,128,128));
        try(OnnxInspector engine=new OnnxInspector(context())){
            for(String model:new String[]{"ppe","crack","surface320","surface512"}){
                OnnxInspector.Frame result=engine.infer(image,model);assertEquals(model,result.modelId);assertTrue(result.millis>=0);
                for(OnnxInspector.Detection item:result.detections){assertTrue(Float.isFinite(item.score));assertTrue(item.box.width()>0);assertTrue(item.box.height()>0);}
            }
        }finally{image.recycle();}
    }
}
