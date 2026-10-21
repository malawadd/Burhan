package com.burhan.site;

import android.content.Context;
import org.json.JSONObject;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.robolectric.RobolectricTestRunner;
import org.robolectric.RuntimeEnvironment;
import org.robolectric.annotation.Config;
import static org.junit.Assert.*;

@RunWith(RobolectricTestRunner.class)
@Config(sdk=35)
public class CameraPreferencesTest {
    private Context context;
    @Before public void setup(){context=RuntimeEnvironment.getApplication();context.getSharedPreferences("burhan-camera-v1",Context.MODE_PRIVATE).edit().clear().commit();}
    @Test public void sharedDefinitionsSupplyDefaultsAndPersistIndependently() throws Exception {
        assertEquals(16,CameraPreferences.definitions(context).getJSONArray("settings").length());
        JSONObject initial=CameraPreferences.read(context);assertTrue(initial.getBoolean("aiSuggestionHints"));assertFalse(initial.getBoolean("photoMarkup"));
        CameraPreferences.write(context,new JSONObject().put("photoMarkup",true).put("aiSuggestionHints",false).put("lastZoom",2.5));
        JSONObject restored=CameraPreferences.read(context);assertTrue(restored.getBoolean("photoMarkup"));assertFalse(restored.getBoolean("aiSuggestionHints"));assertTrue(restored.getBoolean("undoLastCapture"));assertEquals(2.5,restored.getDouble("lastZoom"),.001);
    }
    @Test public void malformedWritesCannotPartiallyChangePreferences() throws Exception {
        for(JSONObject bad:new JSONObject[]{new JSONObject().put("photoMarkup",true).put("unknown",true),new JSONObject().put("photoMarkup","yes"),new JSONObject().put("lastZoom",0)}){
            try{CameraPreferences.write(context,bad);fail("Malformed preferences accepted");}catch(IllegalArgumentException expected){}
            assertFalse(CameraPreferences.enabled(context,"photoMarkup"));
        }
    }
    @Test public void everySwitchRoundTripsBothValuesWithoutChangingOtherSwitches() throws Exception {
        org.json.JSONArray definitions=CameraPreferences.definitions(context).getJSONArray("settings");
        for(int i=0;i<definitions.length();i++){
            String key=definitions.getJSONObject(i).getString("key");JSONObject before=CameraPreferences.read(context);
            for(boolean enabled:new boolean[]{true,false}){
                CameraPreferences.write(context,new JSONObject().put(key,enabled));assertEquals(enabled,CameraPreferences.enabled(context,key));JSONObject after=CameraPreferences.read(context);
                for(int j=0;j<definitions.length();j++){String other=definitions.getJSONObject(j).getString("key");if(!other.equals(key))assertEquals(before.getBoolean(other),after.getBoolean(other));}
            }
        }
    }
}
