package com.burhan.site;

import android.content.Context;
import org.json.*;
import org.junit.*;
import org.junit.runner.RunWith;
import org.robolectric.RobolectricTestRunner;
import org.robolectric.RuntimeEnvironment;
import org.robolectric.annotation.Config;
import static org.junit.Assert.*;

@RunWith(RobolectricTestRunner.class) @Config(sdk=35)
public class ResearchCatalogueTest {
    private Context context;
    @Before public void before(){context=RuntimeEnvironment.getApplication();context.getSharedPreferences("burhan-camera-v1",0).edit().clear().commit();}
    @After public void after(){context.getSharedPreferences("burhan-camera-v1",0).edit().clear().commit();}
    @Test public void expandedTasksAreOptInAndHaveOnlyFiveAutomaticBindings()throws Exception {
        assertFalse(ResearchCatalogue.enabled(context));JSONArray tasks=ResearchCatalogue.tasks(context);assertEquals(30,tasks.length());int automatic=0;
        for(int i=0;i<tasks.length();i++){JSONObject task=tasks.getJSONObject(i);assertTrue(task.getString("id").startsWith("research:"));assertEquals(task.getString("id"),ResearchCatalogue.task(context,task.getString("id")).getString("id"));assertTrue(task.getBoolean("researchOnly"));if(task.has("modelId")){automatic++;assertEquals("research",task.getString("mode"));}else assertEquals("manual",task.getString("mode"));}
        assertEquals(5,automatic);CameraPreferences.write(context,new JSONObject().put("expandedInspectionModels",true));assertTrue(ResearchCatalogue.enabled(context));assertTrue(ResearchCatalogue.capabilities(context).getBoolean("expandedInspectionModels"));
    }
    @Test public void unknownTaskHasNoModelBindingAndFamilyTitlesAreTranslated(){assertNull(ResearchCatalogue.task(context,"research:unknown:invented"));assertEquals("الجدران والتشطيبات",ResearchCatalogue.familyTitle(context,"walls_finishes",true));}
}
