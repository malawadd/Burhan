package com.burhan.site;

import android.app.Dialog;
import android.content.Context;
import android.content.res.Configuration;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Rect;
import android.os.Bundle;
import android.os.Looper;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.TextView;
import androidx.appcompat.app.AppCompatActivity;
import com.google.android.material.button.MaterialButton;
import org.json.JSONArray;
import org.json.JSONObject;
import org.junit.After;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.robolectric.Robolectric;
import org.robolectric.RobolectricTestRunner;
import org.robolectric.RuntimeEnvironment;
import org.robolectric.android.controller.ActivityController;
import org.robolectric.annotation.Config;
import org.robolectric.annotation.GraphicsMode;
import org.robolectric.shadows.ShadowDialog;
import java.io.File;
import java.io.FileOutputStream;
import java.util.ArrayList;
import java.util.List;
import static org.junit.Assert.*;
import static org.robolectric.Shadows.shadowOf;

/** Actual Android/Material Views, without starting CameraX, a model, or a network request. */
@RunWith(RobolectricTestRunner.class)
@Config(sdk=35)
@GraphicsMode(GraphicsMode.Mode.NATIVE)
public class InspectionCameraUiTest {
    private ActivityController<FixtureActivity> controller;
    private InspectionCameraUi ui;
    private final Recorder events=new Recorder();
    private static float requestedFontScale=1;

    public static class FixtureActivity extends AppCompatActivity {
        @Override protected void attachBaseContext(Context base){
            Configuration configuration=new Configuration(base.getResources().getConfiguration());
            configuration.fontScale=requestedFontScale;
            super.attachBaseContext(base.createConfigurationContext(configuration));
        }
        @Override public void onCreate(Bundle state){
            setTheme(R.style.AppTheme_InspectionCamera);
            super.onCreate(state);
        }
    }

    @After public void tearDown(){
        if(ui!=null)ui.dispose();
        if(controller!=null)controller.pause().stop().destroy();
        requestedFontScale=1;
    }

    @Test public void englishCompact320x640KeepsCameraAndCaptureActionsVisible() throws Exception {
        fixture(320,640,1,false);
        screenshot("inspection-en-320x640-font100");
        checkMainControls(320,640);
    }

    @Test public void english360x800RendersActualMaterialCameraHierarchy() throws Exception {
        fixture(360,800,1,false);
        screenshot("inspection-en-360x800-font100");
        checkMainControls(360,800);
        assertTrue(ui.captureButton instanceof MaterialButton);
        assertTrue(ui.captureButton.getText().toString().contains("Capture"));
    }

    @Test public void arabic360x800LargeTextKeepsEssentialControlsReachable() throws Exception {
        fixture(360,800,1.6f,true);
        screenshot("inspection-ar-360x800-font160");
        checkMainControls(360,800);
        assertEquals(View.LAYOUT_DIRECTION_RTL,ui.root.getLayoutDirection());
        assertTrue(ui.captureButton.getText().toString().contains("التقاط"));
    }

    @Test public void englishCompactAtDoubleFontScalePreservesPersistentActions() throws Exception {
        fixture(320,640,2,false);
        screenshot("inspection-en-320x640-font200");
        checkMainControls(320,640);
    }

    @Test public void landscape640x360UsesCameraBesideControls() throws Exception {
        fixture(640,360,1,false);
        screenshot("inspection-en-640x360-font100");
        checkMainControls(640,360);
        Rect camera=bounds(ui.cameraViewport),capture=bounds(ui.captureButton);
        assertTrue("Landscape should put the camera beside the action panel",camera.right<=capture.left);
    }

    @Test public void arabicLandscapeAtDoubleFontScalePreservesPersistentActions() throws Exception {
        fixture(640,360,2,true);
        screenshot("inspection-ar-640x360-font200");
        checkMainControls(640,360);
    }

    @Test public void compactDoubleFontWarningKeepsCameraAndPersistentActionsVisible() throws Exception {
        fixture(320,640,2,false);
        ui.setStatus("Storage low. Export a backup and free space before saving more evidence.",InspectionCameraUi.WARNING);
        layout(320,640);
        screenshot("inspection-en-320x640-font200-warning");
        checkMainControls(320,640);
        assertEquals(View.VISIBLE,ui.statusBanner.getVisibility());
        assertTrue(ui.statusBanner.getText().toString().contains("Storage low"));
        assertEquals(0,events.captures);
    }

    @Test public void readinessCountAndIndependentGuideCameraStateDoNotTriggerCaptures() throws Exception {
        fixture(320,640,2,false);
        ui.setReady(false);
        assertFalse(ui.captureButton.isEnabled());
        ui.setReady(true);
        assertTrue(ui.captureButton.isEnabled());
        ui.captureButton.performClick();
        assertEquals(1,events.captures);
        ui.setSavedCount(3);
        assertTrue(ui.reviewButton.getText().toString().contains("3"));
        assertTrue(ui.reviewButton.getContentDescription().toString().contains("3 saved evidence"));
        ui.setGuideState("paused");
        assertEquals("Guide paused",ui.guideButton.getText().toString());
        assertEquals("Pause camera",ui.pauseButton.getText().toString());
        ui.setCameraPaused(true);
        assertEquals("Resume camera",ui.pauseButton.getText().toString());
        assertEquals("Guide paused",ui.guideButton.getText().toString());
        layout(320,640);
        screenshot("inspection-en-320x640-font200-camera-paused");
        checkMainControls(320,640);
        ui.setGuideState("unavailable");
        assertEquals("Guide offline",ui.guideButton.getText().toString());
        assertTrue("Losing remote guidance must not disable local capture",ui.captureButton.isEnabled());
        ui.pauseButton.performClick();ui.importButton.performClick();ui.reviewButton.performClick();
        assertEquals(1,events.cameraPauses);assertEquals(1,events.imports);assertEquals(1,events.reviews);
        assertEquals("Changing states and secondary actions must not capture evidence",1,events.captures);
    }

    @Test public void manualInspectionKeepsCaptureAndSurfaceOptionsWithoutGuideControls() throws Exception {
        fixture(360,800,1,false,false);
        assertEquals(View.GONE,ui.guideButton.getVisibility());
        ui.setTask("Surface condition","surface");ui.setSurfaceMode(true);
        ui.optionsButton.performClick();shadowOf(Looper.getMainLooper()).idle();
        Dialog details=ShadowDialog.getLatestDialog();
        assertNotNull(details);assertTrue(details.isShowing());
        assertEquals(View.VISIBLE,ui.detailed.getVisibility());
        assertNotNull(ui.detailed.getParent());
        assertNull(findButton(details.getWindow().getDecorView(),"Pause guide"));
        assertEquals(0,events.captures);
        ui.dismissSheets();
        ui.captureButton.performClick();assertEquals(1,events.captures);
    }

    @Test public void unpairedGuideStillOffersInstalledChecksWithoutRemoteSuggestions() throws Exception {
        fixture(360,800,1,false);
        ui.setGuideState("unpaired");ui.setSuggestions(null);
        assertTrue("Unpaired guidance must preserve local capture",ui.captureButton.isEnabled());
        ui.optionsButton.performClick();shadowOf(Looper.getMainLooper()).idle();
        Dialog details=ShadowDialog.getLatestDialog();
        assertNotNull(details);assertTrue(details.isShowing());
        assertNotNull(findText(details.getWindow().getDecorView(),"Available checks"));
        Button crack=findButton(details.getWindow().getDecorView(),"Cracks / joints");
        assertNotNull("Installed detector choices must not depend on remote suggestions",crack);
        crack.performClick();
        assertEquals("crack:visual-cracks",events.selectedTask);
        assertEquals(0,events.captures);
        assertTrue(ui.captureButton.isEnabled());
    }

    @Test public void clearingSceneSuggestionsRemovesOldChoicesFromAnAlreadyOpenSheet() throws Exception {
        fixture(360,800,1,false);
        String oldTitle="Inspect the old outdoor opening";
        ui.setSuggestions(new JSONArray().put(new JSONObject().put("mode","crack").put("taskId","visual-cracks").put("title",oldTitle)));
        ui.showDetails();shadowOf(Looper.getMainLooper()).idle();
        Dialog details=ShadowDialog.getLatestDialog();
        assertNotNull(findButton(details.getWindow().getDecorView(),oldTitle));
        ui.setSuggestions(null);shadowOf(Looper.getMainLooper()).idle();
        assertTrue("Scene updates should preserve an open details sheet",details.isShowing());
        assertNull("Obsolete scene suggestions must disappear immediately",findButton(details.getWindow().getDecorView(),oldTitle));
        assertNotNull("Installed choices remain available",findButton(details.getWindow().getDecorView(),"Cracks / joints"));
        assertEquals(0,events.captures);
    }

    @Test public void fullGuidanceAlternativesAndGuideControlsAreAvailableInSheetWithoutCapturing() throws Exception {
        fixture(360,800,1,false);
        String full="Capture a closer view.\nHold still and focus on the vertical corner. Then photograph both sides separately so the reviewer can distinguish a joint from a surface crack.";
        String reason="The visible line may be a construction joint. An image alone does not confirm a defect.";
        ui.setGuidance(full,reason);
        ui.setSuggestions(new JSONArray().put(new JSONObject().put("mode","crack").put("taskId","visual-cracks").put("title","Check cracks and joints").put("reason","Use the installed detector, then review observations.")));
        ui.guideButton.performClick();
        shadowOf(Looper.getMainLooper()).idle();
        Dialog details=ShadowDialog.getLatestDialog();
        assertNotNull(details);assertTrue(details.isShowing());
        assertNotNull("Full instructions must remain available outside the short main card",findText(details.getWindow().getDecorView(),full));
        assertNotNull(findText(details.getWindow().getDecorView(),reason));
        Button alternative=findButton(details.getWindow().getDecorView(),"Check cracks and joints");
        assertNotNull(alternative);alternative.performClick();
        assertEquals("crack:visual-cracks",events.selectedTask);
        assertEquals(0,events.captures);
        ui.showDetails();shadowOf(Looper.getMainLooper()).idle();
        details=ShadowDialog.getLatestDialog();
        Button pause=findButton(details.getWindow().getDecorView(),"Pause guide");
        assertNotNull(pause);pause.performClick();
        assertEquals(1,events.guidePauses);
        assertEquals(0,events.cameraPauses);
        assertEquals(0,events.captures);
    }

    @Test public void subtleHintSelectsWithoutCapturingAndHidesForWarningsAndDetails() throws Exception {
        fixture(360,800,1,false);
        ui.setTaskLocked(false);ui.setSelectedTask("");ui.setTask("Explore this area","manual");
        ui.setSuggestions(new JSONArray().put(new JSONObject().put("mode","crack").put("taskId","visual-cracks").put("title","Wall joints")));
        layout(360,800);
        assertEquals(View.VISIBLE,ui.suggestionHint.getVisibility());
        assertTrue(ui.suggestionHint.getText().toString().startsWith("✦ Try:"));screenshot("inspection-en-360x800-subtle-hint");
        ui.suggestionHint.performClick();assertEquals("crack:visual-cracks",events.selectedTask);assertEquals(0,events.captures);
        Rect hint=new Rect(),shutter=new Rect();ui.suggestionHint.getGlobalVisibleRect(hint);ui.pauseButton.getGlobalVisibleRect(shutter);assertTrue(hint.bottom<=shutter.top);
        ui.setStatus("Storage low",InspectionCameraUi.WARNING);assertEquals(View.GONE,ui.suggestionHint.getVisibility());
        ui.setStatus("",InspectionCameraUi.INFO);assertEquals(View.VISIBLE,ui.suggestionHint.getVisibility());
        ui.showDetails();assertEquals(View.GONE,ui.suggestionHint.getVisibility());ui.dismissSheets();assertEquals(View.VISIBLE,ui.suggestionHint.getVisibility());
        ui.setHintsEnabled(false);assertEquals(View.GONE,ui.suggestionHint.getVisibility());
    }

    @Test public void hintHorizontalSwipeDismissesWithoutSelectingOrCapturing() throws Exception {
        fixture(360,800,1,false);
        ui.setSuggestions(new JSONArray().put(new JSONObject().put("mode","crack").put("taskId","visual-cracks").put("title","Wall joints")));
        layout(360,800);
        long now=android.os.SystemClock.uptimeMillis();
        ui.suggestionHint.dispatchTouchEvent(android.view.MotionEvent.obtain(now,now,android.view.MotionEvent.ACTION_DOWN,140,20,0));
        ui.suggestionHint.dispatchTouchEvent(android.view.MotionEvent.obtain(now,now+150,android.view.MotionEvent.ACTION_UP,20,20,0));
        assertEquals(View.GONE,ui.suggestionHint.getVisibility());assertNull(events.selectedTask);assertEquals(0,events.captures);
    }

    private void fixture(int width,int height,float fontScale,boolean arabic){
        fixture(width,height,fontScale,arabic,true);
    }

    private void fixture(int width,int height,float fontScale,boolean arabic,boolean smart){
        requestedFontScale=fontScale;
        RuntimeEnvironment.setQualifiers("w"+width+"dp-h"+height+"dp-mdpi");
        controller=Robolectric.buildActivity(FixtureActivity.class).setup();
        FixtureActivity activity=controller.get();
        ui=new InspectionCameraUi(activity,arabic,smart,events);
        activity.setContentView(ui.root);
        ui.setTask(arabic?"زاوية الجدار / الفاصل":"Wall corner / joint","manual");
        ui.setGuidance(arabic?"التقط صورة أقرب للزاوية.\nثبّت الهاتف واضغط على الزاوية للتركيز.":"Capture a closer view.\nHold still and tap the corner to focus.",arabic?"قد يكون الخط فاصلاً بين أجزاء الجدار.":"The line may be a joint between wall surfaces.");
        ui.setQuality(arabic?"ثبّت الهاتف للتركيز":"Hold still to focus");
        ui.setProvider("codex");ui.setGuideState("ready");ui.setReady(true);
        layout(width,height);
    }

    private void layout(int width,int height){
        int widthSpec=View.MeasureSpec.makeMeasureSpec(width,View.MeasureSpec.EXACTLY);
        int heightSpec=View.MeasureSpec.makeMeasureSpec(height,View.MeasureSpec.EXACTLY);
        // onSizeChanged posts orientation reflow; measure once more after processing it.
        ui.root.measure(widthSpec,heightSpec);ui.root.layout(0,0,width,height);
        shadowOf(Looper.getMainLooper()).idle();
        ui.root.measure(widthSpec,heightSpec);ui.root.layout(0,0,width,height);
    }

    private void checkMainControls(int width,int height){
        List<View> actions=new ArrayList<>();
        actions.add(ui.importButton);actions.add(ui.captureButton);actions.add(ui.reviewButton);
        for(View action:actions){
            assertFullyVisible(action,width,height);
            assertTrue("Touch target too narrow: "+((TextView)action).getText(),action.getWidth()>=48);
            assertTrue("Touch target too short: "+((TextView)action).getText(),action.getHeight()>=48);
            TextView text=(TextView)action;
            assertNotNull(text.getLayout());
            assertTrue("Action label is vertically clipped: "+text.getText(),text.getLayout().getHeight()+text.getCompoundPaddingTop()+text.getCompoundPaddingBottom()<=text.getHeight());
            for(int line=0;line<text.getLayout().getLineCount();line++)assertEquals("Essential action label must not be ellipsized: "+text.getText(),0,text.getLayout().getEllipsisCount(line));
        }
        for(int i=0;i<actions.size();i++)for(int j=i+1;j<actions.size();j++){
            assertFalse("Persistent actions overlap",Rect.intersects(bounds(actions.get(i)),bounds(actions.get(j))));
        }
        assertFullyVisible(ui.cameraViewport,width,height);
        assertTrue("Camera must retain useful height",ui.cameraViewport.getHeight()>=96);
        assertTrue("Camera must retain useful width",ui.cameraViewport.getWidth()>=160);
        assertFalse("Capture must stay outside the camera viewport",Rect.intersects(bounds(ui.cameraViewport),bounds(ui.captureButton)));
        assertFullyVisible(ui.optionsButton,width,height);
        assertButtonWordsStayWhole(ui.root);
    }

    private void assertButtonWordsStayWhole(View view){
        if(view.getVisibility()!=View.VISIBLE)return;
        if(view instanceof Button){
            Rect visible=bounds(view);
            if(!visible.intersect(0,0,ui.root.getWidth(),ui.root.getHeight()))return;
            View ancestor=(View)view.getParent();
            while(ancestor!=ui.root){
                if(!visible.intersect(bounds(ancestor)))return;
                ancestor=(View)ancestor.getParent();
            }
            TextView text=(TextView)view;
            if(text.getLayout()!=null){
                String value=text.getText().toString();
                for(int line=0;line<text.getLayout().getLineCount()-1;line++){
                    int end=text.getLayout().getLineEnd(line);
                    if(end>0&&end<value.length())assertFalse("A button label strands part of a word on the next line: "+value,
                        Character.isLetter(value.charAt(end-1))&&Character.isLetter(value.charAt(end)));
                }
            }
        }
        if(view instanceof ViewGroup){ViewGroup group=(ViewGroup)view;for(int i=0;i<group.getChildCount();i++)assertButtonWordsStayWhole(group.getChildAt(i));}
    }

    @Test public void expandedChecksAreOptInAndExperimentalStatusFitsCompactCamera()throws Exception {
        fixture(360,800,1,false);
        CameraPreferences.write(controller.get(),new JSONObject().put("expandedInspectionModels",false));
        ui.showChecks();shadowOf(Looper.getMainLooper()).idle();
        assertNull(findText(ShadowDialog.getLatestDialog().getWindow().getDecorView(),"Peeling paint"));ui.dismissSheets();
        try{
            CameraPreferences.write(controller.get(),new JSONObject().put("expandedInspectionModels",true));
            ui.showChecks();shadowOf(Looper.getMainLooper()).idle();
            assertNotNull(findText(ShadowDialog.getLatestDialog().getWindow().getDecorView(),"Peeling paint"));
            assertNotNull(findText(ShadowDialog.getLatestDialog().getWindow().getDecorView(),"Walls and finishes"));
            ui.dismissSheets();ui.setSelectedTask("research:walls_finishes:peeling_paint");ui.setTaskLocked(true);ui.setTask("Peeling paint","research");
            assertTrue(ui.taskMode.getText().toString().contains("Experimental"));screenshot("inspection-en-360x800-expanded-research");checkMainControls(360,800);
        }finally{CameraPreferences.write(controller.get(),new JSONObject().put("expandedInspectionModels",false));}
    }

    private void assertFullyVisible(View view,int width,int height){
        Rect rectangle=bounds(view);
        assertTrue("View has empty bounds: "+view,rectangle.width()>0&&rectangle.height()>0);
        assertTrue("View escapes screen: "+rectangle,new Rect(0,0,width,height).contains(rectangle));
        View ancestor=(View)view.getParent();
        while(ancestor!=ui.root){
            assertTrue("View is clipped by ancestor: "+rectangle,bounds(ancestor).contains(rectangle));
            ancestor=(View)ancestor.getParent();
        }
    }

    private Rect bounds(View view){
        Rect result=new Rect(0,0,view.getWidth(),view.getHeight());
        ui.root.offsetDescendantRectToMyCoords(view,result);
        return result;
    }

    private TextView findText(View view,String expected){
        if(view instanceof TextView&&expected.contentEquals(((TextView)view).getText()))return (TextView)view;
        if(view instanceof ViewGroup){ViewGroup parent=(ViewGroup)view;for(int i=0;i<parent.getChildCount();i++){TextView result=findText(parent.getChildAt(i),expected);if(result!=null)return result;}}
        return null;
    }

    private Button findButton(View view,String expected){TextView result=findText(view,expected);return result instanceof Button?(Button)result:null;}

    private void screenshot(String name) throws Exception {
        File directory=new File(System.getProperty("user.dir"),"../../../output/native-ui").getCanonicalFile();
        assertTrue(directory.isDirectory()||directory.mkdirs());
        Bitmap bitmap=Bitmap.createBitmap(ui.root.getWidth(),ui.root.getHeight(),Bitmap.Config.ARGB_8888);
        ui.root.draw(new Canvas(bitmap));
        try(FileOutputStream stream=new FileOutputStream(new File(directory,name+".png"))){assertTrue(bitmap.compress(Bitmap.CompressFormat.PNG,100,stream));}
        bitmap.recycle();
    }

    private static final class Recorder implements InspectionCameraUi.Listener {
        int captures,imports,reviews,cameraPauses,guidePauses;String selectedTask;
        public void onBack(){} public void onCapture(){captures++;} public void onImport(){imports++;} public void onReview(){reviews++;}
        public void onToggleCameraPause(){cameraPauses++;} public void onToggleTorch(){} public void onToggleZoom(){}
        public void onToggleGuide(){guidePauses++;} public void onToggleTaskLock(){} public void onCompleteTask(){} public void onSkipTask(){}
        public void onCorrectContext(String correction){} public void onSelectTask(String mode,String taskId){selectedTask=mode+":"+taskId;}
    }
}
