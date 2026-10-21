package com.burhan.site;

import android.Manifest;
import android.content.*;
import android.content.pm.PackageManager;
import android.graphics.*;
import android.net.Uri;
import android.os.*;
import android.view.*;
import android.widget.*;
import androidx.activity.OnBackPressedCallback;
import androidx.activity.result.*;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.*;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.content.ContextCompat;
import androidx.core.view.*;
import com.google.common.util.concurrent.ListenableFuture;
import org.json.*;
import java.io.*;
import java.nio.ByteBuffer;
import java.time.Instant;
import java.util.*;
import java.util.concurrent.*;

/** Camera and inference stay native; only completed, durable evidence crosses the bridge. */
public class ScanActivity extends AppCompatActivity {
    private final ExecutorService worker=Executors.newSingleThreadExecutor();
    private final Object frameLock=new Object();
    private final Handler handler=new Handler(Looper.getMainLooper());
    private final Runnable staleCheck=new Runnable(){@Override public void run(){if(active&&!paused&&lastCompleted>0&&SystemClock.elapsedRealtime()-lastCompleted>3000&&!staleNotice){staleNotice=true;overlay.update(null);ui.setReady(false);showStatus(text("Waiting for a fresh analysed frame…","بانتظار إطار حديث بعد التحليل…"));}if(!isDestroyed())handler.postDelayed(this,1000);}};
    private volatile boolean paused=false,active=false,finishing=false,ready=false,captureRequested=false;
    private boolean staleNotice;
    private final CheckSelection checks=new CheckSelection();
    private final java.util.concurrent.atomic.AtomicBoolean captureBusy=new java.util.concurrent.atomic.AtomicBoolean();
    private final java.util.concurrent.atomic.AtomicBoolean saving=new java.util.concurrent.atomic.AtomicBoolean();
    private final LinkedHashMap<String,BurhanDatabase.CaptureRow> sessionCaptures=new LinkedHashMap<>();
    private JSONObject cameraPreferences=new JSONObject();
    private ImageCapture imageCapture;
    private String captureSource="analysis",retakeId;
    private long lastShutter;
    private boolean focusLocked,reviewing,settingsOpen,pausedBeforeSettings;
    private String proposalTurn,hintDismissedTurn;
    private int captureVersion;
    private boolean gestureScaled;
    private JSONArray proposalChoices=new JSONArray();
    private CaptureReview review;
    private CameraAids aids;
    private boolean arabic,autoSave,torch,followUp,binding;
    private String inspectionId,findingId;
    private volatile String mode;
    private boolean smartEnabled;
    private volatile boolean smartPaused,taskLocked;
    private volatile String activeTaskId="",correction="",smartGuidance="",smartReason="",smartScene="unknown";
    private SmartInspectionClient smartClient;
    private volatile String missingGuideState="unpaired";
    private SmartQuality smartQuality;
    private JSONObject quality=new JSONObject();
    private int sceneVersion=1;
    private int acceptedPlanVersion=-1;
    private volatile boolean importedFrame;
    private boolean importedPlanningScheduled;
    private long lastQuality,lastRemote;
    private InspectionCameraUi ui;
    private final Map<String,String> taskTitles=new ConcurrentHashMap<>();
    private final Set<String> completedTasks=Collections.synchronizedSet(new LinkedHashSet<>());
    private final Set<String> appliedActions=new LinkedHashSet<>();
    private final JSONArray smartSteps=new JSONArray();
    private OnnxInspector inspector;
    private OnnxInspector.Frame latest;
    private PreviewView preview;
    private Overlay overlay;
    private Button pauseButton,torchButton,zoomButton;
    private CheckBox detailed;
    private ProcessCameraProvider cameraProvider;
    private androidx.camera.core.Camera camera;
    private ImageAnalysis analysis;
    private PowerManager power;
    private PowerManager.OnThermalStatusChangedListener thermalListener;
    private int saved;
    private long lastSave,lastAnalysis,errorAt,lastCompleted;
    private final List<Track> tracks=new ArrayList<>();
    private static class Track {String label;RectF box;int hits;long started,seen,savedAt;}
    private final ActivityResultLauncher<String> permission=registerForActivityResult(new ActivityResultContracts.RequestPermission(),granted->{if(granted)bindCamera();else showStatus(text("Camera permission denied. Allow it in Settings or import a photo.","إذن الكاميرا مرفوض. اسمح به من الإعدادات أو استورد صورة."));});
    private final ActivityResultLauncher<String[]> photo=registerForActivityResult(new ActivityResultContracts.OpenDocument(),uri->{if(uri!=null)importImage(uri);});
    private String text(String english,String ar){return arabic?ar:english;}
    @Override public void onCreate(Bundle bundle){
        super.onCreate(bundle);inspectionId=getIntent().getStringExtra("inspectionId");mode=getIntent().getStringExtra("mode");arabic="ar".equals(getIntent().getStringExtra("language"));autoSave=getIntent().getBooleanExtra("autoSave",true);followUp=getIntent().getBooleanExtra("followUp",false);findingId=getIntent().getStringExtra("findingId");
        if(inspectionId==null||mode==null){finish();return;}
        cameraPreferences=CameraPreferences.read(this);smartEnabled=getIntent().getBooleanExtra("smart",false);activeTaskId=getIntent().getStringExtra("taskId");if(activeTaskId==null||smartEnabled)activeTaskId="";if(smartEnabled)mode="manual";taskLocked=!activeTaskId.isEmpty()||!mode.equals("manual");
        WindowCompat.setDecorFitsSystemWindows(getWindow(),false);
        WindowInsetsControllerCompat systemBars=new WindowInsetsControllerCompat(getWindow(),getWindow().getDecorView());systemBars.setAppearanceLightStatusBars(false);systemBars.setAppearanceLightNavigationBars(false);
        buildInterface();getOnBackPressedDispatcher().addCallback(this,new OnBackPressedCallback(true){@Override public void handleOnBackPressed(){finishScan();}});
        power=(PowerManager)getSystemService(POWER_SERVICE);thermalListener=level->{if(level>=PowerManager.THERMAL_STATUS_SEVERE){paused=true;overlay.freeze(true);ui.setCameraPaused(true);ui.setReady(false);applyPreferences();showStatus(text("Phone is hot. Analysis paused; let it cool before resuming.","الهاتف ساخن. أُوقف التحليل مؤقتاً؛ انتظر حتى يبرد قبل الاستئناف."));}};power.addThermalStatusListener(ContextCompat.getMainExecutor(this),thermalListener);
        worker.execute(()->{try{loadTaskTitles();if(smartEnabled)initSmart();if(!mode.equals("manual")){inspector=new OnnxInspector(this);inspector.load(modelId(false));}ready=true;showInfo(mode.equals("manual")?text("Camera evidence ready · manual inspection","جاهز لتوثيق الأدلة · فحص يدوي"):text("CPU model ready · experimental observations","النموذج جاهز · ملاحظات تجريبية"));}catch(Exception e){showStatus(text("Model unavailable: ","النموذج غير متاح: ")+e.getMessage());}});
        if(ContextCompat.checkSelfPermission(this,Manifest.permission.CAMERA)==PackageManager.PERMISSION_GRANTED)preview.post(this::bindCamera);else permission.launch(Manifest.permission.CAMERA);
        handler.postDelayed(staleCheck,1000);
    }
    private void buildInterface(){
        ui=new InspectionCameraUi(this,arabic,smartEnabled,new InspectionCameraUi.Listener(){
            public void onBack(){finishScan();}
            public void onCapture(){requestCapture();}
            public void onImport(){if(captureBusy.get()||saving.get()){showInfo(text("Wait for this capture to save.","انتظر حفظ الالتقاط."));return;}photo.launch(new String[]{"image/*"});}
            public void onReview(){openReview();}
            public void onToggleCameraPause(){togglePause();}
            public void onToggleTorch(){toggleTorch();}
            public void onToggleZoom(){toggleZoom();}
            public void onToggleGuide(){toggleGuide();}
            public void onToggleTaskLock(){stopCheck("stopped");}
            public void onDismissSuggestion(String id){worker.execute(()->{checks.dismiss(id,SystemClock.elapsedRealtime());hintDismissedTurn=proposalTurn;event("dismissed",id);refreshProposals();});}
            public void onSuggestAgain(){worker.execute(()->{checks.suggestAgain();hintDismissedTurn=null;advanceContext();lastRemote=0;});}
            public void onCameraSettings(){pausedBeforeSettings=paused;settingsOpen=true;paused=true;ui.dismissSheets();ui.setSuggestions(null);if(smartClient!=null)smartClient.pause(true);startActivity(new Intent(ScanActivity.this,CameraSettingsActivity.class).putExtra("language",arabic?"ar":"en"));}
            public void onCompleteTask(){finishTask(true);}
            public void onSkipTask(){finishTask(false);}
            public void onCorrectContext(String value){applyCorrection(value);}
            public void onSelectTask(String next,String task){chooseTask(next,task);}
        });
        preview=ui.preview;overlay=new Overlay(this,arabic);ui.cameraLayer.addView(overlay,new FrameLayout.LayoutParams(-1,-1));
        preview.setContentDescription(text("Live inspection camera. Tap an area to request focus.","كاميرا الفحص المباشر. اضغط على منطقة لطلب التركيز."));
        ScaleGestureDetector scale=new ScaleGestureDetector(this,new ScaleGestureDetector.SimpleOnScaleGestureListener(){@Override public boolean onScale(ScaleGestureDetector detector){if(pref("pinchToZoom")&&camera!=null&&!paused){ZoomState zoom=camera.getCameraInfo().getZoomState().getValue();if(zoom!=null)setZoom(zoom.getZoomRatio()*detector.getScaleFactor());}return true;}});
        GestureDetector gestures=new GestureDetector(this,new GestureDetector.SimpleOnGestureListener(){@Override public boolean onDown(MotionEvent e){return true;}@Override public boolean onSingleTapUp(MotionEvent e){if(!scale.isInProgress())focus(e.getX(),e.getY(),false);return true;}@Override public void onLongPress(MotionEvent e){if(pref("focusLock")&&!scale.isInProgress())focus(e.getX(),e.getY(),true);}});
        preview.setOnTouchListener((view,e)->{if(e.getActionMasked()==MotionEvent.ACTION_DOWN)gestureScaled=false;if(e.getPointerCount()>1){gestureScaled=true;MotionEvent cancelled=MotionEvent.obtain(e);cancelled.setAction(MotionEvent.ACTION_CANCEL);gestures.onTouchEvent(cancelled);cancelled.recycle();}if(pref("pinchToZoom"))scale.onTouchEvent(e);if(!gestureScaled&&e.getPointerCount()==1&&!scale.isInProgress())gestures.onTouchEvent(e);return true;});
        aids=new CameraAids(this,arabic);ui.cameraLayer.addView(aids,new FrameLayout.LayoutParams(-1,-1));
        pauseButton=ui.pauseButton;torchButton=ui.torchButton;zoomButton=ui.zoomButton;detailed=ui.detailed;
        ui.setSelectedTask(activeTaskId);ui.setTaskLocked(taskLocked);ui.setTask(currentTaskTitle(),mode);ui.setSurfaceMode(mode.equals("surface"));ui.setReady(false);
        setContentView(ui.root);
    }
    private int dp(int value){return Math.round(value*getResources().getDisplayMetrics().density);}
    private void showStatus(String message){runOnUiThread(()->{if(!isDestroyed())ui.setStatus(message,InspectionCameraUi.WARNING);});}
    private void showInfo(String message){runOnUiThread(()->{if(!isDestroyed())ui.setStatus(message,InspectionCameraUi.INFO);});}
    private void toggleGuide(){
        smartPaused=!smartPaused;if(smartClient!=null)smartClient.pause(smartPaused);cloudStatus(smartPaused?"paused":smartClient==null?missingGuideState:"ready");
        if(!smartPaused&&!worker.isShutdown())worker.execute(this::scheduleImportedPlanning);
    }
    private void loadTaskTitles(){
        try{
            JSONObject catalogue=readAsset("public/catalogue.json"),translated=arabic?readAsset("public/catalogue-ar.json").optJSONObject("tasks"):null;
            JSONArray tasks=catalogue.getJSONArray("tasks");for(int i=0;i<tasks.length();i++){JSONObject task=tasks.getJSONObject(i);String id=task.getString("id");taskTitles.put(id,translated!=null?translated.optString(id,task.getString("title")):task.getString("title"));}
        }catch(Exception ignored){/* Mode labels remain useful if the optional task title cannot be read. */}
        runOnUiThread(()->ui.setTask(currentTaskTitle(),mode));
    }
    private JSONObject readAsset(String path) throws Exception {
        try(InputStream input=getAssets().open(path);ByteArrayOutputStream output=new ByteArrayOutputStream()){byte[] buffer=new byte[8192];int n;while((n=input.read(buffer))!=-1)output.write(buffer,0,n);return new JSONObject(output.toString("UTF-8"));}
    }
    private String currentTaskTitle(){
        if(!activeTaskId.isEmpty()&&taskTitles.containsKey(activeTaskId))return taskTitles.get(activeTaskId);
        if(smartEnabled&&activeTaskId.isEmpty())return text("Explore this area","استكشف المنطقة");
        switch(mode){case "crack":return text("Visible cracks / joints","الشقوق والفواصل المرئية");case "surface":return text("Surface condition","حالة السطح");case "ppe":return text("PPE observations","ملاحظات معدات الوقاية");default:return smartEnabled?text("Explore this area","استكشف المنطقة"):text("Manual inspection","فحص يدوي");}
    }
    private void initSmart(){
        try{
            smartQuality=new SmartQuality();JSONObject settings=SmartSettings.get(this);String url=getIntent().getStringExtra("backendUrl"),token=getIntent().getStringExtra("backendToken");if(url==null||url.isEmpty())url=settings.getString("url");if(token==null||token.isEmpty())token=settings.getString("token");
            if(token.isEmpty()){cloudStatus("unpaired");return;}
            JSONObject initial=new JSONObject().put("inspectionId",inspectionId).put("language",arabic?"ar":"en").put("mode",mode).put("stage",getIntent().getStringExtra("stage")).put("taskId",getIntent().getStringExtra("taskId")).put("location",getIntent().getStringExtra("location"));
            smartClient=new SmartInspectionClient(this,url,token,getIntent().getStringExtra("smartSessionId"),initial,new SmartInspectionClient.Listener(){
                public void status(String value){cloudStatus(value);}
                public void plan(String turnId,int version,String provider,JSONObject result){if(!finishing&&!worker.isShutdown())worker.execute(()->applySmartPlan(turnId,version,provider,result));}
            });cloudStatus("ready");
        }catch(Exception e){cloudStatus("unavailable");}
    }
    private void cloudStatus(String state){if(smartClient==null&&(state.equals("unpaired")||state.equals("unavailable")))missingGuideState=state;runOnUiThread(()->{if(ui!=null&&!isDestroyed())ui.setGuideState(state);});}
    private void smartFrame(OnnxInspector.Frame frame,long now){
        try{
            if(smartQuality!=null&&now-lastQuality>=1000){lastQuality=now;quality=smartQuality.measure(frame.image);int measured=quality.optInt("sceneVersion",1);if(measured>smartQualityVersion){checks.scene(measured);completedTasks.clear();proposalChoices=new JSONArray();advanceContext();smartQualityVersion=measured;}String hint=quality.optBoolean("underexposed")?text("Too dark: add light or use the torch.","الإضاءة ضعيفة: أضف إضاءة أو استخدم المصباح."):quality.optBoolean("overexposed")?text("Bright glare: change the camera angle.","وهج قوي: غيّر زاوية الكاميرا."):quality.optBoolean("blurred")?text("Low visible detail: hold still, refocus, or move closer.","تفاصيل قليلة: ثبّت الكاميرا وأعد التركيز أو اقترب."):"";runOnUiThread(()->{if(ui!=null&&!isDestroyed())ui.setQuality(hint);});}
            if(smartClient==null||smartPaused||smartClient.busy()||now-lastRemote<10000||!active||settingsOpen||reviewing){if(paused&&importedFrame&&!smartPaused)scheduleImportedPlanning();return;}
            lastRemote=now;JSONArray observations=new JSONArray();for(int i=0;i<Math.min(20,frame.detections.size());i++){OnnxInspector.Detection detection=frame.detections.get(i);observations.put(new JSONObject().put("label",detection.label).put("score",detection.score).put("bbox",new JSONArray(new double[]{detection.box.left,detection.box.top,detection.box.width(),detection.box.height()})));}
            JSONArray completed=new JSONArray();synchronized(completedTasks){for(String task:completedTasks)completed.put(task);}
            JSONObject context=new JSONObject().put("mode",mode).put("locked",taskLocked).put("activeTaskId",activeTaskId).put("completedTaskIds",completed).put("correction",correction).put("evidenceCount",saved).put("observations",observations).put("quality",quality).put("selectionMode","confirm").put("selectionConfirmed",!activeTaskId.isEmpty()).put("dismissedTaskIds",new JSONArray(checks.dismissedIds(now)));
            int width=frame.image.getWidth(),height=frame.image.getHeight();double ratio=Math.min(1,768d/Math.max(width,height));Bitmap upload=ratio<1?Bitmap.createScaledBitmap(frame.image,(int)(width*ratio),(int)(height*ratio),true):frame.image;
            try(ByteArrayOutputStream bytes=new ByteArrayOutputStream()){upload.compress(Bitmap.CompressFormat.JPEG,75,bytes);smartClient.submit("data:image/jpeg;base64,"+android.util.Base64.encodeToString(bytes.toByteArray(),android.util.Base64.NO_WRAP),sceneVersion,context);}finally{if(upload!=frame.image)upload.recycle();}
        }catch(Exception e){cloudStatus("unavailable");}
    }
    private int smartQualityVersion=1;
    private static boolean validMode(String value){return Arrays.asList("manual","crack","surface","ppe").contains(value);}
    private void applySmartPlan(String turnId,int version,String provider,JSONObject plan){
        if(finishing||smartPaused||!active||reviewing||settingsOpen||version!=sceneVersion)return;
        try{
            if(!plan.optBoolean("needsHumanReview",false))throw new IllegalArgumentException("Human review required");
            JSONObject scene=plan.getJSONObject("scene");smartScene=scene.getString("environment");
            String target=plan.optString("guidanceTaskId","");if(target.equals("null"))target="";
            // Older responses without an explicit target are safe only in Explore.
            if(target.equals(activeTaskId)){smartGuidance=bounded(plan.getString("guidance"),1200);acceptedPlanVersion=version;}
            smartReason=bounded(plan.optString("reason"),1200);proposalTurn=turnId;proposalChoices=plan.optJSONArray("suggestions");if(proposalChoices==null)proposalChoices=new JSONArray();
            JSONArray results=new JSONArray(),actions=plan.optJSONArray("actions");
            if(actions!=null)for(int i=0;i<actions.length();i++){JSONObject action=actions.getJSONObject(i);String id=action.getString("id"),key=turnId+":"+id;if(!appliedActions.add(key))continue;String type=action.optString("type");boolean registered=Arrays.asList("select_detector","request_view","request_manual_check","ask_context","prepare_finding").contains(type);results.put(new JSONObject().put("actionId",id).put("status",registered?"accepted":"rejected").put("detail",type.equals("select_detector")?"Proposal only; waiting for inspector selection":"Guidance displayed; no detector executed"));}
            for(int i=0;i<proposalChoices.length();i++){JSONObject choice=proposalChoices.optJSONObject(i);if(choice!=null)event("proposed",choice.optString("taskId"));}
            if(smartSteps.length()<100)smartSteps.put(new JSONObject().put("at",Instant.now().toString()).put("instruction",smartGuidance).put("reason",smartReason).put("mode",mode).put("scene",smartScene));
            runOnUiThread(()->{ui.setScene(smartScene);ui.setProvider(provider);ui.setGuidance(smartGuidance,smartReason);});refreshProposals();cloudStatus("ready");if(smartClient!=null)smartClient.acknowledge(turnId,version,results);
        }catch(Exception e){cloudStatus("unavailable");}
    }
    private void event(String type,String task){if(smartClient!=null&&!task.isEmpty())smartClient.event(type,task,sceneVersion,proposalTurn);}
    private void refreshProposals(){JSONArray filtered=new JSONArray();for(int i=0;i<proposalChoices.length();i++){JSONObject choice=proposalChoices.optJSONObject(i);if(choice!=null&&!choice.optString("taskId").equals(activeTaskId)&&!checks.suppressed(choice.optString("taskId"),SystemClock.elapsedRealtime()))filtered.put(choice);}runOnUiThread(()->{ui.setHintsEnabled(pref("aiSuggestionHints")&&!reviewing&&!settingsOpen&&(proposalTurn==null||!proposalTurn.equals(hintDismissedTurn)));ui.setSuggestions(filtered);});}
    private static String bounded(String value,int max){return value.length()>max?value.substring(0,max):value;}
    private void advanceContext(){
        if(captureRequested){captureRequested=false;captureBusy.set(false);retakeId=null;}sceneVersion++;acceptedPlanVersion=-1;if(smartClient!=null)smartClient.invalidate();
        proposalChoices=new JSONArray();
        if(smartEnabled)runOnUiThread(()->{if(!isDestroyed()){ui.setGuidance(smartPaused?text("View changed. Capture evidence or resume the guide.","تغيّر المشهد. التقط دليلاً أو استأنف المساعد."):text("View changed. Hold steady for updated guidance.","تغيّر المشهد. ثبّت الهاتف لانتظار إرشاد جديد."),"");ui.setSuggestions(null);}});
        if(paused&&importedFrame&&!reviewing&&!settingsOpen)scheduleImportedPlanning();
    }
    private void scheduleImportedPlanning(){
        if(importedPlanningScheduled||finishing||smartPaused||smartClient==null||!paused||!importedFrame)return;importedPlanningScheduled=true;
        handler.postDelayed(()->{if(!worker.isShutdown())worker.execute(()->{importedPlanningScheduled=false;if(!finishing&&active&&paused&&importedFrame&&!smartPaused){OnnxInspector.Frame frame;synchronized(frameLock){frame=latest;}if(frame!=null)smartFrame(frame,SystemClock.elapsedRealtime());}});},Math.max(1800,10000-(SystemClock.elapsedRealtime()-lastRemote)));
    }
    private void selectMode(String next) throws Exception {
        // This method only runs on the same executor as inference and capture.
        if(!validMode(next))throw new IllegalArgumentException("Unsupported detector");
        if(!next.equals("manual")){if(inspector==null)inspector=new OnnxInspector(this);inspector.load(next.equals("surface")?"surface320":next);}
        else if(inspector!=null)inspector.close();
        OnnxInspector.Frame refreshed=null;
        if(paused&&importedFrame){OnnxInspector.Frame previous;synchronized(frameLock){previous=latest;}if(previous!=null){if(next.equals("manual")){refreshed=new OnnxInspector.Frame();refreshed.image=previous.image;refreshed.modelId="manual";}else refreshed=inspector.infer(previous.image,next.equals("surface")?"surface320":next);synchronized(frameLock){latest=refreshed;}}}
        else synchronized(frameLock){if(latest!=null&&paused){refreshed=new OnnxInspector.Frame();refreshed.image=latest.image;refreshed.modelId="manual";latest=refreshed;}else latest=null;}
        mode=next;ready=true;tracks.clear();
        final OnnxInspector.Frame display=refreshed;runOnUiThread(()->{if(display!=null)overlay.update(display);else overlay.update(null);ui.setSurfaceMode(mode.equals("surface"));if(display!=null)ui.setReady(true);});
    }
    private void chooseTask(String next,String task){
        worker.execute(()->{if(next.equals("manual")&&task.isEmpty()){stopCheckOnWorker("stopped");return;}if(task.equals(activeTaskId)&&next.equals(mode)&&taskLocked){stopCheckOnWorker("stopped");return;}try{advanceContext();selectMode(next);if(!activeTaskId.isEmpty())event("stopped",activeTaskId);activeTaskId=task;checks.selected=task;taskLocked=!task.isEmpty()||!next.equals("manual");correction="Inspector selected: "+task;smartGuidance=text("Check selected. Capture evidence or wait for guidance.","اختير الفحص. التقط دليلاً أو انتظر الإرشاد.");event("selected",task);runOnUiThread(()->{ui.setTaskLocked(taskLocked);ui.setSelectedTask(activeTaskId);ui.setTask(currentTaskTitle(),mode);ui.setGuidance(smartGuidance,"");});refreshProposals();}catch(Exception e){stopCheckOnWorker("stopped");showStatus(text("Detector unavailable. Exploring with manual evidence.","الكاشف غير متاح. الاستكشاف والتوثيق اليدوي متاحان."));}});
    }
    private void stopCheck(String outcome){worker.execute(()->stopCheckOnWorker(outcome));}
    private void stopCheckOnWorker(String outcome){String previous=activeTaskId;event(outcome,previous);if(!previous.isEmpty()&&(outcome.equals("completed")||outcome.equals("skipped"))){completedTasks.add(previous);checks.finish(previous);}activeTaskId="";checks.selected="";taskLocked=false;advanceContext();try{selectMode("manual");}catch(Exception ignored){}smartGuidance=text("Exploring. Point at the next area.","استكشاف. وجّه الكاميرا إلى المنطقة التالية.");smartReason="";runOnUiThread(()->{ui.setTaskLocked(false);ui.setSelectedTask("");ui.setTask(currentTaskTitle(),"manual");ui.setGuidance(smartGuidance,"");});refreshProposals();}
    private void finishTask(boolean complete){stopCheck(complete?"completed":"skipped");}
    private void applyCorrection(String value){
        worker.execute(()->{correction=value.trim().substring(0,Math.min(500,value.trim().length()));advanceContext();cloudStatus(smartPaused?"paused":smartClient==null?missingGuideState:"ready");runOnUiThread(()->ui.setGuidance(text("Context updated. Hold steady for new guidance.","حُدّث الوصف. ثبّت الهاتف لانتظار إرشاد جديد."),""));});
    }
    private String modelId(boolean detail){return mode.equals("surface")?(detail?"surface512":"surface320"):mode;}
    private void bindCamera(){
        if(finishing||isDestroyed()||isFinishing()||binding||ContextCompat.checkSelfPermission(this,Manifest.permission.CAMERA)!=PackageManager.PERMISSION_GRANTED)return;
        if(preview.getWidth()==0||preview.getHeight()==0){preview.postDelayed(this::bindCamera,32);return;}binding=true;
        ListenableFuture<ProcessCameraProvider> future=ProcessCameraProvider.getInstance(this);
        future.addListener(()->{try{
            cameraProvider=future.get();releaseFocus();torch=false;cameraProvider.unbindAll();int rotation=preview.getDisplay().getRotation();
            Preview cameraPreview=new Preview.Builder().setTargetRotation(rotation).build();cameraPreview.setSurfaceProvider(preview.getSurfaceProvider());
            analysis=new ImageAnalysis.Builder().setTargetRotation(rotation).setTargetResolution(new android.util.Size(640,480)).setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST).setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888).build();analysis.setAnalyzer(worker,this::analyse);
            UseCaseGroup.Builder group=new UseCaseGroup.Builder().addUseCase(cameraPreview).addUseCase(analysis);ViewPort viewport=preview.getViewPort();if(viewport!=null)group.setViewPort(viewport);imageCapture=null;
            if(pref("higherDetailPhotos")){imageCapture=new ImageCapture.Builder().setTargetRotation(rotation).setCaptureMode(ImageCapture.CAPTURE_MODE_MAXIMIZE_QUALITY).build();group.addUseCase(imageCapture);}
            try{camera=cameraProvider.bindToLifecycle(this,CameraSelector.DEFAULT_BACK_CAMERA,group.build());}catch(IllegalArgumentException unsupported){cameraProvider.unbindAll();imageCapture=null;UseCaseGroup.Builder fallback=new UseCaseGroup.Builder().addUseCase(cameraPreview).addUseCase(analysis);if(viewport!=null)fallback.setViewPort(viewport);camera=cameraProvider.bindToLifecycle(this,CameraSelector.DEFAULT_BACK_CAMERA,fallback.build());showStatus(text("Higher-detail capture unsupported here. Standard capture active.","الالتقاط المفصل غير مدعوم هنا. الالتقاط العادي نشط."));}
            torchButton.setEnabled(camera.getCameraInfo().hasFlashUnit());ui.setTorch(false);setZoom(pref("rememberLastZoom")?(float)cameraPreferences.optDouble("lastZoom",1):1f);
        }catch(Exception e){showStatus(text("Cannot open rear camera: ","تعذر فتح الكاميرا الخلفية: ")+e.getMessage());}finally{binding=false;}},ContextCompat.getMainExecutor(this));
    }
    private Bitmap imageBitmap(ImageProxy image){
        ImageProxy.PlaneProxy plane=image.getPlanes()[0];ByteBuffer buffer=plane.getBuffer();android.graphics.Rect crop=image.getCropRect();int width=crop.width(),height=crop.height();int[] pixels=new int[width*height];
        for(int y=0;y<height;y++)for(int x=0;x<width;x++){int index=(y+crop.top)*plane.getRowStride()+(x+crop.left)*plane.getPixelStride();int r=buffer.get(index)&255,g=buffer.get(index+1)&255,b=buffer.get(index+2)&255;pixels[y*width+x]=Color.rgb(r,g,b);}
        Bitmap source=Bitmap.createBitmap(pixels,width,height,Bitmap.Config.ARGB_8888);int rotation=image.getImageInfo().getRotationDegrees();if(rotation==0)return source;Matrix matrix=new Matrix();matrix.postRotate(rotation);Bitmap rotated=Bitmap.createBitmap(source,0,0,width,height,matrix,true);source.recycle();return rotated;
    }
    private void analyse(ImageProxy image){
        Bitmap bitmap=null;
        try{
            if(!active||paused||finishing)return;
            long now=SystemClock.elapsedRealtime();if(now-lastAnalysis<160)return;lastAnalysis=now;
            bitmap=imageBitmap(image);if(importedFrame){importedFrame=false;advanceContext();}OnnxInspector.Frame frame;
            if(mode.equals("manual")||!ready){frame=new OnnxInspector.Frame();frame.image=bitmap;frame.modelId="manual";}
            else frame=inspector.infer(bitmap,modelId(false));
            synchronized(frameLock){latest=frame;}
            if(smartEnabled)smartFrame(frame,now);
            long completed=SystemClock.elapsedRealtime(),interval=lastCompleted==0?Math.max(160,frame.millis):completed-lastCompleted;lastCompleted=completed;
            runOnUiThread(()->{if(!isDestroyed()){overlay.update(frame);if(staleNotice){staleNotice=false;ui.setStatus("",InspectionCameraUi.INFO);}ui.setReady(true);ui.setTechnicalDetails((mode.equals("manual")?text("Guided manual evidence · automatic detector inactive","دليل يدوي موجه · الكاشف التلقائي غير نشط"):String.format(Locale.ROOT,text("On-device · %d ms · %.1f analyses/s · %d candidates","على الجهاز · %d مللي ثانية · %.1f تحليل/ث · %d نتائج"),frame.millis,1000d/Math.max(1,interval),frame.detections.size()))+" · "+(imageCapture==null?text("Capture: analysis frame","الالتقاط: إطار التحليل"):text("Capture: dedicated photo, up to 2048 px","الالتقاط: صورة مستقلة حتى ٢٠٤٨ بكسل")));}});
            boolean requested=captureRequested;captureRequested=false;
            if(requested){captureSource="analysis";save(frame,frame.detections);return;}
            if(autoSave&&active&&!captureBusy.get()&&!paused&&!reviewing&&!settingsOpen&&!followUp&&ready&&!mode.equals("manual")){List<OnnxInspector.Detection> stable=stable(frame,now);if(!stable.isEmpty()&&now-lastSave>=1000&&captureBusy.compareAndSet(false,true)){captureSource="analysis";save(frame,stable);}}
        }catch(Exception e){
            if(captureRequested&&bitmap!=null){captureRequested=false;try{OnnxInspector.Frame manual=new OnnxInspector.Frame();manual.image=bitmap;manual.modelId="manual";save(manual,manual.detections);showStatus(text("Analysis failed; photo saved as manual evidence.","تعذر التحليل؛ حُفظت الصورة كدليل يدوي."));}catch(Exception storage){showStatus(storage.getMessage());}}
            else{long now=SystemClock.elapsedRealtime();if(now-errorAt>3000){errorAt=now;showStatus(text("Analysis failed; manual evidence remains available: ","تعذر التحليل؛ يمكن حفظ الأدلة يدوياً: ")+e.getMessage());}}
        }
        finally{image.close();}
    }
    private List<OnnxInspector.Detection> stable(OnnxInspector.Frame frame,long now){
        tracks.removeIf(track->now-track.seen>2000);List<OnnxInspector.Detection> selected=new ArrayList<>();Set<Track> used=new HashSet<>();
        for(OnnxInspector.Detection detection:frame.detections){Track match=null;float best=.35f;for(Track track:tracks)if(!used.contains(track)&&track.label.equals(detection.label)){float overlap=OnnxInspector.iou(track.box,detection.box);if(overlap>best){best=overlap;match=track;}}
            if(match==null){match=new Track();match.label=detection.label;match.started=now;tracks.add(match);}else if(now-match.started>2000&&match.hits<3){match.started=now;match.hits=0;}
            used.add(match);match.hits++;match.seen=now;match.box=new RectF(detection.box);
            if(match.hits>=3&&now-match.started<=2000&&match.savedAt==0&&now-lastSave>=1000){selected.add(detection);match.savedAt=now;}
        }
        return selected;
    }
    private void requestCapture(){requestCapture(null);}
    private void requestCapture(String replacement){
        if(finishing||settingsOpen||reviewing||saving.get())return;if(!captureBusy.compareAndSet(false,true))return;retakeId=replacement;
        long now=SystemClock.elapsedRealtime();if(pref("doubleTapProtection")&&now-lastShutter<600){captureBusy.set(false);retakeId=null;return;}lastShutter=now;captureVersion=sceneVersion;
        if(!paused&&camera==null){captureBusy.set(false);retakeId=null;showStatus(text("Allow camera permission or use Import.","اسمح بالكاميرا أو استخدم الاستيراد."));return;}
        if(!paused&&imageCapture!=null&&pref("higherDetailPhotos")){
            ui.setReady(false);imageCapture.takePicture(worker,new ImageCapture.OnImageCapturedCallback(){@Override public void onCaptureSuccess(ImageProxy image){try{if(finishing||captureVersion!=sceneVersion)throw new IOException("Check or scene changed; capture cancelled");ByteBuffer bytes=image.getPlanes()[0].getBuffer();byte[] jpeg=new byte[bytes.remaining()];bytes.get(jpeg);Bitmap bitmap=BitmapFactory.decodeByteArray(jpeg,0,jpeg.length);if(bitmap==null)throw new IOException("Photo decoding failed");int rotation=image.getImageInfo().getRotationDegrees();if(rotation!=0){Matrix m=new Matrix();m.postRotate(rotation);Bitmap rotated=Bitmap.createBitmap(bitmap,0,0,bitmap.getWidth(),bitmap.getHeight(),m,true);bitmap.recycle();bitmap=rotated;}bitmap=boundedImage(bitmap,2048);OnnxInspector.Frame frame=new OnnxInspector.Frame();frame.image=bitmap;frame.modelId="manual";captureSource="photo";save(frame,frame.detections);}catch(Exception e){captureBusy.set(false);retakeId=null;showStatus(text("Photo not saved. Previous evidence is safe: ","لم تُحفظ الصورة. الدليل السابق محفوظ: ")+e.getMessage());runOnUiThread(()->ui.setReady(true));}finally{image.close();}}@Override public void onError(ImageCaptureException e){captureBusy.set(false);retakeId=null;showStatus(text("Photo capture failed. Try again.","تعذر التقاط الصورة. حاول مجدداً."));runOnUiThread(()->ui.setReady(true));}});return;
        }
        if(paused){worker.execute(()->{try{OnnxInspector.Frame frame;synchronized(frameLock){frame=latest;}if(frame==null)throw new IOException(text("Resume camera to capture evidence","استأنف الكاميرا لالتقاط الدليل"));captureSource=importedFrame?"import":"analysis";save(frame,frame.detections);}catch(Exception e){captureBusy.set(false);retakeId=null;showStatus(e.getMessage());}});}
        else{captureRequested=true;ui.setReady(false);showInfo(text("Capturing evidence…","جارٍ التقاط الدليل…"));}
    }
    private static Bitmap boundedImage(Bitmap source,int max){double ratio=Math.min(1,max/(double)Math.max(source.getWidth(),source.getHeight()));if(ratio==1)return source;Bitmap result=Bitmap.createScaledBitmap(source,(int)Math.round(source.getWidth()*ratio),(int)Math.round(source.getHeight()*ratio),true);source.recycle();return result;}
    private void save(OnnxInspector.Frame frame,List<OnnxInspector.Detection> detections) throws Exception {
        if(!saving.compareAndSet(false,true))return;try{saveDurably(frame,detections);}finally{saving.set(false);captureBusy.set(false);retakeId=null;}
    }
    private void saveDurably(OnnxInspector.Frame frame,List<OnnxInspector.Detection> detections) throws Exception {
        if(getFilesDir().getUsableSpace()<30L*1024*1024)throw new IOException(text("Storage low. Export a backup and free space.","المساحة منخفضة. صدّر نسخة احتياطية وحرّر مساحة."));
        String id=UUID.randomUUID().toString();File dir=new File(getFilesDir(),"evidence");if(!dir.exists()&&!dir.mkdirs())throw new IOException("Evidence storage unavailable");
        File original=new File(dir,id+"-original.jpg"),annotated=new File(dir,id+"-annotated.jpg");
        try {
            writeJpeg(frame.image,original);if(captureSource.equals("photo")){Bitmap retained=BitmapFactory.decodeFile(original.getAbsolutePath());if(!mode.equals("manual")&&ready){frame=inspector.infer(retained,modelId(detailed.isChecked()));detections=frame.detections;}else{frame.image=retained;} }Bitmap marked=frame.image.copy(Bitmap.Config.ARGB_8888,true);Canvas canvas=new Canvas(marked);Overlay.drawDetections(canvas,frame.detections,frame.image.getWidth(),frame.image.getHeight(),arabic);writeJpeg(marked,annotated);marked.recycle();
            JSONObject evidence=new JSONObject().put("id",id).put("original",Uri.fromFile(original).toString()).put("annotated",Uri.fromFile(annotated).toString()).put("capturedAt",frame.time).put("width",frame.image.getWidth()).put("height",frame.image.getHeight()).put("captureSource",captureSource).put("detectorBaseline",Uri.fromFile(annotated).toString());JSONArray observations=new JSONArray();
            for(OnnxInspector.Detection detection:detections){JSONObject observation=detection.json(frame.modelId,frame.frameId,frame.time,frame.millis,frame.image.getWidth(),frame.image.getHeight());if(frame.modelInfo!=null)observation.put("model",frame.modelInfo);observations.put(observation);}
            JSONObject capture=new JSONObject().put("inspectionId",inspectionId).put("evidence",evidence).put("observations",observations).put("followUp",followUp).put("checkTaskId",activeTaskId);
            if(smartClient!=null&&smartClient.sessionId()!=null&&acceptedPlanVersion==sceneVersion)evidence.put("smartContext",new JSONObject().put("sessionId",smartClient.sessionId()).put("provider",smartClient.provider()).put("sceneVersion",acceptedPlanVersion).put("guidance",smartGuidance).put("mode",mode).put("activeTaskId",activeTaskId));
            if(findingId!=null)capture.put("findingId",findingId);
            BurhanDatabase.CaptureRow row=new BurhanDatabase.CaptureRow();row.id=id;row.inspectionId=inspectionId;row.json=capture.toString();row.createdAt=System.currentTimeMillis();String replacing=retakeId;boolean replace=replacing!=null&&sessionCaptures.containsKey(replacing);if(replace)BurhanDatabase.get(this).store().replaceCapture(replacing,row);else BurhanDatabase.get(this).store().capture(row);synchronized(sessionCaptures){if(replace)sessionCaptures.remove(replacing);sessionCaptures.put(id,row);}
            lastSave=SystemClock.elapsedRealtime();saved=sessionCaptures.size();runOnUiThread(()->{ui.setStatus("",InspectionCameraUi.INFO);ui.setSavedCount(saved);ui.captureSaved();ui.setReady(true);if(pref("undoLastCapture")&&!replace){long undoUntil=SystemClock.elapsedRealtime()+5000;com.google.android.material.snackbar.Snackbar.make(ui.root,text("Evidence saved","حُفظ الدليل"),5000).setAction(text("Undo","تراجع"),v->{if(SystemClock.elapsedRealtime()<=undoUntil)undoCapture(id);}).show();}});
        }catch(Exception e){original.delete();annotated.delete();throw e;}
    }
    private void writeJpeg(Bitmap bitmap,File target) throws IOException {try(FileOutputStream output=new FileOutputStream(target)){if(!bitmap.compress(Bitmap.CompressFormat.JPEG,92,output))throw new IOException("Image encoding failed");output.getFD().sync();}}
    private void importImage(Uri uri){
        if(!captureBusy.compareAndSet(false,true))return;paused=true;overlay.freeze(true);ui.setCameraPaused(true);ui.setReady(false);boolean detail=detailed.isChecked();showStatus(text("Analysing imported photo…","جارٍ تحليل الصورة المستوردة…"));
        worker.execute(()->{try{advanceContext();importedFrame=true;
            Bitmap bitmap=ImageDecoder.decodeBitmap(ImageDecoder.createSource(getContentResolver(),uri),(decoder,info,source)->{decoder.setAllocator(ImageDecoder.ALLOCATOR_SOFTWARE);int max=Math.max(info.getSize().getWidth(),info.getSize().getHeight());if(max>2048)decoder.setTargetSampleSize((int)Math.ceil(max/2048d));});
            OnnxInspector.Frame frame;if(mode.equals("manual")||!ready){frame=new OnnxInspector.Frame();frame.image=bitmap;frame.modelId="manual";}else frame=inspector.infer(bitmap,modelId(detail));synchronized(frameLock){latest=frame;}captureSource="import";save(frame,frame.detections);if(smartEnabled)smartFrame(frame,SystemClock.elapsedRealtime());
            runOnUiThread(()->{overlay.update(frame);ui.setReady(true);showInfo(text("Imported photo saved. Resume camera to continue.","حُفظت الصورة المستوردة. استأنف الكاميرا للمتابعة."));});
        }catch(Exception e){showStatus(text("Photo could not be analysed: ","تعذر تحليل الصورة: ")+e.getMessage());}finally{captureBusy.set(false);retakeId=null;}});
    }
    private void togglePause(){releaseFocus();if(paused&&power!=null&&power.getCurrentThermalStatus()>=PowerManager.THERMAL_STATUS_SEVERE){showStatus(text("Let the phone cool before resuming analysis.","انتظر حتى يبرد الهاتف قبل استئناف التحليل."));return;}paused=!paused;overlay.freeze(paused);ui.setCameraPaused(paused);applyPreferences();if(!paused){ui.setReady(false);showInfo(text("Camera resumed","استؤنفت الكاميرا"));}}
    private void toggleTorch(){if(camera!=null&&camera.getCameraInfo().hasFlashUnit()){torch=!torch;camera.getCameraControl().enableTorch(torch);ui.setTorch(torch);}}
    private void toggleZoom(){if(camera==null)return;ZoomState state=camera.getCameraInfo().getZoomState().getValue();if(state==null)return;float next=state.getZoomRatio()>1.1f?state.getMinZoomRatio():Math.min(2,state.getMaxZoomRatio());setZoom(next);}
    private boolean pref(String key){return cameraPreferences.optBoolean(key,false);}
    private void applyPreferences(){cameraPreferences=CameraPreferences.read(this);if(ui!=null){ui.setHintsEnabled(pref("aiSuggestionHints")&&!reviewing&&!settingsOpen);if(!pref("focusLock"))releaseFocus();overlay.setDetectionsVisible(pref("liveDetectionOverlays"));}if(active&&!paused&&!reviewing&&!settingsOpen&&pref("keepScreenAwake"))getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);else getWindow().clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);if(aids!=null){aids.configure(pref("framingGrid"),pref("cameraLevel"));if(active&&!paused)aids.start();else aids.stop();}}
    private void setZoom(float requested){if(camera==null)return;ZoomState state=camera.getCameraInfo().getZoomState().getValue();if(state==null)return;float zoom=Math.max(state.getMinZoomRatio(),Math.min(state.getMaxZoomRatio(),requested));camera.getCameraControl().setZoomRatio(zoom);ui.setZoom(zoom);if(pref("rememberLastZoom"))try{CameraPreferences.write(this,new JSONObject().put("lastZoom",zoom));}catch(Exception ignored){}}
    private void focus(float x,float y,boolean lock){if(camera==null||paused)return;if(focusLocked&&!lock){releaseFocus();return;}MeteringPoint point=preview.getMeteringPointFactory().createPoint(x,y);FocusMeteringAction.Builder action=new FocusMeteringAction.Builder(point,FocusMeteringAction.FLAG_AF|FocusMeteringAction.FLAG_AE);if(lock)action.disableAutoCancel();else action.setAutoCancelDuration(3,TimeUnit.SECONDS);camera.getCameraControl().startFocusAndMetering(action.build());focusLocked=lock;ui.showFocus(x,y);ui.setFocusLocked(lock);if(lock)showInfo(text("Focus locked. Tap to release.","ثُبّت التركيز. اضغط لتحريره."));}
    private void releaseFocus(){if(focusLocked&&camera!=null)camera.getCameraControl().cancelFocusAndMetering();focusLocked=false;if(ui!=null)ui.setFocusLocked(false);}
    @Override public boolean onKeyDown(int key,KeyEvent event){if((key==KeyEvent.KEYCODE_VOLUME_UP||key==KeyEvent.KEYCODE_VOLUME_DOWN)&&active&&!paused&&!reviewing&&!settingsOpen&&pref("volumeButtonCapture")){if(event.getRepeatCount()==0)requestCapture();return true;}return super.onKeyDown(key,event);}
    @Override public boolean onKeyUp(int key,KeyEvent event){if((key==KeyEvent.KEYCODE_VOLUME_UP||key==KeyEvent.KEYCODE_VOLUME_DOWN)&&active&&!reviewing&&!settingsOpen&&pref("volumeButtonCapture"))return true;return super.onKeyUp(key,event);}
    private void undoCapture(String id){worker.execute(()->{BurhanDatabase.CaptureRow row;synchronized(sessionCaptures){row=sessionCaptures.get(id);}if(row==null)return;BurhanDatabase.get(this).store().acknowledge(Collections.singletonList(id));synchronized(sessionCaptures){sessionCaptures.remove(id);saved=sessionCaptures.size();}runOnUiThread(()->{ui.setSavedCount(saved);showInfo(text("Capture removed","حُذف الالتقاط"));});});}
    private void openReview(){if(!pref("quickCaptureReview")){finishScan();return;}if(captureBusy.get()||saving.get()){showInfo(text("Wait for this capture to save.","انتظر حفظ الالتقاط."));return;}boolean wasPaused=paused;reviewing=true;paused=true;ui.setHintsEnabled(false);overlay.freeze(true);if(smartClient!=null)smartClient.pause(true);applyPreferences();List<BurhanDatabase.CaptureRow> rows;synchronized(sessionCaptures){rows=new ArrayList<>(sessionCaptures.values());}review=new CaptureReview(this,arabic,rows,worker,activeTaskId,new CaptureReview.Listener(){public void closed(){reviewing=false;paused=wasPaused;overlay.freeze(paused);ui.setCameraPaused(paused);if(smartClient!=null)smartClient.pause(smartPaused);applyPreferences();refreshProposals();}public void allFindings(){finishScan();}public void retake(String id){reviewing=false;paused=wasPaused;overlay.freeze(paused);if(smartClient!=null)smartClient.pause(smartPaused);applyPreferences();refreshProposals();requestCapture(id);}public void updated(BurhanDatabase.CaptureRow row){synchronized(sessionCaptures){sessionCaptures.put(row.id,row);}}});review.show();}
    private void finishScan(){if(finishing)return;finishing=true;active=false;getWindow().clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);if(review!=null)review.dismiss();if(smartClient!=null)smartClient.close();ui.setReady(false);ui.dismissSheets();showInfo(text("Finishing and preserving evidence…","جارٍ إنهاء الفحص وحفظ الأدلة…"));if(analysis!=null)analysis.clearAnalyzer();if(cameraProvider!=null)cameraProvider.unbindAll();worker.execute(()->{try{if(inspector!=null)inspector.close();}catch(Exception ignored){}Intent result=new Intent();try{if(smartClient!=null&&smartClient.sessionId()!=null)result.putExtra("smart",new JSONObject().put("sessionId",smartClient.sessionId()).put("provider",smartClient.provider()).put("scene",smartScene).put("steps",smartSteps).toString());}catch(Exception ignored){}runOnUiThread(()->{setResult(RESULT_OK,result);finish();});});}
    @Override protected void onResume(){super.onResume();active=true;applyPreferences();if(settingsOpen){settingsOpen=false;paused=pausedBeforeSettings;ui.setCameraPaused(paused);preview.post(this::bindCamera);applyPreferences();refreshProposals();}if(smartClient!=null)smartClient.pause(smartPaused||reviewing||settingsOpen);if(importedFrame&&!worker.isShutdown())worker.execute(this::scheduleImportedPlanning);if(preview!=null&&cameraProvider==null&&ContextCompat.checkSelfPermission(this,Manifest.permission.CAMERA)==PackageManager.PERMISSION_GRANTED)preview.post(this::bindCamera);}
    @Override public void onConfigurationChanged(android.content.res.Configuration configuration){
        super.onConfigurationChanged(configuration);
        // Preserve the session, guide history and task; rebuild CameraX's viewport/rotation.
        if(!worker.isShutdown())worker.execute(this::advanceContext);
        if(ui!=null)ui.reflow();if(preview!=null)preview.post(()->{if(active&&!finishing)bindCamera();});
    }
    @Override protected void onPause(){active=false;if(captureRequested){captureRequested=false;captureBusy.set(false);retakeId=null;}getWindow().clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);releaseFocus();if(aids!=null)aids.stop();if(smartClient!=null)smartClient.pause(true);if(torch&&camera!=null){camera.getCameraControl().enableTorch(false);torch=false;if(ui!=null)ui.setTorch(false);}super.onPause();}
    @Override protected void onStop(){super.onStop();}
    @Override protected void onDestroy(){active=false;if(ui!=null)ui.dismissSheets();if(aids!=null)aids.stop();if(smartClient!=null)smartClient.close();handler.removeCallbacks(staleCheck);if(power!=null&&thermalListener!=null)power.removeThermalStatusListener(thermalListener);if(analysis!=null)analysis.clearAnalyzer();if(cameraProvider!=null)cameraProvider.unbindAll();worker.execute(()->{try{if(inspector!=null)inspector.close();}catch(Exception ignored){}});worker.shutdown();super.onDestroy();}

    private static final class Overlay extends View {
        private OnnxInspector.Frame frame;private boolean frozen,showDetections=true;private final boolean arabic;
        Overlay(Context context,boolean ar){super(context);arabic=ar;setImportantForAccessibility(View.IMPORTANT_FOR_ACCESSIBILITY_NO);}
        void setDetectionsVisible(boolean value){showDetections=value;invalidate();}
        void update(OnnxInspector.Frame next){frame=next;invalidate();}
        void freeze(boolean value){frozen=value;invalidate();}
        @Override protected void onDraw(Canvas canvas){super.onDraw(canvas);if(frame==null)return;float scale=Math.min(getWidth()/(float)frame.image.getWidth(),getHeight()/(float)frame.image.getHeight());float dx=(getWidth()-frame.image.getWidth()*scale)/2,dy=(getHeight()-frame.image.getHeight()*scale)/2;if(frozen)canvas.drawColor(Color.BLACK);canvas.save();canvas.translate(dx,dy);canvas.scale(scale,scale);if(frozen)canvas.drawBitmap(frame.image,0,0,null);if(showDetections)drawDetections(canvas,frame.detections,frame.image.getWidth(),frame.image.getHeight(),arabic);canvas.restore();}
        static void drawDetections(Canvas canvas,List<OnnxInspector.Detection> detections,int width,int height,boolean arabic){
            Paint stroke=new Paint(Paint.ANTI_ALIAS_FLAG);stroke.setColor(Color.rgb(255,212,59));stroke.setStyle(Paint.Style.STROKE);stroke.setStrokeWidth(Math.max(2,width/320f));Paint text=new Paint(Paint.ANTI_ALIAS_FLAG);text.setTextSize(Math.max(16,width/32f));text.setTypeface(Typeface.DEFAULT_BOLD);Paint background=new Paint();background.setColor(Color.rgb(255,212,59));
            for(OnnxInspector.Detection detection:detections){detection.drawMask(canvas,width,height);canvas.drawRect(detection.box,stroke);String label=displayName(detection.label,arabic)+" "+Math.round(detection.score*100)+"%";float top=Math.max(0,detection.box.top-text.getTextSize()-6),left=Math.min(detection.box.left,Math.max(0,width-text.measureText(label)-8));canvas.drawRect(left,top,left+text.measureText(label)+8,top+text.getTextSize()+6,background);text.setColor(Color.BLACK);canvas.drawText(label,left+4,top+text.getTextSize(),text);}
        }
        static String displayName(String label,boolean arabic){if(!arabic)return label;switch(label){case "crack":return "شرخ";case "alligator_crack":return "تشققات شبكية";case "wetspot":return "بقعة رطوبة";case "efflorescence":return "تزهير ملحي";case "rust":return "مظهر صدأ";case "rockpocket":return "تعشيش";case "hollowareas":return "حدود مناطق معلّمة";case "cavity":return "تجويف ظاهر";case "spalling":return "تقشر سطحي";case "graffiti":return "كتابات سطحية";case "weathering":return "تجوية";case "residual_formwork":return "بقايا شدات";case "exposed_rebar":return "حديد مكشوف";case "washout":return "تآكل سطحي";case "Hardhat":return "خوذة";case "Mask":return "كمامة";case "NO-Hardhat":return "بدون خوذة";case "NO-Mask":return "بدون كمامة";case "NO-Safety Vest":return "بدون سترة سلامة";case "Person":return "شخص";case "Safety Cone":return "مخروط سلامة";case "Safety Vest":return "سترة سلامة";case "machinery":return "معدات";case "vehicle":return "مركبة";default:return label;}}
    }
}
