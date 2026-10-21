package com.burhan.site;

import android.Manifest;
import android.content.*;
import android.content.pm.PackageManager;
import android.graphics.*;
import android.graphics.drawable.GradientDrawable;
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
    private final Runnable staleCheck=new Runnable(){@Override public void run(){if(active&&!paused&&lastCompleted>0&&SystemClock.elapsedRealtime()-lastCompleted>3000){overlay.update(null);showStatus(text("Waiting for a fresh analysed frame…","بانتظار إطار حديث بعد التحليل…"));}if(!isDestroyed())handler.postDelayed(this,1000);}};
    private volatile boolean paused=false,active=false,finishing=false,ready=false,captureRequested=false;
    private boolean arabic,autoSave,torch,followUp,binding;
    private String inspectionId,mode,findingId;
    private OnnxInspector inspector;
    private OnnxInspector.Frame latest;
    private PreviewView preview;
    private Overlay overlay;
    private TextView status,count;
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
        WindowCompat.setDecorFitsSystemWindows(getWindow(),false);
        WindowInsetsControllerCompat systemBars=new WindowInsetsControllerCompat(getWindow(),getWindow().getDecorView());systemBars.setAppearanceLightStatusBars(false);systemBars.setAppearanceLightNavigationBars(false);
        buildInterface();getOnBackPressedDispatcher().addCallback(this,new OnBackPressedCallback(true){@Override public void handleOnBackPressed(){finishScan();}});
        power=(PowerManager)getSystemService(POWER_SERVICE);thermalListener=level->{if(level>=PowerManager.THERMAL_STATUS_SEVERE){paused=true;overlay.freeze(true);pauseButton.setText(text("Resume","استئناف"));showStatus(text("Phone is hot. Analysis paused; let it cool before resuming.","الهاتف ساخن. أُوقف التحليل مؤقتاً؛ انتظر حتى يبرد قبل الاستئناف."));}};power.addThermalStatusListener(ContextCompat.getMainExecutor(this),thermalListener);
        worker.execute(()->{try{if(!mode.equals("manual")){inspector=new OnnxInspector(this);inspector.load(modelId(false));}ready=true;showStatus(mode.equals("manual")?text("Camera evidence ready · manual inspection","جاهز لتوثيق الأدلة · فحص يدوي"):text("CPU model ready · experimental observations","النموذج جاهز · ملاحظات تجريبية"));}catch(Exception e){showStatus(text("Model unavailable: ","النموذج غير متاح: ")+e.getMessage());}});
        if(ContextCompat.checkSelfPermission(this,Manifest.permission.CAMERA)==PackageManager.PERMISSION_GRANTED)preview.post(this::bindCamera);else permission.launch(Manifest.permission.CAMERA);
        handler.postDelayed(staleCheck,1000);
    }
    private void buildInterface(){
        LinearLayout root=new LinearLayout(this);root.setOrientation(LinearLayout.VERTICAL);root.setBackgroundColor(Color.rgb(17,19,21));root.setLayoutDirection(arabic?View.LAYOUT_DIRECTION_RTL:View.LAYOUT_DIRECTION_LTR);
        ViewCompat.setOnApplyWindowInsetsListener(root,(view,insets)->{androidx.core.graphics.Insets bars=insets.getInsets(WindowInsetsCompat.Type.systemBars());view.setPadding(dp(12)+bars.left,bars.top+dp(8),dp(12)+bars.right,bars.bottom+dp(8));return insets;});
        TextView heading=label(text("BURHAN / LIVE INSPECTION","بُرهان / الفحص المباشر"),20,true);root.addView(heading);root.addView(label(text("Experimental visible observations · review every finding","ملاحظات بصرية تجريبية · راجع كل نتيجة"),13,false));
        FrameLayout viewport=new FrameLayout(this);preview=new PreviewView(this);preview.setImplementationMode(PreviewView.ImplementationMode.COMPATIBLE);preview.setScaleType(PreviewView.ScaleType.FIT_CENTER);viewport.addView(preview,new FrameLayout.LayoutParams(-1,-1));overlay=new Overlay(this,arabic);viewport.addView(overlay,new FrameLayout.LayoutParams(-1,-1));root.addView(viewport,new LinearLayout.LayoutParams(-1,0,1));
        status=label(text("Loading bundled model…","جارٍ تحميل النموذج المضمّن…"),14,false);status.setMinHeight(dp(46));root.addView(status);count=label(text("0 evidence captures saved","٠ أدلة محفوظة"),14,true);root.addView(count);
        detailed=new CheckBox(this);detailed.setText(text("Detailed surface photo analysis (512)","تحليل صورة السطح بالتفصيل (٥١٢)"));detailed.setTextColor(Color.WHITE);detailed.setMinHeight(dp(48));if(mode.equals("surface"))root.addView(detailed);
        LinearLayout controls=new LinearLayout(this);controls.setOrientation(LinearLayout.HORIZONTAL);pauseButton=button(text("Pause","إيقاف مؤقت"),v->togglePause());torchButton=button(text("Torch","إضاءة"),v->toggleTorch());zoomButton=button("1×",v->toggleZoom());addControl(controls,pauseButton);addControl(controls,torchButton);addControl(controls,zoomButton);root.addView(controls);
        LinearLayout actions=new LinearLayout(this);actions.setOrientation(LinearLayout.HORIZONTAL);addControl(actions,button(text("Import","استيراد"),v->photo.launch(new String[]{"image/*"})));Button capture=button(text("Capture","التقاط"),v->requestCapture());capture.setTextColor(Color.BLACK);capture.setBackgroundColor(Color.rgb(255,212,59));addControl(actions,capture);addControl(actions,button(text("Done","تم"),v->finishScan()));root.addView(actions);setContentView(root);
    }
    private TextView label(String value,int size,boolean bold){TextView view=new TextView(this);view.setText(value);view.setTextColor(Color.rgb(244,242,234));view.setTextSize(size);view.setPadding(0,dp(5),0,dp(5));if(bold)view.setTypeface(null,Typeface.BOLD);return view;}
    private Button button(String label,View.OnClickListener listener){Button button=new Button(this);button.setText(label);button.setAllCaps(false);button.setTextSize(13);button.setTextColor(Color.WHITE);button.setMinHeight(dp(56));GradientDrawable bg=new GradientDrawable();bg.setColor(Color.rgb(29,32,36));bg.setStroke(dp(2),Color.rgb(125,131,140));bg.setCornerRadius(dp(3));button.setBackground(bg);button.setOnClickListener(listener);button.setContentDescription(label);return button;}
    private void addControl(LinearLayout layout,Button button){LinearLayout.LayoutParams params=new LinearLayout.LayoutParams(0,dp(56),1);params.setMargins(dp(3),dp(4),dp(3),dp(4));layout.addView(button,params);}
    private int dp(int value){return Math.round(value*getResources().getDisplayMetrics().density);}
    private void showStatus(String message){runOnUiThread(()->{if(!isDestroyed())status.setText(message);});}
    private String modelId(boolean detail){return mode.equals("surface")?(detail?"surface512":"surface320"):mode;}
    private void bindCamera(){
        if(finishing||isDestroyed()||isFinishing()||binding||ContextCompat.checkSelfPermission(this,Manifest.permission.CAMERA)!=PackageManager.PERMISSION_GRANTED)return;
        if(preview.getWidth()==0||preview.getHeight()==0){preview.postDelayed(this::bindCamera,32);return;}binding=true;
        ListenableFuture<ProcessCameraProvider> future=ProcessCameraProvider.getInstance(this);
        future.addListener(()->{try{
            cameraProvider=future.get();cameraProvider.unbindAll();int rotation=preview.getDisplay().getRotation();
            Preview cameraPreview=new Preview.Builder().setTargetRotation(rotation).build();cameraPreview.setSurfaceProvider(preview.getSurfaceProvider());
            analysis=new ImageAnalysis.Builder().setTargetRotation(rotation).setTargetResolution(new android.util.Size(640,480)).setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST).setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888).build();analysis.setAnalyzer(worker,this::analyse);
            UseCaseGroup.Builder group=new UseCaseGroup.Builder().addUseCase(cameraPreview).addUseCase(analysis);ViewPort viewport=preview.getViewPort();if(viewport!=null)group.setViewPort(viewport);
            camera=cameraProvider.bindToLifecycle(this,CameraSelector.DEFAULT_BACK_CAMERA,group.build());torchButton.setEnabled(camera.getCameraInfo().hasFlashUnit());
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
            bitmap=imageBitmap(image);OnnxInspector.Frame frame;
            if(mode.equals("manual")||!ready){frame=new OnnxInspector.Frame();frame.image=bitmap;frame.modelId="manual";}
            else frame=inspector.infer(bitmap,modelId(false));
            synchronized(frameLock){latest=frame;}
            long completed=SystemClock.elapsedRealtime(),interval=lastCompleted==0?Math.max(160,frame.millis):completed-lastCompleted;lastCompleted=completed;
            runOnUiThread(()->{if(!isDestroyed()){overlay.update(frame);if(ready)status.setText(mode.equals("manual")?text("Manual camera evidence","توثيق بالكاميرا يدوياً"):String.format(Locale.ROOT,text("CPU · %d ms · %.1f analyses/s · %d candidates","المعالج · %d مللي ثانية · %.1f تحليل/ث · %d نتائج"),frame.millis,1000d/Math.max(1,interval),frame.detections.size()));}});
            boolean requested=captureRequested;captureRequested=false;
            if(requested){save(frame,frame.detections);return;}
            if(autoSave&&!followUp&&ready&&!mode.equals("manual")){List<OnnxInspector.Detection> stable=stable(frame,now);if(!stable.isEmpty()&&now-lastSave>=1000)save(frame,stable);}
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
    private void requestCapture(){
        if(finishing)return;
        if(!paused&&camera==null){showStatus(text("Open camera permission in Settings, or use Import.","اسمح بالكاميرا من الإعدادات أو استخدم الاستيراد."));return;}
        if(paused){OnnxInspector.Frame frame; synchronized(frameLock){frame=latest;}if(frame==null){showStatus(text("Resume camera to capture evidence","استأنف الكاميرا لالتقاط الدليل"));return;}worker.execute(()->{try{save(frame,frame.detections);}catch(Exception e){showStatus(e.getMessage());}});}
        else{captureRequested=true;showStatus(text("Capturing the next analysed frame…","جارٍ التقاط الإطار التالي بعد تحليله…"));}
    }
    private void save(OnnxInspector.Frame frame,List<OnnxInspector.Detection> detections) throws Exception {
        if(getFilesDir().getUsableSpace()<30L*1024*1024)throw new IOException(text("Storage low. Export a backup and free space.","المساحة منخفضة. صدّر نسخة احتياطية وحرّر مساحة."));
        String id=UUID.randomUUID().toString();File dir=new File(getFilesDir(),"evidence");if(!dir.exists()&&!dir.mkdirs())throw new IOException("Evidence storage unavailable");
        File original=new File(dir,id+"-original.jpg"),annotated=new File(dir,id+"-annotated.jpg");
        try {
            writeJpeg(frame.image,original);Bitmap marked=frame.image.copy(Bitmap.Config.ARGB_8888,true);Canvas canvas=new Canvas(marked);Overlay.drawDetections(canvas,frame.detections,frame.image.getWidth(),frame.image.getHeight(),arabic);writeJpeg(marked,annotated);marked.recycle();
            JSONObject evidence=new JSONObject().put("id",id).put("original",Uri.fromFile(original).toString()).put("annotated",Uri.fromFile(annotated).toString()).put("capturedAt",frame.time).put("width",frame.image.getWidth()).put("height",frame.image.getHeight());JSONArray observations=new JSONArray();
            for(OnnxInspector.Detection detection:detections){JSONObject observation=detection.json(frame.modelId,frame.frameId,frame.time,frame.millis,frame.image.getWidth(),frame.image.getHeight());if(frame.modelInfo!=null)observation.put("model",frame.modelInfo);observations.put(observation);}
            JSONObject capture=new JSONObject().put("inspectionId",inspectionId).put("evidence",evidence).put("observations",observations).put("followUp",followUp);
            if(findingId!=null)capture.put("findingId",findingId);
            BurhanDatabase.CaptureRow row=new BurhanDatabase.CaptureRow();row.id=id;row.inspectionId=inspectionId;row.json=capture.toString();row.createdAt=System.currentTimeMillis();BurhanDatabase.get(this).store().capture(row);
            lastSave=SystemClock.elapsedRealtime();saved++;runOnUiThread(()->{count.setText(String.format(Locale.ROOT,text("%d evidence captures saved · review on Done","%d أدلة محفوظة · راجعها بعد الضغط على تم"),saved));});
        }catch(Exception e){original.delete();annotated.delete();throw e;}
    }
    private void writeJpeg(Bitmap bitmap,File target) throws IOException {try(FileOutputStream output=new FileOutputStream(target)){if(!bitmap.compress(Bitmap.CompressFormat.JPEG,92,output))throw new IOException("Image encoding failed");output.getFD().sync();}}
    private void importImage(Uri uri){
        paused=true;overlay.freeze(true);pauseButton.setText(text("Resume","استئناف"));boolean detail=detailed.isChecked();showStatus(text("Analysing imported photo…","جارٍ تحليل الصورة المستوردة…"));
        worker.execute(()->{try{
            Bitmap bitmap=ImageDecoder.decodeBitmap(ImageDecoder.createSource(getContentResolver(),uri),(decoder,info,source)->{decoder.setAllocator(ImageDecoder.ALLOCATOR_SOFTWARE);int max=Math.max(info.getSize().getWidth(),info.getSize().getHeight());if(max>1600)decoder.setTargetSampleSize((int)Math.ceil(max/1600d));});
            OnnxInspector.Frame frame;if(mode.equals("manual")||!ready){frame=new OnnxInspector.Frame();frame.image=bitmap;frame.modelId="manual";}else frame=inspector.infer(bitmap,modelId(detail));synchronized(frameLock){latest=frame;}save(frame,frame.detections);
            runOnUiThread(()->{overlay.update(frame);showStatus(text("Photo saved. Resume to return to live camera.","حُفظت الصورة. اضغط استئناف للعودة إلى الكاميرا."));});
        }catch(Exception e){showStatus(text("Photo could not be analysed: ","تعذر تحليل الصورة: ")+e.getMessage());}});
    }
    private void togglePause(){if(paused&&power!=null&&power.getCurrentThermalStatus()>=PowerManager.THERMAL_STATUS_SEVERE){showStatus(text("Let the phone cool before resuming analysis.","انتظر حتى يبرد الهاتف قبل استئناف التحليل."));return;}paused=!paused;overlay.freeze(paused);pauseButton.setText(paused?text("Resume","استئناف"):text("Pause","إيقاف مؤقت"));if(paused)showStatus(text("Analysis paused · displayed results belong to the last frame","التحليل متوقف · النتائج المعروضة تخص الإطار السابق"));}
    private void toggleTorch(){if(camera!=null&&camera.getCameraInfo().hasFlashUnit()){torch=!torch;camera.getCameraControl().enableTorch(torch);torchButton.setText(torch?text("Torch on","الإضاءة تعمل"):text("Torch","إضاءة"));}}
    private void toggleZoom(){if(camera==null)return;ZoomState state=camera.getCameraInfo().getZoomState().getValue();if(state==null)return;float next=state.getZoomRatio()>1.1f?state.getMinZoomRatio():Math.min(2,state.getMaxZoomRatio());camera.getCameraControl().setZoomRatio(next);zoomButton.setText(String.format(Locale.ROOT,"%.1f×",next));}
    private void finishScan(){if(finishing)return;finishing=true;active=false;showStatus(text("Finishing and preserving evidence…","جارٍ إنهاء الفحص وحفظ الأدلة…"));if(analysis!=null)analysis.clearAnalyzer();if(cameraProvider!=null)cameraProvider.unbindAll();worker.execute(()->{try{if(inspector!=null)inspector.close();}catch(Exception ignored){}runOnUiThread(()->{setResult(RESULT_OK);finish();});});}
    @Override protected void onResume(){super.onResume();active=true;if(preview!=null&&cameraProvider==null&&ContextCompat.checkSelfPermission(this,Manifest.permission.CAMERA)==PackageManager.PERMISSION_GRANTED)preview.post(this::bindCamera);}
    @Override protected void onPause(){active=false;if(torch&&camera!=null){camera.getCameraControl().enableTorch(false);torch=false;}super.onPause();}
    @Override protected void onStop(){worker.execute(()->{try{if(inspector!=null)inspector.close();}catch(Exception ignored){}});super.onStop();}
    @Override protected void onDestroy(){active=false;handler.removeCallbacks(staleCheck);if(power!=null&&thermalListener!=null)power.removeThermalStatusListener(thermalListener);if(analysis!=null)analysis.clearAnalyzer();if(cameraProvider!=null)cameraProvider.unbindAll();worker.execute(()->{try{if(inspector!=null)inspector.close();}catch(Exception ignored){}});worker.shutdown();super.onDestroy();}

    private static final class Overlay extends View {
        private OnnxInspector.Frame frame;private boolean frozen;private final boolean arabic;
        Overlay(Context context,boolean ar){super(context);arabic=ar;setImportantForAccessibility(View.IMPORTANT_FOR_ACCESSIBILITY_NO);}
        void update(OnnxInspector.Frame next){frame=next;invalidate();}
        void freeze(boolean value){frozen=value;invalidate();}
        @Override protected void onDraw(Canvas canvas){super.onDraw(canvas);if(frame==null)return;float scale=Math.min(getWidth()/(float)frame.image.getWidth(),getHeight()/(float)frame.image.getHeight());float dx=(getWidth()-frame.image.getWidth()*scale)/2,dy=(getHeight()-frame.image.getHeight()*scale)/2;if(frozen)canvas.drawColor(Color.BLACK);canvas.save();canvas.translate(dx,dy);canvas.scale(scale,scale);if(frozen)canvas.drawBitmap(frame.image,0,0,null);drawDetections(canvas,frame.detections,frame.image.getWidth(),frame.image.getHeight(),arabic);canvas.restore();}
        static void drawDetections(Canvas canvas,List<OnnxInspector.Detection> detections,int width,int height,boolean arabic){
            Paint stroke=new Paint(Paint.ANTI_ALIAS_FLAG);stroke.setColor(Color.rgb(255,212,59));stroke.setStyle(Paint.Style.STROKE);stroke.setStrokeWidth(Math.max(2,width/320f));Paint text=new Paint(Paint.ANTI_ALIAS_FLAG);text.setTextSize(Math.max(16,width/32f));text.setTypeface(Typeface.DEFAULT_BOLD);Paint background=new Paint();background.setColor(Color.rgb(255,212,59));
            for(OnnxInspector.Detection detection:detections){detection.drawMask(canvas,width,height);canvas.drawRect(detection.box,stroke);String label=displayName(detection.label,arabic)+" "+Math.round(detection.score*100)+"%";float top=Math.max(0,detection.box.top-text.getTextSize()-6),left=Math.min(detection.box.left,Math.max(0,width-text.measureText(label)-8));canvas.drawRect(left,top,left+text.measureText(label)+8,top+text.getTextSize()+6,background);text.setColor(Color.BLACK);canvas.drawText(label,left+4,top+text.getTextSize(),text);}
        }
        static String displayName(String label,boolean arabic){if(!arabic)return label;switch(label){case "crack":return "شرخ";case "alligator_crack":return "تشققات شبكية";case "wetspot":return "بقعة رطوبة";case "efflorescence":return "تزهير ملحي";case "rust":return "مظهر صدأ";case "rockpocket":return "تعشيش";case "hollowareas":return "حدود مناطق معلّمة";case "cavity":return "تجويف ظاهر";case "spalling":return "تقشر سطحي";case "graffiti":return "كتابات سطحية";case "weathering":return "تجوية";case "residual_formwork":return "بقايا شدات";case "exposed_rebar":return "حديد مكشوف";case "washout":return "تآكل سطحي";case "Hardhat":return "خوذة";case "Mask":return "كمامة";case "NO-Hardhat":return "بدون خوذة";case "NO-Mask":return "بدون كمامة";case "NO-Safety Vest":return "بدون سترة سلامة";case "Person":return "شخص";case "Safety Cone":return "مخروط سلامة";case "Safety Vest":return "سترة سلامة";case "machinery":return "معدات";case "vehicle":return "مركبة";default:return label;}}
    }
}
