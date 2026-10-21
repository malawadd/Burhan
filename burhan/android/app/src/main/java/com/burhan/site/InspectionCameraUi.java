package com.burhan.site;

import android.app.Activity;
import android.content.res.ColorStateList;
import android.graphics.Color;
import android.graphics.Typeface;
import android.graphics.drawable.GradientDrawable;
import android.text.TextUtils;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.FrameLayout;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;
import androidx.camera.view.PreviewView;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;
import com.google.android.material.bottomsheet.BottomSheetDialog;
import com.google.android.material.button.MaterialButton;
import com.google.android.material.textfield.TextInputEditText;
import com.google.android.material.textfield.TextInputLayout;
import org.json.JSONArray;
import org.json.JSONObject;

/** Camera-only presentation. Camera, detector, persistence and agent state belong to ScanActivity. */
public final class InspectionCameraUi {
    public interface Listener {
        void onBack(); void onCapture(); void onImport(); void onReview();
        void onToggleCameraPause(); void onToggleTorch(); void onToggleZoom();
        void onToggleGuide(); void onToggleTaskLock(); void onCompleteTask(); void onSkipTask();
        void onCorrectContext(String correction); void onSelectTask(String mode,String taskId);
        default void onDismissSuggestion(String taskId){} default void onSuggestAgain(){} default void onCameraSettings(){}
    }
    public static final int INFO=0, WARNING=1, ERROR=2;
    private static final int BG=0xff111315, PANEL=0xff1d2024, INK=0xfff4f2ea, MUTED=0xffb9bfc6, YELLOW=0xffffd43b;
    public final LinearLayout root;
    public final PreviewView preview;
    /** Add the inference overlay here, after preview. Camera controls are in a separate layer above it. */
    public final FrameLayout cameraLayer, cameraViewport;
    public final Button backButton,optionsButton,guideButton,pauseButton,torchButton,zoomButton,importButton,captureButton,reviewButton;
    public final CheckBox detailed;
    private final Activity activity;
    private final int topHeight,toolHeight,footerHeight;
    private final boolean arabic,smart;
    private final Listener listener;
    private final LinearLayout body,summary,actionPanel,footer;
    public final TextView taskTitle,taskMode,nextInstruction,qualityText,statusBanner,pausedLabel;
    private final ScrollView summaryScroll;
    private TextView sheetGuidance,sheetReason,sheetTask,sheetTaskMode,sheetGuideState,sheetScene,sheetStatus,sheetTechnical;
    private LinearLayout sheetSuggestions;
    private TextView sheetChecksHeading;
    private final Button detailsButton;
    private BottomSheetDialog sheet;
    private LinearLayout sheetContent;
    private boolean landscape,ready,cameraPaused,torch,locked,focusLocked;
    private int availableHeight;
    private String guideState="ready",provider="",scene="unknown",task="",mode="manual",guidance="",reason="",quality="",status="",technical="";
    private int statusImportance=INFO,saved;
    private JSONArray suggestions=new JSONArray();
    public final Button suggestionHint;
    private JSONObject hint;
    private String selectedTaskId="";
    private boolean hintsEnabled=true;
    private float hintDownX,hintDownY;

    public InspectionCameraUi(Activity activity,boolean arabic,boolean smart,Listener listener){
        this.activity=activity;this.arabic=arabic;this.smart=smart;this.listener=listener;
        float extra=Math.max(0,activity.getResources().getConfiguration().fontScale-1);topHeight=Math.round(52+extra*28);toolHeight=Math.round(52+extra*30);footerHeight=extra>.3f?Math.round(146+extra*28):84;
        root=new LinearLayout(activity){@Override protected void onMeasure(int width,int height){int nextHeight=View.MeasureSpec.getSize(height);if(availableHeight!=nextHeight&&summaryScroll!=null)summaryScroll.forceLayout();availableHeight=nextHeight;super.onMeasure(width,height);}@Override protected void onSizeChanged(int w,int h,int ow,int oh){super.onSizeChanged(w,h,ow,oh);boolean horizontal=w>h&&w>dp(560);if(horizontal!=landscape){landscape=horizontal;post(InspectionCameraUi.this::reflow);}}};
        root.setOrientation(LinearLayout.VERTICAL);root.setBackgroundColor(BG);root.setLayoutDirection(arabic?View.LAYOUT_DIRECTION_RTL:View.LAYOUT_DIRECTION_LTR);
        ViewCompat.setOnApplyWindowInsetsListener(root,(view,insets)->{androidx.core.graphics.Insets bars=insets.getInsets(WindowInsetsCompat.Type.systemBars()|WindowInsetsCompat.Type.displayCutout());view.setPadding(bars.left,bars.top,bars.right,bars.bottom);return insets;});
        LinearLayout top=row();top.setPadding(dp(4),0,dp(4),0);top.setGravity(Gravity.CENTER_VERTICAL);
        MaterialButton back=button("",v->listener.onBack(),false);back.setIcon(androidx.core.content.ContextCompat.getDrawable(activity,R.drawable.ic_camera_back));back.setIconTint(ColorStateList.valueOf(INK));back.setIconSize(dp(24));back.setIconPadding(0);back.setIconGravity(MaterialButton.ICON_GRAVITY_TEXT_START);backButton=back;backButton.setContentDescription(t("Back","رجوع"));top.addView(backButton,new LinearLayout.LayoutParams(dp(48),dp(topHeight)));
        TextView brand=label("BURHAN",16,true);brand.setGravity(Gravity.CENTER_VERTICAL);top.addView(brand,new LinearLayout.LayoutParams(0,dp(topHeight),1));if(extra>.3f)brand.setVisibility(View.GONE);
        guideButton=button(t("Guide ready","المساعد جاهز"),v->showDetails(),false);guideButton.setTextSize(13);guideButton.setMaxLines(2);guideButton.setVisibility(smart?View.VISIBLE:View.GONE);top.addView(guideButton,extra>.3f?new LinearLayout.LayoutParams(0,dp(topHeight),1):new LinearLayout.LayoutParams(dp(arabic?120:112),dp(topHeight)));
        optionsButton=button("⋮",v->showDetails(),false);optionsButton.setTextSize(24);optionsButton.setContentDescription(t("Inspection options","خيارات الفحص"));top.addView(optionsButton,new LinearLayout.LayoutParams(dp(48),dp(topHeight)));root.addView(top);
        body=new LinearLayout(activity);root.addView(body,new LinearLayout.LayoutParams(-1,0,1));
        cameraViewport=new FrameLayout(activity);cameraViewport.setBackgroundColor(Color.BLACK);
        cameraLayer=new FrameLayout(activity);cameraViewport.addView(cameraLayer,new FrameLayout.LayoutParams(-1,-1));
        preview=new PreviewView(activity);preview.setImplementationMode(PreviewView.ImplementationMode.COMPATIBLE);preview.setScaleType(PreviewView.ScaleType.FIT_CENTER);cameraLayer.addView(preview,new FrameLayout.LayoutParams(-1,-1));
        pausedLabel=label(t("Camera paused","الكاميرا متوقفة"),14,true);pausedLabel.setGravity(Gravity.CENTER);pausedLabel.setPadding(dp(12),dp(5),dp(12),dp(5));pausedLabel.setBackground(shape(0xdd111315,0xff777f88,1,4));FrameLayout.LayoutParams pausedParams=new FrameLayout.LayoutParams(-2,-2,Gravity.TOP|Gravity.CENTER_HORIZONTAL);pausedParams.topMargin=dp(10);cameraViewport.addView(pausedLabel,pausedParams);pausedLabel.setVisibility(View.GONE);
        LinearLayout cameraTools=row();cameraTools.setGravity(Gravity.CENTER_VERTICAL);cameraTools.setPadding(dp(8),dp(4),dp(8),dp(4));cameraTools.setBackgroundColor(0x99000000);
        torchButton=button(t("Flash off","الفلاش مغلق"),v->listener.onToggleTorch(),false);zoomButton=button("1×",v->listener.onToggleZoom(),false);pauseButton=button(t("Pause camera","إيقاف الكاميرا"),v->listener.onToggleCameraPause(),false);
        if(extra>.3f){cameraTools.addView(torchButton,new LinearLayout.LayoutParams(0,dp(toolHeight),1.1f));cameraTools.addView(pauseButton,new LinearLayout.LayoutParams(0,dp(toolHeight),1.35f));cameraTools.addView(zoomButton,new LinearLayout.LayoutParams(0,dp(toolHeight),.55f));zoomButton.setPadding(dp(2),dp(6),dp(2),dp(6));}
        else{addWeighted(cameraTools,torchButton,toolHeight);addWeighted(cameraTools,pauseButton,toolHeight);addWeighted(cameraTools,zoomButton,toolHeight);}
        FrameLayout.LayoutParams toolsParams=new FrameLayout.LayoutParams(-1,-2,Gravity.BOTTOM);cameraViewport.addView(cameraTools,toolsParams);
        suggestionHint=button("",v->{if(hint!=null)listener.onSelectTask(hint.optString("mode","manual"),hint.optString("taskId",""));},false);suggestionHint.setTextSize(13);suggestionHint.setMaxLines(1);suggestionHint.setEllipsize(TextUtils.TruncateAt.END);suggestionHint.setBackgroundTintList(ColorStateList.valueOf(0xe61d2024));suggestionHint.setVisibility(View.GONE);
        FrameLayout.LayoutParams hintParams=new FrameLayout.LayoutParams(-2,dp(48),Gravity.BOTTOM|Gravity.CENTER_HORIZONTAL);hintParams.bottomMargin=dp(toolHeight+10);hintParams.leftMargin=dp(12);hintParams.rightMargin=dp(12);cameraViewport.addView(suggestionHint,hintParams);
        suggestionHint.setOnTouchListener((v,e)->{if(e.getAction()==android.view.MotionEvent.ACTION_DOWN){hintDownX=e.getX();hintDownY=e.getY();return true;}if(e.getAction()==android.view.MotionEvent.ACTION_UP){float dx=Math.abs(e.getX()-hintDownX),dy=Math.abs(e.getY()-hintDownY);if(dx>dp(42)&&dx>dy){dismissHint();return true;}if(dx<dp(12)&&dy<dp(12))v.performClick();return true;}return true;});
        ViewCompat.replaceAccessibilityAction(suggestionHint,androidx.core.view.accessibility.AccessibilityNodeInfoCompat.AccessibilityActionCompat.ACTION_DISMISS,t("Dismiss suggestion","إخفاء الاقتراح"),(v,args)->{dismissHint();return true;});
        actionPanel=new LinearLayout(activity);actionPanel.setOrientation(LinearLayout.VERTICAL);actionPanel.setBackgroundColor(BG);
        summary=new LinearLayout(activity);summary.setOrientation(LinearLayout.VERTICAL);summary.setPadding(dp(16),dp(9),dp(16),dp(3));
        LinearLayout taskHeading=row();taskHeading.setGravity(Gravity.CENTER_VERTICAL);LinearLayout taskNames=new LinearLayout(activity);taskNames.setOrientation(LinearLayout.VERTICAL);TextView eyebrow=label(t("CURRENT CHECK","الفحص الحالي"),12,true);eyebrow.setTextColor(MUTED);taskNames.addView(eyebrow);
        taskTitle=label("",17,true);taskTitle.setMaxLines(1);taskTitle.setEllipsize(TextUtils.TruncateAt.END);taskNames.addView(taskTitle);taskHeading.addView(taskNames,new LinearLayout.LayoutParams(0,-2,1));Button change=button(t("Change","تغيير"),v->showChecks(),false);change.setTextSize(13);change.setTextColor(YELLOW);change.setVisibility(smart?View.VISIBLE:View.GONE);taskHeading.addView(change,new LinearLayout.LayoutParams(dp(extra>.3f?108:80),dp(toolHeight)));summary.addView(taskHeading);
        taskMode=label("",12,false);taskMode.setTextColor(MUTED);taskMode.setMaxLines(1);taskMode.setEllipsize(TextUtils.TruncateAt.END);summary.addView(taskMode);
        LinearLayout detailRow=row();detailRow.setGravity(Gravity.CENTER_VERTICAL);nextInstruction=label("",15,true);nextInstruction.setMaxLines(2);nextInstruction.setEllipsize(TextUtils.TruncateAt.END);detailRow.addView(nextInstruction,new LinearLayout.LayoutParams(0,-2,1));detailsButton=button(t("Details","التفاصيل"),v->showDetails(),false);detailsButton.setTextSize(13);detailsButton.setTextColor(YELLOW);detailRow.addView(detailsButton,new LinearLayout.LayoutParams(dp(extra>.3f?108:80),dp(toolHeight)));summary.addView(detailRow);
        qualityText=label("",13,false);qualityText.setTextColor(YELLOW);qualityText.setMaxLines(1);qualityText.setEllipsize(TextUtils.TruncateAt.END);qualityText.setVisibility(View.GONE);summary.addView(qualityText);
        summaryScroll=new ScrollView(activity){@Override protected void onMeasure(int width,int height){if(!landscape&&availableHeight>0){float fontScale=activity.getResources().getConfiguration().fontScale;int warningHeight=statusImportance>INFO?3*(int)Math.ceil(13*fontScale*1.25)+14:0;int max=Math.max(dp(48),availableHeight-root.getPaddingTop()-root.getPaddingBottom()-dp(topHeight+footerHeight+180+warningHeight));height=View.MeasureSpec.makeMeasureSpec(max,View.MeasureSpec.AT_MOST);}super.onMeasure(width,height);}};summaryScroll.setFillViewport(false);summaryScroll.addView(summary);actionPanel.addView(summaryScroll,new LinearLayout.LayoutParams(-1,-2));
        statusBanner=label("",13,true);statusBanner.setPadding(dp(12),dp(7),dp(12),dp(7));statusBanner.setMaxLines(3);statusBanner.setVisibility(View.GONE);ViewCompat.setAccessibilityLiveRegion(statusBanner,ViewCompat.ACCESSIBILITY_LIVE_REGION_POLITE);actionPanel.addView(statusBanner);
        footer=row();footer.setGravity(Gravity.CENTER_VERTICAL);footer.setPadding(dp(8),dp(4),dp(8),dp(8));
        importButton=button(t("Import","استيراد"),v->listener.onImport(),false);reviewButton=button(t("Review · 0","مراجعة · ٠"),v->listener.onReview(),false);captureButton=button(extra>.3f?t("Capture evidence","التقاط دليل"):t("Capture\nevidence","التقاط\nدليل"),v->listener.onCapture(),true);captureButton.setTextSize(16);captureButton.setMaxLines(2);importButton.setPadding(dp(2),dp(6),dp(2),dp(6));reviewButton.setPadding(dp(2),dp(6),dp(2),dp(6));
        if(extra>.3f){footer.setOrientation(LinearLayout.VERTICAL);footer.addView(captureButton,new LinearLayout.LayoutParams(-1,dp(Math.round(72+extra*16))));LinearLayout secondary=row();LinearLayout.LayoutParams secondaryParams=new LinearLayout.LayoutParams(-1,-2);secondaryParams.topMargin=dp(6);footer.addView(secondary,secondaryParams);secondary.addView(importButton,new LinearLayout.LayoutParams(0,dp(Math.round(56+extra*12)),1));secondary.addView(reviewButton,new LinearLayout.LayoutParams(0,dp(Math.round(56+extra*12)),1));}
        else{LinearLayout.LayoutParams sideParams=new LinearLayout.LayoutParams(0,dp(72),1);footer.addView(importButton,sideParams);LinearLayout.LayoutParams shutterParams=new LinearLayout.LayoutParams(0,dp(72),1.7f);shutterParams.setMargins(dp(6),0,dp(6),0);footer.addView(captureButton,shutterParams);footer.addView(reviewButton,new LinearLayout.LayoutParams(0,dp(72),1));}actionPanel.addView(footer);
        detailed=new CheckBox(activity);detailed.setText(t("Detailed surface photo analysis (512)","تحليل صورة السطح بالتفصيل (٥١٢)"));detailed.setTextColor(INK);detailed.setButtonTintList(ColorStateList.valueOf(YELLOW));detailed.setMinHeight(dp(48));detailed.setVisibility(View.GONE);
        setTask("","manual");setGuidance("","");setReady(false);reflow();
    }
    public void reflow(){
        body.removeAllViews();body.setOrientation(landscape?LinearLayout.HORIZONTAL:LinearLayout.VERTICAL);
        if(landscape){body.addView(cameraViewport,new LinearLayout.LayoutParams(0,-1,1));int width=Math.min(dp(340),Math.max(dp(280),root.getWidth()*44/100));body.addView(actionPanel,new LinearLayout.LayoutParams(width,-1));summaryScroll.setLayoutParams(new LinearLayout.LayoutParams(-1,0,1));}
        else{body.addView(cameraViewport,new LinearLayout.LayoutParams(-1,0,1));body.addView(actionPanel,new LinearLayout.LayoutParams(-1,-2));summaryScroll.setLayoutParams(new LinearLayout.LayoutParams(-1,-2));}
    }
    public void onConfigurationChanged(){reflow();}
    public void setStatus(String value,int importance){boolean geometryChanged=statusImportance!=importance;status=value==null?"":value;statusImportance=importance;statusBanner.setText(status);statusBanner.setTextColor(importance==ERROR?0xffffc4ba:YELLOW);statusBanner.setBackgroundColor(importance==ERROR?0xff421e1b:0xff352e16);statusBanner.setVisibility(importance>INFO&&!status.isEmpty()?View.VISIBLE:View.GONE);if(geometryChanged)summaryScroll.requestLayout();refreshSheet();refreshHint();}
    public void setQuality(String value){quality=value==null?"":value;qualityText.setText(quality);qualityText.setVisibility(quality.isEmpty()?View.GONE:View.VISIBLE);qualityText.setContentDescription(quality);}
    public void setSavedCount(int value){saved=value;reviewButton.setText(t("Review · ","مراجعة · ")+value);reviewButton.setContentDescription(t("Review "+value+" saved evidence captures","مراجعة "+value+" أدلة محفوظة"));}
    public void setReady(boolean value){ready=value;captureButton.setEnabled(value);captureButton.setAlpha(value?1:.55f);captureButton.setContentDescription(value?t("Capture evidence","التقاط دليل"):t("Capture unavailable until a fresh camera frame or imported photo is ready","الالتقاط غير متاح حتى يصبح إطار الكاميرا أو الصورة المستوردة جاهزاً"));}
    public void setCameraPaused(boolean value){cameraPaused=value;pauseButton.setText(value?t("Resume camera","استئناف الكاميرا"):t("Pause camera","إيقاف الكاميرا"));pausedLabel.setText(value?t("Camera paused","الكاميرا متوقفة"):t("Focus locked · tap to release","التركيز مثبت · اضغط للتحرير"));pausedLabel.setVisibility(value||focusLocked?View.VISIBLE:View.GONE);}
    public void setFocusLocked(boolean value){focusLocked=value;setCameraPaused(cameraPaused);}
    public void setTorch(boolean value){torch=value;torchButton.setText(value?t("Flash on","الفلاش يعمل"):t("Flash off","الفلاش مغلق"));torchButton.setTextColor(value?YELLOW:INK);}
    public void setZoom(String value){zoomButton.setText(value);zoomButton.setContentDescription(t("Camera zoom ","تقريب الكاميرا ")+value);}
    public void setZoom(float value){setZoom(String.format(java.util.Locale.ROOT,"%s×",value==(int)value?Integer.toString((int)value):String.format(java.util.Locale.ROOT,"%.1f",value)));}
    public void setTechnicalDetails(String value){technical=value==null?"":value;if(sheetTechnical!=null)sheetTechnical.setText(technical);}
    public void setTask(String title,String detector){task=title==null?"":title;mode=detector==null?"manual":detector;taskTitle.setText(task.isEmpty()?(smart?t("Looking around","استكشاف المنطقة"):modeTitle(mode)):task);taskMode.setText(smart&&selectedTaskId.isEmpty()&&"manual".equals(mode)?t("Exploring · detector inactive","استكشاف · الكاشف غير نشط"):"research".equals(mode)?t("Experimental · candidate observations · confirm findings","تجريبي · ملاحظات مرشحة · أكّد النتائج"):"manual".equals(mode)?t("Manual evidence · you confirm findings","توثيق يدوي · أنت تؤكد النتائج"):t("On-device "+modeTitle(mode).toLowerCase()+" · review observations","كشف على الجهاز: "+modeTitle(mode)+" · راجع الملاحظات"));refreshSheet();}
    public void setGuidance(String value,String explanation){guidance=value==null?"":value;reason=explanation==null?"":explanation;String shortText=guidance.trim();if(shortText.isEmpty())shortText=smart?t("Point at the area you want to inspect.","وجّه الكاميرا إلى المنطقة المراد فحصها."):t("Frame the area, then capture evidence.","ضع المنطقة في الإطار ثم التقط دليلاً.");int end=shortText.indexOf('\n');if(end>0)shortText=shortText.substring(0,end);if(shortText.length()>180)shortText=shortText.substring(0,177)+"…";nextInstruction.setText(shortText);nextInstruction.setContentDescription(shortText+t(". Open Details for all instructions.",". افتح التفاصيل لعرض جميع التعليمات."));summaryScroll.scrollTo(0,0);refreshSheet();if(sheetContent!=null&&sheetContent.getParent() instanceof ScrollView)((ScrollView)sheetContent.getParent()).smoothScrollTo(0,0);}
    public void setSuggestions(JSONArray value){suggestions=value==null?new JSONArray():value;hint=suggestions.optJSONObject(0);renderSuggestions();refreshHint();}
    public void setHintsEnabled(boolean value){hintsEnabled=value;refreshHint();}
    private void dismissHint(){if(hint!=null)listener.onDismissSuggestion(hint.optString("taskId",""));hint=null;refreshHint();}
    private void refreshHint(){boolean visible=smart&&hintsEnabled&&hint!=null&&sheet==null&&statusImportance<WARNING;boolean entering=visible&&suggestionHint.getVisibility()!=View.VISIBLE;suggestionHint.setVisibility(visible?View.VISIBLE:View.GONE);if(visible){suggestionHint.setText(t("✦ Try: ","✦ جرّب: ")+hint.optString("title",modeTitle(hint.optString("mode")))+" ›");if(entering&&android.animation.ValueAnimator.areAnimatorsEnabled()){suggestionHint.setAlpha(0);suggestionHint.setTranslationY(dp(6));suggestionHint.animate().alpha(1).translationY(0).setDuration(160).start();}}}
    public void setSelectedTask(String id){selectedTaskId=id==null?"":id;}
    public void setTaskLocked(boolean value){locked=value;refreshSheet();}
    public void setSurfaceMode(boolean value){detailed.setVisibility(value?View.VISIBLE:View.GONE);}
    public void setScene(String value){scene=value==null?"unknown":value;if(sheetContent!=null&&sheetScene!=null){sheetScene.setText(t("Scene: ","المشهد: ")+sceneTitle());sheetScene.setVisibility("unknown".equals(scene)?View.GONE:View.VISIBLE);}}
    public void setProvider(String value){provider=value==null?"":value;refreshSheet();}
    public void setGuideState(String state){guideState=state==null?"unavailable":state;String value;switch(guideState){case "working":value=t("Guide thinking…","المساعد يفكّر…");break;case "paused":value=t("Guide paused","المساعد متوقف");break;case "unpaired":value=t("Guide unpaired","المساعد غير مرتبط");break;case "unavailable":value=t("Guide offline","المساعد دون اتصال");break;case "trace_pending":value=t("Guide active","المساعد يعمل");break;default:value=t("Guide ready","المساعد جاهز");}guideButton.setText(value);guideButton.setTextColor("unavailable".equals(guideState)||"unpaired".equals(guideState)?MUTED:YELLOW);guideButton.setContentDescription(value+t(". Open guide details and controls.",". افتح تفاصيل المساعد وعناصر التحكم."));refreshSheet();}
    public void captureSaved(){String message=t("Evidence saved. "+saved+" captures ready for review.","حُفظ الدليل. "+saved+" أدلة جاهزة للمراجعة.");if(CameraPreferences.enabled(activity,"captureHaptics"))captureButton.performHapticFeedback(android.view.HapticFeedbackConstants.VIRTUAL_KEY);reviewButton.announceForAccessibility(message);}
    /** Acknowledges a focus request, without claiming autofocus succeeded. Coordinates use preview bounds. */
    public void showFocus(float x,float y){View ring=new View(activity);ring.setBackground(shape(Color.TRANSPARENT,YELLOW,2,4));ring.setImportantForAccessibility(View.IMPORTANT_FOR_ACCESSIBILITY_NO);FrameLayout.LayoutParams params=new FrameLayout.LayoutParams(dp(56),dp(56));params.leftMargin=Math.round(Math.max(0,Math.min(cameraViewport.getWidth()-dp(56),x-dp(28))));params.topMargin=Math.round(Math.max(0,Math.min(cameraViewport.getHeight()-dp(56),y-dp(28))));cameraViewport.addView(ring,params);ring.postDelayed(()->cameraViewport.removeView(ring),800);}
    public void showDetails(){dismissSheets();sheet=new BottomSheetDialog(activity);final BottomSheetDialog opened=sheet;ScrollView scroll=new ScrollView(activity);scroll.setFillViewport(false);sheetContent=new LinearLayout(activity);sheetContent.setOrientation(LinearLayout.VERTICAL);sheetContent.setPadding(dp(20),dp(12),dp(20),dp(28));sheetContent.setBackgroundColor(PANEL);sheetContent.setLayoutDirection(arabic?View.LAYOUT_DIRECTION_RTL:View.LAYOUT_DIRECTION_LTR);scroll.addView(sheetContent);sheet.setContentView(scroll);sheet.setOnDismissListener(dialog->{if(sheet==opened){sheetContent=null;sheet=null;refreshHint();}});rebuildSheet();sheet.show();refreshHint();sheet.getBehavior().setState(com.google.android.material.bottomsheet.BottomSheetBehavior.STATE_EXPANDED);sheet.getBehavior().setSkipCollapsed(true);}
    public void showChecks(){showDetails();if(sheetChecksHeading!=null){final BottomSheetDialog opened=sheet;sheetChecksHeading.post(()->{if(sheet==opened&&sheetContent!=null&&sheetContent.getParent() instanceof ScrollView)((ScrollView)sheetContent.getParent()).smoothScrollTo(0,sheetChecksHeading.getTop());});}}
    public void dismissSheets(){if(sheet!=null)sheet.dismiss();sheet=null;sheetContent=null;refreshHint();}
    public void dispose(){dismissSheets();}
    private void refreshSheet(){if(sheetContent==null)return;if(sheetTask!=null)sheetTask.setText(task.isEmpty()?modeTitle(mode):task);if(sheetTaskMode!=null)sheetTaskMode.setText(taskMode.getText());if(sheetGuidance!=null)sheetGuidance.setText(guidance.isEmpty()?nextInstruction.getText():guidance);if(sheetReason!=null){sheetReason.setText(reason);sheetReason.setVisibility(reason.isEmpty()?View.GONE:View.VISIBLE);}if(sheetGuideState!=null)sheetGuideState.setText(guideExplanation());if(sheetStatus!=null)sheetStatus.setText(status);}
    private void rebuildSheet(){
        if(sheetContent==null)return;sheetContent.removeAllViews();TextView handle=label("━━━━",14,false);handle.setTextColor(MUTED);handle.setGravity(Gravity.CENTER);sheetContent.addView(handle);
        addSheetButton(t("Camera & Capture settings","إعدادات الكاميرا والالتقاط"),()->listener.onCameraSettings());sheetContent.addView(label(t("Inspection details","تفاصيل الفحص"),22,true));sheetTask=label(task.isEmpty()?modeTitle(mode):task,17,true);sheetContent.addView(sheetTask);sheetTaskMode=label(taskMode.getText().toString(),13,false);sheetContent.addView(sheetTaskMode);
        if(smart){sheetGuideState=label(guideExplanation(),13,false);sheetContent.addView(sheetGuideState);sheetScene=label(t("Scene: ","المشهد: ")+sceneTitle(),13,false);sheetScene.setVisibility("unknown".equals(scene)?View.GONE:View.VISIBLE);sheetContent.addView(sheetScene);}
        sheetContent.addView(label(t("Next step","الخطوة التالية"),14,true));sheetGuidance=label(guidance.isEmpty()?nextInstruction.getText().toString():guidance,16,false);sheetContent.addView(sheetGuidance);sheetContent.addView(label(t("Why this check","سبب اقتراح الفحص"),14,true));sheetReason=label(reason,14,false);sheetReason.setVisibility(reason.isEmpty()?View.GONE:View.VISIBLE);sheetContent.addView(sheetReason);if(!quality.isEmpty())sheetContent.addView(label(quality,14,true));
        sheetContent.addView(label(t("Camera status","حالة الكاميرا"),13,true));sheetStatus=label(status,13,false);sheetContent.addView(sheetStatus);sheetTechnical=label(technical,12,false);sheetTechnical.setTextColor(MUTED);sheetContent.addView(sheetTechnical);
        if(smart){
            if(locked){addSheetButton(t("Complete current check","إتمام الفحص الحالي"),()->listener.onCompleteTask());addSheetButton(t("Skip current check","تخطي الفحص الحالي"),()->listener.onSkipTask());}
            sheetSuggestions=new LinearLayout(activity);sheetSuggestions.setOrientation(LinearLayout.VERTICAL);sheetContent.addView(sheetSuggestions);renderSuggestions();
            sheetChecksHeading=label(t("Available checks","الفحوص المتاحة"),16,true);sheetContent.addView(sheetChecksHeading);sheetContent.addView(label(t("These checks work without the cloud guide. You review and confirm all observations.","تعمل هذه الفحوص دون المساعد السحابي. راجع جميع الملاحظات وأكّدها."),13,false));
            addAvailableCheck("manual","",t("Manual evidence","توثيق يدوي"),t("Photograph and document the area yourself.","صوّر المنطقة ووثّقها بنفسك."));
            addAvailableCheck("crack","visual-cracks",modeTitle("crack"),t("On-device crack detector · distinguish joints during review.","كاشف الشقوق على الجهاز · ميّز الفواصل عند المراجعة."));
            addAvailableCheck("surface","visual-surfaces",modeTitle("surface"),t("On-device detector · visible surface observations.","كاشف على الجهاز · ملاحظات حالة السطح الظاهرة."));
            addAvailableCheck("ppe","visual-ppe",modeTitle("ppe"),t("On-device detector · protective equipment observations.","كاشف على الجهاز · ملاحظات معدات الوقاية."));
            if(ResearchCatalogue.enabled(activity)){
                JSONArray expanded=ResearchCatalogue.tasks(activity);String family="";
                for(int i=0;i<expanded.length();i++){JSONObject check=expanded.optJSONObject(i);if(check==null)continue;
                    String nextFamily=check.optString("family","");if(!family.equals(nextFamily)){family=nextFamily;sheetContent.addView(label(ResearchCatalogue.familyTitle(activity,family,arabic),16,true));}
                    addAvailableCheck(check.optString("mode","manual"),check.optString("id"),arabic?check.optString("titleAr",check.optString("title")):check.optString("title"),t(check.optString("scope", "Manual evidence; confirm findings."),check.optString("scopeAr",check.optString("scope"))));
                }
            }
            sheetContent.addView(label(t("Guide controls","التحكم بالمساعد"),16,true));addSheetButton("paused".equals(guideState)?t("Resume guide","استئناف المساعد"):t("Pause guide","إيقاف المساعد مؤقتاً"),()->listener.onToggleGuide());sheetContent.addView(label(t("Pausing the guide stops new cloud snapshots. The camera and on-device detection continue.","إيقاف المساعد يمنع إرسال لقطات سحابية جديدة. تستمر الكاميرا والكشف على الجهاز."),13,false));if(locked)addSheetButton(t("Stop check · Explore","إيقاف الفحص · استكشاف"),()->listener.onToggleTaskLock());addSheetButton(t("Suggest again","اقترح مجدداً"),()->listener.onSuggestAgain());
            sheetContent.addView(label(t("Correct the guide","تصحيح معلومات المساعد"),16,true));TextInputLayout inputLayout=new TextInputLayout(activity);inputLayout.setHint(t("Describe what you are inspecting","صف ما تفحصه"));TextInputEditText input=new TextInputEditText(activity);input.setMinLines(2);input.setMaxLines(4);input.setTextColor(INK);input.setFilters(new android.text.InputFilter[]{new android.text.InputFilter.LengthFilter(500)});inputLayout.addView(input,new LinearLayout.LayoutParams(-1,-2));sheetContent.addView(inputLayout,new LinearLayout.LayoutParams(-1,-2));MaterialButton apply=button(t("Send correction","إرسال التصحيح"),v->{String correction=input.getText()==null?"":input.getText().toString().trim();if(correction.isEmpty()){inputLayout.setError(t("Describe the correction first","اكتب التصحيح أولاً"));return;}dismissSheets();listener.onCorrectContext(correction);},false);sheetContent.addView(apply,new LinearLayout.LayoutParams(-1,-2));
        }
        if(detailed.getParent() instanceof ViewGroup)((ViewGroup)detailed.getParent()).removeView(detailed);sheetContent.addView(detailed,new LinearLayout.LayoutParams(-1,-2));sheetContent.addView(label(t("Experimental visual observations. Review every finding before including it in a report.","ملاحظات بصرية تجريبية. راجع كل نتيجة قبل تضمينها في التقرير."),12,false));addSheetButton(t("Close details","إغلاق التفاصيل"),()->{});
    }
    private void addAvailableCheck(String detector,String taskId,String title,String description){MaterialButton choice=button(title,v->{dismissSheets();listener.onSelectTask(detector,taskId);},false);choice.setGravity(Gravity.START|Gravity.CENTER_VERTICAL);choice.setCheckable(true);choice.setChecked(locked&&taskId.equals(selectedTaskId));choice.setTextColor(YELLOW);choice.setMaxLines(3);sheetContent.addView(choice,new LinearLayout.LayoutParams(-1,-2));sheetContent.addView(label(description,13,false));}
    private void renderSuggestions(){if(sheetContent==null||sheetSuggestions==null)return;sheetSuggestions.removeAllViews();if(suggestions.length()==0){sheetSuggestions.setVisibility(View.GONE);return;}sheetSuggestions.setVisibility(View.VISIBLE);sheetSuggestions.addView(label(t("Other suggested checks","فحوص مقترحة أخرى"),16,true));for(int i=0;i<Math.min(6,suggestions.length());i++){JSONObject suggestion=suggestions.optJSONObject(i);if(suggestion==null)continue;final String detector=suggestion.optString("mode","manual"),id=suggestion.optString("taskId","");String title=suggestion.optString("title",modeTitle(detector));MaterialButton choice=button(title,v->{dismissSheets();listener.onSelectTask(detector,id);},false);choice.setGravity(Gravity.START|Gravity.CENTER_VERTICAL);choice.setTextColor(YELLOW);choice.setMaxLines(3);sheetSuggestions.addView(choice,new LinearLayout.LayoutParams(-1,-2));String explanation=suggestion.optString("reason","");if(!explanation.isEmpty())sheetSuggestions.addView(label(explanation,13,false));sheetSuggestions.addView(label("manual".equals(detector)?t("Manual evidence","توثيق يدوي"):t("Installed on-device detector","كاشف مثبت يعمل على الجهاز"),12,false));}}
    private String sceneTitle(){switch(scene){case "indoor":return t("Indoors","داخلي");case "outdoor":return t("Outdoors","خارجي");case "transitional":return t("Indoor / outdoor boundary","بين الداخل والخارج");default:return t("Not determined","غير محدد");}}
    private String guideExplanation(){switch(guideState){case "working":return t("Selected snapshot is being interpreted online. Camera analysis stays on this phone.","تُفسَّر اللقطة المختارة عبر الإنترنت. يستمر تحليل الكاميرا على هذا الهاتف.");case "paused":return t("Guide paused. No new cloud snapshots are sent.","المساعد متوقف. لا تُرسل لقطات سحابية جديدة.");case "unpaired":return t("Pair in Smart Inspection settings to enable guidance. Offline evidence capture remains available.","اربط الجهاز في إعدادات الفحص الذكي لتفعيل الإرشاد. يبقى التقاط الأدلة دون اتصال متاحاً.");case "unavailable":return t("Remote guide unavailable. Offline detection and saved guidance continue.","المساعد السحابي غير متاح. يستمر الكشف دون اتصال والإرشاد المحفوظ.");default:return t("Guide active"+(provider.isEmpty()?"":" · "+provider)+". Selected snapshots are sent online at most every 10 seconds.","المساعد يعمل"+(provider.isEmpty()?"":" · "+provider)+". تُرسل اللقطات المختارة عبر الإنترنت كل ١٠ ثوانٍ كحد أدنى.");}}
    private void addSheetButton(String value,Runnable action){MaterialButton control=button(value,v->{dismissSheets();action.run();},false);control.setBackgroundTintList(ColorStateList.valueOf(0xff2b3036));control.setGravity(Gravity.CENTER);control.setMaxLines(3);LinearLayout.LayoutParams params=new LinearLayout.LayoutParams(-1,-2);params.topMargin=dp(7);sheetContent.addView(control,params);}
    private String modeTitle(String detector){switch(detector){case "crack":return t("Cracks / joints","الشقوق / الفواصل");case "surface":return t("Surface condition","حالة السطح");case "ppe":return t("Protective equipment","معدات الوقاية");case "research":return t("Experimental inspection","فحص تجريبي");default:return t("Visual inspection","الفحص البصري");}}
    private LinearLayout row(){LinearLayout result=new LinearLayout(activity);result.setOrientation(LinearLayout.HORIZONTAL);return result;}
    private void addWeighted(LinearLayout layout,View child,int height){layout.addView(child,new LinearLayout.LayoutParams(0,dp(height),1));}
    private TextView label(String value,int size,boolean bold){TextView view=new TextView(activity);view.setText(value);view.setTextColor(INK);view.setTextSize(size);view.setIncludeFontPadding(false);view.setPadding(0,dp(4),0,dp(4));if(bold)view.setTypeface(null,Typeface.BOLD);return view;}
    private MaterialButton button(String value,View.OnClickListener click,boolean primary){MaterialButton button=new MaterialButton(activity);button.setText(value);button.setAllCaps(false);button.setTextSize(13);button.setMinHeight(dp(48));button.setMinimumHeight(dp(48));button.setMinWidth(0);button.setMinimumWidth(0);button.setInsetTop(0);button.setInsetBottom(0);button.setCornerRadius(dp(5));button.setStrokeWidth(0);button.setBackgroundTintList(ColorStateList.valueOf(primary?YELLOW:Color.TRANSPARENT));button.setTextColor(primary?BG:INK);button.setPadding(dp(8),dp(6),dp(8),dp(6));button.setOnClickListener(click);button.setMaxLines(2);return button;}
    private GradientDrawable shape(int fill,int stroke,int width,int radius){GradientDrawable drawable=new GradientDrawable();drawable.setColor(fill);drawable.setStroke(dp(width),stroke);drawable.setCornerRadius(dp(radius));return drawable;}
    private int dp(int value){return Math.round(value*activity.getResources().getDisplayMetrics().density);}
    private String t(String english,String translation){return arabic?translation:english;}
}
