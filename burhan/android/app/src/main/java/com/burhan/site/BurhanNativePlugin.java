package com.burhan.site;

import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.net.Uri;
import android.util.Base64;
import android.print.PrintAttributes;
import android.print.PrintManager;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import androidx.activity.result.ActivityResult;
import androidx.core.content.FileProvider;
import com.getcapacitor.*;
import com.getcapacitor.annotation.*;
import org.json.*;
import java.util.*;
import java.util.concurrent.*;
import java.io.*;

@CapacitorPlugin(name = "BurhanNative")
public class BurhanNativePlugin extends Plugin {
    private final ExecutorService io = Executors.newSingleThreadExecutor();
    private WebView printView;
    private volatile boolean scanning;
    private final Map<String,File> backupFiles=new ConcurrentHashMap<>();
    @PluginMethod public void beginBackup(PluginCall call){
        String filename=call.getString("filename","Burhan-backup.zip");if(!filename.matches("[A-Za-z0-9._-]+\\.zip")){call.reject("Valid ZIP filename required");return;}
        io.execute(()->{try{File directory=new File(getContext().getCacheDir(),"shared-backups");if(!directory.exists()&&!directory.mkdirs())throw new IOException("Backup folder unavailable");String id=UUID.randomUUID().toString();File archive=new File(directory,id+"-"+filename);if(!archive.createNewFile())throw new IOException("Cannot start backup archive");backupFiles.put(id,archive);JSObject result=new JSObject();result.put("id",id);call.resolve(result);}catch(Exception e){call.reject("Cannot start backup",e);}});
    }
    @PluginMethod public void appendBackup(PluginCall call){
        String id=call.getString("id"),base64=call.getString("base64");if(id==null||base64==null||base64.length()>800000){call.reject("Valid backup chunk required (512 KB maximum)");return;}
        io.execute(()->{try{File archive=backupFiles.get(id);if(archive==null)throw new IOException("Backup session expired");byte[] bytes=Base64.decode(base64,Base64.DEFAULT);if(archive.length()+bytes.length>300L*1024*1024)throw new IOException("Backup exceeds 300 MB");try(FileOutputStream output=new FileOutputStream(archive,true)){output.write(bytes);}call.resolve();}catch(Exception e){call.reject("Cannot append backup",e);}});
    }
    @PluginMethod public void finishBackup(PluginCall call){
        String id=call.getString("id");if(id==null){call.reject("Backup session required");return;}
        io.execute(()->{try{File archive=backupFiles.remove(id);if(archive==null||archive.length()==0)throw new IOException("Backup archive not found");try(FileOutputStream output=new FileOutputStream(archive,true)){output.getFD().sync();}shareArchive(archive,call);}catch(Exception e){call.reject("Cannot finish backup",e);}});
    }
    @PluginMethod public void cancelBackup(PluginCall call){
        String id=call.getString("id");if(id==null){call.resolve();return;}io.execute(()->{File archive=backupFiles.remove(id);if(archive!=null)archive.delete();call.resolve();});
    }
    private void shareArchive(File archive,PluginCall call){
        Uri uri=FileProvider.getUriForFile(getContext(),getContext().getPackageName()+".fileprovider",archive);
        Intent share=new Intent(Intent.ACTION_SEND).setType("application/zip").putExtra(Intent.EXTRA_STREAM,uri).addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);share.setClipData(android.content.ClipData.newRawUri(archive.getName(),uri));
        getActivity().runOnUiThread(()->{try{getActivity().startActivity(Intent.createChooser(share,"Save Burhan backup"));call.resolve();}catch(Exception e){call.reject("Cannot open backup sharing",e);}});
    }
    @PluginMethod public void exportBackup(PluginCall call) {
        String base64=call.getString("base64");String filename=call.getString("filename","Burhan-backup.zip");
        if(base64==null||!filename.matches("[A-Za-z0-9._-]+\\.zip")){call.reject("Valid backup archive required");return;}
        io.execute(()->{try {
            File directory=new File(getContext().getCacheDir(),"shared-backups");if(!directory.exists()&&!directory.mkdirs())throw new IOException("Backup folder unavailable");
            File archive=new File(directory,filename);byte[] bytes=Base64.decode(base64,Base64.DEFAULT);
            try(FileOutputStream output=new FileOutputStream(archive)){output.write(bytes);output.getFD().sync();}
            shareArchive(archive,call);
        }catch(Exception e){call.reject("Cannot export backup",e);} });
    }
    @PluginMethod public void loadState(PluginCall call) {
        io.execute(() -> { try { JSObject result = new JSObject(); result.put("json", BurhanDatabase.get(getContext()).store().state()); call.resolve(result); } catch(Exception e) {call.reject("Cannot load saved inspections",e);} });
    }
    @PluginMethod public void saveState(PluginCall call) {
        String json=call.getString("json");
        if(json==null) {call.reject("State JSON required");return;}
        io.execute(() -> {try {
            JSONObject parsed=new JSONObject(json);
            if(parsed.optInt("schemaVersion")!=1) throw new IllegalArgumentException("Unsupported state version");
            BurhanDatabase.get(getContext()).store().replaceState(json);call.resolve();
        }catch(Exception e){call.reject("Cannot save inspection state",e);} });
    }
    @PluginMethod public void getPendingCaptures(PluginCall call) {
        io.execute(() -> {try {call.resolve(captures(call.getString("inspectionId")));}catch(Exception e){call.reject("Cannot recover evidence",e);} });
    }
    @PluginMethod public void acknowledgeCaptures(PluginCall call) {
        io.execute(() -> {try {List<String> ids=new ArrayList<>();JSArray array=call.getArray("ids",new JSArray());for(int i=0;i<array.length();i++)ids.add(array.getString(i));BurhanDatabase.get(getContext()).store().acknowledge(ids);call.resolve();}catch(Exception e){call.reject("Cannot acknowledge evidence",e);} });
    }
    private JSObject captures(String inspection) throws Exception {
        JSArray array=new JSArray();
        List<BurhanDatabase.CaptureRow> rows=inspection==null?BurhanDatabase.get(getContext()).store().pending():BurhanDatabase.get(getContext()).store().pending(inspection);
        for(BurhanDatabase.CaptureRow row:rows) array.put(new JSONObject(row.json));
        JSObject result=new JSObject();result.put("captures",array);return result;
    }
    @PluginMethod public void scan(PluginCall call) {
        if(scanning){call.reject("Camera already open");return;}
        String id=call.getString("inspectionId");String mode=call.getString("mode","manual");
        if(id==null || !Arrays.asList("manual","ppe","crack","surface").contains(mode)){call.reject("Valid inspection and mode required");return;}
        scanning=true;
        Intent intent=new Intent(getActivity(), ScanActivity.class);
        intent.putExtra("inspectionId",id);intent.putExtra("mode",mode);intent.putExtra("autoSave",call.getBoolean("autoSave",true));intent.putExtra("language",call.getString("language","en"));intent.putExtra("followUp",call.getBoolean("followUp",false));intent.putExtra("findingId",call.getString("findingId"));
        startActivityForResult(call,intent,"scanFinished");
    }
    @ActivityCallback private void scanFinished(PluginCall call, ActivityResult result) {
        scanning=false;if(call==null)return;
        io.execute(() -> {try {call.resolve(captures(call.getString("inspectionId")));}catch(Exception e){call.reject("Saved evidence remains recoverable; reload inspection",e);} });
    }
    /** Uses the platform print dialog, supporting Save as PDF, Arabic shaping and paging. */
    @PluginMethod public void printReport(PluginCall call) {
        String html=call.getString("html");if(html==null){call.reject("Report HTML required");return;}
        getActivity().runOnUiThread(() -> {
            if(printView!=null)printView.destroy();
            printView=new WebView(getActivity());printView.getSettings().setJavaScriptEnabled(false);
            printView.setWebViewClient(new WebViewClient(){@Override public void onPageFinished(WebView view,String url){
                try {PrintManager manager=(PrintManager)getActivity().getSystemService(Context.PRINT_SERVICE);String name=call.getString("jobName","Burhan inspection");manager.print(name,view.createPrintDocumentAdapter(name),new PrintAttributes.Builder().setMediaSize(PrintAttributes.MediaSize.ISO_A4).setColorMode(PrintAttributes.COLOR_MODE_COLOR).build());call.resolve();}
                catch(Exception e){call.reject("Could not open PDF printing",e);}
            }});
            printView.loadDataWithBaseURL("https://localhost/",html,"text/html","UTF-8",null);
        });
    }
    @Override protected void handleOnDestroy(){io.shutdown();if(printView!=null)printView.destroy();super.handleOnDestroy();}
}
