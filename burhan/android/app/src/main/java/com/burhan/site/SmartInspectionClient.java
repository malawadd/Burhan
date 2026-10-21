package com.burhan.site;

import android.content.Context;
import android.util.AtomicFile;
import org.json.*;
import java.io.*;
import java.net.*;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.*;

/** All remote work is isolated from the camera/inference executor. */
final class SmartInspectionClient implements AutoCloseable {
    interface Listener {void status(String state);void plan(String turnId,int sceneVersion,String provider,JSONObject result);}
    private final ExecutorService network=Executors.newSingleThreadExecutor();
    private final SmartTurnGate gate=new SmartTurnGate();
    private volatile Future<?> inFlight;
    private final String url,token;private final JSONObject initial;private final Listener listener;
    private final AtomicFile journal;
    private JSONArray pending=new JSONArray();
    private volatile String sessionId,provider="";private volatile boolean closed,paused;private volatile HttpURLConnection connection;
    SmartInspectionClient(Context context,String url,String token,String id,JSONObject initial,Listener listener){this.url=SmartSettings.validUrl(url);this.token=token;sessionId=id;this.initial=initial;this.listener=listener;journal=new AtomicFile(new File(context.getNoBackupFilesDir(),"smart-tool-results.json"));
        network.execute(()->{try(InputStream input=journal.openRead();ByteArrayOutputStream out=new ByteArrayOutputStream()){byte[] buffer=new byte[8192];for(int n;(n=input.read(buffer))!=-1;){if(out.size()+n>1024*1024)throw new IOException("Journal too large");out.write(buffer,0,n);}pending=new JSONArray(out.toString("UTF-8"));}catch(Exception ignored){}flushAcknowledgements();});
    }
    String sessionId(){return sessionId;}String provider(){return provider;}
    boolean busy(){return gate.busy();}
    synchronized void invalidate(){gate.invalidate();Future<?> task=inFlight;if(task!=null)task.cancel(true);HttpURLConnection current=connection;if(current!=null)current.disconnect();}
    synchronized void pause(boolean value){paused=value;invalidate();if(value)listener.status("paused");}
    synchronized void submit(String jpeg,int version,JSONObject context){
        if(closed||paused)return;final int submittedEpoch=gate.begin();if(submittedEpoch<0)return;
        inFlight=network.submit(()->{try{
            listener.status("working");flushAcknowledgements();
            if(sessionId==null||sessionId.isEmpty()){JSONObject session=request("POST","/api/v1/sessions",initial);sessionId=session.getString("id");provider=session.optString("provider","");}
            if(closed||paused||!gate.current(submittedEpoch))return;
            JSONObject turn=request("POST","/api/v1/sessions/"+segment(sessionId)+"/turns",new JSONObject().put("image",jpeg).put("sceneVersion",version).put("context",context));String turnId=turn.getString("id");
            long deadline=android.os.SystemClock.elapsedRealtime()+180000;
            while(!closed&&!paused&&gate.current(submittedEpoch)&&android.os.SystemClock.elapsedRealtime()<deadline){
                JSONObject update=request("GET","/api/v1/sessions/"+segment(sessionId)+"/turns/"+segment(turnId),null);String status=update.getString("status");
                if(status.equals("completed")){provider=update.optString("provider",provider);if(gate.current(submittedEpoch)&&!paused&&!closed)listener.plan(turnId,update.getInt("sceneVersion"),provider,update.getJSONObject("result"));return;}
                if(status.equals("failed"))throw new IOException("Remote inspection unavailable");
                if(status.equals("cancelled")||status.equals("superseded"))return;
                Thread.sleep(1800);
            }
            if(!closed&&!paused)throw new IOException("Inspection planning timed out");
        }catch(Exception e){if(!closed&&!paused&&gate.current(submittedEpoch))listener.status("unavailable");}finally{gate.finish(submittedEpoch);}});
    }
    void event(String type,String task,int version,String turnId){if(closed)return;network.execute(()->{try{JSONObject body=new JSONObject().put("eventId",java.util.UUID.randomUUID().toString()).put("type",type).put("taskId",task).put("sceneVersion",version);if(turnId!=null)body.put("sourceTurnId",turnId);pending.put(new JSONObject().put("origin",url).put("sessionId",sessionId==null?"":sessionId).put("endpoint","events").put("body",body));saveAcknowledgements();flushAcknowledgements();}catch(Exception e){listener.status("trace_pending");}});}
    void acknowledge(String turnId,int version,JSONArray results){
        if(closed||sessionId==null)return;String session=sessionId;network.execute(()->{try{pending.put(new JSONObject().put("origin",url).put("sessionId",session).put("body",new JSONObject().put("turnId",turnId).put("sceneVersion",version).put("results",results)));while(pending.length()>120)pending.remove(0);saveAcknowledgements();flushAcknowledgements();}catch(Exception ignored){listener.status("trace_pending");}});
    }
    private void saveAcknowledgements(){FileOutputStream output=null;try{output=journal.startWrite();output.write(pending.toString().getBytes(StandardCharsets.UTF_8));journal.finishWrite(output);}catch(Exception e){if(output!=null)journal.failWrite(output);listener.status("trace_pending");}}
    private void flushAcknowledgements(){
        for(int i=0;i<pending.length()&&!paused;){JSONObject item=pending.optJSONObject(i);if(item==null||!url.equals(item.optString("origin"))||item.has("terminalStatus")){i++;continue;}try{String target=item.optString("sessionId");if(target.isEmpty()){if(sessionId==null){i++;continue;}target=sessionId;item.put("sessionId",target);}request("POST","/api/v1/sessions/"+segment(target)+"/"+item.optString("endpoint","tool-results"),item.getJSONObject("body"));pending.remove(i);saveAcknowledgements();}catch(HttpFailure e){if(e.status==404||e.status==409){try{item.put("terminalStatus","obsolete");saveAcknowledgements();}catch(Exception ignored){}i++;}else{listener.status("trace_pending");break;}}catch(Exception e){listener.status("trace_pending");break;}}
    }
    private static String segment(String value) throws UnsupportedEncodingException{return URLEncoder.encode(value,"UTF-8");}
    private static class HttpFailure extends IOException {final int status;HttpFailure(int status){super("Backend HTTP "+status);this.status=status;}}
    private JSONObject request(String method,String path,JSONObject body) throws Exception {
        if(Thread.currentThread().isInterrupted())throw new InterruptedIOException("Request cancelled");
        HttpURLConnection current=(HttpURLConnection)new URL(url+path).openConnection();connection=current;current.setRequestMethod(method);current.setInstanceFollowRedirects(false);current.setConnectTimeout(12000);current.setReadTimeout(20000);current.setRequestProperty("Authorization","Bearer "+token);current.setRequestProperty("Accept","application/json");current.setRequestProperty("User-Agent","Burhan/1.2.0 (Android; SmartInspection)");
        try{
            if(body!=null){current.setDoOutput(true);current.setRequestProperty("Content-Type","application/json");byte[] bytes=body.toString().getBytes(StandardCharsets.UTF_8);current.setFixedLengthStreamingMode(bytes.length);try(OutputStream out=current.getOutputStream()){out.write(bytes);}}
            int code=current.getResponseCode();if(code<200||code>=300)throw new HttpFailure(code);
            try(InputStream input=current.getInputStream();ByteArrayOutputStream output=new ByteArrayOutputStream()){byte[] buffer=new byte[8192];for(int n;(n=input.read(buffer))!=-1;){if(output.size()+n>1024*1024)throw new IOException("Backend response too large");output.write(buffer,0,n);}String text=output.toString("UTF-8");return text.isEmpty()?new JSONObject():new JSONObject(text);}
        }finally{current.disconnect();if(connection==current)connection=null;}
    }
    @Override public synchronized void close(){if(closed)return;closed=true;invalidate();
        network.execute(()->{flushAcknowledgements();if(sessionId!=null)try{request("DELETE","/api/v1/sessions/"+segment(sessionId),null);}catch(Exception ignored){}});network.shutdown();
    }
}
