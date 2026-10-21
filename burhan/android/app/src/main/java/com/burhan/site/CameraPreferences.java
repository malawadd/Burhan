package com.burhan.site;

import android.content.Context;
import android.content.SharedPreferences;
import org.json.JSONArray;
import org.json.JSONObject;
import java.io.InputStream;
import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;

/** Per-device preferences using the same definitions as the browser settings page. */
public final class CameraPreferences {
    private CameraPreferences() {}
    private static volatile JSONObject definitionsCache;
    private static SharedPreferences store(Context context){return context.getApplicationContext().getSharedPreferences("burhan-camera-v1",Context.MODE_PRIVATE);}
    public static JSONObject definitions(Context context){
        if(definitionsCache!=null)return definitionsCache;
        try(InputStream in=context.getAssets().open("public/camera-settings.json");ByteArrayOutputStream out=new ByteArrayOutputStream()){byte[] buffer=new byte[4096];int count;while((count=in.read(buffer))!=-1)out.write(buffer,0,count);JSONObject definitions=new JSONObject(new String(out.toByteArray(),StandardCharsets.UTF_8));definitionsCache=definitions;return definitions;}
        catch(Exception error){throw new IllegalStateException("Camera setting definitions unavailable",error);}
    }
    public static synchronized JSONObject read(Context context){
        try{
            SharedPreferences saved=store(context);JSONObject result=new JSONObject();JSONArray definitions=definitions(context).getJSONArray("settings");
            for(int i=0;i<definitions.length();i++){JSONObject definition=definitions.getJSONObject(i);String key=definition.getString("key");result.put(key,saved.getBoolean(key,definition.getBoolean("default")));}
            result.put("lastZoom",Math.max(0.1f,Math.min(100f,saved.getFloat("lastZoom",1f))));return result;
        }catch(Exception error){throw new IllegalStateException("Cannot read camera preferences",error);}
    }
    public static synchronized JSONObject write(Context context,JSONObject updates){
        try{
            JSONObject current=read(context);SharedPreferences.Editor editor=store(context).edit();
            for(java.util.Iterator<String> keys=updates.keys();keys.hasNext();){String key=keys.next();Object value=updates.get(key);
                if(!current.has(key))throw new IllegalArgumentException("Unknown camera preference: "+key);
                if("lastZoom".equals(key)){if(!(value instanceof Number))throw new IllegalArgumentException("Invalid zoom");double zoom=((Number)value).doubleValue();if(!Double.isFinite(zoom)||zoom<0.1||zoom>100)throw new IllegalArgumentException("Invalid zoom");editor.putFloat(key,(float)zoom);}
                else{if(!(value instanceof Boolean))throw new IllegalArgumentException("Boolean camera preference required");editor.putBoolean(key,(Boolean)value);}
            }
            if(!editor.commit())throw new IllegalStateException("Camera preferences could not be stored");return read(context);
        }catch(RuntimeException error){throw error;}catch(Exception error){throw new IllegalArgumentException("Invalid camera preferences",error);}
    }
    public static boolean enabled(Context context,String key){
        JSONArray settings=definitions(context).optJSONArray("settings");boolean fallback=false;
        for(int i=0;settings!=null&&i<settings.length();i++){JSONObject definition=settings.optJSONObject(i);if(definition!=null&&key.equals(definition.optString("key"))){fallback=definition.optBoolean("default");break;}}
        return store(context).getBoolean(key,fallback);
    }
}
