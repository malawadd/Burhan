package com.burhan.site;

import android.content.Context;
import org.json.*;
import java.io.*;

/** Installed, opt-in capabilities; model identities are never accepted from agent output. */
final class ResearchCatalogue {
    static boolean enabled(Context c){return CameraPreferences.enabled(c,"expandedInspectionModels");}
    static JSONObject read(Context c,String path)throws Exception {
        try(InputStream in=c.getAssets().open(path);ByteArrayOutputStream out=new ByteArrayOutputStream()){
            byte[] buffer=new byte[8192];for(int n;(n=in.read(buffer))!=-1;)out.write(buffer,0,n);return new JSONObject(out.toString("UTF-8"));
        }
    }
    static JSONArray tasks(Context c){try{return read(c,"public/research-catalogue.json").getJSONArray("tasks");}catch(Exception e){return new JSONArray();}}
    static String familyTitle(Context c,String id,boolean arabic){try{JSONArray families=read(c,"public/research-catalogue.json").getJSONArray("families");for(int i=0;i<families.length();i++){JSONObject family=families.getJSONObject(i);if(id.equals(family.optString("id")))return family.optString(arabic?"titleAr":"title",id);}}catch(Exception ignored){}return id;}
    static JSONObject task(Context c,String id){JSONArray tasks=tasks(c);for(int i=0;i<tasks.length();i++){JSONObject task=tasks.optJSONObject(i);if(task!=null&&id.equals(task.optString("id")))return task;}return null;}
    static JSONObject capabilities(Context c){
        JSONObject result=new JSONObject();JSONArray installed=new JSONArray();
        try{
            try(InputStream in=c.getAssets().open("public/models/manifest.json");ByteArrayOutputStream out=new ByteArrayOutputStream()){
                byte[] b=new byte[8192];for(int n;(n=in.read(b))!=-1;)out.write(b,0,n);String text=out.toString("UTF-8");JSONArray specs=text.trim().startsWith("[")?new JSONArray(text):new JSONObject(text).getJSONArray("models");
                for(int i=0;i<specs.length();i++){JSONObject s=specs.getJSONObject(i);try(InputStream model=c.getAssets().open("public/models/"+s.getString("file"))){installed.put(s.getString("id"));}catch(IOException ignored){}}
            }
            result.put("expandedInspectionModels",enabled(c)).put("installedModelIds",installed);
        }catch(Exception ignored){}
        return result;
    }
}
