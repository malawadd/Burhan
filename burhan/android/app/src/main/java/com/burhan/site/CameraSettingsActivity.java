package com.burhan.site;

import android.app.Activity;
import android.graphics.Color;
import android.os.Bundle;
import android.view.View;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;
import android.widget.Toast;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;
import com.google.android.material.button.MaterialButton;
import com.google.android.material.materialswitch.MaterialSwitch;
import org.json.JSONArray;
import org.json.JSONObject;

/** ScanActivity pauses through its ordinary lifecycle during this settings round trip. */
public final class CameraSettingsActivity extends Activity {
    private boolean arabic;
    @Override public void onCreate(Bundle state){super.onCreate(state);arabic="ar".equals(getIntent().getStringExtra("language"));
        LinearLayout root=new LinearLayout(this);root.setOrientation(LinearLayout.VERTICAL);root.setBackgroundColor(0xff111315);root.setLayoutDirection(arabic?View.LAYOUT_DIRECTION_RTL:View.LAYOUT_DIRECTION_LTR);
        ViewCompat.setOnApplyWindowInsetsListener(root,(view,insets)->{androidx.core.graphics.Insets bars=insets.getInsets(WindowInsetsCompat.Type.systemBars()|WindowInsetsCompat.Type.displayCutout());view.setPadding(bars.left,bars.top,bars.right,bars.bottom);return insets;});
        MaterialButton back=new MaterialButton(this);back.setText(arabic?"تم — الكاميرا والالتقاط":"Done — Camera & Capture");back.setMinHeight(dp(56));back.setOnClickListener(v->finish());root.addView(back,new LinearLayout.LayoutParams(-1,-2));
        ScrollView scroll=new ScrollView(this);LinearLayout content=new LinearLayout(this);content.setOrientation(LinearLayout.VERTICAL);content.setPadding(dp(16),dp(12),dp(16),dp(24));scroll.addView(content);root.addView(scroll,new LinearLayout.LayoutParams(-1,0,1));setContentView(root);
        try{JSONObject values=CameraPreferences.read(this);JSONArray definitions=CameraPreferences.definitions(this).getJSONArray("settings");
            for(String group:new String[]{"assistance","capture","framing","review"}){
                TextView title=new TextView(this);title.setText(groupTitle(group));title.setTextColor(0xffffd43b);title.setTextSize(19);title.setPadding(0,dp(20),0,dp(8));content.addView(title);
                for(int i=0;i<definitions.length();i++){JSONObject definition=definitions.getJSONObject(i);if(!group.equals(definition.getString("group")))continue;String key=definition.getString("key");
                    MaterialSwitch toggle=new MaterialSwitch(this);toggle.setText(definition.getJSONObject("label").getString(arabic?"ar":"en"));toggle.setTextColor(Color.WHITE);toggle.setTextSize(16);toggle.setMinHeight(dp(56));toggle.setChecked(values.getBoolean(key));toggle.setPadding(0,dp(8),0,dp(8));content.addView(toggle,new LinearLayout.LayoutParams(-1,-2));
                    TextView description=new TextView(this);description.setText(definition.getJSONObject("description").getString(arabic?"ar":"en"));description.setTextColor(0xffb9bfc6);description.setTextSize(14);description.setPadding(0,0,0,dp(12));content.addView(description);
                    toggle.setOnCheckedChangeListener((button,checked)->{try{CameraPreferences.write(this,new JSONObject().put(key,checked));}catch(Exception error){Toast.makeText(this,arabic?"تعذر حفظ الإعداد":"Could not save setting",Toast.LENGTH_LONG).show();}});
                }
            }
        }catch(Exception error){Toast.makeText(this,arabic?"تعذر تحميل الإعدادات":"Could not load camera settings",Toast.LENGTH_LONG).show();finish();}
    }
    private String groupTitle(String group){switch(group){case "assistance":return arabic?"المساعدة":"Assistance";case "capture":return arabic?"الالتقاط":"Capture";case "framing":return arabic?"التكوين":"Framing";default:return arabic?"المراجعة والمشاركة":"Review & sharing";}}
    private int dp(int value){return Math.round(value*getResources().getDisplayMetrics().density);}
}
