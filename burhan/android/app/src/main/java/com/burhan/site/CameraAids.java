package com.burhan.site;

import android.content.Context;
import android.graphics.*;
import android.hardware.*;
import android.view.View;

/** Orientation is the phone's roll, never a claim about a construction surface. */
final class CameraAids extends View implements SensorEventListener {
    private final SensorManager manager;private final boolean ar;private boolean grid,level;private float roll;
    CameraAids(Context context,boolean ar){super(context);this.ar=ar;manager=(SensorManager)context.getSystemService(Context.SENSOR_SERVICE);setImportantForAccessibility(IMPORTANT_FOR_ACCESSIBILITY_NO);setClickable(false);}
    void configure(boolean grid,boolean level){this.grid=grid;this.level=level;invalidate();}
    void start(){manager.unregisterListener(this);if(level){Sensor sensor=manager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);if(sensor!=null)manager.registerListener(this,sensor,SensorManager.SENSOR_DELAY_UI);}}
    void stop(){manager.unregisterListener(this);}
    @Override public void onSensorChanged(SensorEvent event){roll=(float)Math.toDegrees(Math.atan2(-event.values[0],event.values[1]));invalidate();}
    @Override public void onAccuracyChanged(Sensor sensor,int accuracy){}
    @Override protected void onDraw(Canvas canvas){Paint paint=new Paint(Paint.ANTI_ALIAS_FLAG);paint.setColor(0x99ffffff);paint.setStrokeWidth(1.5f);if(grid){for(int i=1;i<3;i++){canvas.drawLine(getWidth()*i/3f,0,getWidth()*i/3f,getHeight(),paint);canvas.drawLine(0,getHeight()*i/3f,getWidth(),getHeight()*i/3f,paint);}}if(level){paint.setColor(Math.abs(roll)<2?0xffb8e986:0xffffd43b);paint.setStrokeWidth(3);canvas.save();canvas.rotate(-roll,getWidth()/2f,getHeight()/2f);canvas.drawLine(getWidth()/2f-44,getHeight()/2f,getWidth()/2f+44,getHeight()/2f,paint);canvas.restore();paint.setTextSize(24);canvas.drawText(String.format(java.util.Locale.ROOT,ar?"الهاتف %.0f°":"Phone %.0f°",roll),12,32,paint);}}
}
