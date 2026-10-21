package com.burhan.site;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Color;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.platform.app.InstrumentationRegistry;
import org.json.JSONObject;
import org.junit.Test;
import org.junit.runner.RunWith;
import static org.junit.Assert.*;

/** Actual OpenCV 5 / Android Keystore checks; run on an ARM64 API31+ device. */
@RunWith(AndroidJUnit4.class)
public class SmartInspectionTest {
    @Test public void sceneShiftRequiresThreeMeasurementsAndReportsExposure() throws Exception {
        SmartQuality quality=new SmartQuality();Bitmap image=Bitmap.createBitmap(320,240,Bitmap.Config.ARGB_8888);
        try{
            image.eraseColor(Color.BLACK);JSONObject first=quality.measure(image);assertTrue(first.getBoolean("underexposed"));assertTrue(first.getBoolean("blurred"));assertTrue(first.getString("opencvVersion").startsWith("5."));
            image.eraseColor(Color.WHITE);assertEquals(1,quality.measure(image).getInt("sceneVersion"));assertEquals(1,quality.measure(image).getInt("sceneVersion"));JSONObject third=quality.measure(image);assertEquals(2,third.getInt("sceneVersion"));assertTrue(third.getBoolean("overexposed"));assertEquals(2,quality.measure(image).getInt("sceneVersion"));
        }finally{image.recycle();}
    }
    @Test public void pairingTokenIsEncryptedAndHttpsOriginIsRequired() throws Exception {
        Context context=InstrumentationRegistry.getInstrumentation().getTargetContext();JSONObject previous=SmartSettings.get(context);String marker="test-pairing-"+java.util.UUID.randomUUID();
        try{
            SmartSettings.set(context,"https://merhaba-api.kiyan.finance/",marker);JSONObject loaded=SmartSettings.get(context);assertEquals(marker,loaded.getString("token"));assertEquals("https://merhaba-api.kiyan.finance",loaded.getString("url"));String stored=context.getSharedPreferences("smart-pairing",Context.MODE_PRIVATE).getString("token","");assertFalse(stored.contains(marker));assertTrue(stored.contains(":"));
            for(String bad:new String[]{"http://example.com","https://user:password@example.com","https://example.com/path","https://example.com?token=secret"}){try{SmartSettings.validUrl(bad);fail("Accepted invalid origin");}catch(IllegalArgumentException expected){}}
        }finally{SmartSettings.set(context,previous.getString("url"),previous.getString("token"));}
    }
}
