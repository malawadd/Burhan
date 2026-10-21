package com.burhan.site;
import android.graphics.*;
import android.view.*;
import org.json.*;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.robolectric.RobolectricTestRunner;
import org.robolectric.RuntimeEnvironment;
import org.robolectric.annotation.*;
import static org.junit.Assert.*;
@RunWith(RobolectricTestRunner.class) @Config(sdk=35) @GraphicsMode(GraphicsMode.Mode.NATIVE)
public class PhotoMarkupTest {
    @Test public void drawingNormalizesLetterboxingAndLeavesDetectorBaselineUnchanged() throws Exception {
        Bitmap baseline=Bitmap.createBitmap(100,100,Bitmap.Config.ARGB_8888);baseline.eraseColor(Color.BLACK);CaptureReview.MarkupView view=new CaptureReview.MarkupView(RuntimeEnvironment.getApplication(),baseline,null);view.layout(0,0,200,100);
        view.onTouchEvent(MotionEvent.obtain(0,0,MotionEvent.ACTION_DOWN,60,10,0));view.onTouchEvent(MotionEvent.obtain(0,10,MotionEvent.ACTION_UP,140,90,0));
        JSONArray points=view.shapes.getJSONObject(0).getJSONArray("points");assertEquals(.1,points.getDouble(0),.001);assertEquals(.9,points.getDouble(2),.001);Bitmap derivative=view.renderImage();assertEquals(Color.BLACK,baseline.getPixel(50,50));assertNotEquals(Color.BLACK,derivative.getPixel(50,50));assertEquals(100,derivative.getWidth());derivative.recycle();view.release();
    }
}
