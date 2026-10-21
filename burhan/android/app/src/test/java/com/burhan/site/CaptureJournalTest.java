package com.burhan.site;
import androidx.room.Room;
import org.junit.*;
import org.junit.runner.RunWith;
import org.robolectric.RobolectricTestRunner;
import org.robolectric.RuntimeEnvironment;
import org.robolectric.annotation.Config;
import static org.junit.Assert.*;
@RunWith(RobolectricTestRunner.class) @Config(sdk=35)
public class CaptureJournalTest {
    private BurhanDatabase database;
    @Before public void before(){database=Room.inMemoryDatabaseBuilder(RuntimeEnvironment.getApplication(),BurhanDatabase.class).allowMainThreadQueries().build();}
    @After public void after(){database.close();}
    private BurhanDatabase.CaptureRow row(String id){BurhanDatabase.CaptureRow row=new BurhanDatabase.CaptureRow();row.id=id;row.inspectionId="current";row.json="{\"id\":\""+id+"\"}";return row;}
    @Test public void retakeReplacesOnlyTargetAfterInsertionAndUndoTargetsExactId(){BurhanDatabase.Store store=database.store();store.capture(row("earlier"));store.capture(row("latest"));store.replaceCapture("latest",row("replacement"));assertNotNull(store.captureById("earlier"));assertNull(store.captureById("latest"));assertNotNull(store.captureById("replacement"));store.acknowledge(java.util.Collections.singletonList("replacement"));assertNotNull(store.captureById("earlier"));assertEquals(1,store.pending().size());}
    @Test public void failedRetakeRollsBackAndKeepsPreviousEvidence(){BurhanDatabase.Store store=database.store();store.capture(row("latest"));store.capture(row("collision"));try{store.replaceCapture("latest",row("collision"));fail("Conflicting insertion must fail atomically");}catch(RuntimeException expected){}assertNotNull(store.captureById("latest"));assertEquals(2,store.pending().size());}
    @Test public void anotherInspectionCannotReplaceExistingCapture(){BurhanDatabase.Store store=database.store();store.capture(row("latest"));BurhanDatabase.CaptureRow replacement=row("new");replacement.inspectionId="other";try{store.replaceCapture("latest",replacement);fail();}catch(IllegalArgumentException expected){}assertNotNull(store.captureById("latest"));assertNull(store.captureById("new"));}
}
