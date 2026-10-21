package com.burhan.site;

import android.content.Context;
import androidx.annotation.NonNull;
import androidx.room.*;
import java.util.List;

/** The state transaction and capture journal survive WebView/Activity process death. */
@Database(entities = {BurhanDatabase.StateRow.class, BurhanDatabase.CaptureRow.class}, version = 1, exportSchema = false)
public abstract class BurhanDatabase extends RoomDatabase {
    @Entity(tableName = "app_state") public static class StateRow {
        @PrimaryKey public int id = 1;
        @NonNull public String json = "{}";
    }
    @Entity(tableName = "capture_journal") public static class CaptureRow {
        @PrimaryKey @NonNull public String id = "";
        @NonNull public String inspectionId = "";
        @NonNull public String json = "{}";
        public long createdAt;
    }
    @Dao public interface Store {
        @Query("SELECT json FROM app_state ORDER BY id") List<String> stateChunks();
        @Query("DELETE FROM app_state") void clearState();
        @Insert(onConflict = OnConflictStrategy.REPLACE) void saveState(StateRow row);
        default String state(){List<String> chunks=stateChunks();return chunks.isEmpty()?null:String.join("",chunks);}
        @Transaction default void replaceState(String json){clearState();for(int offset=0,id=1;offset<json.length();id++){int end=Math.min(json.length(),offset+131072);if(end<json.length()&&Character.isHighSurrogate(json.charAt(end-1)))end--;StateRow row=new StateRow();row.id=id;row.json=json.substring(offset,end);saveState(row);offset=end;}}
        @Insert(onConflict = OnConflictStrategy.IGNORE) void capture(CaptureRow row);
        @Update void updateCapture(CaptureRow row);
        @Insert(onConflict = OnConflictStrategy.ABORT) void insertReplacement(CaptureRow row);
        @Query("SELECT * FROM capture_journal WHERE id=:id") CaptureRow captureById(String id);
        @Transaction default void replaceCapture(String oldId,CaptureRow replacement){CaptureRow old=captureById(oldId);if(old==null||!old.inspectionId.equals(replacement.inspectionId))throw new IllegalArgumentException("Retake target is not in this inspection");insertReplacement(replacement);acknowledge(java.util.Collections.singletonList(oldId));}
        @Query("SELECT * FROM capture_journal ORDER BY createdAt") List<CaptureRow> pending();
        @Query("SELECT * FROM capture_journal WHERE inspectionId=:inspection ORDER BY createdAt") List<CaptureRow> pending(String inspection);
        @Query("DELETE FROM capture_journal WHERE id IN (:ids)") void acknowledge(List<String> ids);
    }
    public abstract Store store();
    private static volatile BurhanDatabase instance;
    public static BurhanDatabase get(Context context) {
        if(instance == null) synchronized(BurhanDatabase.class) {
            if(instance == null) instance = Room.databaseBuilder(context.getApplicationContext(), BurhanDatabase.class, "burhan.db").build();
        }
        return instance;
    }
}
