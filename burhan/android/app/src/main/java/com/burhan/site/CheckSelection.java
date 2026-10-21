package com.burhan.site;

import java.util.*;

/** Stable-scene decisions are independent of request versions and camera mode. */
final class CheckSelection {
    String selected="";
    private int scene=1;
    private final Set<String> finished=new LinkedHashSet<>();
    private final Map<String,Dismissal> dismissed=new LinkedHashMap<>();
    private static final class Dismissal {int scene;long until;Dismissal(int s,long u){scene=s;until=u;}}
    void scene(int value){if(scene!=value){scene=value;finished.clear();}}
    boolean suppressed(String id,long now){Dismissal d=dismissed.get(id);return finished.contains(id)||(d!=null&&(d.scene==scene||now<d.until));}
    void dismiss(String id,long now){dismissed.put(id,new Dismissal(scene,now+60000));}
    void finish(String id){if(!id.isEmpty())finished.add(id);selected="";}
    void suggestAgain(){dismissed.clear();}
    Set<String> dismissedIds(long now){Set<String> ids=new LinkedHashSet<>();for(String id:dismissed.keySet())if(suppressed(id,now))ids.add(id);return ids;}
}
