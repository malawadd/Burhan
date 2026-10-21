package com.burhan.site;
import org.junit.Test;
import static org.junit.Assert.*;
public class CheckSelectionTest {
    @Test public void dismissalRequiresSceneChangeAndSixtySeconds(){CheckSelection checks=new CheckSelection();checks.dismiss("crack",1000);assertTrue(checks.suppressed("crack",90000));checks.scene(2);assertTrue(checks.suppressed("crack",2000));assertFalse(checks.suppressed("crack",61000));}
    @Test public void completionIsSceneScopedAndDoesNotChangeAnotherSelection(){CheckSelection checks=new CheckSelection();checks.selected="crack";checks.finish("crack");assertEquals("",checks.selected);assertTrue(checks.suppressed("crack",100));checks.selected="ppe";checks.dismiss("crack",100);assertEquals("ppe",checks.selected);checks.scene(2);assertFalse(checks.suppressed("crack",60100));}
    @Test public void suggestAgainClearsDismissalsButKeepsCompletedChecksFiltered(){CheckSelection checks=new CheckSelection();checks.finish("crack");checks.dismiss("ppe",10);checks.suggestAgain();assertTrue(checks.suppressed("crack",11));assertFalse(checks.suppressed("ppe",11));}
}
