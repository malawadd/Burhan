package com.burhan.site;

import org.junit.Test;
import static org.junit.Assert.*;

public class SmartTurnGateTest {
    @Test public void invalidationImmediatelyAllowsLatestAndOldCompletionCannotClearIt(){
        SmartTurnGate gate=new SmartTurnGate();int old=gate.begin();assertTrue(gate.busy());assertEquals(-1,gate.begin());gate.invalidate();assertFalse(gate.busy());assertFalse(gate.current(old));int latest=gate.begin();assertTrue(gate.current(latest));gate.finish(old);assertTrue(gate.busy());gate.finish(latest);assertFalse(gate.busy());
    }
}
