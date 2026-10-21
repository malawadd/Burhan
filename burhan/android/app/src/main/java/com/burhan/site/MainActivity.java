package com.burhan.site;

import com.getcapacitor.BridgeActivity;

public class MainActivity extends BridgeActivity {
    @Override public void onCreate(android.os.Bundle savedInstanceState) {
        registerPlugin(BurhanNativePlugin.class);
        super.onCreate(savedInstanceState);
    }
}
