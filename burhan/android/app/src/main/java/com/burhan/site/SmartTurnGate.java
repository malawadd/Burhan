package com.burhan.site;

/** A cancelled request cannot keep the next scene busy or clear its busy flag later. */
final class SmartTurnGate {
    private int generation;
    private boolean busy;
    synchronized int begin(){if(busy)return -1;busy=true;return generation;}
    synchronized void invalidate(){generation++;busy=false;}
    synchronized boolean current(int value){return value==generation;}
    synchronized boolean busy(){return busy;}
    synchronized void finish(int value){if(value==generation)busy=false;}
}
