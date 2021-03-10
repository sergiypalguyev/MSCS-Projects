package edu.gatech.seclass;

import org.junit.Test;

/**
 * Created by Astro on 11/2/2016.
 */

public class BuggyClassTestSC2 {
    public BuggyClass classInstance = new BuggyClass();
    @Test
    public void buggyMethod2SC2Test1 () {
        classInstance.buggyMethod2(9,3); // First 50%
    }
    @Test
    public void buggyMethod2SC2Test2 () {
        classInstance.buggyMethod2(-1,-1);// Last 50%
    }
}
