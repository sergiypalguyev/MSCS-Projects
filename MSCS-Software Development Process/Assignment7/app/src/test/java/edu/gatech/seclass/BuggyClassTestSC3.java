package edu.gatech.seclass;

import org.junit.Test;

/**
 * Created by Astro on 11/2/2016.
 */

public class BuggyClassTestSC3 {
    public BuggyClass classInstance = new BuggyClass();
    @Test
    public void buggyMethod3SC3Test1() {
        classInstance.buggyMethod3(1,3); //100% with fault
    }
}
