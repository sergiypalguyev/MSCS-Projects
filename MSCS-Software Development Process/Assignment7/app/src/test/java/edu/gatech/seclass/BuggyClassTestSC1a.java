package edu.gatech.seclass;

import org.junit.Test;

/**
 * Created by Astro on 11/1/2016.
 */

public class BuggyClassTestSC1a {

    public BuggyClass classInstance = new BuggyClass();
    @Test
    public void buggyMethod1SC1aTest1 () {
        classInstance.buggyMethod1(9,3);//First 50%
    }
    @Test
    public void buggyMethod1SC1aTest2 () {
        classInstance.buggyMethod1(-1,-1);//Last 50%
    }
}