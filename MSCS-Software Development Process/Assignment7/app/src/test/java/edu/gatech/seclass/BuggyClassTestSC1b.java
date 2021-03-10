package edu.gatech.seclass;

import org.junit.Test;

/**
 * Created by Astro on 11/1/2016.
 */

public class BuggyClassTestSC1b {

    public BuggyClass classInstance = new BuggyClass();

    @Test
    public void buggyMethod1SC1bTest1 (){
        classInstance.buggyMethod1(3,0);//First 50% Fail here
    }
}