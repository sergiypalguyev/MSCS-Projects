package edu.gatech.seclass;

import org.junit.Test;

/**
 * Created by Astro on 11/2/2016.
 */

public class BuggyClassTestBC2 {
    public BuggyClass classInstance = new BuggyClass();
    @Test
    public void buggyMethod2BC2Test1() {
        classInstance.buggyMethod2(9,3); //if branch
    }
    @Test
    public void buggyMethod2BC2Test2() {
        classInstance.buggyMethod2(-1,1); //else if branch
    }
    @Test
    public void buggyMethod2BC2Test3() {
        classInstance.buggyMethod2(0,-1); //Hidden else branch where a=0
    }
}
