package edu.gatech.seclass;

import org.junit.Test;

/**
 * Created by Astro on 11/2/2016.
 */

public class BuggyClassTestBC3 {
    public BuggyClass classInstance = new BuggyClass();
    @Test
    public void buggyMethod3BC3Test1() {
        classInstance.buggyMethod3(9,3); //if branch
    }
    @Test
    public void buggyMethod3BC3Test2() {
        classInstance.buggyMethod3(-3,3); //Hidden else branch
    }

}
