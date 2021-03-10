package edu.gatech.seclass;

import android.support.v7.app.AppCompatActivity;

public class BuggyClass extends AppCompatActivity {

    // one suite with 100% statement = no fault
    // one suite with 50% statement = fault
    public int buggyMethod1(int a, int b) {

        if (a > 0) {
            a = a / b;
            return a;
        } else if (b < 0) {
            b = b / a;
            return b;
        }
        return 0;
    }

    // one suite with 100% statement = no fault
    // every suite with >50% branch  = fault
    public void buggyMethod2(int a, int b) {

        if (a>0) {
            a += 1; // Arbitrary arithmetic
        }else if (a<0) {
            a -= 1; // Arbitrary arithmetic
        }

        b = b/a;
    }

    // one suite with 100% branch = no fault
    // one suite 100% statement, not 100% branch = fault
    public void buggyMethod3(int a, int b){

        if (a>0) {
            a -= 1; // Arbitrary arithmetic
        }

        b = b/a;
    }

    // every suite with 100% statement = fault
    // one suite with 100% branch = no fault
    public void buggyMethod4(){
        // This is impossible.
        // If 100% statement coverage fails for every statement suite.
        // then there is at least one branch which will fail 100% of the time.
    }

    // one suite with 100% statement = no fault
    // division by zero is not revealed
    public void buggyMethod5(){
        // Impossible
        // If 100% statement coverage is expected, the divide by zero statement will be executed.
        // Only possibility to execute the line of code without fault, is to redefine the meaning of the division symbol.
        // Unfortunately division symbol reassignment is not allowed for this assignment.
    }
}