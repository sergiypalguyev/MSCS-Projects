/*
Author: Sergiy Palguyev
ID:     spalguyev3
Email:  sergiy.palguyev@gatch.edu
Due:    17 Sep 2016

Main class file
*/

package edu.gatech.seclass.tipcalculator;

import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.EditText;
import android.widget.Toast;

public class TipCalculatorActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_tip_calculator);
    }

    //Static collection of tip definitions
    public class Tips{
        public static final double FIFTEEN = 0.15;
        public static final double TWENTY = 0.20;
        public static final double TWENTYFIVE = 0.25;

        public static final double FIFTEEN_TOTAL = 1.15;
        public static final double TWENTY_TOTAL = 1.20;
        public static final double TWENTYFIVE_TOTAL = 1.25;
    }

    // Method to throw toast if check amount is invalid
    private double throwCheckAmountToast(EditText checkAmt){
        String sResponse = null;
        double checkAmount = 0;
        try{
            checkAmount = Double.parseDouble(checkAmt.getText().toString());
            if (checkAmount <= 0){sResponse = "Check amount must be larger than 0";}
        }
        catch(NullPointerException ex){
            sResponse = "Check amount is null.";
        }
        catch(NumberFormatException ex) {
            sResponse = "Check amount is not a number.";
        }
        catch(Exception ex){
            sResponse = "Unhandled Exception: " + ex.toString();
        }

        if(sResponse != null) {
            Toast toast = Toast.makeText(getApplicationContext(), sResponse, Toast.LENGTH_SHORT);
            toast.show();
        }

        return checkAmount;
    }

    // Method to throw toast if party size is invalid
    private int throwPartySizeToast(EditText partySz){
        String sResponse = null;
        int partySize = 0;
        try{
            partySize = Integer.parseInt(partySz.getText().toString());
            if (partySize <= 0){sResponse = "Party size must be larger than 0";}
        }
        catch(NullPointerException ex){
            sResponse = "Party size is null.";
        }
        catch(NumberFormatException ex) {
            sResponse = "Party size is not a number.";
        }
        catch(Exception ex){
            sResponse = "Unhandled Exception: " + ex.toString();
        }

        if(sResponse != null) {
            Toast toast = Toast.makeText(getApplicationContext(), sResponse, Toast.LENGTH_SHORT);
            toast.show();
        }

        return partySize;
    }

    // Method to update tips and values
    public void updateTipsAndTotals(int viewID, double setValue){
        String sResponse = null;
        int partySize = 0;
        try{
            EditText editText = (EditText) findViewById(viewID);
            editText.setText("$" + (Integer.toString((int) Math.round(setValue))));
        }
        catch(NullPointerException ex){
            sResponse = "Tip or total is null.";
        }
        catch(NumberFormatException ex) {
            sResponse = "Tip or total is not a number.";
        }
        catch(Exception ex){
            sResponse = "Unhandled Exception: " + ex.toString();
        }

        if(sResponse != null) {
            Toast toast = Toast.makeText(getApplicationContext(), sResponse, Toast.LENGTH_SHORT);
            toast.show();
        }
    }

    // Main method to handle "Calculate Tip" button click.
    public void handleClick(View view){

        //Get the check amount value. Throw toast if invalid.
        double checkAmount = throwCheckAmountToast((EditText) findViewById(R.id.checkAmountValue));

        //Get the party size value. Throw toast if invalid.
        int partySize = throwPartySizeToast((EditText) findViewById((R.id.partySizeValue)));

        //If inputs are ok, set the outputs.
        if(checkAmount > 0 && partySize > 0) {

            // Set the tip values
            updateTipsAndTotals(R.id.fifteenPercentTipValue,(checkAmount * Tips.FIFTEEN) / partySize );
            updateTipsAndTotals(R.id.twentyPercentTipValue, (checkAmount * Tips.TWENTY) / partySize);
            updateTipsAndTotals(R.id.twentyfivePercentTipValue, (checkAmount * Tips.TWENTYFIVE) / partySize);

            // Set the total values
            updateTipsAndTotals(R.id.fifteenPercentTotalValue, (checkAmount * Tips.FIFTEEN_TOTAL) / partySize);
            updateTipsAndTotals(R.id.twentyPercentTotalValue, (checkAmount * Tips.TWENTY_TOTAL) / partySize);
            updateTipsAndTotals(R.id.twentyfivePercentTotalValue, (checkAmount * Tips.TWENTYFIVE_TOTAL) / partySize);

        }
    }
}
