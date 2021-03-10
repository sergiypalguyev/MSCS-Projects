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
    public class Tips
    {
        public static final double FIFTEEN = 0.15;
        public static final double TWENTY = 0.20;
        public static final double TWENTYFIVE = 0.25;

    }

    public void handleClick(View view)
    {
        EditText checkAmt = (EditText) findViewById(R.id.checkAmountValue);
        if (checkAmt.getText() == null)
        {
            Toast toast = Toast.makeText(getApplicationContext(), "Check amount is empty", Toast.LENGTH_SHORT);
            toast.show();
        }
        double checkAmount = Double.parseDouble(checkAmt.getText().toString());

        EditText partySz = (EditText) findViewById((R.id.partySizeValue));
        if (partySz.getText() == null)
        {
            Toast toast = Toast.makeText(getApplicationContext(), "Party size is empty", Toast.LENGTH_SHORT);
            toast.show();
        }
        int partySize = Integer.parseInt(partySz.getText().toString());

        double fifteenPercentTip = checkAmount * Tips.FIFTEEN;
        double twentyPercentTip = checkAmount * Tips.TWENTY;
        double twentyfivePercentTip = checkAmount * Tips.TWENTYFIVE;

        double fifteenPercentTotal = (checkAmount + fifteenPercentTip)/partySize;
        double twentyPercentTotal = (checkAmount + twentyPercentTip)/partySize;
        double twentyfivePercentTotal = (checkAmount + twentyfivePercentTip)/partySize;

        EditText fifteenTip = (EditText) findViewById(R.id.fifteenPercentTipValue);
        fifteenTip.setText((int) Math.round(fifteenPercentTip));

        EditText twentyTip = (EditText) findViewById(R.id.twentyPercentTipValue);
        fifteenTip.setText((int) Math.round(twentyPercentTip));

        EditText twentyfiveTip = (EditText) findViewById(R.id.twentyfivePercentTipValue);
        fifteenTip.setText((int) Math.round(twentyfivePercentTip));

        EditText fifteenTotal = (EditText) findViewById(R.id.fifteenPercentTotalValue);
        fifteenTip.setText((int) Math.round(fifteenPercentTotal));

        EditText twentyTotal = (EditText) findViewById(R.id.twentyPercentTotalValue);
        fifteenTip.setText((int) Math.round(twentyPercentTotal));

        EditText twentyfiveTotal = (EditText) findViewById(R.id.twentyfivePercentTotalValue);
        fifteenTip.setText((int) Math.round(twentyfivePercentTotal));
    }
}
