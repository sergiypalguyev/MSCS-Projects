package edu.gatech.seclass;

/**
 * Created by Sergiy Palguyev
 * sergiy.palguyev@gatech.edu
 * spalguyev3
 * Finished: 10 Sep,2016
 */
public class MyCustomString
        implements MyCustomStringInterface
{
    private String m_String = null;

    /**
     * Returns the current string. If the string is null, it should return null.
     */
    public String getString(){
        if (this.m_String == null) {
            return null;
        }
        else {
            return this.m_String;
        }
    }

    /**
     * Sets the value of the current string.
     */
    public void setString(String string){
        this.m_String = string;
    }

    /**
     * Returns the number of numbers in the current string, where a number is defined as a
     * contiguous sequence of digits.
     */
    public int countNumbers(){
        //for each letter in a string see if it is a number and count.

        if(m_String == null)
            throw new NullPointerException("countNumbers: encountered a null pointer.");

        int count = 0;
        if(!m_String.equals("")) {
            String tempString = m_String.replaceAll("[^0-9]+", " ");
            String[] strBits = tempString.trim().split(" ");
            count = strBits.length;
        }

        return count;
    }


    /**
     * Returns a string that consists of all and only the characters in positions n, 2n, 3n, and
     * so on in the current string, starting either from the beginning or from the end of the
     * string. The characters in the resulting string should be in the same order and with the
     * same case as in the current string.
     */
    public String getEveryNthCharacterFromBeginningOrEnd(int n, boolean startFromEnd){

        if(m_String == null)
            throw new NullPointerException("getEveryNthCharacterFromBeginningOrEnd: encountered a null pointer.");
        if(n<=0)
            throw new IllegalArgumentException("getEveryNthCharacterFromBeginningOrEnd: encountered an illegal argument");

        String tempString;

        if (m_String.length()<n){
            return "";
        }
        else if (!startFromEnd){

            tempString = charPicker(n,m_String);

            return tempString;
        }
        else if (startFromEnd){
            String revdString = new StringBuffer(m_String).reverse().toString();

            tempString = new StringBuilder(charPicker(n,revdString)).reverse().toString();

            return tempString;
        }
        else{
            return null;
        }
    }

    // Helper function to perform picking characters out of a string. The order is determined in calling function.
    private String charPicker(int n, String string){
        String tempString = "";
        int curChar = 0;

        while((curChar+=n) <= string.length()){
            tempString += string.charAt(curChar-1);
        }
        return tempString;
    }

    // Helper function to translate digits to their word representation.
    private String intToChar (int n){
        switch(n){
            case 0:return "Zero";
            case 1:return "One";
            case 2:return "Two";
            case 3:return "Three";
            case 4:return "Four";
            case 5:return "Five";
            case 6:return "Six";
            case 7:return "Seven";
            case 8:return "Eight";
            case 9:return "Nine";
            default: return null;
        }

    }

    /**
     * Replace the individual digits in the current string, between startPosition and endPosition
     * (included), with the corresponding English names of those digits, with the first letter
     * capitalized. The first character in the string is considered to be in Position 1.
     * Unlike for the previous method, digits are converted individually, even if contiguous.
     */
    public void convertDigitsToNamesInSubstring(int startPosition, int endPosition) {

        if(endPosition < startPosition)
            throw new  IllegalArgumentException("convertDigitsToNamesInSubstring: encountered an illegal argument");
        if(m_String != null && startPosition < endPosition &&  (startPosition <= 0 || m_String.length() < endPosition))
            throw new MyIndexOutOfBoundsException("convertDigitsToNamesInSubstring: encountered an index out of bounds");
        if(startPosition < endPosition && 0< startPosition && 0 < endPosition && m_String == null)
            throw new NullPointerException("convertDigitsToNamesInSubstring: encountered a null pointer");

        startPosition -= 1;
        String tempString = "";
        for (int i = startPosition; i < endPosition; i++) {
            char tempChar = m_String.charAt(i);

            if (Character.isDigit(tempChar)) {
                tempString += intToChar(Character.getNumericValue(tempChar));
            }
            else{
                tempString += tempChar;
            }
        }

        m_String = new StringBuilder(m_String).replace(startPosition,endPosition,tempString).toString();
    }
}
