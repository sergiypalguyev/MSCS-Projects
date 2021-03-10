package edu.gatech.seclass;

import org.junit.After;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;

import static org.junit.Assert.assertEquals;


public class MyCustomStringTest {

    // variable of MyCustomStringInterface type
    private MyCustomStringInterface mycustomstring;

    // Instance of MyCustomString class as MyCustomStringInterface type. Allowed since MyCustomString inherits from the interface.
    @Before
    public void setUp() {
        mycustomstring = new MyCustomString();
    }

    //Destructor when all tests are finished.
    @After
    public void tearDown() {
        mycustomstring = null;
    }

    // Class instantiation for expected exception used throughout the test cases.
    @Rule
    public ExpectedException thrown = ExpectedException.none();

    // This test checks whether numbers within words will be counted correctly by the method countNumbers()
    @Test
    public void testCountNumbers1() {
        mycustomstring.setString("I'd b3tt3r put s0me d161ts in this 5tr1n6, right?");
        assertEquals(7, mycustomstring.countNumbers());
    }

    //This method checks whether the method countNumbers() is able to count numbers across multi-line and tabbed strings.
    @Test
    public void testCountNumbers2() {

        mycustomstring.setString("I'd\nb3tt3r\tput\ns0me\td161ts\nin\tthis\n5tr1n6,\tright?");
        assertEquals(7, mycustomstring.countNumbers());
    }

    // This test checks whether an empty string correctly returns a value of 0 by the method countNumbers().
    @Test
    public void testCountNumbers3() {
        mycustomstring.setString("");
        assertEquals(0, mycustomstring.countNumbers());
    }

    // This method checks whether the method countNumbers() is able to count strings with extremely large numbers (i.e. 128-bit, 256-bit, etc.)
    @Test
    public void testCountNumbers4() {
        mycustomstring.setString("255 " +
                "65535 " +
                "4294967295 " +
                "18446744073709551615 " +
                "340282366920938463463374607431768211455 " +
                "115792089237316195423570985008687907853269984665640564039457584007913129639935");
        assertEquals(6, mycustomstring.countNumbers());
    }

    // This method checks whether the method countNumbers() is able to count numbers in a string with special characters.
    @Test
    public void testCountNumbers5() {
        mycustomstring.setString("!!! ?!?!?!?! !@#$ %^&*()~`*_ -1 2 345 8 -9 -3 -4234 432 ?!? !?!..~`== -+_!@ #$%^ &*( )/* ");
        assertEquals(8, mycustomstring.countNumbers());
    }

    // This test checks whether method countNumbers() suitably throws a NullPointerException if mycustomstring is null
    @Test
    public void testCountNumbers6() {
        mycustomstring.setString(null);

        thrown.expect(NullPointerException.class);
        thrown.expectMessage("countNumbers: encountered a null pointer.");

        mycustomstring.countNumbers();
    }

    // This test checks whether method getEveryNthCharacterFromBeginningOrEnd() can get et every 3rd character from beginning.
    @Test
    public void testGetEveryNthCharacterFromBeginningOrEnd1() {
        mycustomstring.setString("I'd b3tt3r put s0me d161ts in this 5tr1n6, right?");
        assertEquals("d33p md1  i51,it", mycustomstring.getEveryNthCharacterFromBeginningOrEnd(3, false));
    }

    // This test checks whether method getEveryNthCharacterFromBeginningOrEnd() can get et every 3rd character from end.
    @Test
    public void testGetEveryNthCharacterFromBeginningOrEnd2() {
        mycustomstring.setString("I'd b3tt3r put s0me d161ts in this 5tr1n6, right?");
        assertEquals("'bt t0 6snh r6rh", mycustomstring.getEveryNthCharacterFromBeginningOrEnd(3, true));
    }

    // This test checks whether method getEveryNthCharacterFromBeginningOrEnd() can get et every single character from beginning.
    // This test checks method getEveryNthCharacterFromBeginningOrEnd() variable n lower bound, from beginning.
    @Test
    public void testGetEveryNthCharacterFromBeginningOrEnd3() {
        mycustomstring.setString("I'd b3tt3r put s0me d161ts in this 5tr1n6, right?");
        assertEquals("I'd b3tt3r put s0me d161ts in this 5tr1n6, right?", mycustomstring.getEveryNthCharacterFromBeginningOrEnd(1, false));
    }

    // This test checks whether method getEveryNthCharacterFromBeginningOrEnd() can get et every single character from end.
    // This test checks method getEveryNthCharacterFromBeginningOrEnd() variable n lower bound, from end.
    @Test
    public void testGetEveryNthCharacterFromBeginningOrEnd4() {
        mycustomstring.setString("I'd b3tt3r put s0me d161ts in this 5tr1n6, right?");
        assertEquals("I'd b3tt3r put s0me d161ts in this 5tr1n6, right?", mycustomstring.getEveryNthCharacterFromBeginningOrEnd(1, true));
    }

    // This test checks method getEveryNthCharacterFromBeginningOrEnd() variable n upper bound, from beginning.
    // Note: From end is currently not tested.
    @Test
    public void testGetEveryNthCharacterFromBeginningOrEnd5() {
        mycustomstring.setString("I'd b3tt3r put s0me d161ts in this 5tr1n6, right?");
        assertEquals("?", mycustomstring.getEveryNthCharacterFromBeginningOrEnd(49, false));
    }

    // This test checks whether method getEveryNthCharacterFromBeginningOrEnd() will return an empty string is n is larger than string length, from beginning.
    @Test
    public void testGetEveryNthCharacterFromBeginningOrEnd6() {
        mycustomstring.setString("I'd b3tt3r put s0me d161ts in this 5tr1n6, right?");
        assertEquals("", mycustomstring.getEveryNthCharacterFromBeginningOrEnd(60, false));
    }

    // This test checks whether method getEveryNthCharacterFromBeginningOrEnd() will return an empty string is n is larger than string length, from end.
    @Test
    public void testGetEveryNthCharacterFromBeginningOrEnd7()  {
        mycustomstring.setString("I'd b3tt3r put s0me d161ts in this 5tr1n6, right?");
        assertEquals("", mycustomstring.getEveryNthCharacterFromBeginningOrEnd(60, true));
    }

    /*
    Simple stress test on the method getEveryNthCharacterFromBeginningOrEnd() algorithm, to see if characters are lost over many repetitions.
    Every character from mycustomstring is read from beginning and stored back into mycustomstring.
    Same process is repeated in next iteration from end.
    This back and forth reassignment is carried out a total of 1000 times and the final string is evaluated to contain the original string.
    */
    @Test
    public void testGetEveryNthCharacterFromBeginningOrEnd8() {
        mycustomstring.setString("I'd b3tt3r put s0me d161ts in this 5tr1n6, right?");
        int i = 0;
        do{
            if(i%2 == 0)
                mycustomstring.setString(mycustomstring.getEveryNthCharacterFromBeginningOrEnd(1,false));
            else
                mycustomstring.setString(mycustomstring.getEveryNthCharacterFromBeginningOrEnd(1,true));
            i++;
        }while(i<=999);

        assertEquals("I'd b3tt3r put s0me d161ts in this 5tr1n6, right?", mycustomstring.getEveryNthCharacterFromBeginningOrEnd(1, false));
    }

    //This method checks whether the method getEveryNthCharacterFromBeginningOrEnd() is able to get characters across multi-line and tabbed strings.
    @Test
    public void testGetEveryNthCharacterFromBeginningOrEnd9() {
        mycustomstring.setString("I'd\nb3tt3r\tput\ns0me\td161ts\nin\tthis\n5tr1n6,\tright?");
        assertEquals("'bt\tt0\t6snh\nr6rh", mycustomstring.getEveryNthCharacterFromBeginningOrEnd(3, true));

    }

    // This test checks whether method getEveryNthCharacterFromBeginningOrEnd() suitably throws a IllegalArgumentException if n is 0.
    @Test
    public void testGetEveryNthCharacterFromBeginningOrEnd10() {
        mycustomstring.setString("I'd b3tt3r put s0me d161ts in this 5tr1n6, right?");

        thrown.expect(IllegalArgumentException.class);
        thrown.expectMessage("getEveryNthCharacterFromBeginningOrEnd: encountered an illegal argument");

        mycustomstring.getEveryNthCharacterFromBeginningOrEnd(0,true);
    }

    // This test checks whether method getEveryNthCharacterFromBeginningOrEnd() suitably throws a IllegalArgumentException if n is less than 0, from end.
    @Test
    public void testGetEveryNthCharacterFromBeginningOrEnd11() {
        mycustomstring.setString("I'd b3tt3r put s0me d161ts in this 5tr1n6, right?");

        thrown.expect(IllegalArgumentException.class);
        thrown.expectMessage("getEveryNthCharacterFromBeginningOrEnd: encountered an illegal argument");

        mycustomstring.getEveryNthCharacterFromBeginningOrEnd(-1,true);
    }

    // This test checks whether method getEveryNthCharacterFromBeginningOrEnd() suitably throws a IllegalArgumentException if n is 0, from beginning.
    @Test
    public void testGetEveryNthCharacterFromBeginningOrEnd12()  {
        mycustomstring.setString("I'd b3tt3r put s0me d161ts in this 5tr1n6, right?");

        thrown.expect(IllegalArgumentException.class);
        thrown.expectMessage("getEveryNthCharacterFromBeginningOrEnd: encountered an illegal argument");

        mycustomstring.getEveryNthCharacterFromBeginningOrEnd(-5,false);
    }

    // This test checks whether method getEveryNthCharacterFromBeginningOrEnd() suitably throws a NullPointerException if mycustomstring is null, from end.
    @Test
    public void testGetEveryNthCharacterFromBeginningOrEnd13() {

        mycustomstring.setString(null);

        thrown.expect(NullPointerException.class);
        thrown.expectMessage("getEveryNthCharacterFromBeginningOrEnd: encountered a null pointer.");

        mycustomstring.getEveryNthCharacterFromBeginningOrEnd(3,true);
    }

    // This test checks whether method getEveryNthCharacterFromBeginningOrEnd() suitably throws a NullPointerException if mycustomstring is null, from beginning.
    @Test
    public void testGetEveryNthCharacterFromBeginningOrEnd14() {

        mycustomstring.setString(null);

        thrown.expect(NullPointerException.class);
        thrown.expectMessage("getEveryNthCharacterFromBeginningOrEnd: encountered a null pointer.");

        mycustomstring.getEveryNthCharacterFromBeginningOrEnd(5,false);
    }

    // This test checks whether method ConvertDigitsToNamesInSubstring() convert digits in a substring to word equivalents.
    @Test
    public void testConvertDigitsToNamesInSubstring1() {
        mycustomstring.setString("I'd b3tt3r put s0me d161ts in this 5tr1n6, right?");
        mycustomstring.convertDigitsToNamesInSubstring(17, 23);
        assertEquals("I'd b3tt3r put sZerome dOneSix1ts in this 5tr1n6, right?", mycustomstring.getString());
    }

    /*
    This test checks whether method ConvertDigitsToNamesInSubstring() convert digits in a substring to word equivalents.
    The substring is the entire valid string.
    Numbers are specifically placed at first and last position to check if the algorithm will get them.
    */
    @Test
    public void testConvertDigitsToNamesInSubstring2() {
        mycustomstring.setString("2I'd b3tt3r put s0me d161ts in this 5tr1n6, right?7");
        mycustomstring.convertDigitsToNamesInSubstring(1, 51);
        assertEquals("TwoI'd bThreettThreer put sZerome dOneSixOnets in this FivetrOnenSix, right?Seven", mycustomstring.getString());
    }

    // This test checks whether method ConvertDigitsToNamesInSubstring() convert digits in a substring of only 1 character.
    @Test
    public void testConvertDigitsToNamesInSubstring3() {
        mycustomstring.setString("3");
        mycustomstring.convertDigitsToNamesInSubstring(1,1);
        assertEquals("Three", mycustomstring.getString());
    }

    // This test checks whether method ConvertDigitsToNamesInSubstring() suitably throws a IllegalArgumentException if starting character is ahead of ending character.
    @Test
    public void testConvertDigitsToNamesInSubstring4() {
        mycustomstring.setString("I'd b3tt3r put s0me d161ts in this 5tr1n6, right?");

        thrown.expect(IllegalArgumentException.class);
        thrown.expectMessage("convertDigitsToNamesInSubstring: encountered an illegal argument");

        mycustomstring.convertDigitsToNamesInSubstring(6,5);
    }

    // This test checks whether method ConvertDigitsToNamesInSubstring() suitably throws a MyIndexOutOfBoundsException if starting character is 0.
    @Test
    public void testConvertDigitsToNamesInSubstring5() {
        mycustomstring.setString("I'd b3tt3r put s0me d161ts in this 5tr1n6, right?");

        thrown.expect(MyIndexOutOfBoundsException.class);
        thrown.expectMessage("convertDigitsToNamesInSubstring: encountered an index out of bounds");

        mycustomstring.convertDigitsToNamesInSubstring(0,7);
    }

    // This test checks whether method ConvertDigitsToNamesInSubstring() suitably throws a MyIndexOutOfBoundsException if starting character is less than 0.
    @Test
    public void testConvertDigitsToNamesInSubstring6() {
        mycustomstring.setString("I'd b3tt3r put s0me d161ts in this 5tr1n6, right?");

        thrown.expect(MyIndexOutOfBoundsException.class);
        thrown.expectMessage("convertDigitsToNamesInSubstring: encountered an index out of bounds");

        mycustomstring.convertDigitsToNamesInSubstring(-5,6);
    }

    // This test checks whether method ConvertDigitsToNamesInSubstring() suitably throws a MyIndexOutOfBoundsException if ending character is greater than string length.
    @Test
    public void testConvertDigitsToNamesInSubstring7() {
        mycustomstring.setString("I'd b3tt3r put s0me d161ts in this 5tr1n6, right?");

        thrown.expect(MyIndexOutOfBoundsException.class);
        thrown.expectMessage("convertDigitsToNamesInSubstring: encountered an index out of bounds");

        mycustomstring.convertDigitsToNamesInSubstring(5,66);
    }

    // This test checks whether method ConvertDigitsToNamesInSubstring() suitably throws a NullPointerException if mycustomstring is null.
    @Test
    public void testConvertDigitsToNamesInSubstring8() {
        mycustomstring.setString(null);

        thrown.expect(NullPointerException.class);
        thrown.expectMessage("convertDigitsToNamesInSubstring: encountered a null pointer");

        mycustomstring.convertDigitsToNamesInSubstring(5,6);
    }

}
