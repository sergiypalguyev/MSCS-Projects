package edu.gatech.seclass.replace;

import org.junit.After;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.*;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;

import static org.junit.Assert.*;

public class MainTest {

    private ByteArrayOutputStream outStream;
    private ByteArrayOutputStream errStream;
    private PrintStream outOrig;
    private PrintStream errOrig;
    private Charset charset = StandardCharsets.UTF_8;

    @Rule
    public TemporaryFolder temporaryFolder = new TemporaryFolder();

    @Before
    public void setUp() throws Exception {
        outStream = new ByteArrayOutputStream();
        PrintStream out = new PrintStream(outStream);
        errStream = new ByteArrayOutputStream();
        PrintStream err = new PrintStream(errStream);
        outOrig = System.out;
        errOrig = System.err;
        System.setOut(out);
        System.setErr(err);
    }

    @After
    public void tearDown() throws Exception {
        System.setOut(outOrig);
        System.setErr(errOrig);
    }

    // Some utilities

    private File createTmpFile() throws IOException {
        File tmpfile = temporaryFolder.newFile();
        tmpfile.deleteOnExit();
        return tmpfile;
    }

    private File createInputFile1() throws Exception {
        File file1 =  createTmpFile();
        FileWriter fileWriter = new FileWriter(file1);

        fileWriter.write("Howdy Bill,\n" +
                "This is a test file for the replace utility\n" +
                "Let's make sure it has at least a few lines\n" +
                "so that we can create some interesting test cases...\n" +
                "And let's say \"howdy bill\" again!");

        fileWriter.close();
        return file1;
    }

    private File createInputFile2() throws Exception {
        File file1 =  createTmpFile();
        FileWriter fileWriter = new FileWriter(file1);

        fileWriter.write("Howdy Bill,\n" +
                "This is another test file for the replace utility\n" +
                "that contains a list:\n" +
                "-a) Item 1\n" +
                "-b) Item 2\n" +
                "...\n" +
                "and says \"howdy Bill\" twice");

        fileWriter.close();
        return file1;
    }

    private File createInputFile3() throws Exception {
        File file1 =  createTmpFile();
        FileWriter fileWriter = new FileWriter(file1);

        fileWriter.write("Howdy Bill, have you learned your abc and 123?\n" +
                "It is important to know your abc and 123," +
                "so you should study it\n" +
                "and then repeat with me: abc and 123");

        fileWriter.close();
        return file1;
    }

    private String getFileContent(String filename) {
        String content = null;
        try {
            content = new String(Files.readAllBytes(Paths.get(filename)), charset);
        } catch (IOException e) {
            e.printStackTrace();
        }
        return content;
    }

    // Actual test cases

    @Test
    public void mainTest1() throws Exception {
        File inputFile1 = createInputFile1();
        File inputFile2 = createInputFile2();
        File inputFile3 = createInputFile3();

        String args[] = {"-i", "Howdy", "Hello", "--", inputFile1.getPath(), inputFile2.getPath(), inputFile3.getPath()};
        Main.main(args);

        String expected1 = "Hello Bill,\n" +
                "This is a test file for the replace utility\n" +
                "Let's make sure it has at least a few lines\n" +
                "so that we can create some interesting test cases...\n" +
                "And let's say \"Hello bill\" again!";
        String expected2 = "Hello Bill,\n" +
                "This is another test file for the replace utility\n" +
                "that contains a list:\n" +
                "-a) Item 1\n" +
                "-b) Item 2\n" +
                "...\n" +
                "and says \"Hello Bill\" twice";
        String expected3 = "Hello Bill, have you learned your abc and 123?\n" +
                "It is important to know your abc and 123," +
                "so you should study it\n" +
                "and then repeat with me: abc and 123";

        String actual1 = getFileContent(inputFile1.getPath());
        String actual2 = getFileContent(inputFile2.getPath());
        String actual3 = getFileContent(inputFile3.getPath());

        assertEquals("The files differ!", expected1, actual1);
        assertEquals("The files differ!", expected2, actual2);
        assertEquals("The files differ!", expected3, actual3);

        assertFalse(Files.exists(Paths.get(inputFile1.getPath() + ".bck")));
        assertFalse(Files.exists(Paths.get(inputFile2.getPath() + ".bck")));
        assertFalse(Files.exists(Paths.get(inputFile3.getPath() + ".bck")));
    }

    @Test
    public void mainTest2() throws Exception {
        File inputFile1 = createInputFile1();
        File inputFile2 = createInputFile2();

        String args[] = {"-b", "-f", "Bill", "William", "--", inputFile1.getPath(), inputFile2.getPath()};
        Main.main(args);

        String expected1 = "Howdy William,\n" +
                "This is a test file for the replace utility\n" +
                "Let's make sure it has at least a few lines\n" +
                "so that we can create some interesting test cases...\n" +
                "And let's say \"howdy bill\" again!";
        String expected2 = "Howdy William,\n" +
                "This is another test file for the replace utility\n" +
                "that contains a list:\n" +
                "-a) Item 1\n" +
                "-b) Item 2\n" +
                "...\n" +
                "and says \"howdy Bill\" twice";

        String actual1 = getFileContent(inputFile1.getPath());
        String actual2 = getFileContent(inputFile2.getPath());

        assertEquals("The files differ!", expected1, actual1);
        assertEquals("The files differ!", expected2, actual2);
        assertTrue(Files.exists(Paths.get(inputFile1.getPath() + ".bck")));
        assertTrue(Files.exists(Paths.get(inputFile2.getPath() + ".bck")));
    }

    @Test
    public void mainTest3() throws Exception {
        File inputFile = createInputFile3();

        String args[] = {"-f", "-l", "abc", "ABC", "--", inputFile.getPath()};
        Main.main(args);

        String expected = "Howdy Bill, have you learned your ABC and 123?\n" +
                "It is important to know your abc and 123," +
                "so you should study it\n" +
                "and then repeat with me: ABC and 123";

        String actual = getFileContent(inputFile.getPath());

        assertEquals("The files differ!", expected, actual);
        assertFalse(Files.exists(Paths.get(inputFile.getPath() + ".bck")));
    }

    @Test
    public void mainTest4() throws Exception {
        File inputFile = createInputFile3();

        String args[] = {"123", "<numbers removed>", "--", inputFile.getPath()};
        Main.main(args);

        String expected = "Howdy Bill, have you learned your abc and <numbers removed>?\n" +
                "It is important to know your abc and <numbers removed>," +
                "so you should study it\n" +
                "and then repeat with me: abc and <numbers removed>";

        String actual = getFileContent(inputFile.getPath());

        assertEquals("The files differ!", expected, actual);
        assertFalse(Files.exists(Paths.get(inputFile.getPath() + ".bck")));
    }

    @Test
    public void mainTest5() throws Exception {
        File inputFile = createInputFile2();

        String args1[] = {"-b", "--", "-a", "1", "--", inputFile.getPath()};
        Main.main(args1);
        String args2[] = {"--", "-b", "2", "--", inputFile.getPath()};
        Main.main(args2);

        String expected = "Howdy Bill,\n" +
                "This is another test file for the replace utility\n" +
                "that contains a list:\n" +
                "1) Item 1\n" +
                "2) Item 2\n" +
                "...\n" +
                "and says \"howdy Bill\" twice";

        String actual = getFileContent(inputFile.getPath());

        assertEquals("The files differ!", expected, actual);
        assertTrue(Files.exists(Paths.get(inputFile.getPath() + ".bck")));
    }

    @Test
    public void mainTest6() {
        String args[] = {"blah",};
        Main.main(args);
        assertEquals("Usage: Replace [-b] [-f] [-l] [-i] <from> <to> -- <filename> [<filename>]*", errStream.toString().trim());
    }


    @Test
    // Test Case 1  		<single>
    // Options :  None
    public void Test1(){
        File inputFile = null; // TODO
        String expected = null; // TODO

        String filePath = null;
        if(inputFile != null) {
            filePath = inputFile.getPath();
        }

        String[] args = {"abc","xyz", "--", filePath};
        Main.main(args);

        String actual = null;
        if (inputFile != null){
            actual = getFileContent(inputFile.getPath());
        }

        assertEquals("The files are different!",expected, actual);
    }
    @Test
    // Test Case 2  		<single>
    // Length of the string :  Zero
    public void Test2(){
        File inputFile = null; // TODO
        String expected = null; // TODO

        String filePath = null;
        if(inputFile != null) {
            filePath = inputFile.getPath();
        }

        String[] args = {"-b","","z","--",filePath};
        Main.main(args);

        String actual = null;
        if (inputFile != null){
            actual = getFileContent(inputFile.getPath());
        }
        assertEquals("The files are different!",expected, actual); }
    @Test
    // Test Case 3  		<single>
    // Length of the string :  One
    public void Test3(){
        File inputFile = null; // TODO
        String expected = null; // TODO

        String filePath = null;
        if(inputFile != null) {
            filePath = inputFile.getPath();
        }

        String[] args = {"-b","a","z","--",filePath};
        Main.main(args);

        String actual = null;
        if (inputFile != null){
            actual = getFileContent(inputFile.getPath());
        }
        assertEquals("The files are different!",expected, actual);}
    @Test
    //    Test Case 4  		<error>
    //    Length of the string :  Longer than the file
    public void Test4()throws Exception{
        File inputFile = null; // TODO
        String expected = null; // TODO

        String filePath = null;
        if(inputFile != null) {
            filePath = inputFile.getPath();
        }

        String[] args = {"-b","abc","xyz","--",filePath};
        Main.main(args);

        String actual = null;
        if (inputFile != null){
            actual = getFileContent(inputFile.getPath());
        }
        assertEquals("The files are different!",expected, actual); }
    @Test
    //    Test Case 5  		<single>
    //    Presence of special characters :  None
    public void Test5(){
        File inputFile = null; // TODO
        String expected = null; // TODO

        String filePath = null;
        if(inputFile != null) {
            filePath = inputFile.getPath();
        }

        String[] args = {"-b","abc","xyz","--",filePath};
        Main.main(args);

        String actual = null;
        if (inputFile != null){
            actual = getFileContent(inputFile.getPath());
        }
        assertEquals("The files are different!",expected, actual);  }
    @Test
    //    Test Case 6  		<single>
    //    Length of the string :  Zero
    public void Test6(){
        File inputFile = null; // TODO
        String expected = null; // TODO

        String filePath = null;
        if(inputFile != null) {
            filePath = inputFile.getPath();
        }

        String[] args = {"-b","a","","--",filePath};
        Main.main(args);

        String actual = null;
        if (inputFile != null){
            actual = getFileContent(inputFile.getPath());
        }
        assertEquals("The files are different!",expected, actual);  }
    @Test
    //    Test Case 7  		<single>
    //    Length of the string :  One
    public void Test7(){
        File inputFile = null; // TODO
        String expected = null; // TODO

        String filePath = null;
        if(inputFile != null) {
            filePath = inputFile.getPath();
        }

        String[] args = {"-b","a","a","--",filePath};
        Main.main(args);

        String actual = null;
        if (inputFile != null){
            actual = getFileContent(inputFile.getPath());
        }
        assertEquals("The files are different!",expected, actual);  }
    @Test
    //    Test Case 8  		<error>
    //    Length of the string :  Longer than the file
    public void Test8()throws Exception{
        File inputFile = null; // TODO
        String expected = null; // TODO

        String filePath = null;
        if(inputFile != null) {
            filePath = inputFile.getPath();
        }

        String[] args = {"-b","a","xyz","--",filePath};
        Main.main(args);

        String actual = null;
        if (inputFile != null){
            actual = getFileContent(inputFile.getPath());
        }
        assertEquals("The files are different!",expected, actual);   }
    @Test
    //    Test Case 9  		<single>
    //    Presence of special characters :  None
    public void Test9(){
        File inputFile = null; // TODO
        String expected = null; // TODO

        String filePath = null;
        if(inputFile != null) {
            filePath = inputFile.getPath();
        }

        String[] args = {"-b","abc","xyz","--",filePath};
        Main.main(args);

        String actual = null;
        if (inputFile != null){
            actual = getFileContent(inputFile.getPath());
        }

        assertEquals("The files are different!",expected, actual);   }
    @Test
    //    Test Case 10 		<single>
    //    Count :  Zero
    public void Test10(){
        File inputFile = null; // TODO
        String expected = null; // TODO

        String filePath = null;
        if(inputFile != null) {
            filePath = inputFile.getPath();
        }

        String[] args = {"-b","abc","xyz","--"};
        Main.main(args);

        String actual = null;
        if (inputFile != null){
            actual = getFileContent(inputFile.getPath());
        }
        assertEquals("The files are different!",expected, actual);   }
    @Test
    //    Test Case 11 		<single>
    //    Count :  One
    public void Test11(){
        File inputFile = null; // TODO
        String expected = null; // TODO

        String filePath = null;
        if(inputFile != null) {
            filePath = inputFile.getPath();
        }

        String[] args = {"-b","abc","xyz","--",filePath};
        Main.main(args);

        String actual = null;
        if (inputFile != null){
            actual = getFileContent(inputFile.getPath());
        }
        assertEquals("The files are different!",expected, actual);   }
    @Test
    //    Test Case 12 		<single>
    //    Size :  Empty
    public void Test12(){
        File inputFile = null; // TODO
        String expected = null; // TODO

        String filePath = null;
        if(inputFile != null) {
            filePath = inputFile.getPath();
        }

        String[] args = {"-b","abc","xyz","--",filePath};
        Main.main(args);

        String actual = null;
        if (inputFile != null){
            actual = getFileContent(inputFile.getPath());
        }
        assertEquals("The files are different!",expected, actual);   }
    @Test
    //    Test Case 13 		<single>
    //    Number of occurrences of the string fragment in the file :  None
    public void Test13(){
        File inputFile = null; // TODO
        String expected = null; // TODO

        String filePath = null;
        if(inputFile != null) {
            filePath = inputFile.getPath();
        }

        String[] args = {"-b","abc","xyz","--",filePath};
        Main.main(args);

        String actual = null;
        if (inputFile != null){
            actual = getFileContent(inputFile.getPath());
        }
        assertEquals("The files are different!",expected, actual);   }
    @Test
    //    Test Case 14 		<single>
    //    Number of occurrences of the string fragment in one line :  None
    public void Test14(){
        File inputFile = null; // TODO
        String expected = null; // TODO

        String filePath = null;
        if(inputFile != null) {
            filePath = inputFile.getPath();
        }

        String[] args = {"-b","abc","xyz","--",filePath};
        Main.main(args);

        String actual = null;
        if (inputFile != null){
            actual = getFileContent(inputFile.getPath());
        }
        assertEquals("The files are different!",expected, actual);   }
    @Test
    //    Test Case 15 		<single>
    //    Position of the string fragment in the file :  First line
    public void Test15(){
        File inputFile = null; // TODO
        String expected = null; // TODO

        String filePath = null;
        if(inputFile != null) {
            filePath = inputFile.getPath();
        }

        String[] args = {"-b","abc","xyz","--",filePath};
        Main.main(args);

        String actual = null;
        if (inputFile != null){
            actual = getFileContent(inputFile.getPath());
        }
        assertEquals("The files are different!",expected, actual);  }
    @Test
    //    Test Case 16 		<single>
    //    Position of the string fragment in the file :  Last line
    public void Test16(){
        File inputFile = null; // TODO
        String expected = null; // TODO

        String filePath = null;
        if(inputFile != null) {
            filePath = inputFile.getPath();
        }

        String[] args = {"-b","abc","xyz","--",filePath};
        Main.main(args);

        String actual = null;
        if (inputFile != null){
            actual = getFileContent(inputFile.getPath());
        }
        assertEquals("The files are different!",expected, actual);   }
    @Test
    //    Test Case 17 		<error>
    //    Presence of a file corresponding to the name :  Not present
    public void Test17(){
        File inputFile = null; // TODO
        String expected = null; // TODO

        String filePath = null;
        if(inputFile != null) {
            filePath = inputFile.getPath();
        }

        String[] args = {"-b","abc","xyz","--",filePath};
        Main.main(args);

        String actual = null;
        if (inputFile != null){
            actual = getFileContent(inputFile.getPath());
        }
        assertEquals("The files are different!",expected, actual);   }
    @Test
    //    Test Case 18 		(Key = 2.3.2.3.2.3.2.2.2.3.2.)
    //    Options                                                  :  -b
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  One
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  One
    //    Count                                                    :  Many
    //    Size                                                     :  Not empty
    //    Number of occurrences of the string fragment in the file :  One
    //    Number of occurrences of the string fragment in one line :  One
    //    Position of the string fragment in the file              :  Any
    //    Presence of a file corresponding to the name             :  Present
    public void Test18(){
        File inputFile = null; // TODO
        String expected = null; // TODO

        String filePath = null;
        if(inputFile != null) {
            filePath = inputFile.getPath();
        }

        String[] args = {"-b","abc!","xyz@","--",filePath};
        Main.main(args);

        String actual = null;
        if (inputFile != null){
            actual = getFileContent(inputFile.getPath());
        }
        assertEquals("The files are different!",expected, actual);   }
    @Test
    //    Test Case 19 		(Key = 2.3.2.3.2.3.2.2.3.3.2.)
    //    Options                                                  :  -b
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  One
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  One
    //    Count                                                    :  Many
    //    Size                                                     :  Not empty
    //    Number of occurrences of the string fragment in the file :  One
    //    Number of occurrences of the string fragment in one line :  Many
    //    Position of the string fragment in the file              :  Any
    //    Presence of a file corresponding to the name             :  Present
    public void Test19(){
        File inputFile = null; // TODO
        String expected = null; // TODO

        String filePath = null;
        if(inputFile != null) {
            filePath = inputFile.getPath();
        }

        String[] args = {"-b","abc#","xyz$","--",filePath};
        Main.main(args);

        String actual = null;
        if (inputFile != null){
            actual = getFileContent(inputFile.getPath());
        }
        assertEquals("The files are different!",expected, actual);    }
    @Test
    //    Test Case 20 		(Key = 2.3.2.3.2.3.2.3.2.3.2.)
    //    Options                                                  :  -b
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  One
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  One
    //    Count                                                    :  Many
    //    Size                                                     :  Not empty
    //    Number of occurrences of the string fragment in the file :  Many
    //    Number of occurrences of the string fragment in one line :  One
    //    Position of the string fragment in the file              :  Any
    //    Presence of a file corresponding to the name             :  Present
    public void Test20()throws Exception{
        File inputFile = null; // TODO
        String expected = null; // TODO

        String filePath = null;
        if(inputFile != null) {
            filePath = inputFile.getPath();
        }
        String[] args = {"-b","abc%","xyz^","--",filePath};
        Main.main(args);

        String actual = null;
        if (inputFile != null){
            actual = getFileContent(inputFile.getPath());
        }
        assertEquals("The files are different!",expected, actual);   }
    @Test
    //    Test Case 21 		(Key = 2.3.2.3.2.3.2.3.3.3.2.)
    //    Options                                                  :  -b
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  One
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  One
    //    Count                                                    :  Many
    //    Size                                                     :  Not empty
    //    Number of occurrences of the string fragment in the file :  Many
    //    Number of occurrences of the string fragment in one line :  Many
    //    Position of the string fragment in the file              :  Any
    //    Presence of a file corresponding to the name             :  Present
    public void Test21()throws Exception{
        File inputFile = null; // TODO
        String expected = null; // TODO

        String filePath = null;
        if(inputFile != null) {
            filePath = inputFile.getPath();
        }
        String[] args = {"-b","abc*","x(yz","--",filePath};
        Main.main(args);

        String actual = null;
        if (inputFile != null){
            actual = getFileContent(inputFile.getPath());
        }
        assertEquals("The files are different!",expected, actual);   }
    @Test
    //    Test Case 22 		(Key = 2.3.2.3.3.3.2.2.2.3.2.)
    //    Options                                                  :  -b
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  One
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  Many
    //    Count                                                    :  Many
    //    Size                                                     :  Not empty
    //    Number of occurrences of the string fragment in the file :  One
    //    Number of occurrences of the string fragment in one line :  One
    //    Position of the string fragment in the file              :  Any
    //    Presence of a file corresponding to the name             :  Present
    public void Test22()throws Exception{
        File inputFile = null; // TODO
        String expected = null; // TODO

        String filePath = null;
        if(inputFile != null) {
            filePath = inputFile.getPath();
        }
        String[] args = {"-b","a)bc","x+yz","--",filePath};
        Main.main(args);

        String actual = null;
        if (inputFile != null){
            actual = getFileContent(inputFile.getPath());
        }
        assertEquals("The files are different!",expected, actual);   }
    @Test
    //    Test Case 23 		(Key = 2.3.2.3.3.3.2.2.3.3.2.)
    //    Options                                                  :  -b
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  One
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  Many
    //    Count                                                    :  Many
    //    Size                                                     :  Not empty
    //    Number of occurrences of the string fragment in the file :  One
    //    Number of occurrences of the string fragment in one line :  Many
    //    Position of the string fragment in the file              :  Any
    //    Presence of a file corresponding to the name             :  Present
    public void Test23()throws Exception{
        File inputFile = null; // TODO
        String expected = null; // TODO

        String filePath = null;
        if(inputFile != null) {
            filePath = inputFile.getPath();
        }

        String[] args = {"-b","ab~c","xy|z","--",filePath};
        Main.main(args);

        String actual = null;
        if (inputFile != null){
            actual = getFileContent(inputFile.getPath());
        }
        assertEquals("The files are different!",expected, actual);   }
    @Test
    //    Test Case 24 		(Key = 2.3.2.3.3.3.2.3.2.3.2.)
    //    Options                                                  :  -b
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  One
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  Many
    //    Count                                                    :  Many
    //    Size                                                     :  Not empty
    //    Number of occurrences of the string fragment in the file :  Many
    //    Number of occurrences of the string fragment in one line :  One
    //    Position of the string fragment in the file              :  Any
    //    Presence of a file corresponding to the name             :  Present
    public void Test24()throws Exception{
        File inputFile = null; // TODO
        String expected = null; // TODO

        String filePath = null;
        if(inputFile != null) {
            filePath = inputFile.getPath();
        }

        String[] args = {"-b","ab~c","x!#@$y|z","--",filePath};
        Main.main(args);

        String actual = null;
        if (inputFile != null){
            actual = getFileContent(inputFile.getPath());
        }
        assertEquals("The files are different!",expected, actual);   }
    @Test
    //    Test Case 25 		(Key = 2.3.2.3.3.3.2.3.3.3.2.)
    //    Options                                                  :  -b
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  One
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  Many
    //    Count                                                    :  Many
    //    Size                                                     :  Not empty
    //    Number of occurrences of the string fragment in the file :  Many
    //    Number of occurrences of the string fragment in one line :  Many
    //    Position of the string fragment in the file              :  Any
    //    Presence of a file corresponding to the name             :  Present
    public void Test25()throws Exception{
        File inputFile = null; // TODO
        String expected = null; // TODO

        String filePath = null;
        if(inputFile != null) {
            filePath = inputFile.getPath();
        }

        String[] args = {"-b","ab~c","xyz!@#$#!#$","--",filePath};
        Main.main(args);

        String actual = null;
        if (inputFile != null){
            actual = getFileContent(inputFile.getPath());
        }
        assertEquals("The files are different!",expected, actual);    }
    @Test
    //    Test Case 26 		(Key = 2.3.3.3.2.3.2.2.2.3.2.)
    //    Options                                                  :  -b
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  Many
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  One
    //    Count                                                    :  Many
    //    Size                                                     :  Not empty
    //    Number of occurrences of the string fragment in the file :  One
    //    Number of occurrences of the string fragment in one line :  One
    //    Position of the string fragment in the file              :  Any
    //    Presence of a file corresponding to the name             :  Present
    public void Test26()throws Exception{
        File inputFile = null; // TODO
        String expected = null; // TODO

        String filePath = null;
        if(inputFile != null) {
            filePath = inputFile.getPath();
        }

        String[] args = {"-b","ab~c","x!#@$y|z","--",filePath};
        Main.main(args);

        String actual = null;
        if (inputFile != null){
            actual = getFileContent(inputFile.getPath());
        }
        assertEquals("The files are different!",expected, actual);   }
    @Test
    //    Test Case 27 		(Key = 2.3.3.3.2.3.2.2.3.3.2.)
    //    Options                                                  :  -b
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  Many
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  One
    //    Count                                                    :  Many
    //    Size                                                     :  Not empty
    //    Number of occurrences of the string fragment in the file :  One
    //    Number of occurrences of the string fragment in one line :  Many
    //    Position of the string fragment in the file              :  Any
    //    Presence of a file corresponding to the name             :  Present
    public void Test27()throws Exception{
        File inputFile = null; // TODO
        String expected = null; // TODO

        String filePath = null;
        if(inputFile != null) {
            filePath = inputFile.getPath();
        }
        String[] args = {"-b","ab~c","x!#@$y|z","--",filePath};
        Main.main(args);

        String actual = null;
        if (inputFile != null){
            actual = getFileContent(inputFile.getPath());
        }
        assertEquals("The files are different!",expected, actual);   }
    @Test
    //    Test Case 28 		(Key = 2.3.3.3.2.3.2.3.2.3.2.)
    //    Options                                                  :  -b
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  Many
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  One
    //    Count                                                    :  Many
    //    Size                                                     :  Not empty
    //    Number of occurrences of the string fragment in the file :  Many
    //    Number of occurrences of the string fragment in one line :  One
    //    Position of the string fragment in the file              :  Any
    //    Presence of a file corresponding to the name             :  Present
    public void Test28()throws Exception{
        File inputFile = null; // TODO
        String expected = null; // TODO

        String filePath = null;
        if(inputFile != null) {
            filePath = inputFile.getPath();
        }
        String[] args = {"-b","ab~c","x!#@$y|z","--",filePath};
        Main.main(args);

        String actual = null;
        if (inputFile != null){
            actual = getFileContent(inputFile.getPath());
        }
        assertEquals("The files are different!",expected, actual);    }
    @Test
    //    Test Case 29 		(Key = 2.3.3.3.2.3.2.3.3.3.2.)
    //    Options                                                  :  -b
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  Many
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  One
    //    Count                                                    :  Many
    //    Size                                                     :  Not empty
    //    Number of occurrences of the string fragment in the file :  Many
    //    Number of occurrences of the string fragment in one line :  Many
    //    Position of the string fragment in the file              :  Any
    //    Presence of a file corresponding to the name             :  Present
    public void Test29()throws Exception{
        File inputFile = null; // TODO
        String expected = null; // TODO

        String filePath = null;
        if(inputFile != null) {
            filePath = inputFile.getPath();
        }

        String[] args = {"-b","ab~c","x!#@$y|z","--",filePath};
        Main.main(args);

        String actual = null;
        if (inputFile != null){
            actual = getFileContent(inputFile.getPath());
        }
        assertEquals("The files are different!",expected, actual);    }
    @Test
    //    Test Case 30 		(Key = 2.3.3.3.3.3.2.2.2.3.2.)
    //    Options                                                  :  -b
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  Many
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  Many
    //    Count                                                    :  Many
    //    Size                                                     :  Not empty
    //    Number of occurrences of the string fragment in the file :  One
    //    Number of occurrences of the string fragment in one line :  One
    //    Position of the string fragment in the file              :  Any
    //    Presence of a file corresponding to the name             :  Present
    public void Test30()throws Exception{
        File inputFile = null; // TODO
        String expected = null; // TODO

        String filePath = null;
        if(inputFile != null) {
            filePath = inputFile.getPath();
        }

        String[] args = {"-b","ab~c","x!#@$y|z","--",filePath};
        Main.main(args);

        String actual = null;
        if (inputFile != null){
            actual = getFileContent(inputFile.getPath());
        }
        assertEquals("The files are different!",expected, actual);  }
    @Test
    //    Test Case 31 		(Key = 2.3.3.3.3.3.2.2.3.3.2.)
    //    Options                                                  :  -b
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  Many
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  Many
    //    Count                                                    :  Many
    //    Size                                                     :  Not empty
    //    Number of occurrences of the string fragment in the file :  One
    //    Number of occurrences of the string fragment in one line :  Many
    //    Position of the string fragment in the file              :  Any
    //    Presence of a file corresponding to the name             :  Present
    public void Test31()throws Exception{
        File inputFile = null; // TODO
        String expected = null; // TODO

        String filePath = null;
        if(inputFile != null) {
            filePath = inputFile.getPath();
        }

        String[] args = {"-b","ab~c","x!#@$y|z","--",filePath};
        Main.main(args);

        String actual = null;
        if (inputFile != null){
            actual = getFileContent(inputFile.getPath());
        }
        assertEquals("The files are different!",expected, actual);   }
    @Test
    //    Test Case 32 		(Key = 2.3.3.3.3.3.2.3.2.3.2.)
    //    Options                                                  :  -b
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  Many
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  Many
    //    Count                                                    :  Many
    //    Size                                                     :  Not empty
    //    Number of occurrences of the string fragment in the file :  Many
    //    Number of occurrences of the string fragment in one line :  One
    //    Position of the string fragment in the file              :  Any
    //    Presence of a file corresponding to the name             :  Present
    public void Test32()throws Exception{
        File inputFile = null; // TODO
        String expected = null; // TODO

        String filePath = null;
        if(inputFile != null) {
            filePath = inputFile.getPath();
        }

        String[] args = {"-b","ab~c","x!#@$y|z","--",filePath};
        Main.main(args);

        String actual = null;
        if (inputFile != null){
            actual = getFileContent(inputFile.getPath());
        }
        assertEquals("The files are different!",expected, actual);   }
    @Test
    //    Test Case 33 		(Key = 2.3.3.3.3.3.2.3.3.3.2.)
    //    Options                                                  :  -b
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  Many
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  Many
    //    Count                                                    :  Many
    //    Size                                                     :  Not empty
    //    Number of occurrences of the string fragment in the file :  Many
    //    Number of occurrences of the string fragment in one line :  Many
    //    Position of the string fragment in the file              :  Any
    //    Presence of a file corresponding to the name             :  Present
    public void Test33()throws Exception {
        File inputFile = null; // TODO
        String expected = null; // TODO

        String filePath = null;
        if(inputFile != null) {
            filePath = inputFile.getPath();
        }

        String[] args = {"-b","ab~c","x!#@$y|z","--",filePath};
        Main.main(args);

        String actual = null;
        if (inputFile != null){
            actual = getFileContent(inputFile.getPath());
        }
        assertEquals("The files are different!",expected, actual);   }
    @Test
    //    Test Case 34 		(Key = 3.3.2.3.2.3.2.2.2.3.2.)
    //    Options                                                  :  -f
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  One
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  One
    //    Count                                                    :  Many
    //    Size                                                     :  Not empty
    //    Number of occurrences of the string fragment in the file :  One
    //    Number of occurrences of the string fragment in one line :  One
    //    Position of the string fragment in the file              :  Any
    //    Presence of a file corresponding to the name             :  Present
    public void Test34()throws Exception{
        File inputFile = null; // TODO
        String expected = null; // TODO

        String filePath = null;
        if(inputFile != null) {
            filePath = inputFile.getPath();
        }

        String[] args = {"-f","ab~c","x!#@$y|z","--",filePath};
        Main.main(args);

        String actual = null;
        if (inputFile != null){
            actual = getFileContent(inputFile.getPath());
        }
        assertEquals("The files are different!",expected, actual);   }
    @Test
    //    Test Case 35 		(Key = 3.3.2.3.2.3.2.2.3.3.2.)
    //    Options                                                  :  -f
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  One
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  One
    //    Count                                                    :  Many
    //    Size                                                     :  Not empty
    //    Number of occurrences of the string fragment in the file :  One
    //    Number of occurrences of the string fragment in one line :  Many
    //    Position of the string fragment in the file              :  Any
    //    Presence of a file corresponding to the name             :  Present
    public void Test35()throws Exception{
        File inputFile = null; // TODO
        String expected = null; // TODO

        String filePath = null;
        if(inputFile != null) {
            filePath = inputFile.getPath();
        }

        String[] args = {"-f","ab~c","x!#@$y|z","--",filePath};
        Main.main(args);

        String actual = null;
        if (inputFile != null){
            actual = getFileContent(inputFile.getPath());
        }
        assertEquals("The files are different!",expected, actual);   }
    @Test
    //    Test Case 36 		(Key = 3.3.2.3.2.3.2.3.2.3.2.)
    //    Options                                                  :  -f
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  One
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  One
    //    Count                                                    :  Many
    //    Size                                                     :  Not empty
    //    Number of occurrences of the string fragment in the file :  Many
    //    Number of occurrences of the string fragment in one line :  One
    //    Position of the string fragment in the file              :  Any
    //    Presence of a file corresponding to the name             :  Present
    public void Test36()throws Exception{
        File inputFile = null; // TODO
        String expected = null; // TODO

        String filePath = null;
        if(inputFile != null) {
            filePath = inputFile.getPath();
        }

        String[] args = {"-f","ab~c","x!#@$y|z","--",filePath};
        Main.main(args);

        String actual = null;
        if (inputFile != null){
            actual = getFileContent(inputFile.getPath());
        }
        assertEquals("The files are different!",expected, actual);  }
    @Test
    //    Test Case 37 		(Key = 3.3.2.3.2.3.2.3.3.3.2.)
    //    Options                                                  :  -f
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  One
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  One
    //    Count                                                    :  Many
    //    Size                                                     :  Not empty
    //    Number of occurrences of the string fragment in the file :  Many
    //    Number of occurrences of the string fragment in one line :  Many
    //    Position of the string fragment in the file              :  Any
    //    Presence of a file corresponding to the name             :  Present
    public void Test37()throws Exception{
        File inputFile = null; // TODO
        String expected = null; // TODO

        String filePath = null;
        if(inputFile != null) {
            filePath = inputFile.getPath();
        }

        String[] args = {"-f","ab~c","x!#@$y|z","--",filePath};
        Main.main(args);

        String actual = null;
        if (inputFile != null){
            actual = getFileContent(inputFile.getPath());
        }
        assertEquals("The files are different!",expected, actual);   }
    @Test
    //    Test Case 38 		(Key = 3.3.2.3.3.3.2.2.2.3.2.)
    //    Options                                                  :  -f
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  One
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  Many
    //    Count                                                    :  Many
    //    Size                                                     :  Not empty
    //    Number of occurrences of the string fragment in the file :  One
    //    Number of occurrences of the string fragment in one line :  One
    //    Position of the string fragment in the file              :  Any
    //    Presence of a file corresponding to the name             :  Present
    public void Test38()throws Exception{
        File inputFile = null; // TODO
        String expected = null; // TODO

        String filePath = null;
        if(inputFile != null) {
            filePath = inputFile.getPath();
        }

        String[] args = {"-f","ab~c","x!#@$y|z","--",filePath};
        Main.main(args);

        String actual = null;
        if (inputFile != null){
            actual = getFileContent(inputFile.getPath());
        }
        assertEquals("The files are different!",expected, actual);    }
    @Test
    //    Test Case 39 		(Key = 3.3.2.3.3.3.2.2.3.3.2.)
    //    Options                                                  :  -f
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  One
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  Many
    //    Count                                                    :  Many
    //    Size                                                     :  Not empty
    //    Number of occurrences of the string fragment in the file :  One
    //    Number of occurrences of the string fragment in one line :  Many
    //    Position of the string fragment in the file              :  Any
    //    Presence of a file corresponding to the name             :  Present
    public void Test39()throws Exception{
        File inputFile = null; // TODO
        String expected = null; // TODO

        String filePath = null;
        if(inputFile != null) {
            filePath = inputFile.getPath();
        }

        String[] args = {"-f","ab~c","x!#@$y|z","--",filePath};
        Main.main(args);

        String actual = null;
        if (inputFile != null){
            actual = getFileContent(inputFile.getPath());
        }
        assertEquals("The files are different!",expected, actual); }
    @Test
    //    Test Case 40 		(Key = 3.3.2.3.3.3.2.3.2.3.2.)
    //    Options                                                  :  -f
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  One
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  Many
    //    Count                                                    :  Many
    //    Size                                                     :  Not empty
    //    Number of occurrences of the string fragment in the file :  Many
    //    Number of occurrences of the string fragment in one line :  One
    //    Position of the string fragment in the file              :  Any
    //    Presence of a file corresponding to the name             :  Present
    public void Test40()throws Exception{
        File inputFile = null; // TODO
        String expected = null; // TODO

        String filePath = null;
        if(inputFile != null) {
            filePath = inputFile.getPath();
        }

        String[] args = {"-f","ab~c","x!#@$y|z","--",filePath};
        Main.main(args);

        String actual = null;
        if (inputFile != null){
            actual = getFileContent(inputFile.getPath());
        }
        assertEquals("The files are different!",expected, actual);   }
    @Test
    //    Test Case 41 		(Key = 3.3.2.3.3.3.2.3.3.3.2.)
    //    Options                                                  :  -f
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  One
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  Many
    //    Count                                                    :  Many
    //    Size                                                     :  Not empty
    //    Number of occurrences of the string fragment in the file :  Many
    //    Number of occurrences of the string fragment in one line :  Many
    //    Position of the string fragment in the file              :  Any
    //    Presence of a file corresponding to the name             :  Present
    public void Test41()throws Exception{
        File inputFile = null; // TODO
        String expected = null; // TODO

        String filePath = null;
        if(inputFile != null) {
            filePath = inputFile.getPath();
        }

        String[] args = {"-f","ab~c","x!#@$y|z","--",filePath};
        Main.main(args);

        String actual = null;
        if (inputFile != null){
            actual = getFileContent(inputFile.getPath());
        }
        assertEquals("The files are different!",expected, actual);   }
    @Test
    //    Test Case 42 		(Key = 3.3.3.3.2.3.2.2.2.3.2.)
    //    Options                                                  :  -f
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  Many
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  One
    //    Count                                                    :  Many
    //    Size                                                     :  Not empty
    //    Number of occurrences of the string fragment in the file :  One
    //    Number of occurrences of the string fragment in one line :  One
    //    Position of the string fragment in the file              :  Any
    //    Presence of a file corresponding to the name             :  Present
    public void Test42()throws Exception{
        File inputFile = null; // TODO
        String expected = null; // TODO

        String filePath = null;
        if(inputFile != null) {
            filePath = inputFile.getPath();
        }

        String[] args = {"-f","ab~c","x!#@$y|z","--",filePath};
        Main.main(args);

        String actual = null;
        if (inputFile != null){
            actual = getFileContent(inputFile.getPath());
        }
        assertEquals("The files are different!",expected, actual);   }
    @Test
    //    Test Case 43 		(Key = 3.3.3.3.2.3.2.2.3.3.2.)
    //    Options                                                  :  -f
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  Many
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  One
    //    Count                                                    :  Many
    //    Size                                                     :  Not empty
    //    Number of occurrences of the string fragment in the file :  One
    //    Number of occurrences of the string fragment in one line :  Many
    //    Position of the string fragment in the file              :  Any
    //    Presence of a file corresponding to the name             :  Present
    public void Test43()throws Exception{
        File inputFile = null; // TODO
        String expected = null; // TODO

        String filePath = null;
        if(inputFile != null) {
            filePath = inputFile.getPath();
        }

        String[] args = {"-f","ab~c","x!#@$y|z","--",filePath};
        Main.main(args);

        String actual = null;
        if (inputFile != null){
            actual = getFileContent(inputFile.getPath());
        }
        assertEquals("The files are different!",expected, actual);    }
    @Test
    //    Test Case 44 		(Key = 3.3.3.3.2.3.2.3.2.3.2.)
    //    Options                                                  :  -f
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  Many
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  One
    //    Count                                                    :  Many
    //    Size                                                     :  Not empty
    //    Number of occurrences of the string fragment in the file :  Many
    //    Number of occurrences of the string fragment in one line :  One
    //    Position of the string fragment in the file              :  Any
    //    Presence of a file corresponding to the name             :  Present
    public void Test44()throws Exception{
        File inputFile = null; // TODO
        String expected = null; // TODO

        String filePath = null;
        if(inputFile != null) {
            filePath = inputFile.getPath();
        }

        String[] args = {"-f","ab~c","x!#@$y|z","--",filePath};
        Main.main(args);

        String actual = null;
        if (inputFile != null){
            actual = getFileContent(inputFile.getPath());
        }
        assertEquals("The files are different!",expected, actual);    }
    @Test
    //    Test Case 45 		(Key = 3.3.3.3.2.3.2.3.3.3.2.)
    //    Options                                                  :  -f
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  Many
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  One
    //    Count                                                    :  Many
    //    Size                                                     :  Not empty
    //    Number of occurrences of the string fragment in the file :  Many
    //    Number of occurrences of the string fragment in one line :  Many
    //    Position of the string fragment in the file              :  Any
    //    Presence of a file corresponding to the name             :  Present
    public void Test45()throws Exception{
        File inputFile = null; // TODO
        String expected = null; // TODO

        String filePath = null;
        if(inputFile != null) {
            filePath = inputFile.getPath();
        }

        String[] args = {"-f","ab~c","x!#@$y|z","--",filePath};
        Main.main(args);

        String actual = null;
        if (inputFile != null){
            actual = getFileContent(inputFile.getPath());
        }
        assertEquals("The files are different!",expected, actual);    }
    @Test
    //    Test Case 46 		(Key = 3.3.3.3.3.3.2.2.2.3.2.)
    //    Options                                                  :  -f
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  Many
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  Many
    //    Count                                                    :  Many
    //    Size                                                     :  Not empty
    //    Number of occurrences of the string fragment in the file :  One
    //    Number of occurrences of the string fragment in one line :  One
    //    Position of the string fragment in the file              :  Any
    //    Presence of a file corresponding to the name             :  Present
    public void Test46()throws Exception{
        File inputFile = null; // TODO
        String expected = null; // TODO

        String filePath = null;
        if(inputFile != null) {
            filePath = inputFile.getPath();
        }

        String[] args = {"-f","ab~c","x!#@$y|z","--",filePath};
        Main.main(args);

        String actual = null;
        if (inputFile != null){
            actual = getFileContent(inputFile.getPath());
        }
        assertEquals("The files are different!",expected, actual);    }
    @Test
    //    Test Case 47 		(Key = 3.3.3.3.3.3.2.2.3.3.2.)
    //    Options                                                  :  -f
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  Many
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  Many
    //    Count                                                    :  Many
    //    Size                                                     :  Not empty
    //    Number of occurrences of the string fragment in the file :  One
    //    Number of occurrences of the string fragment in one line :  Many
    //    Position of the string fragment in the file              :  Any
    //    Presence of a file corresponding to the name             :  Present
    public void Test47()throws Exception{
        File inputFile = null; // TODO
        String expected = null; // TODO

        String filePath = null;
        if(inputFile != null) {
            filePath = inputFile.getPath();
        }

        String[] args = {"-f","ab~c","x!#@$y|z","--",filePath};
        Main.main(args);

        String actual = null;
        if (inputFile != null){
            actual = getFileContent(inputFile.getPath());
        }
        assertEquals("The files are different!",expected, actual);   }
    @Test
    //    Test Case 48 		(Key = 3.3.3.3.3.3.2.3.2.3.2.)
    //    Options                                                  :  -f
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  Many
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  Many
    //    Count                                                    :  Many
    //    Size                                                     :  Not empty
    //    Number of occurrences of the string fragment in the file :  Many
    //    Number of occurrences of the string fragment in one line :  One
    //    Position of the string fragment in the file              :  Any
    //    Presence of a file corresponding to the name             :  Present
    public void Test48()throws Exception{
        File inputFile = null; // TODO
        String expected = null; // TODO

        String filePath = null;
        if(inputFile != null) {
            filePath = inputFile.getPath();
        }

        String[] args = {"-f","ab~c","x!#@$y|z","--",filePath};
        Main.main(args);

        String actual = null;
        if (inputFile != null){
            actual = getFileContent(inputFile.getPath());
        }
        assertEquals("The files are different!",expected, actual);    }
    @Test
    //    Test Case 49 		(Key = 3.3.3.3.3.3.2.3.3.3.2.)
    //    Options                                                  :  -f
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  Many
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  Many
    //    Count                                                    :  Many
    //    Size                                                     :  Not empty
    //    Number of occurrences of the string fragment in the file :  Many
    //    Number of occurrences of the string fragment in one line :  Many
    //    Position of the string fragment in the file              :  Any
    //    Presence of a file corresponding to the name             :  Present
    public void Test49()throws Exception {
        File inputFile = null; // TODO
        String expected = null; // TODO

        String filePath = null;
        if(inputFile != null) {
            filePath = inputFile.getPath();
        }

        String[] args = {"-f","ab~c","x!#@$y|z","--",filePath};
        Main.main(args);

        String actual = null;
        if (inputFile != null){
            actual = getFileContent(inputFile.getPath());
        }
        assertEquals("The files are different!",expected, actual);   }
    @Test
    //    Test Case 50 		(Key = 4.3.2.3.2.3.2.2.2.3.2.)
    //    Options                                                  :  -l
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  One
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  One
    //    Count                                                    :  Many
    //    Size                                                     :  Not empty
    //    Number of occurrences of the string fragment in the file :  One
    //    Number of occurrences of the string fragment in one line :  One
    //    Position of the string fragment in the file              :  Any
    //    Presence of a file corresponding to the name             :  Present
    public void Test50()throws Exception{
        File inputFile = null; // TODO
        String expected = null; // TODO

        String filePath = null;
        if(inputFile != null) {
            filePath = inputFile.getPath();
        }

        String[] args = {"-f","ab~c","x!#@$y|z","--",filePath};
        Main.main(args);

        String actual = null;
        if (inputFile != null){
            actual = getFileContent(inputFile.getPath());
        }
        assertEquals("The files are different!",expected, actual);   }
    @Test
    //    Test Case 51 		(Key = 4.3.2.3.2.3.2.2.3.3.2.)
    //    Options                                                  :  -l
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  One
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  One
    //    Count                                                    :  Many
    //    Size                                                     :  Not empty
    //    Number of occurrences of the string fragment in the file :  One
    //    Number of occurrences of the string fragment in one line :  Many
    //    Position of the string fragment in the file              :  Any
    //    Presence of a file corresponding to the name             :  Present
    public void Test51()throws Exception{
        File inputFile = null; // TODO
        String expected = null; // TODO

        String filePath = null;
        if(inputFile != null) {
            filePath = inputFile.getPath();
        }

        String[] args = {"-l","ab~c","x!#@$y|z","--",filePath};
        Main.main(args);

        String actual = null;
        if (inputFile != null){
            actual = getFileContent(inputFile.getPath());
        }
        assertEquals("The files are different!",expected, actual);   }
    @Test
    //    Test Case 52 		(Key = 4.3.2.3.2.3.2.3.2.3.2.)
    //    Options                                                  :  -l
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  One
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  One
    //    Count                                                    :  Many
    //    Size                                                     :  Not empty
    //    Number of occurrences of the string fragment in the file :  Many
    //    Number of occurrences of the string fragment in one line :  One
    //    Position of the string fragment in the file              :  Any
    //    Presence of a file corresponding to the name             :  Present
    public void Test52()throws Exception{
        File inputFile = null; // TODO
        String expected = null; // TODO

        String filePath = null;
        if(inputFile != null) {
            filePath = inputFile.getPath();
        }

        String[] args = {"-l","ab~c","x!#@$y|z","--",filePath};
        Main.main(args);

        String actual = null;
        if (inputFile != null){
            actual = getFileContent(inputFile.getPath());
        }
        assertEquals("The files are different!",expected, actual);   }
    @Test
    //    Test Case 53 		(Key = 4.3.2.3.2.3.2.3.3.3.2.)
    //    Options                                                  :  -l
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  One
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  One
    //    Count                                                    :  Many
    //    Size                                                     :  Not empty
    //    Number of occurrences of the string fragment in the file :  Many
    //    Number of occurrences of the string fragment in one line :  Many
    //    Position of the string fragment in the file              :  Any
    //    Presence of a file corresponding to the name             :  Present
    public void Test53()throws Exception{
        File inputFile = null; // TODO
        String expected = null; // TODO

        String filePath = null;
        if(inputFile != null) {
            filePath = inputFile.getPath();
        }

        String[] args = {"-l","ab~c","x!#@$y|z","--",filePath};
        Main.main(args);

        String actual = null;
        if (inputFile != null){
            actual = getFileContent(inputFile.getPath());
        }
        assertEquals("The files are different!",expected, actual);   }
    @Test
    //    Test Case 54 		(Key = 4.3.2.3.3.3.2.2.2.3.2.)
    //    Options                                                  :  -l
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  One
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  Many
    //    Count                                                    :  Many
    //    Size                                                     :  Not empty
    //    Number of occurrences of the string fragment in the file :  One
    //    Number of occurrences of the string fragment in one line :  One
    //    Position of the string fragment in the file              :  Any
    //    Presence of a file corresponding to the name             :  Present
    public void Test54()throws Exception{
        File inputFile = null; // TODO
        String expected = null; // TODO

        String filePath = null;
        if(inputFile != null) {
            filePath = inputFile.getPath();
        }

        String[] args = {"-l","ab~c","x!#@$y|z","--",filePath};
        Main.main(args);

        String actual = null;
        if (inputFile != null){
            actual = getFileContent(inputFile.getPath());
        }
        assertEquals("The files are different!",expected, actual);  }
    @Test
    //    Test Case 55 		(Key = 4.3.2.3.3.3.2.2.3.3.2.)
    //    Options                                                  :  -l
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  One
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  Many
    //    Count                                                    :  Many
    //    Size                                                     :  Not empty
    //    Number of occurrences of the string fragment in the file :  One
    //    Number of occurrences of the string fragment in one line :  Many
    //    Position of the string fragment in the file              :  Any
    //    Presence of a file corresponding to the name             :  Present
    public void Test55()throws Exception{
        File inputFile = null; // TODO
        String expected = null; // TODO

        String filePath = null;
        if(inputFile != null) {
            filePath = inputFile.getPath();
        }

        String[] args = {"-l","ab~c","x!#@$y|z","--",filePath};
        Main.main(args);

        String actual = null;
        if (inputFile != null){
            actual = getFileContent(inputFile.getPath());
        }
        assertEquals("The files are different!",expected, actual);   }
    @Test
    //    Test Case 56 		(Key = 4.3.2.3.3.3.2.3.2.3.2.)
    //    Options                                                  :  -l
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  One
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  Many
    //    Count                                                    :  Many
    //    Size                                                     :  Not empty
    //    Number of occurrences of the string fragment in the file :  Many
    //    Number of occurrences of the string fragment in one line :  One
    //    Position of the string fragment in the file              :  Any
    //    Presence of a file corresponding to the name             :  Present
    public void Test56()throws Exception{
        File inputFile = null; // TODO
        String expected = null; // TODO

        String filePath = null;
        if(inputFile != null) {
            filePath = inputFile.getPath();
        }

        String[] args = {"-l","ab~c","x!#@$y|z","--",filePath};
        Main.main(args);

        String actual = null;
        if (inputFile != null){
            actual = getFileContent(inputFile.getPath());
        }
        assertEquals("The files are different!",expected, actual);   }
    @Test
    //    Test Case 57 		(Key = 4.3.2.3.3.3.2.3.3.3.2.)
    //    Options                                                  :  -l
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  One
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  Many
    //    Count                                                    :  Many
    //    Size                                                     :  Not empty
    //    Number of occurrences of the string fragment in the file :  Many
    //    Number of occurrences of the string fragment in one line :  Many
    //    Position of the string fragment in the file              :  Any
    //    Presence of a file corresponding to the name             :  Present
    public void Test57()throws Exception{
        File inputFile = null; // TODO
        String expected = null; // TODO

        String filePath = null;
        if(inputFile != null) {
            filePath = inputFile.getPath();
        }

        String[] args = {"-l","ab~c","x!#@$y|z","--",filePath};
        Main.main(args);

        String actual = null;
        if (inputFile != null){
            actual = getFileContent(inputFile.getPath());
        }
        assertEquals("The files are different!",expected, actual);   }
    @Test
    //    Test Case 58 		(Key = 4.3.3.3.2.3.2.2.2.3.2.)
    //    Options                                                  :  -l
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  Many
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  One
    //    Count                                                    :  Many
    //    Size                                                     :  Not empty
    //    Number of occurrences of the string fragment in the file :  One
    //    Number of occurrences of the string fragment in one line :  One
    //    Position of the string fragment in the file              :  Any
    //    Presence of a file corresponding to the name             :  Present
    public void Test58()throws Exception{
        File inputFile = null; // TODO
        String expected = null; // TODO

        String filePath = null;
        if(inputFile != null) {
            filePath = inputFile.getPath();
        }

        String[] args = {"-l","ab~c","x!#@$y|z","--",filePath};
        Main.main(args);

        String actual = null;
        if (inputFile != null){
            actual = getFileContent(inputFile.getPath());
        }
        assertEquals("The files are different!",expected, actual);   }
    @Test
    //    Test Case 59 		(Key = 4.3.3.3.2.3.2.2.3.3.2.)
    //    Options                                                  :  -l
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  Many
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  One
    //    Count                                                    :  Many
    //    Size                                                     :  Not empty
    //    Number of occurrences of the string fragment in the file :  One
    //    Number of occurrences of the string fragment in one line :  Many
    //    Position of the string fragment in the file              :  Any
    //    Presence of a file corresponding to the name             :  Present
    public void Test59()throws Exception{
        File inputFile = null; // TODO
        String expected = null; // TODO

        String filePath = null;
        if(inputFile != null) {
            filePath = inputFile.getPath();
        }

        String[] args = {"-l","ab~c","x!#@$y|z","--",filePath};
        Main.main(args);

        String actual = null;
        if (inputFile != null){
            actual = getFileContent(inputFile.getPath());
        }
        assertEquals("The files are different!",expected, actual);    }
    @Test
    //    Test Case 60 		(Key = 4.3.3.3.2.3.2.3.2.3.2.)
    //    Options                                                  :  -l
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  Many
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  One
    //    Count                                                    :  Many
    //    Size                                                     :  Not empty
    //    Number of occurrences of the string fragment in the file :  Many
    //    Number of occurrences of the string fragment in one line :  One
    //    Position of the string fragment in the file              :  Any
    //    Presence of a file corresponding to the name             :  Present
    public void Test60()throws Exception{
        File inputFile = null; // TODO
        String expected = null; // TODO

        String filePath = null;
        if(inputFile != null) {
            filePath = inputFile.getPath();
        }

        String[] args = {"-l","ab~c","x!#@$y|z","--",filePath};
        Main.main(args);

        String actual = null;
        if (inputFile != null){
            actual = getFileContent(inputFile.getPath());
        }
        assertEquals("The files are different!",expected, actual);   }
    @Test
    //    Test Case 61 		(Key = 4.3.3.3.2.3.2.3.3.3.2.)
    //    Options                                                  :  -l
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  Many
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  One
    //    Count                                                    :  Many
    //    Size                                                     :  Not empty
    //    Number of occurrences of the string fragment in the file :  Many
    //    Number of occurrences of the string fragment in one line :  Many
    //    Position of the string fragment in the file              :  Any
    //    Presence of a file corresponding to the name             :  Present
    public void Test61()throws Exception{
        File inputFile = null; // TODO
        String expected = null; // TODO

        String filePath = null;
        if(inputFile != null) {
            filePath = inputFile.getPath();
        }

        String[] args = {"-l","ab~c","x!#@$y|z","--",filePath};
        Main.main(args);

        String actual = null;
        if (inputFile != null){
            actual = getFileContent(inputFile.getPath());
        }
        assertEquals("The files are different!",expected, actual);   }
    @Test
    //    Test Case 62 		(Key = 4.3.3.3.3.3.2.2.2.3.2.)
    //    Options                                                  :  -l
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  Many
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  Many
    //    Count                                                    :  Many
    //    Size                                                     :  Not empty
    //    Number of occurrences of the string fragment in the file :  One
    //    Number of occurrences of the string fragment in one line :  One
    //    Position of the string fragment in the file              :  Any
    //    Presence of a file corresponding to the name             :  Present
    public void Test62()throws Exception{
        File inputFile = null; // TODO
        String expected = null; // TODO

        String filePath = null;
        if(inputFile != null) {
            filePath = inputFile.getPath();
        }

        String[] args = {"-l","ab~c","x!#@$y|z","--",filePath};
        Main.main(args);

        String actual = null;
        if (inputFile != null){
            actual = getFileContent(inputFile.getPath());
        }
        assertEquals("The files are different!",expected, actual);   }
    @Test
    //    Test Case 63 		(Key = 4.3.3.3.3.3.2.2.3.3.2.)
    //    Options                                                  :  -l
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  Many
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  Many
    //    Count                                                    :  Many
    //    Size                                                     :  Not empty
    //    Number of occurrences of the string fragment in the file :  One
    //    Number of occurrences of the string fragment in one line :  Many
    //    Position of the string fragment in the file              :  Any
    //    Presence of a file corresponding to the name             :  Present
    public void Test63()throws Exception{
        File inputFile = null; // TODO
        String expected = null; // TODO

        String filePath = null;
        if(inputFile != null) {
            filePath = inputFile.getPath();
        }

        String[] args = {"-l","ab~c","x!#@$y|z","--",filePath};
        Main.main(args);

        String actual = null;
        if (inputFile != null){
            actual = getFileContent(inputFile.getPath());
        }
        assertEquals("The files are different!",expected, actual);   }
    @Test
    //    Test Case 64 		(Key = 4.3.3.3.3.3.2.3.2.3.2.)
    //    Options                                                  :  -l
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  Many
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  Many
    //    Count                                                    :  Many
    //    Size                                                     :  Not empty
    //    Number of occurrences of the string fragment in the file :  Many
    //    Number of occurrences of the string fragment in one line :  One
    //    Position of the string fragment in the file              :  Any
    //    Presence of a file corresponding to the name             :  Present
    public void Test64()throws Exception{
        File inputFile = null; // TODO
        String expected = null; // TODO

        String filePath = null;
        if(inputFile != null) {
            filePath = inputFile.getPath();
        }

        String[] args = {"-l","ab~c","x!#@$y|z","--",filePath};
        Main.main(args);

        String actual = null;
        if (inputFile != null){
            actual = getFileContent(inputFile.getPath());
        }
        assertEquals("The files are different!",expected, actual);   }
    @Test
    //    Test Case 65 		(Key = 4.3.3.3.3.3.2.3.3.3.2.)
    //    Options                                                  :  -l
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  Many
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  Many
    //    Count                                                    :  Many
    //    Size                                                     :  Not empty
    //    Number of occurrences of the string fragment in the file :  Many
    //    Number of occurrences of the string fragment in one line :  Many
    //    Position of the string fragment in the file              :  Any
    //    Presence of a file corresponding to the name             :  Present
    public void Test65()throws Exception{
        File inputFile = null; // TODO
        String expected = null; // TODO

        String filePath = null;
        if(inputFile != null) {
            filePath = inputFile.getPath();
        }

        String[] args = {"-l","ab~c","x!#@$y|z","--",filePath};
        Main.main(args);

        String actual = null;
        if (inputFile != null){
            actual = getFileContent(inputFile.getPath());
        }
        assertEquals("The files are different!",expected, actual);   }
    @Test
    //    Test Case 66 		(Key = 5.3.2.3.2.3.2.2.2.3.2.)
    //    Options                                                  :  -i
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  One
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  One
    //    Count                                                    :  Many
    //    Size                                                     :  Not empty
    //    Number of occurrences of the string fragment in the file :  One
    //    Number of occurrences of the string fragment in one line :  One
    //    Position of the string fragment in the file              :  Any
    //    Presence of a file corresponding to the name             :  Present
    public void Test66()throws Exception{
        File inputFile = null; // TODO
        String expected = null; // TODO

        String filePath = null;
        if(inputFile != null) {
            filePath = inputFile.getPath();
        }

        String[] args = {"-i","ab~c","x!#@$y|z","--",filePath};
        Main.main(args);

        String actual = null;
        if (inputFile != null){
            actual = getFileContent(inputFile.getPath());
        }
        assertEquals("The files are different!",expected, actual);   }
    @Test
    //    Test Case 67 		(Key = 5.3.2.3.2.3.2.2.3.3.2.)
    //    Options                                                  :  -i
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  One
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  One
    //    Count                                                    :  Many
    //    Size                                                     :  Not empty
    //    Number of occurrences of the string fragment in the file :  One
    //    Number of occurrences of the string fragment in one line :  Many
    //    Position of the string fragment in the file              :  Any
    //    Presence of a file corresponding to the name             :  Present
    public void Test67()throws Exception{
        File inputFile = null; // TODO
        String expected = null; // TODO

        String filePath = null;
        if(inputFile != null) {
            filePath = inputFile.getPath();
        }

        String[] args = {"-i","ab~c","x!#@$y|z","--",filePath};
        Main.main(args);

        String actual = null;
        if (inputFile != null){
            actual = getFileContent(inputFile.getPath());
        }
        assertEquals("The files are different!",expected, actual);   }
    @Test
    //    Test Case 68 		(Key = 5.3.2.3.2.3.2.3.2.3.2.)
    //    Options                                                  :  -i
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  One
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  One
    //    Count                                                    :  Many
    //    Size                                                     :  Not empty
    //    Number of occurrences of the string fragment in the file :  Many
    //    Number of occurrences of the string fragment in one line :  One
    //    Position of the string fragment in the file              :  Any
    //    Presence of a file corresponding to the name             :  Present
    public void Test68()throws Exception{
        File inputFile = null; // TODO
        String expected = null; // TODO

        String filePath = null;
        if(inputFile != null) {
            filePath = inputFile.getPath();
        }

        String[] args = {"-i","ab~c","x!#@$y|z","--",filePath};
        Main.main(args);

        String actual = null;
        if (inputFile != null){
            actual = getFileContent(inputFile.getPath());
        }
        assertEquals("The files are different!",expected, actual);   }
    @Test
    //    Test Case 69 		(Key = 5.3.2.3.2.3.2.3.3.3.2.)
    //    Options                                                  :  -i
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  One
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  One
    //    Count                                                    :  Many
    //    Size                                                     :  Not empty
    //    Number of occurrences of the string fragment in the file :  Many
    //    Number of occurrences of the string fragment in one line :  Many
    //    Position of the string fragment in the file              :  Any
    //    Presence of a file corresponding to the name             :  Present
    public void Test69()throws Exception{
        File inputFile = null; // TODO
        String expected = null; // TODO

        String filePath = null;
        if(inputFile != null) {
            filePath = inputFile.getPath();
        }

        String[] args = {"-i","ab~c","x!#@$y|z","--",filePath};
        Main.main(args);

        String actual = null;
        if (inputFile != null){
            actual = getFileContent(inputFile.getPath());
        }
        assertEquals("The files are different!",expected, actual);   }
    @Test
    //    Test Case 70 		(Key = 5.3.2.3.3.3.2.2.2.3.2.)
    //    Options                                                  :  -i
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  One
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  Many
    //    Count                                                    :  Many
    //    Size                                                     :  Not empty
    //    Number of occurrences of the string fragment in the file :  One
    //    Number of occurrences of the string fragment in one line :  One
    //    Position of the string fragment in the file              :  Any
    //    Presence of a file corresponding to the name             :  Present
    public void Test70()throws Exception{
        File inputFile = null; // TODO
        String expected = null; // TODO

        String filePath = null;
        if(inputFile != null) {
            filePath = inputFile.getPath();
        }

        String[] args = {"-i","ab~c","x!#@$y|z","--",filePath};
        Main.main(args);

        String actual = null;
        if (inputFile != null){
            actual = getFileContent(inputFile.getPath());
        }
        assertEquals("The files are different!",expected, actual);   }
    @Test
    //    Test Case 71 		(Key = 5.3.2.3.3.3.2.2.3.3.2.)
    //    Options                                                  :  -i
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  One
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  Many
    //    Count                                                    :  Many
    //    Size                                                     :  Not empty
    //    Number of occurrences of the string fragment in the file :  One
    //    Number of occurrences of the string fragment in one line :  Many
    //    Position of the string fragment in the file              :  Any
    //    Presence of a file corresponding to the name             :  Present
    public void Test71()throws Exception{
        File inputFile = null; // TODO
        String expected = null; // TODO

        String filePath = null;
        if(inputFile != null) {
            filePath = inputFile.getPath();
        }

        String[] args = {"-i","ab~c","x!#@$y|z","--",filePath};
        Main.main(args);

        String actual = null;
        if (inputFile != null){
            actual = getFileContent(inputFile.getPath());
        }
        assertEquals("The files are different!",expected, actual);   }
    @Test
    //    Test Case 72 		(Key = 5.3.2.3.3.3.2.3.2.3.2.)
    //    Options                                                  :  -i
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  One
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  Many
    //    Count                                                    :  Many
    //    Size                                                     :  Not empty
    //    Number of occurrences of the string fragment in the file :  Many
    //    Number of occurrences of the string fragment in one line :  One
    //    Position of the string fragment in the file              :  Any
    //    Presence of a file corresponding to the name             :  Present
    public void Test72()throws Exception{
        File inputFile = null; // TODO
        String expected = null; // TODO

        String filePath = null;
        if(inputFile != null) {
            filePath = inputFile.getPath();
        }

        String[] args = {"-i","ab~c","x!#@$y|z","--",filePath};
        Main.main(args);

        String actual = null;
        if (inputFile != null){
            actual = getFileContent(inputFile.getPath());
        }
        assertEquals("The files are different!",expected, actual);    }
    @Test
    //    Test Case 73 		(Key = 5.3.2.3.3.3.2.3.3.3.2.)
    //    Options                                                  :  -i
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  One
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  Many
    //    Count                                                    :  Many
    //    Size                                                     :  Not empty
    //    Number of occurrences of the string fragment in the file :  Many
    //    Number of occurrences of the string fragment in one line :  Many
    //    Position of the string fragment in the file              :  Any
    //    Presence of a file corresponding to the name             :  Present
    public void Test73()throws Exception{
        File inputFile = null; // TODO
        String expected = null; // TODO

        String filePath = null;
        if(inputFile != null) {
            filePath = inputFile.getPath();
        }

        String[] args = {"-i","ab~c","x!#@$y|z","--",filePath};
        Main.main(args);

        String actual = null;
        if (inputFile != null){
            actual = getFileContent(inputFile.getPath());
        }
        assertEquals("The files are different!",expected, actual);    }
    @Test
    //    Test Case 74 		(Key = 5.3.3.3.2.3.2.2.2.3.2.)
    //    Options                                                  :  -i
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  Many
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  One
    //    Count                                                    :  Many
    //    Size                                                     :  Not empty
    //    Number of occurrences of the string fragment in the file :  One
    //    Number of occurrences of the string fragment in one line :  One
    //    Position of the string fragment in the file              :  Any
    //    Presence of a file corresponding to the name             :  Present
    public void Test74()throws Exception{
        File inputFile = null; // TODO
        String expected = null; // TODO

        String filePath = null;
        if(inputFile != null) {
            filePath = inputFile.getPath();
        }

        String[] args = {"-i","ab~c","x!#@$y|z","--",filePath};
        Main.main(args);

        String actual = null;
        if (inputFile != null){
            actual = getFileContent(inputFile.getPath());
        }
        assertEquals("The files are different!",expected, actual);   }
    @Test
    //    Test Case 75 		(Key = 5.3.3.3.2.3.2.2.3.3.2.)
    //    Options                                                  :  -i
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  Many
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  One
    //    Count                                                    :  Many
    //    Size                                                     :  Not empty
    //    Number of occurrences of the string fragment in the file :  One
    //    Number of occurrences of the string fragment in one line :  Many
    //    Position of the string fragment in the file              :  Any
    //    Presence of a file corresponding to the name             :  Present
    public void Test75()throws Exception{
        File inputFile = null; // TODO
        String expected = null; // TODO

        String filePath = null;
        if(inputFile != null) {
            filePath = inputFile.getPath();
        }

        String[] args = {"-i","ab~c","x!#@$y|z","--",filePath};
        Main.main(args);

        String actual = null;
        if (inputFile != null){
            actual = getFileContent(inputFile.getPath());
        }
        assertEquals("The files are different!",expected, actual);   }
    @Test
    //    Test Case 76 		(Key = 5.3.3.3.2.3.2.3.2.3.2.)
    //    Options                                                  :  -i
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  Many
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  One
    //    Count                                                    :  Many
    //    Size                                                     :  Not empty
    //    Number of occurrences of the string fragment in the file :  Many
    //    Number of occurrences of the string fragment in one line :  One
    //    Position of the string fragment in the file              :  Any
    //    Presence of a file corresponding to the name             :  Present
    public void Test76()throws Exception{
        File inputFile = null; // TODO
        String expected = null; // TODO

        String filePath = null;
        if(inputFile != null) {
            filePath = inputFile.getPath();
        }

        String[] args = {"-i","ab~c","x!#@$y|z","--",filePath};
        Main.main(args);

        String actual = null;
        if (inputFile != null){
            actual = getFileContent(inputFile.getPath());
        }
        assertEquals("The files are different!",expected, actual);    }
    @Test
    //    Test Case 77 		(Key = 5.3.3.3.2.3.2.3.3.3.2.)
    //    Options                                                  :  -i
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  Many
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  One
    //    Count                                                    :  Many
    //    Size                                                     :  Not empty
    //    Number of occurrences of the string fragment in the file :  Many
    //    Number of occurrences of the string fragment in one line :  Many
    //    Position of the string fragment in the file              :  Any
    //    Presence of a file corresponding to the name             :  Present
    public void Test77()throws Exception{
        File inputFile = null; // TODO
        String expected = null; // TODO

        String filePath = null;
        if(inputFile != null) {
            filePath = inputFile.getPath();
        }

        String[] args = {"-i","ab~c","x!#@$y|z","--",filePath};
        Main.main(args);

        String actual = null;
        if (inputFile != null){
            actual = getFileContent(inputFile.getPath());
        }
        assertEquals("The files are different!",expected, actual);   }
    @Test
    //    Test Case 78 		(Key = 5.3.3.3.3.3.2.2.2.3.2.)
    //    Options                                                  :  -i
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  Many
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  Many
    //    Count                                                    :  Many
    //    Size                                                     :  Not empty
    //    Number of occurrences of the string fragment in the file :  One
    //    Number of occurrences of the string fragment in one line :  One
    //    Position of the string fragment in the file              :  Any
    //    Presence of a file corresponding to the name             :  Present
    public void Test78()throws Exception{
        File inputFile = null; // TODO
        String expected = null; // TODO

        String filePath = null;
        if(inputFile != null) {
            filePath = inputFile.getPath();
        }

        String[] args = {"-i","ab~c","x!#@$y|z","--",filePath};
        Main.main(args);

        String actual = null;
        if (inputFile != null){
            actual = getFileContent(inputFile.getPath());
        }
        assertEquals("The files are different!",expected, actual);   }
    @Test
    //    Test Case 79 		(Key = 5.3.3.3.3.3.2.2.3.3.2.)
    //    Options                                                  :  -i
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  Many
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  Many
    //    Count                                                    :  Many
    //    Size                                                     :  Not empty
    //    Number of occurrences of the string fragment in the file :  One
    //    Number of occurrences of the string fragment in one line :  Many
    //    Position of the string fragment in the file              :  Any
    //    Presence of a file corresponding to the name             :  Present
    public void Test79()throws Exception{
        File inputFile = null; // TODO
        String expected = null; // TODO

        String filePath = null;
        if(inputFile != null) {
            filePath = inputFile.getPath();
        }

        String[] args = {"-i","ab~c","x!#@$y|z","--",filePath};
        Main.main(args);

        String actual = null;
        if (inputFile != null){
            actual = getFileContent(inputFile.getPath());
        }
        assertEquals("The files are different!",expected, actual);    }
    @Test
    //    Test Case 80 		(Key = 5.3.3.3.3.3.2.3.2.3.2.)
    //    Options                                                  :  -i
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  Many
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  Many
    //    Count                                                    :  Many
    //    Size                                                     :  Not empty
    //    Number of occurrences of the string fragment in the file :  Many
    //    Number of occurrences of the string fragment in one line :  One
    //    Position of the string fragment in the file              :  Any
    //    Presence of a file corresponding to the name             :  Present
    public void Test80()throws Exception{
        File inputFile = null; // TODO
        String expected = null; // TODO

        String filePath = null;
        if(inputFile != null) {
            filePath = inputFile.getPath();
        }

        String[] args = {"-i","ab~c","x!#@$y|z","--",filePath};
        Main.main(args);

        String actual = null;
        if (inputFile != null){
            actual = getFileContent(inputFile.getPath());
        }
        assertEquals("The files are different!",expected, actual);   }
    @Test
    //    Test Case 81 		(Key = 5.3.3.3.3.3.2.3.3.3.2.)
    //    Options                                                  :  -i
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  Many
    //    Length of the string                                     :  More than one
    //    Presence of special characters                           :  Many
    //    Count                                                    :  Many
    //    Size                                                     :  Not empty
    //    Number of occurrences of the string fragment in the file :  Many
    //    Number of occurrences of the string fragment in one line :  Many
    //    Position of the string fragment in the file              :  Any
    //    Presence of a file corresponding to the name             :  Present
    public void Test81()throws Exception{
        File inputFile = null; // TODO
        String expected = null; // TODO

        String filePath = null;
        if(inputFile != null) {
            filePath = inputFile.getPath();
        }

        String[] args = {"-i","ab~c","x!#@$y|z","--",filePath};
        Main.main(args);

        String actual = null;
        if (inputFile != null){
            actual = getFileContent(inputFile.getPath());
        }
        assertEquals("The files are different!",expected, actual);   }
}