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

public class MainTestAddOn {

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
        File file1 = createTmpFile();
        FileWriter fileWriter = new FileWriter(file1);

        fileWriter.write("Howdy Bill," + System.lineSeparator() +
                "This is a test file for the replace utility" + System.lineSeparator() +
                "Let's make sure it has at least a few lines" + System.lineSeparator() +
                "so that we can create some interesting test cases..." + System.lineSeparator() +
                "And let's say \"howdy bill\" again!");

        fileWriter.close();
        return file1;
    }

    private File createInputFile2() throws Exception {
        File file1 = createTmpFile();
        FileWriter fileWriter = new FileWriter(file1);

        fileWriter.write("Howdy Bill," + System.lineSeparator() +
                "This is another test file for the replace utility" + System.lineSeparator() +
                "that contains a list:" + System.lineSeparator() +
                "-a) Item 1" + System.lineSeparator() +
                "-b) Item 2" + System.lineSeparator() +
                "..." + System.lineSeparator() +
                "and says \"howdy Bill\" twice");

        fileWriter.close();
        return file1;
    }

    private File createInputFile3() throws Exception {
        File file1 = createTmpFile();
        FileWriter fileWriter = new FileWriter(file1);

        fileWriter.write("Howdy Bill, have you learned your abc and 123?" + System.lineSeparator() +
                "It is important to know your abc and 123," +
                "so you should study it" + System.lineSeparator() +
                "and then repeat with me: abc and 123");

        fileWriter.close();
        return file1;
    }

    private File createInputFile4() throws Exception {
        File file = createTmpFile();
        FileWriter fileWriter = new FileWriter(file);

        fileWriter.write("");

        fileWriter.close();
        return file;
    }

    private File createInputFile5() throws Exception {
        File file = createTmpFile();
        FileWriter fileWriter = new FileWriter(file);

        fileWriter.write("First line: not replaced" + System.lineSeparator() +
                "Second line: not replaced" + System.lineSeparator() +
                "Third line: not replaced" + System.lineSeparator() +
                "Last line: not replaced" + System.lineSeparator());

        fileWriter.close();
        return file;
    }

    private File createInputFile6() throws Exception {
        File file = createTmpFile();
        FileWriter fileWriter = new FileWriter(file);

        fileWriter.write("ababab");

        fileWriter.close();
        return file;
    }

    private File createInputFile7() throws Exception {
        File file = createTmpFile();
        FileWriter fileWriter = new FileWriter(file);

        fileWriter.write("The goal here is to replace string \"-i\" with" + System.lineSeparator() +
                "string \"-f\". Since we may also want to do multiple replacements," + System.lineSeparator() +
                "we will repeat the two strings here: -i and -f");

        fileWriter.close();
        return file;
    }

    private File createInputFile8() throws Exception {
        File file = createTmpFile();
        FileWriter fileWriter = new FileWriter(file);

        fileWriter.write("Let's have some numbers in the file: 12345678");

        fileWriter.close();
        return file;
    }

    private File createInputFile9() throws Exception {
        File file = createTmpFile();
        FileWriter fileWriter = new FileWriter(file);

        fileWriter.write("-- -- -- --");

        fileWriter.close();
        return file;
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
    public void mainTest7() {
        String args[] = {"-a", "-b"};
        Main.main(args);
        assertEquals("Usage: Replace [-b] [-f] [-l] [-i] <from> <to> -- <filename> [<filename>]*", errStream.toString().trim());
    }

    @Test
    public void mainTest8() throws Exception {
        File inputFile = createInputFile1();
        String args[] = {"Howdy", "Hello", "--", inputFile.getPath()};
        Main.main(args);
        String expected = "Hello Bill," + System.lineSeparator() +
                "This is a test file for the replace utility" + System.lineSeparator() +
                "Let's make sure it has at least a few lines" + System.lineSeparator() +
                "so that we can create some interesting test cases..." + System.lineSeparator() +
                "And let's say \"howdy bill\" again!";
        String actual = getFileContent(inputFile.getPath());
        assertEquals("The files differ!", expected, actual);
        assertFalse(Files.exists(Paths.get(inputFile.getPath() + ".bck")));
        assertEquals("", errStream.toString().trim());
    }

    @Test
    public void mainTest9() throws Exception {
        File inputFile = createInputFile1();
        String args[] = {"howdy", "Hello", "--", inputFile.getPath()};
        Main.main(args);
        String expected = "Howdy Bill," + System.lineSeparator() +
                "This is a test file for the replace utility" + System.lineSeparator() +
                "Let's make sure it has at least a few lines" + System.lineSeparator() +
                "so that we can create some interesting test cases..." + System.lineSeparator() +
                "And let's say \"Hello bill\" again!";
        String actual = getFileContent(inputFile.getPath());
        assertEquals("The files differ!", expected, actual);
        assertFalse(Files.exists(Paths.get(inputFile.getPath() + ".bck")));
        assertEquals("", errStream.toString().trim());
    }

    @Test
    public void mainTest10() throws Exception {
        File inputFile = createInputFile1();
        String args[] = {"-i", "howdy", "Hello", "--", inputFile.getPath()};
        Main.main(args);
        String expected = "Hello Bill," + System.lineSeparator() +
                "This is a test file for the replace utility" + System.lineSeparator() +
                "Let's make sure it has at least a few lines" + System.lineSeparator() +
                "so that we can create some interesting test cases..." + System.lineSeparator() +
                "And let's say \"Hello bill\" again!";
        String actual = getFileContent(inputFile.getPath());
        assertEquals("The files differ!", expected, actual);
        assertFalse(Files.exists(Paths.get(inputFile.getPath() + ".bck")));
        assertEquals("", errStream.toString().trim());
    }

    @Test
    public void mainTest11() throws Exception {
        File inputFile = createInputFile1();
        String args[] = {"-f", "-l", "-i", "Bill", "William", "--", inputFile.getPath()};
        Main.main(args);
        String expected = "Howdy William," + System.lineSeparator() +
                "This is a test file for the replace utility" + System.lineSeparator() +
                "Let's make sure it has at least a few lines" + System.lineSeparator() +
                "so that we can create some interesting test cases..." + System.lineSeparator() +
                "And let's say \"howdy William\" again!";
        String actual = getFileContent(inputFile.getPath());
        assertEquals("The files differ!", expected, actual);
        assertFalse(Files.exists(Paths.get(inputFile.getPath() + ".bck")));
        assertEquals("", errStream.toString().trim());
    }

    @Test
    public void mainTest12() throws Exception {
        File inputFile = createInputFile1();
        String args[] = {"lines" + System.lineSeparator() + "so that ", "lines." +
                System.lineSeparator() + "In this way, ", "--", inputFile.getPath()};
        Main.main(args);
        String expected = "Howdy Bill," + System.lineSeparator() +
                "This is a test file for the replace utility" + System.lineSeparator() +
                "Let's make sure it has at least a few lines." + System.lineSeparator() +
                "In this way, we can create some interesting test cases..." + System.lineSeparator() +
                "And let's say \"howdy bill\" again!";
        String actual = getFileContent(inputFile.getPath());
        assertEquals("The files differ!", expected, actual);
        assertFalse(Files.exists(Paths.get(inputFile.getPath() + ".bck")));
        assertEquals("", errStream.toString().trim());
    }

    @Test
    public void mainTest13() throws Exception {
        File inputFile = createInputFile1();
        String args[] = {"-b", "Let's make sure", "Let's make sure", "--", inputFile.getPath()};
        Main.main(args);
        String expected = getFileContent(inputFile.getPath() + ".bck");
        String actual = getFileContent(inputFile.getPath());
        assertEquals("The files differ!", expected, actual);
        assertTrue(Files.exists(Paths.get(inputFile.getPath() + ".bck")));
        assertEquals("", errStream.toString().trim());
    }

    @Test
    public void mainTest14() throws Exception {
        File inputFile = createInputFile1();
        String expected = getFileContent(inputFile.getPath());
        String args1[] = {"Let's make sure", "We hope", "--", inputFile.getPath()};
        Main.main(args1);
        String args2[] = {"We hope", "Let's make sure", "--", inputFile.getPath()};
        Main.main(args2);
        String actual = getFileContent(inputFile.getPath());
        assertEquals("The files differ!", expected, actual);
        assertFalse(Files.exists(Paths.get(inputFile.getPath() + ".bck")));
        assertEquals("", errStream.toString().trim());
    }

    @Test
    public void mainTest15() throws Exception {
        File inputFile = createInputFile1();
        String args1[] = {"-b", "-f", "-i", "bill", "William", "--", inputFile.getPath()};
        Main.main(args1);
        String actual1 = getFileContent(inputFile.getPath());
        File inputFile2 = createInputFile1();
        String args2[] = {"-f", "-i", "-b", "bill", "William", "--", inputFile2.getPath()};
        Main.main(args2);
        String actual2 = getFileContent(inputFile.getPath());
        File inputFile3 = createInputFile1();
        String args3[] = {"-i", "-b", "-f", "bill", "William", "--", inputFile3.getPath()};
        Main.main(args3);
        String actual3 = getFileContent(inputFile.getPath());
        File inputFile4 = createInputFile1();
        String args4[] = {"-i", "-f", "-b", "bill", "William", "--", inputFile4.getPath()};
        Main.main(args4);
        String actual4 = getFileContent(inputFile.getPath());
        String expected = "Howdy William," + System.lineSeparator() +
                "This is a test file for the replace utility" + System.lineSeparator() +
                "Let's make sure it has at least a few lines" + System.lineSeparator() +
                "so that we can create some interesting test cases..." + System.lineSeparator() +
                "And let's say \"howdy bill\" again!";
        assertEquals("The files differ!", expected, actual1);
        assertEquals("The files differ!", expected, actual2);
        assertEquals("The files differ!", expected, actual3);
        assertEquals("The files differ!", expected, actual4);
        assertTrue(Files.exists(Paths.get(inputFile.getPath() + ".bck")));
        assertEquals("", errStream.toString().trim());
    }

    @Test
    public void mainTest16() throws Exception {
        File inputFile = createInputFile1();
        String expected = getFileContent(inputFile.getPath());
        String args[] = {"Let's", "Let us", inputFile.getPath()};
        Main.main(args);
        assertEquals("Usage: Replace [-b] [-f] [-l] [-i] <from> <to> -- <filename> [<filename>]*", errStream.toString().trim());
    }

    @Test
    public void mainTest17() throws Exception {
        File inputFile = createInputFile1();
        String args[] = {"s", "5", "--", inputFile.getPath()};
        Main.main(args);
        String expected = "Howdy Bill," + System.lineSeparator() +
                "Thi5 i5 a te5t file for the replace utility" + System.lineSeparator() +
                "Let'5 make 5ure it ha5 at lea5t a few line5" + System.lineSeparator() +
                "5o that we can create 5ome intere5ting te5t ca5e5..." + System.lineSeparator() +
                "And let'5 5ay \"howdy bill\" again!";
        String actual = getFileContent(inputFile.getPath());
        assertEquals("The files differ!", expected, actual);
        assertFalse(Files.exists(Paths.get(inputFile.getPath() + ".bck")));
        assertEquals("", errStream.toString().trim());
    }

    @Test
    public void mainTest18() throws Exception {
        File inputFile = createInputFile1();
        String args[] = {"l", "1", "--", inputFile.getPath()};
        Main.main(args);
        String expected = "Howdy Bi11," + System.lineSeparator() +
                "This is a test fi1e for the rep1ace uti1ity" + System.lineSeparator() +
                "Let's make sure it has at 1east a few 1ines" + System.lineSeparator() +
                "so that we can create some interesting test cases..." + System.lineSeparator() +
                "And 1et's say \"howdy bi11\" again!";
        String actual = getFileContent(inputFile.getPath());
        assertEquals("The files differ!", expected, actual);
        assertFalse(Files.exists(Paths.get(inputFile.getPath() + ".bck")));
        assertEquals("", errStream.toString().trim());
    }

    @Test
    public void mainTest19() throws Exception {
        File inputFile = createInputFile1();
        String args[] = {"-i", "l", "1", "--", inputFile.getPath()};
        Main.main(args);
        String expected = "Howdy Bi11," + System.lineSeparator() +
                "This is a test fi1e for the rep1ace uti1ity" + System.lineSeparator() +
                "1et's make sure it has at 1east a few 1ines" + System.lineSeparator() +
                "so that we can create some interesting test cases..." + System.lineSeparator() +
                "And 1et's say \"howdy bi11\" again!";
        String actual = getFileContent(inputFile.getPath());
        assertEquals("The files differ!", expected, actual);
        assertFalse(Files.exists(Paths.get(inputFile.getPath() + ".bck")));
        assertEquals("", errStream.toString().trim());
    }

    @Test
    public void mainTest20() throws Exception {
        File inputFile = createInputFile1();
        String args[] = {"-i", "-f", "let's", "Let us", "Let's", "let us", "--", inputFile.getPath()};
        Main.main(args);
        String expected = "Howdy Bill," + System.lineSeparator() +
                "This is a test file for the replace utility" + System.lineSeparator() +
                "Let us make sure it has at least a few lines" + System.lineSeparator() +
                "so that we can create some interesting test cases..." + System.lineSeparator() +
                "And let us say \"howdy bill\" again!";
        String actual = getFileContent(inputFile.getPath());
        assertEquals("The files differ!", expected, actual);
        assertFalse(Files.exists(Paths.get(inputFile.getPath() + ".bck")));
        assertEquals("", errStream.toString().trim());
    }

    @Test
    public void mainTest21() throws Exception {
        File inputFile = createInputFile7();
        String args[] = {"-i", "-f", "--", inputFile.getPath()};
        Main.main(args);
        assertEquals("Usage: Replace [-b] [-f] [-l] [-i] <from> <to> -- <filename> [<filename>]*", errStream.toString().trim());
    }

    @Test
    public void mainTest22() throws Exception {
        File inputFile = createInputFile7();
        String args[] = {"-i", "-f", "the goal", "The objective", "--", inputFile.getPath()};
        Main.main(args);
        String expected = "The objective here is to replace string \"-i\" with" + System.lineSeparator() +
                "string \"-f\". Since we may also want to do multiple replacements," + System.lineSeparator() +
                "we will repeat the two strings here: -i and -f";
        String actual = getFileContent(inputFile.getPath());
        assertEquals("The files differ!", expected, actual);
        assertFalse(Files.exists(Paths.get(inputFile.getPath() + ".bck")));
        assertEquals("", errStream.toString().trim());
    }

    @Test
    public void mainTest23() throws Exception {
        File inputFile = createInputFile7();
        String args[] = {"the goal", "The objective", "-i", "-f", "--", inputFile.getPath()};
        Main.main(args);
        String expected = "The goal here is to replace string \"-f\" with" + System.lineSeparator() +
                "string \"-f\". Since we may also want to do multiple replacements," + System.lineSeparator() +
                "we will repeat the two strings here: -f and -f";
        String actual = getFileContent(inputFile.getPath());
        assertEquals("The files differ!", expected, actual);
        assertFalse(Files.exists(Paths.get(inputFile.getPath() + ".bck")));
        assertEquals("", errStream.toString().trim());
    }

    @Test
    public void mainTest24() throws Exception {
        File inputFile = createInputFile7();
        String args[] = {"--", "-i", "-f", "--", inputFile.getPath()};
        Main.main(args);
        String expected = "The goal here is to replace string \"-f\" with" + System.lineSeparator() +
                "string \"-f\". Since we may also want to do multiple replacements," + System.lineSeparator() +
                "we will repeat the two strings here: -f and -f";
        String actual = getFileContent(inputFile.getPath());
        assertEquals("The files differ!", expected, actual);
        assertFalse(Files.exists(Paths.get(inputFile.getPath() + ".bck")));
        assertEquals("", errStream.toString().trim());
    }

    @Test
    public void mainTest25() throws Exception {
        File inputFile = createInputFile7();
        String args[] = {"-f", "--", "-i", "-f", "--", inputFile.getPath()};
        Main.main(args);
        String expected = "The goal here is to replace string \"-f\" with" + System.lineSeparator() +
                "string \"-f\". Since we may also want to do multiple replacements," + System.lineSeparator() +
                "we will repeat the two strings here: -i and -f";
        String actual = getFileContent(inputFile.getPath());
        assertEquals("The files differ!", expected, actual);
        assertFalse(Files.exists(Paths.get(inputFile.getPath() + ".bck")));
        assertEquals("", errStream.toString().trim());
    }

    @Test
    public void mainTest26() throws Exception {
        File inputFile = createInputFile7();
        String args[] = {"-l", "--", "-i", "-f", "--", inputFile.getPath()};
        Main.main(args);
        String expected = "The goal here is to replace string \"-i\" with" + System.lineSeparator() +
                "string \"-f\". Since we may also want to do multiple replacements," + System.lineSeparator() +
                "we will repeat the two strings here: -f and -f";
        String actual = getFileContent(inputFile.getPath());
        assertEquals("The files differ!", expected, actual);
        assertFalse(Files.exists(Paths.get(inputFile.getPath() + ".bck")));
        assertEquals("", errStream.toString().trim());
    }

    @Test
    public void mainTest27() throws Exception {
        File inputFile = createInputFile7();
        String args[] = {"-f", "-l", "--", "-i", "-f", "--", inputFile.getPath()};
        Main.main(args);
        String expected = "The goal here is to replace string \"-f\" with" + System.lineSeparator() +
                "string \"-f\". Since we may also want to do multiple replacements," + System.lineSeparator() +
                "we will repeat the two strings here: -f and -f";
        String actual = getFileContent(inputFile.getPath());
        assertEquals("The files differ!", expected, actual);
        assertFalse(Files.exists(Paths.get(inputFile.getPath() + ".bck")));
        assertEquals("", errStream.toString().trim());
    }

    @Test
    public void mainTest28() throws Exception {
        File inputFile = createInputFile8();
        String args[] = {"Let's have some numbers in the file: 12345678", "New content", "--",
                inputFile.getPath()};
        Main.main(args);
        String expected = "New content";
        String actual = getFileContent(inputFile.getPath());
        assertEquals("The files differ!", expected, actual);
        assertFalse(Files.exists(Paths.get(inputFile.getPath() + ".bck")));
        assertEquals("", errStream.toString().trim());
    }

    @Test
    public void mainTest29() throws Exception {
        File inputFile = createInputFile8();
        String args[] = {"Let's have some numbers in the file: 12345678", "", "--", inputFile.getPath()};
        Main.main(args);
        String expected = "";
        String actual = getFileContent(inputFile.getPath());
        assertEquals("The files differ!", expected, actual);
        assertFalse(Files.exists(Paths.get(inputFile.getPath() + ".bck")));
        assertEquals("", errStream.toString().trim());
    }

    @Test
    public void mainTest30() throws Exception {
        File inputFile = createInputFile8();
        String args[] = {"", "Hey", "--", inputFile.getPath()};
        Main.main(args);
        String expected = "Let's have some numbers in the file: 12345678";
        String actual = getFileContent(inputFile.getPath());
        assertEquals("The files differ!", expected, actual);
        assertEquals("Usage: Replace [-b] [-f] [-l] [-i] <from> <to> -- <filename> [<filename>]*", errStream.toString().trim());
    }

    @Test
    public void mainTest31() throws Exception {
        File inputFile = createInputFile8();
        String args[] = {"--", inputFile.getPath()};
        Main.main(args);
        String expected = "Let's have some numbers in the file: 12345678";
        String actual = getFileContent(inputFile.getPath());
        assertEquals("The files differ!", expected, actual);
        assertEquals("Usage: Replace [-b] [-f] [-l] [-i] <from> <to> -- <filename> [<filename>]*", errStream.toString().trim());
    }

    @Test
    public void mainTest32() throws Exception {
        File inputFile = createInputFile1();
        inputFile.delete();
        String args[] = {"a", "b", "--", inputFile.getPath()};
        Main.main(args);
        assertEquals("File " + inputFile.getName() + " not found", errStream.toString().trim());
    }

    @Test
    public void mainTest33() throws Exception {
        File inputFile1 = createInputFile1();
        File inputFile2 = createInputFile2();
        File inputFile3 = createInputFile3();
        inputFile2.delete();

        String args[] = {"-i", "Howdy", "Hello", "--", inputFile1.getPath(), inputFile2.getPath(), inputFile3.getPath()};
        Main.main(args);

        String expected1 = "Hello Bill," + System.lineSeparator() +
                "This is a test file for the replace utility" + System.lineSeparator() +
                "Let's make sure it has at least a few lines" + System.lineSeparator() +
                "so that we can create some interesting test cases..." + System.lineSeparator() +
                "And let's say \"Hello bill\" again!";
        String expected3 = "Hello Bill, have you learned your abc and 123?" + System.lineSeparator() +
                "It is important to know your abc and 123," +
                "so you should study it" + System.lineSeparator() +
                "and then repeat with me: abc and 123";

        String actual1 = getFileContent(inputFile1.getPath());
        String actual3 = getFileContent(inputFile3.getPath());

        assertEquals("The files differ!", expected1, actual1);
        assertEquals("The files differ!", expected3, actual3);

        assertEquals("File " + inputFile2.getName() + " not found", errStream.toString().trim());

        assertFalse(Files.exists(Paths.get(inputFile1.getPath() + ".bck")));
        assertFalse(Files.exists(Paths.get(inputFile2.getPath() + ".bck")));
        assertFalse(Files.exists(Paths.get(inputFile3.getPath() + ".bck")));
    }

    @Test
    public void mainTest34() throws Exception {
        File inputFile1 = createInputFile1();
        File inputFile2 = createInputFile2();
        File inputFile3 = createInputFile3();
        inputFile1.delete();
        inputFile2.delete();

        String args[] = {"-i", "Howdy", "Hello", "--", inputFile1.getPath(), inputFile2.getPath(), inputFile3.getPath()};
        Main.main(args);

        String expected3 = "Hello Bill, have you learned your abc and 123?" + System.lineSeparator() +
                "It is important to know your abc and 123," +
                "so you should study it" + System.lineSeparator() +
                "and then repeat with me: abc and 123";

        String actual3 = getFileContent(inputFile3.getPath());

        assertEquals("The files differ!", expected3, actual3);

        assertEquals("File " + inputFile1.getName() + " not found" + System.lineSeparator() +
                        "File " + inputFile2.getName() + " not found",
                errStream.toString().trim());

        assertFalse(Files.exists(Paths.get(inputFile1.getPath() + ".bck")));
        assertFalse(Files.exists(Paths.get(inputFile2.getPath() + ".bck")));
        assertFalse(Files.exists(Paths.get(inputFile3.getPath() + ".bck")));
    }

    @Test
    public void mainTest35() throws Exception {
        File inputFile = createInputFile5();
        String args[] = {"not replaced", "replaced", "--", inputFile.getPath()};
        Main.main(args);
        String expected = "First line: replaced" + System.lineSeparator() +
                "Second line: replaced" + System.lineSeparator() +
                "Third line: replaced" + System.lineSeparator() +
                "Last line: replaced" + System.lineSeparator();
        String actual = getFileContent(inputFile.getPath());
        assertEquals("The files differ!", expected, actual);
        assertFalse(Files.exists(Paths.get(inputFile.getPath() + ".bck")));
        assertEquals("", errStream.toString().trim());
    }

    @Test
    public void mainTest36() throws Exception {
        File inputFile = createInputFile5();
        String args[] = {"-f", "not replaced", "replaced", "--", inputFile.getPath()};
        Main.main(args);
        String expected = "First line: replaced" + System.lineSeparator() +
                "Second line: not replaced" + System.lineSeparator() +
                "Third line: not replaced" + System.lineSeparator() +
                "Last line: not replaced" + System.lineSeparator();
        String actual = getFileContent(inputFile.getPath());
        assertEquals("The files differ!", expected, actual);
        assertFalse(Files.exists(Paths.get(inputFile.getPath() + ".bck")));
        assertEquals("", errStream.toString().trim());
    }

    @Test
    public void mainTest37() throws Exception {
        File inputFile = createInputFile5();
        String args[] = {"-l", "not replaced", "replaced", "--", inputFile.getPath()};
        Main.main(args);
        String expected = "First line: not replaced" + System.lineSeparator() +
                "Second line: not replaced" + System.lineSeparator() +
                "Third line: not replaced" + System.lineSeparator() +
                "Last line: replaced" + System.lineSeparator();
        String actual = getFileContent(inputFile.getPath());
        assertEquals("The files differ!", expected, actual);
        assertFalse(Files.exists(Paths.get(inputFile.getPath() + ".bck")));
        assertEquals("", errStream.toString().trim());
    }

    @Test
    public void mainTest38() throws Exception {
        File inputFile = createInputFile5();
        String args[] = {"-f", "-l", "not replaced", "replaced", "--", inputFile.getPath()};
        Main.main(args);
        String expected = "First line: replaced" + System.lineSeparator() +
                "Second line: not replaced" + System.lineSeparator() +
                "Third line: not replaced" + System.lineSeparator() +
                "Last line: replaced" + System.lineSeparator();
        String actual = getFileContent(inputFile.getPath());
        assertEquals("The files differ!", expected, actual);
        assertFalse(Files.exists(Paths.get(inputFile.getPath() + ".bck")));
        assertEquals("", errStream.toString().trim());
    }

    @Test
    public void mainTest39() throws Exception {
        File inputFile = createInputFile5();
        String args[] = {"-l", "-f", "not replaced", "replaced", "--", inputFile.getPath()};
        Main.main(args);
        String expected = "First line: replaced" + System.lineSeparator() +
                "Second line: not replaced" + System.lineSeparator() +
                "Third line: not replaced" + System.lineSeparator() +
                "Last line: replaced" + System.lineSeparator();
        String actual = getFileContent(inputFile.getPath());
        assertEquals("The files differ!", expected, actual);
        assertFalse(Files.exists(Paths.get(inputFile.getPath() + ".bck")));
        assertEquals("", errStream.toString().trim());
    }

    @Test
    public void mainTest40() throws Exception {
        File inputFile = createInputFile6();
        String args[] = {"abab", "<repl>", "--", inputFile.getPath()};
        Main.main(args);
        String expected = "<repl>ab";
        String actual = getFileContent(inputFile.getPath());
        assertEquals("The files differ!", expected, actual);
        assertFalse(Files.exists(Paths.get(inputFile.getPath() + ".bck")));
        assertEquals("", errStream.toString().trim());
    }

    @Test
    public void mainTest41() throws Exception {
        File inputFile = createInputFile6();
        String args[] = {"-l", "abab", "<repl>", "--", inputFile.getPath()};
        Main.main(args);
        String expected = "ab<repl>";
        String actual = getFileContent(inputFile.getPath());
        assertEquals("The files differ!", expected, actual);
        assertFalse(Files.exists(Paths.get(inputFile.getPath() + ".bck")));
        assertEquals("", errStream.toString().trim());
    }

    @Test
    public void mainTest42() throws Exception {
        File inputFile = createInputFile6();
        String args[] = {"-f", "-l", "abab", "<repl>", "--", inputFile.getPath()};
        Main.main(args);
        String expected = "<repl>ab";
        String actual = getFileContent(inputFile.getPath());
        assertEquals("The files differ!", expected, actual);
        assertFalse(Files.exists(Paths.get(inputFile.getPath() + ".bck")));
        assertEquals("", errStream.toString().trim());
    }

    @Test
    public void mainTest43() throws Exception {
        File inputFile = createInputFile6();
        String args[] = {"-f", "-f", "-f", "-l", "-l", "-l", "abab", "<repl>", "--", inputFile.getPath()};
        Main.main(args);
        String expected = "<repl>ab";
        String actual = getFileContent(inputFile.getPath());
        assertEquals("The files differ!", expected, actual);
        assertFalse(Files.exists(Paths.get(inputFile.getPath() + ".bck")));
        assertEquals("", errStream.toString().trim());
    }

    @Test
    public void mainTest44() throws Exception {
        File inputFile = createInputFile6();
        String args[] = {"-l", "-f", "--", "abab", "<repl>", "--", inputFile.getPath()};
        Main.main(args);
        String expected = "<repl>ab";
        String actual = getFileContent(inputFile.getPath());
        assertEquals("The files differ!", expected, actual);
        assertFalse(Files.exists(Paths.get(inputFile.getPath() + ".bck")));
        assertEquals("", errStream.toString().trim());
    }

    @Test
    public void mainTest45() throws Exception {
        File inputFile = createInputFile1();
        String args[] = {"Howdy", "Hello", "replace", "REPLACE", "sure it", "sure that it", "--", inputFile.getPath()};
        Main.main(args);
        String expected = "Hello Bill," + System.lineSeparator() +
                "This is a test file for the REPLACE utility" + System.lineSeparator() +
                "Let's make sure that it has at least a few lines" + System.lineSeparator() +
                "so that we can create some interesting test cases..." + System.lineSeparator() +
                "And let's say \"howdy bill\" again!";
        String actual = getFileContent(inputFile.getPath());
        assertEquals("The files differ!", expected, actual);
        assertFalse(Files.exists(Paths.get(inputFile.getPath() + ".bck")));
        assertEquals("", errStream.toString().trim());
    }

    @Test
    public void mainTest46() throws Exception {
        File inputFile = createInputFile1();
        String args[] = {"Howdy", "Hello", "Howdy", "Hey", "--", inputFile.getPath()};
        Main.main(args);
        String expected = "Hello Bill," + System.lineSeparator() +
                "This is a test file for the replace utility" + System.lineSeparator() +
                "Let's make sure it has at least a few lines" + System.lineSeparator() +
                "so that we can create some interesting test cases..." + System.lineSeparator() +
                "And let's say \"howdy bill\" again!";
        String actual = getFileContent(inputFile.getPath());
        assertEquals("The files differ!", expected, actual);
        assertFalse(Files.exists(Paths.get(inputFile.getPath() + ".bck")));
        assertEquals("", errStream.toString().trim());
    }

    @Test
    public void mainTest47() throws Exception {
        File inputFile = createInputFile1();
        String args[] = {"-i", "Howdy", "Yo!", "howdy", "hello", "--", inputFile.getPath()};
        Main.main(args);
        String expected = "Yo! Bill," + System.lineSeparator() +
                "This is a test file for the replace utility" + System.lineSeparator() +
                "Let's make sure it has at least a few lines" + System.lineSeparator() +
                "so that we can create some interesting test cases..." + System.lineSeparator() +
                "And let's say \"Yo! bill\" again!";
        String actual = getFileContent(inputFile.getPath());
        assertEquals("The files differ!", expected, actual);
        assertFalse(Files.exists(Paths.get(inputFile.getPath() + ".bck")));
        assertEquals("", errStream.toString().trim());
    }

    @Test
    public void mainTest48() throws Exception {
        File inputFile = createInputFile7();
        String args[] = {"-f", "goal", "objective", "-i", "-f", "--", inputFile.getPath()};
        Main.main(args);
        String expected = "The objective here is to replace string \"-f\" with" + System.lineSeparator() +
                "string \"-f\". Since we may also want to do multiple replacements," + System.lineSeparator() +
                "we will repeat the two strings here: -i and -f";
        String actual = getFileContent(inputFile.getPath());
        assertEquals("The files differ!", expected, actual);
        assertFalse(Files.exists(Paths.get(inputFile.getPath() + ".bck")));
        assertEquals("", errStream.toString().trim());
    }

    @Test
    public void mainTest49() throws Exception {
        File inputFile = createInputFile9();
        String args[] = {"--", "--", "++", "--", inputFile.getPath()};
        Main.main(args);
        String expected = "++ ++ ++ ++";
        String actual = getFileContent(inputFile.getPath());
        assertEquals("The files differ!", expected, actual);
        assertFalse(Files.exists(Paths.get(inputFile.getPath() + ".bck")));
        assertEquals("", errStream.toString().trim());
    }

    @Test
    public void mainTest50() throws Exception {
        File inputFile = createInputFile9();
        String args[] = {"-f", "--", "--", "++", "--", inputFile.getPath()};
        Main.main(args);
        String expected = "++ -- -- --";
        String actual = getFileContent(inputFile.getPath());
        assertEquals("The files differ!", expected, actual);
        assertFalse(Files.exists(Paths.get(inputFile.getPath() + ".bck")));
        assertEquals("", errStream.toString().trim());
    }

    @Test
    public void mainTest51() throws Exception {
        File inputFile = createInputFile9();
        String args[] = {"-l", "--", "--", "++", "--", inputFile.getPath()};
        Main.main(args);
        String expected = "-- -- -- ++";
        String actual = getFileContent(inputFile.getPath());
        assertEquals("The files differ!", expected, actual);
        assertFalse(Files.exists(Paths.get(inputFile.getPath() + ".bck")));
        assertEquals("", errStream.toString().trim());
    }

    @Test
    public void mainTest52() throws Exception {
        File inputFile = createInputFile9();
        String args[] = {"-f", "-l", "--", "--", "++", "--", inputFile.getPath()};
        Main.main(args);
        String expected = "++ -- -- ++";
        String actual = getFileContent(inputFile.getPath());
        assertEquals("The files differ!", expected, actual);
        assertFalse(Files.exists(Paths.get(inputFile.getPath() + ".bck")));
        assertEquals("", errStream.toString().trim());
    }

    @Test
    public void mainTest53() throws Exception {
        File inputFile1 = temporaryFolder.newFile("_tmpfile");
        File inputFile2 = temporaryFolder.newFile("_tmpfile.bck");
        inputFile1.deleteOnExit();
        inputFile2.deleteOnExit();
        FileWriter fileWriter1 = new FileWriter(inputFile1);
        FileWriter fileWriter2 = new FileWriter(inputFile2);
        fileWriter1.write("Content of file1");
        fileWriter2.write("Content of file2");
        fileWriter1.close();
        fileWriter2.close();

        String args1[] = {"Content of ", "This is ", "--", inputFile1.getPath()};
        Main.main(args1);
        String args2[] = {"Content of ", "", "--", inputFile2.getPath()};
        Main.main(args2);

        String expected1 = "This is file1";
        String actual1 = getFileContent(inputFile1.getPath());
        String expected2 = "file2";
        String actual2 = getFileContent(inputFile2.getPath());

        assertEquals("The files differ!", expected1, actual1);
        assertEquals("The files differ!", expected2, actual2);
        assertTrue(Files.exists(Paths.get(inputFile2.getPath())));
        assertFalse(Files.exists(Paths.get(inputFile2.getPath() + ".bck")));
        assertEquals("", errStream.toString().trim());
    }

    @Test
    public void mainTest54() throws Exception {
        File inputFile1 = temporaryFolder.newFile("_tmpfile");
        File inputFile2 = temporaryFolder.newFile("_tmpfile.bck");
        inputFile1.deleteOnExit();
        inputFile2.deleteOnExit();
        FileWriter fileWriter1 = new FileWriter(inputFile1);
        FileWriter fileWriter2 = new FileWriter(inputFile2);
        fileWriter1.write("Content of file1");
        fileWriter2.write("Content of file2");
        fileWriter1.close();
        fileWriter2.close();

        String args1[] = {"-b", "Content of ", "This is ", "--", inputFile1.getPath()};
        Main.main(args1);

        String expected1 = "Content of file1";
        String actual1 = getFileContent(inputFile1.getPath());

        assertTrue(Files.exists(Paths.get(inputFile1.getPath() + ".bck")));
        assertEquals("Not performing replace for " + inputFile1.getName() + ": Backup file already exists", errStream.toString().trim());
    }

    @Test
    public void mainTest55() throws Exception {
        File inputFile1 = temporaryFolder.newFile("_tmpfile");
        inputFile1.deleteOnExit();
        FileWriter fileWriter1 = new FileWriter(inputFile1);
        fileWriter1.write("Content of file1");
        fileWriter1.close();

        String args1[] = {"-b", "Content of ", "This is ", "--", inputFile1.getPath()};
        Main.main(args1);
        String args2[] = {"Content of file1", "This is a hack...", "--", inputFile1.getPath() + ".bck"};
        Main.main(args2);

        String expected1 = "This is file1";
        String actual1 = getFileContent(inputFile1.getPath());
        String expected2 = "This is a hack...";
        String actual2 = getFileContent(inputFile1.getPath() + ".bck");

        assertEquals("The files differ!", expected1, actual1);
        assertEquals("The files differ!", expected2, actual2);
        assertTrue(Files.exists(Paths.get(inputFile1.getPath())));
        assertTrue(Files.exists(Paths.get(inputFile1.getPath() + ".bck")));
        assertEquals("", errStream.toString().trim());
    }

    @Test
    public void mainTest56() throws Exception {
        File inputFile = temporaryFolder.newFile("_tmpfile.bck");
        inputFile.deleteOnExit();
        FileWriter fileWriter = new FileWriter(inputFile);
        fileWriter.write("Content of file");
        fileWriter.close();

        String args[] = {"-b", "Content of ", "", "--", inputFile.getPath()};
        Main.main(args);

        String expected = "file";
        String actual = getFileContent(inputFile.getPath());

        assertEquals("The files differ!", expected, actual);
        assertTrue(Files.exists(Paths.get(inputFile.getPath() + ".bck")));
        assertEquals("", errStream.toString().trim());
    }

    @Test
    public void mainTest57() throws Exception {
        File inputFile = createInputFile1();
        String args[] = {"-i", "let's", "Let us", "let us", "we will", "--", inputFile
                .getPath()};
        Main.main(args);
        String expected = "Howdy Bill," + System.lineSeparator() +
                "This is a test file for the replace utility" + System.lineSeparator() +
                "we will make sure it has at least a few lines" + System.lineSeparator() +
                "so that we can create some interesting test cases..." + System.lineSeparator() +
                "And we will say \"howdy bill\" again!";
        String actual = getFileContent(inputFile.getPath());
        assertEquals("The files differ!", expected, actual);
        assertFalse(Files.exists(Paths.get(inputFile.getPath() + ".bck")));
        assertEquals("", errStream.toString().trim());
    }

}
