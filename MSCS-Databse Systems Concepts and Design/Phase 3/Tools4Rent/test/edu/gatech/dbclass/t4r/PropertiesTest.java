package edu.gatech.dbclass.t4r;

import org.junit.*;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;

public class PropertiesTest {

    private File file = new File("t4r.properties");
    private File originalFile = new File("t4r.properties.orig");
    private FileWriter fw;

    @Before
    public void setup() throws IOException {

        //If you have an existing properties file, it will not override it,
        // but rather save a copy and after the unit test is done, revert the copy back.
        if (this.file.exists()) {
            Files.move(this.file.toPath(), this.originalFile.toPath(), StandardCopyOption.REPLACE_EXISTING);
        }

        this.fw = new FileWriter(this.file);
    }

    @After
    public void teardown() throws IOException {

        if (this.originalFile.exists()) {
            Files.move(this.originalFile.toPath(), this.file.toPath(), StandardCopyOption.REPLACE_EXISTING );
        }
    }

    @Ignore
    @Test
    public void testGetDefaultDBhostname() {

        PropertiesUtil pu = PropertiesUtil.getInstance();
        Assert.assertEquals("127.0.0.1", pu.getDBhostname());
    }

    @Ignore
    @Test
    public void testGetDBhostname() throws IOException {

        //Add the property to test.
        this.fw.write("db.hostname=192.168.1.1");
        this.fw.flush();
        this.fw.close();

        PropertiesUtil pu = PropertiesUtil.getInstance();
        Assert.assertEquals("192.168.1.1", pu.getDBhostname());
    }

}
