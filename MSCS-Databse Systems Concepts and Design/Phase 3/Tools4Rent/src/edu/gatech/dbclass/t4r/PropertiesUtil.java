package edu.gatech.dbclass.t4r;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.util.Properties;

public class PropertiesUtil {

    private static PropertiesUtil instance;

    public static PropertiesUtil getInstance() {
        if (instance == null)
        {
            instance = new PropertiesUtil();
        }

        return instance;
    }

    public String getDBhostname() {
        Properties p = new Properties();
        String defaultValue = "127.0.0.1";
        InputStream is = null;

        try {
            is = this.getClass().getResourceAsStream("../../../../../../t4r.properties");
        }
        catch (NullPointerException e) { e.printStackTrace(); }

        if (is == null) {
            try {
                is = new FileInputStream("web/t4r.properties");
            }
            catch (FileNotFoundException e) { e.printStackTrace(); }
        }

        try {
            p.load(is);

            return p.getProperty("db.hostname", defaultValue);
        }
        catch (IOException e) {
            e.printStackTrace();
            return defaultValue;
        }
    }
}
