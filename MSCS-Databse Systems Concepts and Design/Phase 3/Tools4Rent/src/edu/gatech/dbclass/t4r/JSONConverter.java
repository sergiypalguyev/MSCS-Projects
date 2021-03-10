package edu.gatech.dbclass.t4r;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

import java.lang.reflect.Type;

public class JSONConverter {

    private Gson gson;
    private static JSONConverter jc;

    private JSONConverter()
    {
        this.gson = new GsonBuilder()
                .setDateFormat("yyyy-MM-dd hh:mm:ss")
                .setPrettyPrinting()
                .registerTypeAdapterFactory(new PostProcessingEnabler())
                .create();
    }

    public static JSONConverter getInstance() {
        if (jc == null) {
            jc = new JSONConverter();
        }

        return jc;
    }

    public Gson getGson() {
        return this.gson;
    }

}
