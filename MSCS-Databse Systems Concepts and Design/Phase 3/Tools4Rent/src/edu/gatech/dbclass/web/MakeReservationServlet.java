package edu.gatech.dbclass.web;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.google.gson.reflect.TypeToken;
import edu.gatech.dbclass.t4r.Database;
import edu.gatech.dbclass.t4r.JSONConverter;
import edu.gatech.dbclass.t4r.Status;

import javax.servlet.ServletException;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.IOException;
import java.io.PrintWriter;
import java.lang.reflect.Type;
import java.sql.Time;
import java.sql.Timestamp;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

public class MakeReservationServlet extends HttpServlet {

    @Override
    protected void doOptions(HttpServletRequest request, HttpServletResponse response)
            throws ServletException, IOException {
        Options.setAccessControlHeaders(response);
        response.setStatus(HttpServletResponse.SC_OK);
    }

    @Override
    protected void doPost(HttpServletRequest request, HttpServletResponse response)
            throws ServletException, IOException {

        Options.setAccessControlHeaders(response);
        response.setStatus(HttpServletResponse.SC_UNAUTHORIZED);
        response.setContentType("application/json");
        PrintWriter pw = response.getWriter();

        try {
            StringBuffer sb = Options.getRequestData(request);
            JsonParser jp = new JsonParser();
            JsonObject jsonObject = (JsonObject) jp.parse(sb.toString());
            Type collectionType = new TypeToken<Collection<Integer>>(){}.getType();

            int userID = jsonObject.get("userID").getAsInt();
            Timestamp startDate = Timestamp.valueOf(jsonObject.get("startDate").getAsString());
            Timestamp endDate = Timestamp.valueOf(jsonObject.get("endDate").getAsString());
            List<Integer> tools = JSONConverter.getInstance().getGson().fromJson(jsonObject.get("tools"), collectionType);

            Status status = Database.getInstance().addReservation(userID, startDate, endDate, tools);

            if (status.errorCode > 0) {
                response.setStatus(HttpServletResponse.SC_OK);
            }

            pw.println(JSONConverter.getInstance().getGson().toJson(status));
        } catch (Exception e) {
            // crash and burn
            throw new IOException("Error parsing JSON request string");
        }

    }

}
