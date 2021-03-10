package edu.gatech.dbclass.web;

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.google.gson.reflect.TypeToken;
import edu.gatech.dbclass.t4r.Database;
import edu.gatech.dbclass.t4r.JSONConverter;
import edu.gatech.dbclass.t4r.Status;
import edu.gatech.dbclass.t4r.Tool;

import javax.servlet.ServletException;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.IOException;
import java.io.PrintWriter;
import java.lang.reflect.Type;
import java.sql.Timestamp;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;

public class ToolAvailabilityServlet extends HttpServlet {

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

            Timestamp startDate = Timestamp.valueOf(jsonObject.get("startDate").getAsString());
            Timestamp endDate = Timestamp.valueOf(jsonObject.get("endDate").getAsString());

            String type = "";
            String subType = "";
            String subOption = "";
            String powerSource = "";

            if (jsonObject.get("type") != null) {
                type = jsonObject.get("type").getAsString();
                if (type.equalsIgnoreCase("All Tools")) {
                    type = "";
                }
            }
            if (jsonObject.get("subType") != null){
                subType = jsonObject.get("subType").getAsString();
            }
            if (jsonObject.get("subOption") != null) {
                subOption = jsonObject.get("subOption").getAsString();
            }
            if (jsonObject.get("powerSource") != null) {
                powerSource = jsonObject.get("powerSource").getAsString();
            }

            List<Tool> tools = Database.getInstance().CheckToolAvailability(startDate, endDate, type, subType, powerSource, subOption);
            if (tools.size() == 0) {
                Status status = new Status();
                status.errorCode = -2;
                status.errorMessage = "No tools are available with these specifications. Please change the scope!";
                pw.println(JSONConverter.getInstance().getGson().toJson(status));
            }
            else if (tools.size() > 10) {
                Status status = new Status();
                status.errorCode = -2;
                status.errorMessage = "More than 10 tools retrieved, please enter a more unique search.";

                pw.println(JSONConverter.getInstance().getGson().toJson(status));
            }
            else{
                response.setStatus(HttpServletResponse.SC_OK);
                pw.println(JSONConverter.getInstance().getGson().toJson(tools));
            }
        } catch (Exception e) {
            // crash and burn
            throw new IOException("Error parsing JSON request string");
        }
    }

    @Override
    protected void doGet(HttpServletRequest request, HttpServletResponse response)
            throws ServletException, IOException {

        Options.setAccessControlHeaders(response);
        response.setStatus(HttpServletResponse.SC_UNAUTHORIZED);
        response.setContentType("application/json");
        PrintWriter pw = response.getWriter();

        try {
            String type = request.getParameter("type");
            String powerSource = request.getParameter("powerSource");
            String subType = request.getParameter("subType");
            Type collectionType = new TypeToken<List<String>>(){}.getType();
            JsonObject jo = new JsonObject();

            List<String> dropdown;
            //Get all tool types, power sources and subtypes
            if (type == null || type.isEmpty()) {
                List<String> types = Database.getInstance().getAllToolTypes();
                List<String> powerSources = Database.getInstance().getAllToolPowerSources();
                List<String> subTypes = Database.getInstance().getAllToolSubTypes();

                types.add(0, "All Tools");

                jo.add("types", JSONConverter.getInstance().getGson().toJsonTree(types, collectionType));
                jo.add("powerSources", JSONConverter.getInstance().getGson().toJsonTree(powerSources, collectionType));
                jo.add("subTypes", JSONConverter.getInstance().getGson().toJsonTree(subTypes, collectionType));
            }
            //Get the power sources and sub types based on a type
            else if (!type.isEmpty() && (powerSource == null || powerSource.isEmpty()) ) {
                List<String> powerSources = Database.getInstance().fillDropDown(type);
                List<String> subTypes = Database.getInstance().getAllToolSubTypes(type);
                jo.add("powerSources", JSONConverter.getInstance().getGson().toJsonTree(powerSources, collectionType));
                jo.add("subTypes", JSONConverter.getInstance().getGson().toJsonTree(subTypes, collectionType));
            }
            //Get the sub types based on the type and power source
            else if (!type.isEmpty() && !powerSource.isEmpty() && (subType == null || subType.isEmpty()) ) {

                if (type.equalsIgnoreCase("All Tools")) {
                    dropdown = Database.getInstance().getAllToolSubTypesFromPowerSource(powerSource);
                }
                else {
                    dropdown = Database.getInstance().fillDropDown(type, powerSource);
                }
                jo.add("subTypes", JSONConverter.getInstance().getGson().toJsonTree(dropdown, collectionType));
            }
            else if (!type.isEmpty() && !powerSource.isEmpty() && !subType.isEmpty()) {
                dropdown = Database.getInstance().getAllToolSubOptionsFromSubType(subType);
                jo.add("subOptions", JSONConverter.getInstance().getGson().toJsonTree(dropdown, collectionType));
            }
            else {
                dropdown = new ArrayList<>();
                jo.add("error", JSONConverter.getInstance().getGson().toJsonTree(dropdown, collectionType));
            }

            response.setStatus(HttpServletResponse.SC_OK);
            pw.println(JSONConverter.getInstance().getGson().toJson(jo));

        } catch (Exception e) {
            // crash and burn
            throw new IOException("Error parsing JSON request string");
        }

    }
}
