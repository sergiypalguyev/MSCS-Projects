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
import java.util.ArrayList;
import java.util.List;

public class GenerateToolReportServlet extends HttpServlet {

    @Override
    protected void doOptions(HttpServletRequest request, HttpServletResponse response)
            throws ServletException, IOException {
        Options.setAccessControlHeaders(response);
        response.setStatus(HttpServletResponse.SC_OK);
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
            List<Tool> report = Database.getInstance().getToolReport(type);

            if (report != null) {
                JsonObject jo = new JsonObject();
                jo.add("tools", JSONConverter.getInstance().getGson().toJsonTree(report));
                response.setStatus(HttpServletResponse.SC_OK);
                pw.println(JSONConverter.getInstance().getGson().toJson(jo));
            }

        } catch (Exception e) {
            // crash and burn
            throw new IOException(e.getMessage());
        }
    }
}
