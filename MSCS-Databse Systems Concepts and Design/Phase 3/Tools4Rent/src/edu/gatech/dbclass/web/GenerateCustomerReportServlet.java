package edu.gatech.dbclass.web;

import com.google.gson.JsonObject;
import edu.gatech.dbclass.t4r.CustomerReport;
import edu.gatech.dbclass.t4r.Database;
import edu.gatech.dbclass.t4r.JSONConverter;

import javax.servlet.ServletException;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.List;

public class GenerateCustomerReportServlet extends HttpServlet {
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
            List<CustomerReport> report = Database.getInstance().getCustomerReport();

            if (report != null) {
                JsonObject jo = new JsonObject();
                jo.add("customers", JSONConverter.getInstance().getGson().toJsonTree(report));
                response.setStatus(HttpServletResponse.SC_OK);
                pw.println(JSONConverter.getInstance().getGson().toJson(jo));
            }
            } catch (Exception e) {
            // crash and burn
            throw new IOException(e.getMessage());
        }
    }
}
