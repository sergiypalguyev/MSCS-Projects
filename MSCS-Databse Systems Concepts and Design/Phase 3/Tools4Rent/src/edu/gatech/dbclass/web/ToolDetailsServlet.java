package edu.gatech.dbclass.web;

import edu.gatech.dbclass.t4r.Database;
import edu.gatech.dbclass.t4r.JSONConverter;
import edu.gatech.dbclass.t4r.Tool;

import javax.servlet.ServletException;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.IOException;
import java.io.PrintWriter;
import java.math.BigDecimal;

public class ToolDetailsServlet extends HttpServlet {

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
            int toolID = Integer.parseInt(request.getParameter("toolID"));
            Tool tool = Database.getInstance().viewToolDetails(toolID);

            if (tool != null) {
                response.setStatus(HttpServletResponse.SC_OK);
                pw.println(JSONConverter.getInstance().getGson().toJson(tool));
            }
        } catch (Exception e) {
            // crash and burn
            throw new IOException(e.getMessage());
        }
    }
}
