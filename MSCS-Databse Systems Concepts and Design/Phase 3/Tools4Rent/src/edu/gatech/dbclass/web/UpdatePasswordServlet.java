package edu.gatech.dbclass.web;

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import edu.gatech.dbclass.t4r.Database;
import edu.gatech.dbclass.t4r.JSONConverter;

import javax.servlet.ServletException;
import javax.servlet.annotation.WebServlet;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.PrintWriter;

public class UpdatePasswordServlet extends HttpServlet {

    @Override
    protected void doOptions(HttpServletRequest request, HttpServletResponse response)
            throws ServletException, IOException {
        Options.setAccessControlHeaders(response);
        response.setStatus(HttpServletResponse.SC_OK);
    }

    @Override
    protected void doPost(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        Options.setAccessControlHeaders(response);
        response.setStatus(HttpServletResponse.SC_UNAUTHORIZED);

        StringBuffer jb = new StringBuffer();
        String line = null;
        try {
            BufferedReader reader = request.getReader();
            while ((line = reader.readLine()) != null)
                jb.append(line);
        } catch (Exception e) { }

        try {
            JsonParser jp = new JsonParser();
            JsonObject jsonObject = (JsonObject) jp.parse(jb.toString());

            int userID = jsonObject.get("userID").getAsInt();
            String newPassword = jsonObject.get("newPassword").getAsString();

            int statusCode = Database.getInstance().updateUserPassword(userID, newPassword);
            String reason = createReason(statusCode, response);

            if (statusCode == 0) {
                statusCode = Database.getInstance().updateClerkHasLoggedInBefore(userID);
                reason = createReason(statusCode, response);
            }

            JsonObject jo = new JsonObject();
            jo.addProperty("statusCode", statusCode);
            jo.addProperty("reason", reason);

            response.setContentType("application/json");

            PrintWriter pw = response.getWriter();
            pw.println(JSONConverter.getInstance().getGson().toJson(jo));

        } catch (Exception e) {
            // crash and burn
            throw new IOException("Error parsing JSON request string");
        }
    }

    private String createReason(int statusCode, HttpServletResponse response) {
        String reason;

        switch (statusCode) {
            case -1:
                reason = "Problem with database connection. Please contact system administrator.";
                break;

            case -2:
                reason = "User is not found.";
                break;

            case -3:
                reason = "Password is too long, must be less than 20 characters";
                break;

            default:
                reason = "Update password successful.";
                response.setStatus(HttpServletResponse.SC_OK);
                break;
        }

        return reason;
    }

}
