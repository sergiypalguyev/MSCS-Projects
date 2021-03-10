package edu.gatech.dbclass.web;

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import edu.gatech.dbclass.t4r.Database;
import edu.gatech.dbclass.t4r.JSONConverter;

import javax.servlet.ServletException;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.IOException;
import java.io.PrintWriter;

public class LoginServlet extends HttpServlet {

    @Override
    protected void doOptions(HttpServletRequest request, HttpServletResponse response)
            throws ServletException, IOException {
        Options.setAccessControlHeaders(response);
        response.setStatus(HttpServletResponse.SC_OK);
    }

    //Ex. http://localhost:8080/login
    @Override
    protected void doPost(HttpServletRequest request, HttpServletResponse response)
            throws ServletException, IOException {

        Options.setAccessControlHeaders(response);
        response.setStatus(HttpServletResponse.SC_UNAUTHORIZED);

        try {
            StringBuffer sb = Options.getRequestData(request);
            JsonParser jp = new JsonParser();
            JsonObject jsonObject = (JsonObject) jp.parse(sb.toString());
            JsonObject jo = new JsonObject();

            String username = jsonObject.get("username").getAsString();
            String password = jsonObject.get("password").getAsString();
            String userType = jsonObject.get("user").getAsString();

            //Test Case, login with a customer account but select Clerk, it currently return "login success"
            int userID = Database.getInstance().loginToT4R(username, password, userType);
            String reason = createReason(userID, username, response);
            String name = Database.getInstance().getUserName(userID);

            jo.addProperty("userID", userID);
            jo.addProperty("reason", reason);
            jo.addProperty("name", name);

            if (userType.equalsIgnoreCase("clerk") && userID > 0) {
                boolean hasLoggedInBefore = Database.getInstance().getClerkHasLoggedInBefore(userID);
                jo.addProperty("has_logged_in_before", hasLoggedInBefore);
            }

            response.setContentType("application/json");

            PrintWriter pw = response.getWriter();
            pw.println(JSONConverter.getInstance().getGson().toJson(jo));

        } catch (Exception e) {
            // crash and burn
            throw new IOException("Error parsing JSON request string");
        }
    }

    private String createReason(int userID, String username, HttpServletResponse response) {
        String reason;

        switch (userID) {
            case -1:
                reason = "Problem with database connection. Please contact system administrator.";
                break;

            case -2:
                reason = "User " + username + " is not found.";
                break;

            case -3:
                reason = "Incorrect password for " + username;
                break;

            case -4:
                reason = username + " is not a customer";
                break;

            case -5:
                reason = username + " is not a clerk.";
                break;

            default:
                reason = "Login successful.";
                response.setStatus(HttpServletResponse.SC_OK);
                break;
        }

        return reason;
    }
}
