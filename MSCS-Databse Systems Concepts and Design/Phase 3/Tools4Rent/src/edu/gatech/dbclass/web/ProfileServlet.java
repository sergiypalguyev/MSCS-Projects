package edu.gatech.dbclass.web;

import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import edu.gatech.dbclass.t4r.*;

import javax.servlet.ServletException;
import javax.servlet.annotation.WebServlet;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.List;

@WebServlet(name = "ProfileServlet")
public class ProfileServlet extends HttpServlet {

    @Override
    protected void doOptions(HttpServletRequest request, HttpServletResponse response)
            throws ServletException, IOException {
        Options.setAccessControlHeaders(response);
        response.setStatus(HttpServletResponse.SC_OK);
    }

    protected void doGet(HttpServletRequest request, HttpServletResponse response)
            throws ServletException, IOException {

        Options.setAccessControlHeaders(response);
        response.setStatus(HttpServletResponse.SC_UNAUTHORIZED);
        response.setContentType("application/json");
        PrintWriter pw = response.getWriter();

        try {
            int userID = Integer.parseInt(request.getParameter("userID"));

            UserInfo userInfo = Database.getInstance().ViewProfile(userID);

            if (userInfo.getStatus().errorCode == 0) {
                List<Reservation> reservations = Database.getInstance().getCustomerReservations(userID);
                UserProfile userProfile = new UserProfile(userInfo, reservations);

                String jsonUserProfile = JSONConverter.getInstance().getGson().toJson(userProfile);

                response.setStatus(HttpServletResponse.SC_OK);
                pw.println(jsonUserProfile);
            }
            else {
                pw.println(JSONConverter.getInstance().getGson().toJson(userInfo.getStatus()));
            }



        } catch (Exception e) {
            // crash and burn
            throw new IOException("Error parsing JSON request string");
        }
    }
}
