package edu.gatech.dbclass.web;

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import edu.gatech.dbclass.t4r.*;

import javax.servlet.ServletException;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.List;

public class PickUpReservationServlet extends HttpServlet {

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
            Status status;
            StringBuffer sb = Options.getRequestData(request);

            JsonParser parser = new JsonParser();
            JsonObject jo = (JsonObject) parser.parse(sb.toString());
            int customerID = jo.get("customerID").getAsInt();
            CreditCard cc = JSONConverter.getInstance().getGson().fromJson(jo.get("creditCard"), CreditCard.class);

            status = Database.getInstance().updateCreditCard(customerID, cc);

            if (status.errorCode >= 0) {
                response.setStatus(HttpServletResponse.SC_OK);
            }

            pw.println(JSONConverter.getInstance().getGson().toJson(status));
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
            int reservationID = 0;
            int customerID = 0;
            int clerkID = 0;

            if (request.getParameter("reservationID") != null) {
                reservationID = Integer.parseInt(request.getParameter("reservationID"));
            }

            if (request.getParameter("customerID") != null) {
                customerID = Integer.parseInt(request.getParameter("customerID"));
            }

            if (request.getParameter("clerkID") != null) {
                clerkID = Integer.parseInt(request.getParameter("clerkID"));
            }

            //get all the reservations
            if (reservationID == 0) {
                List<ReservationSummary> reservations = Database.getInstance().getReservationsToPickUp();

                response.setStatus(HttpServletResponse.SC_OK);
                pw.println(JSONConverter.getInstance().getGson().toJson(reservations));
            }
            //get the reservation summary for a specific reservation
            else if (reservationID > 0 && customerID == 0 && clerkID == 0) {
                ReservationDetail detail = Database.getInstance().getReservationDetail(reservationID);

                response.setStatus(HttpServletResponse.SC_OK);
                pw.println(JSONConverter.getInstance().getGson().toJson(detail));
            }
            //get the credit card info based on the customer ID
            else if (reservationID > 0 && customerID > 0 && clerkID == 0) {
                ReservationDetail detail = Database.getInstance().getReservationDetail(reservationID);
                CreditCard creditCard = Database.getInstance().getCustomerCreditCard(customerID);

                JsonObject jo = new JsonObject();
                jo.add("reservation", JSONConverter.getInstance().getGson().toJsonTree(detail));
                jo.add("creditCard", JSONConverter.getInstance().getGson().toJsonTree(creditCard));

                response.setStatus(HttpServletResponse.SC_OK);
                pw.println(JSONConverter.getInstance().getGson().toJson(jo));
            }
            //confirm reservation and update credit card (if necessary)
            else if (reservationID > 0 && customerID >= 0 && clerkID > 0) {
                Status status = Database.getInstance().setReservationPickUp(reservationID, clerkID);
                if (status.errorCode >= 0) {
                    response.setStatus(HttpServletResponse.SC_OK);
                }

                pw.println(JSONConverter.getInstance().getGson().toJson(status));
            }
        } catch (Exception e) {
            // crash and burn
            throw new IOException("Error parsing JSON request string");
        }
    }
}
