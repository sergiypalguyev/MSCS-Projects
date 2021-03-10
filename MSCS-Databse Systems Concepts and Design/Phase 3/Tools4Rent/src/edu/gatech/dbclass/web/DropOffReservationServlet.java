package edu.gatech.dbclass.web;

import edu.gatech.dbclass.t4r.*;

import javax.servlet.ServletException;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.List;

public class DropOffReservationServlet extends HttpServlet {

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
            int reservationID = 0;
            int clerkID = 0;

            if (request.getParameter("reservationID") != null) {
                reservationID = Integer.parseInt(request.getParameter("reservationID"));
            }

            if (request.getParameter("clerkID") != null) {
                clerkID = Integer.parseInt(request.getParameter("clerkID"));
            }

            //get all the reservations
            if (reservationID == 0) {
                List<ReservationSummary> reservations = Database.getInstance().getReservationsToDropOff();

                response.setStatus(HttpServletResponse.SC_OK);
                pw.println(JSONConverter.getInstance().getGson().toJson(reservations));
            }
            //get the reservation summary for a specific reservation
            else if (reservationID > 0 && clerkID == 0) {
                ReservationDetail detail = Database.getInstance().getReservationDetail(reservationID);

                response.setStatus(HttpServletResponse.SC_OK);
                pw.println(JSONConverter.getInstance().getGson().toJson(detail));
            }
            //confirm reservation
            else if (reservationID > 0 && clerkID > 0) {
                Status status = Database.getInstance().setReservationDropOff(reservationID, clerkID);

                response.setStatus(HttpServletResponse.SC_OK);
                pw.println(JSONConverter.getInstance().getGson().toJson(status));
            }
        } catch (Exception e) {
            // crash and burn
            throw new IOException("Error parsing JSON request string");
        }
    }
}
