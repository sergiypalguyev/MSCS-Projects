package edu.gatech.dbclass.t4r;

import java.sql.Timestamp;

public class ReservationSummary {

    private int reservationID;
    private int customerID;
    private String customerUsername;
    private Timestamp startDate;
    private Timestamp endDate;

    public ReservationSummary(int reservationID, int customerID, String customerUsername, Timestamp startDate, Timestamp endDate) {
        this.reservationID = reservationID;
        this.customerID = customerID;
        this.customerUsername = customerUsername;
        this.startDate = startDate;
        this.endDate = endDate;
    }

    public int getReservationID() {
        return reservationID;
    }

    public int getCustomerID() {
        return customerID;
    }

    public String getCustomerUsername() {
        return customerUsername;
    }

    public Timestamp getStartDate() {
        return startDate;
    }

    public Timestamp getEndDate() {
        return endDate;
    }
}
