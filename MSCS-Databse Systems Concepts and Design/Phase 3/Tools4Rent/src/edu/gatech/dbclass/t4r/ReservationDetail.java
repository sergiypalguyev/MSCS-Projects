package edu.gatech.dbclass.t4r;

import java.math.BigDecimal;
import java.sql.Timestamp;
import java.util.ArrayList;
import java.util.List;

public class ReservationDetail {

    private int reservationID;
    private int customerID;
    private String firstName;
    private String middleName;
    private String lastName;
    private Timestamp startDate;
    private Timestamp endDate;
    private BigDecimal totalDepositPrice;
    private BigDecimal totalRentalPrice;
    private List<Tool> tools;

    public ReservationDetail(int reservationID, int customerID,
                             String firstName, String middleName, String lastName,
                             Timestamp startDate, Timestamp endDate) {
        this.reservationID = reservationID;
        this.customerID = customerID;
        this.firstName = firstName;
        this.middleName = middleName;
        this.lastName = lastName;
        this.startDate = startDate;
        this.endDate = endDate;
        this.totalDepositPrice = new BigDecimal(0);
        this.totalRentalPrice = new BigDecimal(0);
        this.tools = new ArrayList<>();
    }

    public void addTool(Tool tool) {
        this.tools.add(tool);
    }

    public void addTotalDepositPrice(BigDecimal depositPrice) {
        this.totalDepositPrice = this.totalDepositPrice.add(depositPrice);
    }

    public void addTotalRentalPrice(BigDecimal rentalPrice) {
        this.totalRentalPrice = this.totalRentalPrice.add(rentalPrice);
    }

    public int getReservationID() {
        return reservationID;
    }

    public int getCustomerID() {
        return customerID;
    }

    public String getFirstName() {
        return firstName;
    }

    public String getMiddleName() {
        return middleName;
    }

    public String getLastName() {
        return lastName;
    }

    public BigDecimal getTotalDepositPrice() {
        return totalDepositPrice;
    }

    public BigDecimal getTotalRentalPrice() {
        return totalRentalPrice;
    }

    public List<Tool> getTools() {
        return tools;
    }

    public Timestamp getStartDate() {
        return startDate;
    }

    public Timestamp getEndDate() {
        return endDate;
    }
}
