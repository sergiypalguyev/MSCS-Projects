package edu.gatech.dbclass.t4r;

import java.math.BigDecimal;
import java.sql.Timestamp;
import java.time.Period;
import java.util.ArrayList;
import java.util.List;

public class Reservation {
    private int reservationID;
    private List<Tool> tools;
    private Timestamp startDate;
    private Timestamp endDate;
    private int pickupUserID;
    private int dropoffUserID;
    private BigDecimal totalRentalPrice;
    private BigDecimal totalDepositPrice;
    private int numberOfDays;

    public Reservation() {
        this.tools = new ArrayList<>();
    }

    public int getReservationID() {
        return reservationID;
    }

    public void setReservationID(int reservationID) {
        this.reservationID = reservationID;
    }

    public List<Tool> getTools() {
        return tools;
    }

    public void setTools(ArrayList tools) {
        this.tools = tools;
    }

    public Timestamp getStartDate() {
        return startDate;
    }

    public void setStartDate(Timestamp startDate) {
        this.startDate = startDate;
        setNumberOfDays();
    }

    public Timestamp getEndDate() {
        return endDate;
    }

    public void setEndDate(Timestamp endDate) {
        this.endDate = endDate;
        setNumberOfDays();
    }

    public int getPickupUserID() {
        return pickupUserID;
    }

    public void setPickupUserID(int pickupUserID) {
        this.pickupUserID = pickupUserID;
    }

    public int getDropoffUserID() {
        return dropoffUserID;
    }

    public void setDropoffUserID(int dropoffUserID) {
        this.dropoffUserID = dropoffUserID;
    }

    public BigDecimal getTotalRentalPrice() {
        return totalRentalPrice;
    }

    public void setTotalRentalPrice(BigDecimal totalRentalPrice) {
        this.totalRentalPrice = totalRentalPrice;
    }

    public BigDecimal getTotalDepositPrice() {
        return totalDepositPrice;
    }

    public void setTotalDepositPrice(BigDecimal totalDepositPrice) {
        this.totalDepositPrice = totalDepositPrice;
    }

    public void addTool(Tool tool) {
        this.tools.add(tool);
    }

    private int setNumberOfDays() {
        if (this.startDate != null && this.endDate != null) {
            numberOfDays = Period.between(
                    startDate.toLocalDateTime().toLocalDate(),
                    endDate.toLocalDateTime().toLocalDate())
                    .getDays();
        }

        return numberOfDays;
    }
}
