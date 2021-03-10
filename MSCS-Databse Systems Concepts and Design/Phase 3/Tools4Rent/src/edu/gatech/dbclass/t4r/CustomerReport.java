package edu.gatech.dbclass.t4r;

public class CustomerReport {

    private int customerID;
    private String firstName;
    private String middleName;
    private String lastName;
    private String email;
    private String areaCode;
    private String phoneNumber;
    private String extension;
    private int total_reservations;
    private int total_tools_rented;

    public CustomerReport(int customerID, String firstName, String middleName, String lastName, String email,
                          String areaCode, String phoneNumber, String extension,
                          int total_reservations, int total_tools_rented) {
        this.customerID = customerID;
        this.firstName = firstName;
        this.middleName = middleName;
        this.lastName = lastName;
        this.email = email;
        this.areaCode = areaCode;
        this.phoneNumber = phoneNumber;
        this.extension = extension;
        this.total_reservations = total_reservations;
        this.total_tools_rented = total_tools_rented;
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

    public String getEmail() {
        return email;
    }

    public String getAreaCode() {
        return areaCode;
    }

    public String getPhoneNumber() {
        return phoneNumber;
    }

    public String getExtension() {
        return extension;
    }

    public int getTotal_reservations() {
        return total_reservations;
    }

    public int getTotal_tools_rented() {
        return total_tools_rented;
    }
}
