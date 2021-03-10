package edu.gatech.dbclass.t4r;

import java.sql.Timestamp;
import java.text.SimpleDateFormat;
import java.time.format.DateTimeFormatter;

public class ClerkReport implements Comparable<ClerkReport> {

    private int clerkID;
    private String firstName;
    private String middleName;
    private String lastName;
    private String email;
    private String dateHired;
    private int numPickUps;
    private int numDropOffs;
    private int combinedTotal;

    private static DateTimeFormatter formatter = DateTimeFormatter.ofPattern("MM/dd/yyyy");

    public ClerkReport(int clerkID, String firstName, String middleName, String lastName, String email,
                       Timestamp dateHired, int numPickUps, int numDropOffs) {
        this.clerkID = clerkID;
        this.firstName = firstName;
        this.middleName = middleName;
        this.lastName = lastName;
        this.email = email;
        this.dateHired = dateHired.toLocalDateTime().format(formatter);
        this.numPickUps = numPickUps;
        this.numDropOffs = numDropOffs;
        this.combinedTotal = this.numPickUps + this.numDropOffs;
    }

    public int getClerkID() {
        return clerkID;
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

    public String  getDateHired() {
        return dateHired;
    }

    public int getNumPickUps() {
        return numPickUps;
    }

    public int getNumDropOffs() {
        return numDropOffs;
    }

    public int getCombinedTotal() {
        return combinedTotal;
    }

    @Override
    public int compareTo(ClerkReport o) {

        if (this.combinedTotal > o.combinedTotal) {
            return -1;
        }
        else if (this.combinedTotal < o.combinedTotal) {
            return 1;
        }

        return 0;
    }
}
