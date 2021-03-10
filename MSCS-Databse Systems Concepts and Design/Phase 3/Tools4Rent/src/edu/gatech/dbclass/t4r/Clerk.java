package edu.gatech.dbclass.t4r;

import java.time.LocalDateTime;

public class Clerk extends User {

    private LocalDateTime dateHired;
    private boolean hasLoggedInBefore;
    private int empNum;

    public Clerk()
    {
        super();
        this.dateHired = LocalDateTime.now();
        this.hasLoggedInBefore = false;
        this.empNum = 0;
    }

    public LocalDateTime getDateHired() {
        return dateHired;
    }

    public void setDateHired(LocalDateTime dateHired) {
        this.dateHired = dateHired;
    }

    public boolean hasLoggedInBefore() {
        return hasLoggedInBefore;
    }

    public void setHasLoggedInBefore(boolean hasLoggedInBefore) {
        this.hasLoggedInBefore = hasLoggedInBefore;
    }

    public int getEmpNum() {
        return empNum;
    }

    public void setEmpNum(int empNum) {
        this.empNum = empNum;
    }
}
