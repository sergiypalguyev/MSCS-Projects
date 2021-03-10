package edu.gatech.dbclass.t4r;

public class CreditCard {

    private String name;
    private String number;
    private int exp_month;
    private int exp_year;
    private int cvc;

    public CreditCard(String name, String number, int exp_month, int exp_year, int cvc) {
        this.name = name;
        this.number = number;
        this.exp_month = exp_month;
        this.exp_year = exp_year;
        this.cvc = cvc;
    }

    public String getName() {
        return name;
    }

    public String getNumber() {
        return number;
    }

    public int getExp_month() {
        return exp_month;
    }

    public int getExp_year() {
        return exp_year;
    }

    public int getCvc() {
        return cvc;
    }
}
