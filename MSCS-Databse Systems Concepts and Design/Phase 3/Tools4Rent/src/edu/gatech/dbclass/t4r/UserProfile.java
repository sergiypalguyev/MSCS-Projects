package edu.gatech.dbclass.t4r;

import java.util.List;

public class UserProfile {

    private UserInfo userInfo;
    private List<Reservation> reservations;

    public UserProfile(UserInfo userInfo, List<Reservation> reservations) {
        this.userInfo = userInfo;
        this.reservations = reservations;
    }

    public UserInfo getUserInfo() {
        return userInfo;
    }

    public List<Reservation> getReservations() {
        return reservations;
    }
}
