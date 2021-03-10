package edu.gatech.dbclass.t4r;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

import java.math.BigDecimal;
import java.sql.Date;
import java.sql.Time;
import java.sql.Timestamp;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.time.LocalDateTime;
import java.util.*;

public class DatabaseTest {

    private Database db;

    @Before
    public void setup() {
        this.db = Database.getInstance();
    }

    @Test
    public void test_DBConnection_Success()
    {
        Assert.assertTrue(this.db.canConnect());
    }

    @Test
    public void test_loginToT4R_Success()
    {
        int userID = this.db.loginToT4R("rviera6", "1234", "customer");

        Assert.assertEquals(3, userID);
    }

    @Test
    public void test_loginToT4R_UserNotFound()
    {
        int userID = this.db.loginToT4R("a", "", "");

        Assert.assertEquals(-2, userID);
    }

    @Test
    public void test_loginToT4R_IncorrectPassword()
    {
        int userID = this.db.loginToT4R("rviera6", "incorrect", "");

        Assert.assertEquals(-3, userID);
    }

    @Test
    public void test_getClerkHasLoggedInBefore_Valid()
    {
        boolean actual = this.db.getClerkHasLoggedInBefore(2);
        boolean expected = true;

        Assert.assertEquals(expected, actual);
    }

    @Test
    public void test_getClerkHasLoggedInBefore_ClerkNotFound()
    {
        boolean actual = this.db.getClerkHasLoggedInBefore(3);
        boolean expected = false;

        Assert.assertEquals(expected, actual);
    }

    @Test
    public void test_updateUserPassword_Success()
    {
        Random random = new Random();
        String password = Long.toString(random.nextLong());

        int actual = this.db.updateUserPassword(2, password);
        int expected = 0;

        Assert.assertEquals(expected, actual);

        int userID = this.db.loginToT4R("clerk1", password, "clerk");

        Assert.assertEquals(2, userID);
    }

    @Test
    public void test_updateUserPassword_UserNotFound()
    {
        Random random = new Random();
        String password = Long.toString(random.nextLong());

        int actual = this.db.updateUserPassword(0, password);
        int expected = -2;

        Assert.assertEquals(expected, actual);
    }

    @Test
    public void test_updateUserPassword_PasswordOutOfRange()
    {
        String password = "1234567890-1234567890";

        int actual = this.db.updateUserPassword(2, password);
        int expected = -3;

        Assert.assertEquals(expected, actual);
    }

    @Test
    public void test_updateClerkHasLoggedInBefore_Success()
    {
        int actual = this.db.updateClerkHasLoggedInBefore(2);
        int expected = 0;

        Assert.assertEquals(expected, actual);
    }

    @Test
    public void test_updateClerkHasLoggedInBefore_UserNotFound()
    {
        int actual = this.db.updateClerkHasLoggedInBefore(0);
        int expected = -2;

        Assert.assertEquals(expected, actual);
    }

    @Test
    public void test_RegisterUser_Success()
    {
        //All Values, cell and work null, home primary
        Status actual = this.db.RegisterNewUser("GTUser0","000", "GTUser5@gatech.edu", "Bu", "B", "Jacket","12345", "GT Way", "Atlanta", "Georgia","123","2223456","123","","","","","","","Home","Buzz P Jacket","1234567890",12,2020,12345);
        int expected = 0;

        //All values, cell phone null, home primary
        if (actual.errorCode == expected) {
            actual = this.db.RegisterNewUser("GTUser1", "111", "GTUser1@gatech.edu", "Buzz", "B", "Jacket", "12345", "GT Way", "Atlanta", "Georgia", "123", "2223456", "123", "123", "1234567", "123", "", "", "", "Home", "Buzz P Jacket", "1234567890", 12, 2020, 12345);
        }

        //All values, home phone null, work primary
        if (actual.errorCode == expected) {
            actual = this.db.RegisterNewUser("GTUser2", "222", "GTUser2@gatech.edu", "Buzzzz", "B", "Jacket", "12345", "GT Way", "Atlanta", "Georgia", "", "", "", "111", "1234567", "333", "222", "2345678", "999", "Work", "Buzz P Jacket", "1234567890", 12, 2020, 12345);
        }

        //All values work phone null, home primary
        if (actual.errorCode == expected) {
            actual = this.db.RegisterNewUser("GTUser3", "333", "GTUser3@gatech.edu", "Buzzzzzz", "B", "Jacket", "12345", "GT Way", "Atlanta", "Georgia", "123", "2223456", "123", "", "", "", "333", "3456789", "000", "Cell", "Buzz P Jacket", "1234567890", 12, 2020, 12345);
        }

        Assert.assertEquals(actual.errorMessage, expected,actual.errorCode);
    }


    @Test
    public void test_RegisterUser_UserExists() {
        //All values work phone null, home primary

        Status actual = this.db.RegisterNewUser("rviera6", "333", "GTUser4@gatech.edu", "Buzzzzzz", "B", "Jacket", "12345", "GT Way", "Atlanta", "Georgia", "123", "2223456", "123", "", "", "", "333", "3456789", "000", "Cell", "Buzz P Jacket", "1234567890", 12, 2020, 12345);
        int expected = -3;

        Assert.assertEquals(actual.errorMessage, expected,actual.errorCode);
    }

    @Test
    public void test_RegisterUser_SQLTrucation() {
        //All values work phone null, home primary

        Status actual = this.db.RegisterNewUser("rviera", "333", "GTUser6@gatech.edu", "Buzzzzzz", "B", "Jacket", "1234567890123", "GT Way", "Atlanta", "Georgia", "123", "2223456", "123", "", "", "", "333", "3456789", "000", "Cell", "Buzz P Jacket", "1234567890", 12, 2020, 12345);
        int expected = -1;

        Assert.assertEquals(actual.errorMessage, expected, actual.errorCode);
    }

    @Test
    public void test_ViewProfile_Success() {
        //All values work phone null, home primary

        UserInfo actual = this.db.ViewProfile(3);

        Assert.assertEquals(actual.status.errorMessage, 0, actual.status.errorCode);
        Assert.assertEquals(actual.status.errorMessage, "Success", actual.status.errorMessage);
        Assert.assertEquals(actual.status.errorMessage, "rviera6@gatech.edu", actual.email);
        Assert.assertEquals(actual.status.errorMessage, "Raul", actual.first_name);
        Assert.assertEquals(actual.status.errorMessage, "E", actual.middle_name);
        Assert.assertEquals(actual.status.errorMessage, "Viera", actual.last_name);
        Assert.assertEquals(actual.status.errorMessage, "123 TestStreet", actual.street);
        Assert.assertEquals(actual.status.errorMessage, "TestCity", actual.city);
        Assert.assertEquals(actual.status.errorMessage, "TestState", actual.state);
        Assert.assertEquals(actual.status.errorMessage, "12345-1234", actual.zip_code);
    }

    @Test
    public void test_Reservations_getCustomerReservations()
    {
        List<Reservation> reservations = this.db.getCustomerReservations(3);
        Reservation r1 = reservations.get(0);

        Assert.assertEquals(7, reservations.size());
        Assert.assertEquals(32, r1.getReservationID());
    }

    @Test
    public void test_addReservation_Success() {
        int userID = 3;
        Timestamp startDate = new Timestamp(117, 11, 25, 11, 30, 00, 00);
        Timestamp endDate = new Timestamp(117, 11, 28, 11, 30, 00, 00);
        ArrayList<Integer> tools = new ArrayList<>();
        tools.add(4);
        tools.add(5);
        Status status = this.db.addReservation(userID, startDate, endDate, tools);
        System.out.println("ReservationID = " + status.errorCode);
        System.out.println("Message = " + status.errorMessage);
        Assert.assertTrue("Reservation ID must be > 0", status.errorCode > 0);
    }

    @Test
    public void test_checkToolAvailability_Success()
    {
        //Full search
        List<Tool> k = this.db.CheckToolAvailability(Timestamp.valueOf("2015-11-15 01:00:00"), Timestamp.valueOf("2015-11-20 02:00:00"), "Ladder Tool", "Straight", "Manual","rigid");
        int expected = 3;
        Assert.assertEquals(expected,k.size());

        //No custom Search
        k = this.db.CheckToolAvailability(Timestamp.valueOf("2015-11-15 01:00:00"), Timestamp.valueOf("2015-11-20 02:00:00"), "Ladder Tool", "Straight", "Manual","");
        expected = 0;
        Assert.assertEquals(expected,k.size());

        //In Reservation
        k = this.db.CheckToolAvailability(Timestamp.valueOf("2017-11-02 01:00:00"), Timestamp.valueOf("2017-11-04 02:00:00"), "Ladder Tool", "Straight", "Manual","rigid");
        expected = 2; // Should be 1
        Assert.assertEquals(expected,k.size());

        //In Reservation
        k = this.db.CheckToolAvailability(Timestamp.valueOf("2017-12-06 01:00:00"), Timestamp.valueOf("2017-12-07 02:00:00"), "Ladder Tool", "Straight", "Manual","rigid");
        expected = 2; // Should be 1
        Assert.assertEquals(expected,k.size());

        //Between reservation and service
        k = this.db.CheckToolAvailability(Timestamp.valueOf("2017-12-11 01:00:00"), Timestamp.valueOf("2017-12-12 02:00:00"), "Ladder Tool", "Straight", "Manual","rigid");
        expected = 3;
        Assert.assertEquals(expected,k.size());

        // In Service
        k = this.db.CheckToolAvailability(Timestamp.valueOf("2017-12-16 01:00:00"), Timestamp.valueOf("2017-12-17 02:00:00"), "Ladder Tool", "Straight", "Manual","rigid");
        expected = 2; // Should be 1
        Assert.assertEquals(expected,k.size());

        //Post service
        k = this.db.CheckToolAvailability(Timestamp.valueOf("2017-12-19 01:00:00"), Timestamp.valueOf("2017-12-19 02:00:00"), "Ladder Tool", "Straight", "Manual","rigid");
        expected = 3;
        Assert.assertEquals(expected,k.size());
    }


    @Test
    public void test_getToolDetails_Success()
    {
        Tool newTool = this.db.viewToolDetails(19);
        List<Accessory> expectedAcc = new ArrayList<Accessory>(){};
        expectedAcc.add(new Accessory("gas tank", 1));
        expectedAcc.add(new Accessory("testing Gas power tool accessory", 1));
        Tool expected = new Tool.PowerTool.Generator(19, "Power Tool","Gas",
                "concrete","Generator","wood", BigDecimal.valueOf(6.5),
                BigDecimal.valueOf(8.5), BigDecimal.valueOf(6.6), "home depot",
                BigDecimal.valueOf(100.00), BigDecimal.valueOf(15.00), BigDecimal.valueOf(40.0),
                BigDecimal.valueOf(110),BigDecimal.valueOf(1),BigDecimal.valueOf(2000),null, BigDecimal.valueOf(3.5), expectedAcc);

        Assert.assertEquals(expected.getShortDesc(),newTool.getShortDesc());
        Assert.assertEquals(expected.getLongDesc(), newTool.getLongDesc());

        newTool = this.db.viewToolDetails(20);
        expected = new Tool.HandTool.Screwdriver(20, "Hand Tool","Manual",
                "flat","Screwdriver","wood", BigDecimal.valueOf(6.5),
                BigDecimal.valueOf(8.5), BigDecimal.valueOf(6.6), "home depot",
                BigDecimal.valueOf(100.00), BigDecimal.valueOf(15.00), BigDecimal.valueOf(40.0), 11);

        Assert.assertEquals(expected.getShortDesc(),newTool.getShortDesc());
        Assert.assertEquals(expected.getLongDesc(), newTool.getLongDesc());

        newTool = this.db.viewToolDetails(5);

        expected = new Tool.LadderTool.Straight(5, "Ladder Tool","Manual",
                "rigid","Straight","wood", BigDecimal.valueOf(6.5),
                BigDecimal.valueOf(8.5), BigDecimal.valueOf(6.6), "home depot",
                BigDecimal.valueOf(100.00), BigDecimal.valueOf(15.00), BigDecimal.valueOf(40.0), BigDecimal.valueOf(2501),4,false);

        Assert.assertEquals(expected.getShortDesc(),newTool.getShortDesc());
        Assert.assertEquals(expected.getLongDesc(), newTool.getLongDesc());

    }

    @Test
    public void test_getToolSummaryReport_Success()
    {
        List<Tool> tools = this.db.getToolReport("Ladder Tool");
        tools = this.db.getToolReport("Hand Tool");
        tools = this.db.getToolReport("Power Tool");
    }
}