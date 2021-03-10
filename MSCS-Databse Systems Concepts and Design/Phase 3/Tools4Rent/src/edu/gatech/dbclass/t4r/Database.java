package edu.gatech.dbclass.t4r;

import com.mysql.jdbc.MysqlDataTruncation;

import java.math.BigDecimal;
import java.sql.*;
import java.time.Instant;
import java.time.LocalDateTime;
import java.util.*;

public class Database {

    private static Database instance;
    private Connection conn;
    private String connectionString;
    private String host;
    private PropertiesUtil propertiesUtil;
    private final int VIRTUAL_CLERK_ID = 1;
    private final int RENTAL_THRESHOLD = 50;

    private Database() {
        this.propertiesUtil = PropertiesUtil.getInstance();
        this.host = this.propertiesUtil.getDBhostname();
        this.connectionString =
                String.format("jdbc:mysql://%s/cs6400_fa17_team073?user=gatechUser&password=gatech123&useSSL=false",
                        this.host);
    }

    public static Database getInstance() {
        if (instance == null) {
            try {
                Class.forName("com.mysql.jdbc.Driver");
            } catch (ClassNotFoundException e) {
                e.printStackTrace();
                return null;
            }
            instance = new Database();
        }

        return instance;
    }

    private Connection getConnection() throws SQLException {
        try {
            if (this.conn == null || this.conn.isClosed()) {
                DriverManager.setLoginTimeout(5);
                this.conn = DriverManager.getConnection(this.connectionString);
            }

            return conn;

        } catch (SQLException ex) {
            System.out.println("SQLException: " + ex.getMessage());
            System.out.println("SQLState: " + ex.getSQLState());
            System.out.println("VendorError: " + ex.getErrorCode());
            throw ex;
        }
    }

    public boolean canConnect()
    {
        try {
            Connection conn = getConnection();

            if (conn == null)
                return false;
            else
                return !conn.isClosed();

        } catch (SQLException e) {
            e.printStackTrace();
            return false;
        }
    }

    /**
     * Lookup the DB by the given username and check the password match.
     * @param username The username to lookup in the DB.
     * @param password The password of the user.
     * @return  User ID of the user, or <br>
     *          -1 if SQL Exception, like no connection to DB was established, <br>
     *          -2 if user not found, <br>
     *          -3 if password is incorrect, <br>
     *          -4 is user is not a customer, <br>
     *          -5 if user is not a clerk. <br>
     */
    public int loginToT4R(String username, String password, String userType)
    {
        Connection conn = null;
        PreparedStatement ps = null;
        ResultSet rs = null;

        try {
            conn = getConnection();
            String sql = "SELECT userID, password " +
                         "FROM User " +
                         "WHERE username=? ";

            ps = conn.prepareStatement(sql);
            ps.setString(1, username);

            rs = ps.executeQuery();
            int userID;

            //At least one record found
            if (rs.next()) {
                if (rs.getString("password").equals(password)) {
                    userID = rs.getInt("userID");
                }
                else {
                    return -3;
                }

                if (userType.equalsIgnoreCase("customer")) {
                    return doCustomerExists(userID) ? userID : -4;
                }
                else if (userType.equalsIgnoreCase("clerk")) {
                    return doClerkExists(userID) ? userID : -5;
                }
            }

            //No record found
            return -2;
        }
        catch (SQLException e) {
            e.printStackTrace();
            return -1;
        }
        finally {
            try { if (rs != null) rs.close(); } catch (Exception e) {}
            try { if (ps != null) ps.close(); } catch (Exception e) {}
            try { if (conn != null) conn.close(); } catch (Exception e) {}
        }
    }

    public String getUserName(int userID) {
        Connection conn = null;
        PreparedStatement ps = null;
        ResultSet rs = null;

        try {
            conn = getConnection();
            String sql = "SELECT first_name, middle_name, last_name " +
                    "FROM User " +
                    "WHERE userID=? ";

            ps = conn.prepareStatement(sql);
            ps.setInt(1, userID);

            rs = ps.executeQuery();

            String firstName = "";
            String middleName = "";
            String lastName = "";

            if (rs.next()) {
                 firstName = rs.getString("first_name");
                 middleName = rs.getString("middle_name");
                 lastName = rs.getString("last_name");
            }

            if (middleName == null || middleName.isEmpty()) {
                return firstName + " " + lastName;
            }
            else {
                return firstName + " " + middleName + " " + lastName;
            }
        }
        catch (SQLException e) {
            e.printStackTrace();
            return "";
        }
        finally {
            try { if (rs != null) rs.close(); } catch (Exception e) {}
            try { if (ps != null) ps.close(); } catch (Exception e) {}
            try { if (conn != null) conn.close(); } catch (Exception e) {}
        }
    }

    private boolean doCustomerExists(int userID) {
        Connection conn = null;
        PreparedStatement ps = null;
        ResultSet rs = null;

        try {
            conn = getConnection();
            String sql = "SELECT userID " +
                    "FROM Customer " +
                    "WHERE userID=? ";

            ps = conn.prepareStatement(sql);
            ps.setInt(1, userID);

            rs = ps.executeQuery();

            //At least one record found
            if (rs.next()) {
                return rs.getInt("userID") == userID;
            }

            return false;
        }
        catch (SQLException e) {
            e.printStackTrace();
            return false;
        }
        finally {
            try { if (rs != null) rs.close(); } catch (Exception e) {}
            try { if (ps != null) ps.close(); } catch (Exception e) {}
            try { if (conn != null) conn.close(); } catch (Exception e) {}
        }

    }

    private boolean doClerkExists(int userID) {
        Connection conn = null;
        PreparedStatement ps = null;
        ResultSet rs = null;

        try {
            conn = getConnection();
            String sql = "SELECT userID " +
                    "FROM Clerk " +
                    "WHERE userID=? ";

            ps = conn.prepareStatement(sql);
            ps.setInt(1, userID);

            rs = ps.executeQuery();

            //At least one record found
            if (rs.next()) {
                return rs.getInt("userID") == userID;
            }

            return false;
        }
        catch (SQLException e) {
            e.printStackTrace();
            return false;
        }
        finally {
            try { if (rs != null) rs.close(); } catch (Exception e) {}
            try { if (ps != null) ps.close(); } catch (Exception e) {}
            try { if (conn != null) conn.close(); } catch (Exception e) {}
        }

    }

    /**
     * Lookup the DB by the given userID to verify if the Clerk has
     * logged in before.
     * @param userID The userID to lookup.
     * @return True if has logged in before, False otherwise.
     */
    public boolean getClerkHasLoggedInBefore(int userID)
    {
        Connection conn = null;
        PreparedStatement ps = null;
        ResultSet rs = null;

        try {
            conn = getConnection();
            String sql = "SELECT has_logged_in_before " +
                         "FROM Clerk " +
                         "WHERE userID=?";

            ps = conn.prepareStatement(sql);
            ps.setInt(1, userID);

            rs = ps.executeQuery();

            if (rs.next()) {
                return rs.getBoolean("has_logged_in_before");
            }

            return false;
        }
        catch (SQLException e) {
            e.printStackTrace();
            return false;
        }
        finally {
            try { if (rs != null) rs.close(); } catch (Exception e) {}
            try { if (ps != null) ps.close(); } catch (Exception e) {}
            try { if (conn != null) conn.close(); } catch (Exception e) {}
        }
    }

    /**
     * Update the password for a given user.
     * @param userID The user ID for the user to update.
     * @param password The new password.
     * @return The status code of the update. <br>
     *         0 if successfully updated, <br>
     *         -1 if SQL Exception, like no database connection, <br>
     *         -2 if user not found, <br>
     *         -3 if password is too long.
     */
    public int updateUserPassword(int userID, String password)
    {
        Connection conn = null;
        PreparedStatement ps = null;

        try {
            conn = getConnection();
            String sql = "UPDATE User " +
                         "SET password = ?" +
                         "WHERE userID = ?";

            ps = conn.prepareStatement(sql);
            ps.setString(1, password);
            ps.setInt(2, userID);

            int status = ps.executeUpdate() == 1 ? 0 : -2;
            return status;
        }
        catch (MysqlDataTruncation e) {
            return -3;
        }
        catch (SQLException e) {
            e.printStackTrace();
            return -1;
        }
        finally {
            try { if (ps != null) ps.close(); } catch (Exception e) {}
            try { if (conn != null) conn.close(); } catch (Exception e) {}
        }
    }

    /**
     * Set the Clerk has logged in before to True.
     * @param userID The user ID of the Clerk.
     * @return  The status code of the update. <br>
     *          0 if successful, <br>
     *          -1 if SQL Exception, like no connection to DB was established, <br>
     *          -2 if user not found. <br>
     */
    public int updateClerkHasLoggedInBefore(int userID)
    {
        Connection conn = null;
        PreparedStatement ps = null;

        try {
            conn = getConnection();
            String sql = "UPDATE Clerk " +
                         "SET has_logged_in_before = '1' " +
                         "WHERE userID = ?";

            ps = conn.prepareStatement(sql);
            ps.setInt(1, userID);

            int status = ps.executeUpdate() == 1 ? 0 : -2;
            return status;
        }
        catch (SQLException e) {
            e.printStackTrace();
            return -1;
        }
        finally {
            try { if (ps != null) ps.close(); } catch (Exception e) {}
            try { if (conn != null) conn.close(); } catch (Exception e) {}
        }
    }

    /**
     * Register the new User and all associated data.
     * @param username
     * @param password
     * @param email
     * @param first_name
     * etc..
     * @return  The status code of the update. <br>
     *          0 if successful, <br>
     *          -1 if SQL Exception, like no connection to DB was established, <br>
     *          -2 registration failed <br>
     *          -3 user exists <br>
     */

    public Status RegisterNewUser (String username, String password, String email, String first_name, String middle_name, String last_name, String zip_code, String street, String city, String state, String h_area_code, String h_phone_number, String h_extension, String w_area_code, String w_phone_number, String w_extension, String c_area_code, String c_phone_number, String c_extension, String primary, String name, String cred_num, int exp_month, int exp_year, int cvc)
    {
        Connection conn = null;
        PreparedStatement ps = null;
        ResultSet rs = null;
        Status status = new Status();

        try{
            conn = getConnection();

            String sql = "SELECT username " +
                    "FROM User " +
                    "WHERE username=?";

            ps = conn.prepareStatement(sql);
            ps.setString(1, username);

            rs = ps.executeQuery();

            if (rs.next()) {
                status.errorMessage = "Failed at SELECT username, user exists";
                status.errorCode = -3;
                return status;
            }

            String sq2 = "";
            conn.setAutoCommit(false);

            sq2 = "INSERT INTO User (username, password, email, first_name, middle_name, last_name)" +
                    "VALUES (?, ?, ?, ?, ?, ?);";
            ps = conn.prepareStatement(sq2);
            ps.setString(1, username);
            ps.setString(2, password);
            ps.setString(3, email);
            ps.setString(4, first_name);
            ps.setString(5, middle_name);
            ps.setString(6, last_name);
            status.errorMessage = "Failed at INSERT INTO User";
            status.errorCode = ps.executeUpdate();

            sq2 = "INSERT INTO Customer (userID, zip_code, street, city, state)" +
                    "VALUES (LAST_INSERT_ID(), ?, ?, ?, ?);";
            ps = conn.prepareStatement(sq2);
            ps.setString(1, zip_code);
            ps.setString(2, street);
            ps.setString(3, city);
            ps.setString(4, state);
            if(status.errorCode>0) {
                status.errorMessage = "Failed at INSERT INTO Customer";
                status.errorCode = ps.executeUpdate();
            }

            sq2 = "INSERT INTO HomePhone(userID, area_code, phone_number, extension)" +
                    "VALUES(LAST_INSERT_ID(), ?, ?, ?);";
            ps = conn.prepareStatement(sq2);
            if(h_phone_number != null) {
                ps.setString(1, h_area_code);
                ps.setString(2, h_phone_number);
                ps.setString(3, h_extension);
            }
            else {
                ps.setString(1, null);
                ps.setString(2, null);
                ps.setString(3, null);
            }
            if(status.errorCode>0) {
                status.errorMessage = "Failed at INSERT INTO HomePhone";
                status.errorCode = ps.executeUpdate();
            }

            sq2 = "INSERT INTO WorkPhone(userID, area_code, phone_number, extension)" +
                    "VALUES(LAST_INSERT_ID(), ?, ?, ?);";
            ps = conn.prepareStatement(sq2);
            if(w_phone_number != null) {
                ps.setString(1, w_area_code);
                ps.setString(2, w_phone_number);
                ps.setString(3, w_extension);
            }
            else{
                ps.setString(1, null);
                ps.setString(2, null);
                ps.setString(3, null);
            }
            if(status.errorCode>0) {
                status.errorMessage = "Failed at INSERT INTO WorkPhone";
                status.errorCode = ps.executeUpdate();
            }

            sq2 = "INSERT INTO CellPhone(userID, area_code, phone_number, extension)" +
                    "VALUES(LAST_INSERT_ID(), ?, ?, ?);";
            ps = conn.prepareStatement(sq2);
            if(c_phone_number != null) {
                ps.setString(1, c_area_code);
                ps.setString(2, c_phone_number);
                ps.setString(3, c_extension);
            }
            else{
                ps.setString(1, null);
                ps.setString(2, null);
                ps.setString(3, null);
            }
            if(status.errorCode>0) {
                status.errorMessage = "Failed at INSERT INTO CellPhone";
                status.errorCode = ps.executeUpdate();
            }

            sq2 = "INSERT INTO PrimaryPhone (userID, area_code, phone_number, extension) VALUES (LAST_INSERT_ID(), ?, ?, ?);";
            ps = conn.prepareStatement(sq2);
            switch (primary.toUpperCase()){
                case "HOME":
                    ps.setString(1, h_area_code);
                    ps.setString(2, h_phone_number);
                    ps.setString(3, h_extension);break;
                case "CELL":
                    ps.setString(1, c_area_code);
                    ps.setString(2, c_phone_number);
                    ps.setString(3, c_extension);break;
                case "WORK":
                    ps.setString(1, w_area_code);
                    ps.setString(2, w_phone_number);
                    ps.setString(3, w_extension);break;
                default:break;
            }
            if(status.errorCode>0) {
                status.errorMessage = "Failed at INSERT INTO PrimaryPhone";
                status.errorCode = ps.executeUpdate();
            }

            sq2 = "INSERT INTO CreditCard (userID, name, cred_number, exp_month, exp_year, cvc) VALUES (LAST_INSERT_ID(), ?, ?, ?, ?, ?);";
            ps = conn.prepareStatement(sq2);
            ps.setString(1, name);
            ps.setString(2, cred_num);
            ps.setInt(3, exp_month);
            ps.setInt(4, exp_year);
            ps.setInt(5, cvc);
            if(status.errorCode>0) {
                status.errorMessage = "Failed at INSERT INTO CreditCard";
                status.errorCode = ps.executeUpdate();
            }

            conn.commit();

            if(status.errorCode == 1) status.errorMessage = "Success";
            status.errorCode = status.errorCode == 1 ? 0 : -2;
            return status;
        }
        catch(SQLException e) {
            e.printStackTrace();
            status.errorMessage = "SQL Exception occured";
            status.errorCode = -1;
            return status;
        }
        finally {
            try { if (ps != null) ps.close(); } catch (Exception e) {}
            try { if (conn != null) conn.close(); } catch (Exception e) {}
        }
    }

    public UserInfo ViewProfile (int userID) {

        Connection conn = null;
        PreparedStatement ps = null;
        ResultSet rs = null;
        UserInfo userInfo = new UserInfo();
        userInfo.status = new Status();

        try {
            conn = getConnection();

            String sql = "SELECT email, first_name, middle_name, last_name, zip_code, street, city, state, " +
                    "hp.area_code AS hp_ac, hp.phone_number AS hp_pn, hp.extension AS hp_ext, " +
                    "cp.area_code AS cp_ac, cp.phone_number AS cp_pn, cp.extension AS cp_ext, " +
                    "wp.area_code AS wp_ac, wp.phone_number AS wp_pn, wp.extension AS wp_ext " +
                    "FROM User INNER JOIN Customer on User.userID= Customer.userID " +
                    "LEFT JOIN HomePhone AS hp on User.userID=hp.userID " +
                    "LEFT JOIN CellPhone AS cp on User.userID=cp.userID " +
                    "LEFT JOIN WorkPhone AS wp on User.userID=wp.userID " +
                    "WHERE User.userID = ?;";


            ps = conn.prepareStatement(sql);
            ps.setInt(1, userID);

            rs = ps.executeQuery();

            if (rs.next()) {
                userInfo.email = rs.getString("email");
                userInfo.first_name = rs.getString("first_name");
                userInfo.middle_name = rs.getString("middle_name");
                userInfo.last_name = rs.getString("last_name");
                userInfo.zip_code = rs.getString("zip_code");
                userInfo.street = rs.getString("street");
                userInfo.city = rs.getString("city");
                userInfo.state = rs.getString("state");
                userInfo.homePhone_areaCode = rs.getString("hp_ac");
                userInfo.homePhone_phoneNumber = rs.getString("hp_pn");
                userInfo.homePhone_extension = rs.getString("hp_ext");
                userInfo.cellPhone_areaCode = rs.getString("cp_ac");
                userInfo.cellPhone_phoneNumber = rs.getString("cp_pn");
                userInfo.cellPhone_extension = rs.getString("cp_ext");
                userInfo.workPhone_areaCode = rs.getString("wp_ac");
                userInfo.workPhone_phoneNumber = rs.getString("wp_pn");
                userInfo.workPhone_extension = rs.getString("wp_ext");
            } else {
                userInfo.status.errorMessage = "userID not present";
                userInfo.status.errorCode = -3;
                return userInfo;
            }

            userInfo.status.errorMessage = "Success";
            userInfo.status.errorCode = 0;
            return userInfo;
        } catch (SQLException e) {
            e.printStackTrace();
            userInfo.status.errorMessage = "SQL Exception occurred";
            userInfo.status.errorCode = -1;
            return userInfo;
        } finally {

            try {
                if (ps != null) ps.close();
            } catch (Exception e) {
            }
            try {
                if (conn != null) conn.close();
            } catch (Exception e) {
            }
        }
    }

    public List<Reservation> getCustomerReservations(int userID) {
        Connection conn = null;
        PreparedStatement ps = null;
        ResultSet rs = null;
        List<Reservation> reservations = new ArrayList<>();

        try {
            conn = getConnection();
            String sql = "SELECT reservationID " +
                    "FROM Reservation " +
                    "WHERE customerUserID=? " +
                    "ORDER BY start_date DESC";

            ps = conn.prepareStatement(sql);
            ps.setInt(1, userID);

            rs = ps.executeQuery();

            while (rs.next()) {
                reservations.add(getCustomerReservationsSummary(userID, rs.getInt("reservationID")));
            }

            return reservations;
        }
        catch (SQLException e) {
            e.printStackTrace();
            return reservations;
        }
        finally {
            try { if (rs != null) rs.close(); } catch (Exception e) {}
            try { if (ps != null) ps.close(); } catch (Exception e) {}
            try { if (conn != null) conn.close(); } catch (Exception e) {}
        }
    }

    private Reservation getCustomerReservationsSummary(int userID, int reservationID) {
        Connection conn = null;
        PreparedStatement ps = null;
        ResultSet rs = null;

        try {
            conn = getConnection();
            String sql = "SELECT Reservation.reservationID, Tool.toolNumber, " +
                    "power_source, sub_type, sub_option, " +
                    "start_date, end_date, pickUpUserID, dropOffUserID, " +
                    "DATEDIFF(Reservation.end_date, Reservation.start_date) * ROUND((purchase_price * 0.15), 2) AS rental_price, " +
                    "ROUND((purchase_price * 0.40), 2) AS deposit_price " +
                    "FROM Reservation " +
                    "INNER JOIN IsOf ON IsOf.reservationID=Reservation.reservationID " +
                    "INNER JOIN Tool ON IsOf.toolNumber=Tool.toolNumber " +
                    "WHERE customerUserID=? " +
                    "AND Reservation.reservationID=?";

            ps = conn.prepareStatement(sql);
            ps.setInt(1, userID);
            ps.setInt(2, reservationID);


            rs = ps.executeQuery();

            Reservation reservation = new Reservation();
            while (rs.next()) {
                reservation.setReservationID(rs.getInt("Reservation.reservationID"));
                reservation.setStartDate(rs.getTimestamp("start_date"));
                reservation.setEndDate(rs.getTimestamp("end_date"));
                reservation.setPickupUserID(rs.getInt("pickUpUserID"));
                reservation.setDropoffUserID(rs.getInt("dropOffUserID"));

                Tool tool = new Tool(
                        rs.getInt("Tool.toolNumber"),
                        rs.getString("power_source"),
                        rs.getString("sub_option"),
                        rs.getString("sub_type"),
                        rs.getBigDecimal("rental_price"),
                        rs.getBigDecimal("deposit_price")
                );
                reservation.addTool(tool);
            }

            BigDecimal totalRentalPrice = new BigDecimal(0);
            BigDecimal totalDepositPrice = new BigDecimal(0);
            for (Tool t : reservation.getTools()) {
                totalRentalPrice = totalRentalPrice.add(t.getRentalPrice());
                totalDepositPrice = totalDepositPrice.add(t.getDepositPrice());
            }

            reservation.setTotalRentalPrice(totalRentalPrice);
            reservation.setTotalDepositPrice(totalDepositPrice);

            return reservation;
        }
        catch (SQLException e) {
            e.printStackTrace();
            return null;
        }
        finally {
            try { if (rs != null) rs.close(); } catch (Exception e) {}
            try { if (ps != null) ps.close(); } catch (Exception e) {}
        }
    }

    public Status addReservation(int userID, Timestamp startDate, Timestamp endDate, List<Integer> toolIDs) {
        Connection conn = null;
        PreparedStatement ps = null;
        ResultSet rs = null;
        Status status = new Status();

        try {
            conn = getConnection();
            conn.setAutoCommit(false);

            String sqlInsertReservation = "INSERT INTO Reservation " +
                    "(customerUserID, start_date, end_date, pickUpUserID, dropOffUserID) " +
                    "VALUES " +
                    "(?, " +
                    "STR_TO_DATE(?, '%Y-%m-%d %H:%i:%s'), " +
                    "STR_TO_DATE(?, '%Y-%m-%d %H:%i:%s'), " +
                    "NULL, NULL);";

            String sqlCheckToolAvailable = "SELECT Tool.toolNumber " +
                    "FROM Tool " +
                    "LEFT JOIN SaleOrder " +
                    "ON SaleOrder.toolNumber = Tool.toolNumber " +
                    "LEFT JOIN ServiceOrderRequest " +
                    "ON ServiceOrderRequest.toolNumber = Tool.toolNumber " +
                    "LEFT JOIN IsOf " +
                    "ON IsOf.toolNumber = Tool.toolNumber " +
                    "LEFT JOIN Reservation " +
                    "ON Reservation.reservationID = IsOf.reservationID " +
                    "WHERE SaleOrder.saleOrderID IS NULL " +
                    "AND Tool.toolNumber NOT IN " +
                        "( " +
                        "SELECT Tool.toolNumber FROM Tool " +
                        "LEFT JOIN ServiceOrderRequest " +
                        "ON ServiceOrderRequest.toolNumber = Tool.toolNumber " +
                        "WHERE " +
                            "(" +
                            "(STR_TO_DATE(?, '%Y-%m-%d %H:%i:%s') < ServiceOrderRequest.end_date) " +
                            "AND " +
                            "(STR_TO_DATE(?, '%Y-%m-%d %H:%i:%s') > ServiceOrderRequest.start_date) " +
                            ") " +
                        ") " +
                    "AND Tool.toolNumber NOT IN " +
                        "( " +
                        "SELECT Tool.toolNumber " +
                        "FROM Tool " +
                        "LEFT JOIN IsOf " +
                        "ON IsOf.toolNumber = Tool.toolNumber " +
                        "LEFT JOIN Reservation " +
                        "ON Reservation.reservationID = IsOf.reservationID " +
                        "WHERE " +
                            "(" +
                            "(STR_TO_DATE(?, '%Y-%m-%d %H:%i:%s') < Reservation.end_date) " +
                            "AND " +
                            "(STR_TO_DATE(?, '%Y-%m-%d %H:%i:%s') > Reservation.start_date) " +
                            ")" +
                        ") " +
                    "AND Tool.toolNumber = ?;";

            String sqlInsertTool = "INSERT INTO IsOf (reservationID, toolNumber) " +
                                   "VALUES (LAST_INSERT_ID(), ?);";

            String sqlSelectInsertID = "SELECT LAST_INSERT_ID() AS reservationID";

            ps = conn.prepareStatement(sqlInsertReservation);
            ps.setInt(1, userID);
            ps.setTimestamp(2, startDate);
            ps.setTimestamp(3, endDate);

            int rowCount = ps.executeUpdate();
            if (rowCount == 1) {
                ps = conn.prepareStatement(sqlSelectInsertID);
                rs = ps.executeQuery();

                int reservationID = 0;
                if (rs.next()) {
                    reservationID = rs.getInt("reservationID");
                }

                for (Integer i : toolIDs) {
                    ps = conn.prepareStatement(sqlCheckToolAvailable);
                    ps.setTimestamp(1, startDate);
                    ps.setTimestamp(2, endDate);
                    ps.setTimestamp(3, startDate);
                    ps.setTimestamp(4, endDate);
                    ps.setInt(5, i.intValue());

                    rs = ps.executeQuery();
                    if (rs.next()) {
                        ps = conn.prepareStatement(sqlInsertTool);
                        ps.setInt(1, i.intValue());
                        rowCount = ps.executeUpdate();

                        if (rowCount != 1) {
                            status.errorCode = -3;
                            status.errorMessage = "Error adding tool ID " + i.intValue() + " to reservation.";
                            return status;
                        }
                    }
                    else {  //the Tool is not available
                        status.errorCode = -4;
                        status.errorMessage = "The tool ID " + i.intValue() + " is not available.";
                        return status;
                    }
                }

                conn.commit();
                status.errorCode = reservationID;
                status.errorMessage = "Success";
                return status;
            }

            status.errorCode = -2;
            status.errorMessage = "Error adding reservation.";
            return status;
        }
        catch (SQLException e) {
            e.printStackTrace();
            status.errorCode = -1;
            status.errorMessage = "Problem with database. " + e.getMessage();
            return status;
        }
        finally {
            try { if (rs != null) rs.close(); } catch (Exception e) {}
            try { if (ps != null) ps.close(); } catch (Exception e) {}
            try { if (conn != null) conn.close(); } catch (Exception e) {}
        }
    }

    public List<String> fillDropDown() { return fill("", ""); }
    public List<String> fillDropDown(String toolType) { return fill(toolType, ""); }
    public List<String> fillDropDown(String toolType, String powerSource) { return fill(toolType, powerSource); }
    private List<String> fill(String toolType, String powerSource)
    {
        Connection conn = null;
        PreparedStatement ps = null;
        ResultSet rs = null;
        List<String> dropDown = new ArrayList<>();

        try {
            conn = getConnection();

            if (toolType == "" && powerSource == ""){
                String sql = "SELECT DISTINCT tt_name " +
                        "FROM ToolTypeOption;";
                ps = conn.prepareStatement(sql);
            }
            else if (toolType != "" && powerSource == "") {
                String sql = "SELECT DISTINCT ps_name " +
                        "FROM ToolTypeOption " +
                        "WHERE tt_name = ?;";
                ps = conn.prepareStatement(sql);
                ps.setString(1, toolType);
            }
            else {
                String sql = "SELECT DISTINCT tst_name " +
                        "FROM ToolTypeOption " +
                        "WHERE tt_name = ? " +
                        "AND ps_name = ?;";
                ps = conn.prepareStatement(sql);
                ps.setString(1, toolType);
                ps.setString(2, powerSource);
            }

            rs = ps.executeQuery();

            while (rs.next()) {
                dropDown.add(rs.getString(1));
            }

            return dropDown;
        }
        catch (SQLException e) {
            e.printStackTrace();
            return dropDown;
        }
        finally {
            try { if (rs != null) rs.close(); } catch (Exception e) {}
            try { if (ps != null) ps.close(); } catch (Exception e) {}
            try { if (conn != null) conn.close(); } catch (Exception e) {}
        }
    }

    public List<Tool> CheckToolAvailability(Timestamp startDate, Timestamp endDate, String typeName, String subTypeName, String powerSourceName, String subOption){

        Connection conn = null;
        PreparedStatement ps = null;
        ResultSet rs = null;
        List<Tool> toolList = new ArrayList<>();

        try {
            conn = getConnection();

            String sql = "SELECT DISTINCT Tool.toolNumber, type, sub_type, sub_option, power_source, " +
                    "ROUND((purchase_price * 0.15), 2) AS rental_price, ROUND ((purchase_price * 0.40), 2) " +
                    "AS deposit_price "+
                    "FROM Tool "+
                    "LEFT JOIN SaleOrder "+
                    "ON SaleOrder.toolNumber = Tool.toolNumber " +
                    "LEFT JOIN ServiceOrderRequest " +
                    "ON ServiceOrderRequest.toolNumber = Tool.toolNumber " +
                    "LEFT JOIN IsOf " +
                    "ON IsOf.toolNumber = Tool.toolNumber " +
                    "LEFT JOIN Reservation " +
                    "ON Reservation.reservationID = IsOf.reservationID " +
                    "WHERE SaleOrder.saleOrderID IS NULL " +
                    "AND Tool.toolNumber NOT IN " +
                    "( " +
                    "SELECT Tool.toolNumber " +
                    "FROM Tool " +
                    "LEFT JOIN ServiceOrderRequest " +
                    "ON ServiceOrderRequest.toolNumber = Tool.toolNumber " +
                    "WHERE " +
                    "( " +
                    "(STR_TO_DATE(?, '%Y-%m-%d %H:%i:%s') < ServiceOrderRequest.end_date) " +
                    "AND " +
                    "(STR_TO_DATE(?, '%Y-%m-%d %H:%i:%s') > ServiceOrderRequest.start_date) " +
                    ") "+
                    ") " +
                    "AND Tool.toolNumber NOT IN " +
                    "(" +
                    "SELECT Tool.toolNumber " +
                    "FROM Tool " +
                    "LEFT JOIN IsOf " +
                    "ON IsOf.toolNumber = Tool.toolNumber " +
                    "LEFT JOIN Reservation " +
                    "ON Reservation.reservationID = IsOf.reservationID " +
                    "WHERE " +
                    "( " +
                    "(STR_TO_DATE(?, '%Y-%m-%d %H:%i:%s') < Reservation.end_date) " +
                    "AND " +
                    "(STR_TO_DATE(?, '%Y-%m-%d %H:%i:%s') > Reservation.start_date) " +
                    ") " +
                    ")" +
                    "AND Tool.toolNumber NOT IN" +
                    "( " +
                    "SELECT Tool.toolNumber " +
                    "FROM Tool " +
                    "LEFT JOIN IsOf " +
                    "ON IsOf.toolNumber = Tool.toolNumber " +
                    "LEFT JOIN Reservation " +
                    "ON Reservation.reservationID = IsOf.reservationID " +
                    "WHERE Reservation.pickUpUserID IS NOT NULL " +
                    "GROUP BY Tool.toolNumber " +
                    "HAVING COUNT(IsOf.reservationID) >= 50 " +
                    ") " +
                    "AND Tool.type LIKE ? " +
                    "AND Tool.sub_type LIKE ? " +
                    "AND Tool.power_source LIKE ? " +
                    "AND Tool.sub_option LIKE ?";

            ps = conn.prepareStatement(sql);
            ps.setTimestamp(1, startDate);
            ps.setTimestamp(2, endDate);
            ps.setTimestamp(3, startDate);
            ps.setTimestamp(4, endDate);
            ps.setString(5, typeName.isEmpty() ? "%" : typeName);
            ps.setString(6, subTypeName.isEmpty() ? "%" : subTypeName);
            ps.setString(7, powerSourceName.isEmpty() ? "%" : powerSourceName);
            ps.setString(8, subOption.isEmpty() ? "%" : subOption);

            rs = ps.executeQuery();

            while (rs.next()) {
                Tool tool = new Tool(
                        rs.getInt("Tool.toolNumber"),
                        rs.getString("power_source"),
                        rs.getString("sub_option"),
                        rs.getString("sub_type"),
                        rs.getBigDecimal("rental_price"),
                        rs.getBigDecimal("deposit_price")
                );
                toolList.add(tool);
            }

            return toolList;
        }
        catch (SQLException e) {
            e.printStackTrace();
            return toolList;
        }
        finally {
            try { if (rs != null) rs.close(); } catch (Exception e) {}
            try { if (ps != null) ps.close(); } catch (Exception e) {}
            try { if (conn != null) conn.close(); } catch (Exception e) {}
        }
    }

    public List<String> getAllToolTypes() {
        Connection conn = null;
        PreparedStatement ps = null;
        ResultSet rs = null;
        List<String> types = new ArrayList<>();

        try {
            conn = getConnection();
            String sql = "SELECT DISTINCT tt_name " +
                        "FROM ToolTypeOption;";

            ps = conn.prepareStatement(sql);
            rs = ps.executeQuery();

            while (rs.next()) {
                types.add(rs.getString("tt_name"));
            }

            return types;
        }
        catch (SQLException e) {
            e.printStackTrace();
            return types;
        }
        finally {
            try { if (rs != null) rs.close(); } catch (Exception e) {}
            try { if (ps != null) ps.close(); } catch (Exception e) {}
            try { if (conn != null) conn.close(); } catch (Exception e) {}
        }
    }

    public List<String> getAllToolPowerSources() {
        Connection conn = null;
        PreparedStatement ps = null;
        ResultSet rs = null;
        List<String> powerSources = new ArrayList<>();

        try {
            conn = getConnection();
            String sql = "SELECT DISTINCT ps_name " +
                    "FROM ToolTypeOption;";

            ps = conn.prepareStatement(sql);
            rs = ps.executeQuery();

            while (rs.next()) {
                powerSources.add(rs.getString("ps_name"));
            }

            return powerSources;
        }
        catch (SQLException e) {
            e.printStackTrace();
            return powerSources;
        }
        finally {
            try { if (rs != null) rs.close(); } catch (Exception e) {}
            try { if (ps != null) ps.close(); } catch (Exception e) {}
            try { if (conn != null) conn.close(); } catch (Exception e) {}
        }
    }

    public List<String> getAllToolSubTypes() {
        Connection conn = null;
        PreparedStatement ps = null;
        ResultSet rs = null;
        List<String> subTypes = new ArrayList<>();

        try {
            conn = getConnection();
            String sql = "SELECT DISTINCT tst_name " +
                    "FROM ToolTypeOption;";

            ps = conn.prepareStatement(sql);
            rs = ps.executeQuery();

            while (rs.next()) {
                subTypes.add(rs.getString("tst_name"));
            }

            return subTypes;
        }
        catch (SQLException e) {
            e.printStackTrace();
            return subTypes;
        }
        finally {
            try { if (rs != null) rs.close(); } catch (Exception e) {}
            try { if (ps != null) ps.close(); } catch (Exception e) {}
            try { if (conn != null) conn.close(); } catch (Exception e) {}
        }
    }

    public List<String> getAllToolSubTypes(String type) {
        Connection conn = null;
        PreparedStatement ps = null;
        ResultSet rs = null;
        List<String> subTypes = new ArrayList<>();

        try {
            conn = getConnection();
            String sql = "SELECT DISTINCT tst_name " +
                    "FROM ToolTypeOption " +
                    "WHERE tt_name = ?";

            ps = conn.prepareStatement(sql);
            ps.setString(1, type);

            rs = ps.executeQuery();

            while (rs.next()) {
                subTypes.add(rs.getString("tst_name"));
            }

            return subTypes;
        }
        catch (SQLException e) {
            e.printStackTrace();
            return subTypes;
        }
        finally {
            try { if (rs != null) rs.close(); } catch (Exception e) {}
            try { if (ps != null) ps.close(); } catch (Exception e) {}
            try { if (conn != null) conn.close(); } catch (Exception e) {}
        }
    }

    public List<String> getAllToolSubTypesFromPowerSource(String powerSource) {
        Connection conn = null;
        PreparedStatement ps = null;
        ResultSet rs = null;
        List<String> subTypes = new ArrayList<>();

        try {
            conn = getConnection();
            String sql = "SELECT DISTINCT tst_name " +
                    "FROM ToolTypeOption " +
                    "WHERE ps_name = ?";

            ps = conn.prepareStatement(sql);
            ps.setString(1, powerSource);

            rs = ps.executeQuery();

            while (rs.next()) {
                subTypes.add(rs.getString("tst_name"));
            }

            return subTypes;
        }
        catch (SQLException e) {
            e.printStackTrace();
            return subTypes;
        }
        finally {
            try { if (rs != null) rs.close(); } catch (Exception e) {}
            try { if (ps != null) ps.close(); } catch (Exception e) {}
            try { if (conn != null) conn.close(); } catch (Exception e) {}
        }
    }

    public List<String> getAllToolSubOptionsFromSubType(String subType) {
        Connection conn = null;
        PreparedStatement ps = null;
        ResultSet rs = null;
        List<String> subOptions = new ArrayList<>();

        try {
            conn = getConnection();
            String sql = "SELECT DISTINCT tso_name " +
                    "FROM ToolTypeOption " +
                    "WHERE tst_name = ?";

            ps = conn.prepareStatement(sql);
            ps.setString(1, subType);

            rs = ps.executeQuery();

            while (rs.next()) {
                subOptions.add(rs.getString("tso_name"));
            }

            return subOptions;
        }
        catch (SQLException e) {
            e.printStackTrace();
            return subOptions;
        }
        finally {
            try { if (rs != null) rs.close(); } catch (Exception e) {}
            try { if (ps != null) ps.close(); } catch (Exception e) {}
            try { if (conn != null) conn.close(); } catch (Exception e) {}
        }
    }

    public List<ReservationSummary> getReservationsToPickUp() {
        Connection conn = null;
        PreparedStatement ps = null;
        ResultSet rs = null;
        List<ReservationSummary> reservations = new ArrayList<>();

        try {
            conn = getConnection();
            String sql = "SELECT User.username, customerUserID, end_date, start_date, reservationID " +
                    "FROM Reservation " +
                    "INNER JOIN User " +
                    "ON Reservation.customerUserID = User.userID " +
                    "WHERE pickUpUserID IS NULL " +
                    "AND dropOffUserID IS NULL";

            ps = conn.prepareStatement(sql);
            rs = ps.executeQuery();

            while (rs.next()) {
                reservations.add(new ReservationSummary(
                        rs.getInt("reservationID"),
                        rs.getInt("customerUserID"),
                        rs.getString("User.username"),
                        rs.getTimestamp("start_date"),
                        rs.getTimestamp("end_date")));
            }

            return reservations;
        }
        catch (SQLException e) {
            e.printStackTrace();
            return reservations;
        }
        finally {
            try { if (rs != null) rs.close(); } catch (Exception e) {}
            try { if (ps != null) ps.close(); } catch (Exception e) {}
            try { if (conn != null) conn.close(); } catch (Exception e) {}
        }
    }

    public ReservationDetail getReservationDetail(int reservationID) {
        Connection conn = null;
        PreparedStatement ps = null;
        ResultSet rs = null;
        ReservationDetail reservationDetail = null;

        try {
            conn = getConnection();
            String sql = "SELECT Reservation.reservationID, Reservation.customerUserID, start_date, end_date, " +
                    "first_name, middle_name, last_name, " +
                    "Tool.toolNumber, power_source, sub_type, sub_option, " +
                    "DATEDIFF(end_date, start_date) * ROUND((purchase_price * 0.15), 2) AS rental_price, " +
                    "ROUND((Tool.purchase_price * 0.40), 2) AS deposit_price " +
                    "FROM Reservation " +
                    "INNER JOIN User " +
                    "ON Reservation.customerUserID = User.userID " +
                    "INNER JOIN IsOf " +
                    "ON IsOf.reservationID = Reservation.reservationID " +
                    "INNER JOIN Tool " +
                    "ON IsOf.toolNumber = Tool.toolNumber " +
                    "WHERE Reservation.reservationID = ? ";

            ps = conn.prepareStatement(sql);
            ps.setInt(1, reservationID);

            rs = ps.executeQuery();

            while (rs.next()) {
                if (reservationDetail == null) {
                    reservationDetail = new ReservationDetail(
                            rs.getInt("Reservation.reservationID"),
                            rs.getInt("Reservation.customerUserID"),
                            rs.getString("first_name"),
                            rs.getString("middle_name"),
                            rs.getString("last_name"),
                            rs.getTimestamp("start_date"),
                            rs.getTimestamp("end_date")
                    );
                }

                reservationDetail.addTool(
                        new Tool(
                            rs.getInt("Tool.toolNumber"),
                            rs.getString("power_source"),
                            rs.getString("sub_type"),
                            rs.getString("sub_option"),
                            rs.getBigDecimal("rental_price"),
                            rs.getBigDecimal("deposit_price")
                ));

                reservationDetail.addTotalDepositPrice(rs.getBigDecimal("deposit_price"));
                reservationDetail.addTotalRentalPrice(rs.getBigDecimal("rental_price"));
            }

            return reservationDetail;
        }
        catch (SQLException e) {
            e.printStackTrace();
            return reservationDetail;
        }
        finally {
            try { if (rs != null) rs.close(); } catch (Exception e) {}
            try { if (ps != null) ps.close(); } catch (Exception e) {}
            try { if (conn != null) conn.close(); } catch (Exception e) {}
        }
    }

    public CreditCard getCustomerCreditCard(int customerID) {
        Connection conn = null;
        PreparedStatement ps = null;
        ResultSet rs = null;
        CreditCard creditCard = null;

        try {
            conn = getConnection();
            String sql = "SELECT name, cred_number, exp_month, exp_year, cvc " +
                    "FROM CreditCard " +
                    "WHERE userID = ?";

            ps = conn.prepareStatement(sql);
            ps.setInt(1, customerID);

            rs = ps.executeQuery();


            if (rs.next()) {
               creditCard = new CreditCard(
                       rs.getString("name"),
                       rs.getString("cred_number"),
                       rs.getInt("exp_month"),
                       rs.getInt("exp_year"),
                       rs.getInt("cvc")
               );
            }

            return creditCard;
        }
        catch (SQLException e) {
            e.printStackTrace();
            return creditCard;
        }
        finally {
            try { if (rs != null) rs.close(); } catch (Exception e) {}
            try { if (ps != null) ps.close(); } catch (Exception e) {}
            try { if (conn != null) conn.close(); } catch (Exception e) {}
        }
    }

    public Status setReservationPickUp(int reservationID, int clerkID) {
        Connection conn = null;
        PreparedStatement ps = null;
        Status status = new Status();

        try {
            conn = getConnection();
            String sql = "UPDATE Reservation " +
                    "SET pickUpUserID = ? " +
                    "WHERE reservationID = ?";

            ps = conn.prepareStatement(sql);
            ps.setInt(1, clerkID);
            ps.setInt(2, reservationID);

            status.errorCode = ps.executeUpdate();
            if (status.errorCode > 0) {
                status.errorMessage = "Success";
            }
            else {
                status.errorMessage = "Error updating reservation pick up.";
            }

            return status;
        }
        catch (SQLException e) {
            e.printStackTrace();
            status.errorCode = -1;
            status.errorMessage = e.getMessage();
            return status;
        }
        finally {
            try { if (ps != null) ps.close(); } catch (Exception e) {}
            try { if (conn != null) conn.close(); } catch (Exception e) {}
        }
    }

    public Status updateCreditCard(int userID, CreditCard cc) {
        Connection conn = null;
        PreparedStatement ps = null;
        Status status = new Status();

        try {
            conn = getConnection();
            conn.setAutoCommit(false);

            String sql = "UPDATE CreditCard " +
                    "SET name=? , cred_number=? , exp_month=? , exp_year=? , cvc=? " +
                    "WHERE userID=?";

            ps = conn.prepareStatement(sql);
            ps.setString(1, cc.getName());
            ps.setString(2, cc.getNumber());
            ps.setInt(3, cc.getExp_month());
            ps.setInt(4, cc.getExp_year());
            ps.setInt(5, cc.getCvc());
            ps.setInt(6, userID);

            int rowCount = ps.executeUpdate();
            if (rowCount == 1) {
                status.errorCode = 0;
                status.errorMessage = "Success";

                conn.commit();
                return status;
            }

            status.errorCode = -2;
            status.errorMessage = "Error updating credit card.";
            return status;
        }
        catch (SQLException e) {
            e.printStackTrace();
            status.errorCode = -1;
            status.errorMessage = "Problem with database. " + e.getMessage();
            return status;
        }
        finally {
            try { if (ps != null) ps.close(); } catch (Exception e) {}
            try { if (conn != null) conn.close(); } catch (Exception e) {}
        }
    }

    public List<ReservationSummary> getReservationsToDropOff() {
        Connection conn = null;
        PreparedStatement ps = null;
        ResultSet rs = null;
        List<ReservationSummary> reservations = new ArrayList<>();

        try {
            conn = getConnection();
            String sql = "SELECT User.username, customerUserID, end_date, start_date, reservationID " +
                    "FROM Reservation " +
                    "INNER JOIN User " +
                    "ON Reservation.customerUserID = User.userID " +
                    "WHERE pickUpUserID IS NOT NULL " +
                    "AND dropOffUserID IS NULL";

            ps = conn.prepareStatement(sql);
            rs = ps.executeQuery();

            while (rs.next()) {
                reservations.add(new ReservationSummary(
                        rs.getInt("reservationID"),
                        rs.getInt("customerUserID"),
                        rs.getString("User.username"),
                        rs.getTimestamp("start_date"),
                        rs.getTimestamp("end_date")));
            }

            return reservations;
        }
        catch (SQLException e) {
            e.printStackTrace();
            return reservations;
        }
        finally {
            try { if (rs != null) rs.close(); } catch (Exception e) {}
            try { if (ps != null) ps.close(); } catch (Exception e) {}
            try { if (conn != null) conn.close(); } catch (Exception e) {}
        }
    }

    public Status setReservationDropOff(int reservationID, int clerkID) {
        Connection conn = null;
        PreparedStatement ps = null;
        Status status = new Status();

        try {
            conn = getConnection();
            String sql = "UPDATE Reservation " +
                    "SET dropOffUserID = ? " +
                    "WHERE reservationID = ?";

            ps = conn.prepareStatement(sql);
            ps.setInt(1, clerkID);
            ps.setInt(2, reservationID);

            status.errorCode = ps.executeUpdate();
            if (status.errorCode > 0) {
                status.errorMessage = "Success";

                ReservationDetail detail = getReservationDetail(reservationID);
                for (Tool t: detail.getTools()) {
                    if (getToolReservationCount(t.getToolID()) >= RENTAL_THRESHOLD) {
                        markToolForSale(t.getToolID(), VIRTUAL_CLERK_ID);
                    }
                }
            }
            else {
                status.errorMessage = "Error updating reservation pick up.";
            }

            return status;
        }
        catch (SQLException e) {
            e.printStackTrace();
            status.errorCode = -1;
            status.errorMessage = e.getMessage();
            return status;
        }
        finally {
            try { if (ps != null) ps.close(); } catch (Exception e) {}
            try { if (conn != null) conn.close(); } catch (Exception e) {}
        }
    }

    private int getToolReservationCount(int toolID) {
        Connection conn = null;
        PreparedStatement ps = null;
        ResultSet rs = null;
        int count = 0;

        try {
            conn = getConnection();
            String sql = "SELECT COUNT(Reservation.reservationID) AS count " +
                    "FROM Reservation " +
                    "INNER JOIN IsOf " +
                    "ON Reservation.reservationID = IsOf.reservationID " +
                    "INNER JOIN Tool ON IsOf.toolNumber = Tool.toolNumber " +
                    "WHERE Tool.toolNumber = ? " +
                    "AND dropOffUserID IS NOT NULL";

            ps = conn.prepareStatement(sql);
            ps.setInt(1, toolID);

            rs = ps.executeQuery();

            if (rs.next()) {
                count = rs.getInt("count");

            }

            return count;
        }
        catch (SQLException e) {
            e.printStackTrace();
            return count;
        }
        finally {
            try { if (rs != null) rs.close(); } catch (Exception e) {}
            try { if (ps != null) ps.close(); } catch (Exception e) {}
            try { if (conn != null) conn.close(); } catch (Exception e) {}
        }
    }

    private Status markToolForSale(int toolID, int clerkID) {
        Connection conn = null;
        PreparedStatement ps = null;
        Status status = new Status();

        try {
            conn = getConnection();
            String sql = "INSERT INTO SaleOrder (clerkUserID, customerUserID, toolNumber, for_sale_date, sold_date) " +
                    "VALUES (?, NULL, ? , ?,  NULL)";

            ps = conn.prepareStatement(sql);
            ps.setInt(1, clerkID);
            ps.setInt(2, toolID);
            ps.setTimestamp(3, Timestamp.from(Instant.now()));

            status.errorCode = ps.executeUpdate();
            if (status.errorCode > 0) {
                status.errorMessage = "Success";
            }
            else {
                status.errorMessage = "Error marking tool for sale";
            }

            return status;
        }
        catch (SQLException e) {
            e.printStackTrace();
            status.errorCode = -1;
            status.errorMessage = e.getMessage();
            return status;
        }
        finally {
            try { if (ps != null) ps.close(); } catch (Exception e) {}
            try { if (conn != null) conn.close(); } catch (Exception e) {}
        }
    }

    public Tool viewToolDetails(int toolNumber) {
        Connection conn = null;
        PreparedStatement ps = null;
        ResultSet rs = null;
        Status status = new Status();
        Tool tool = null;
        List<Accessory> accList;

        try {
            conn = getConnection();

            String sql = "SELECT Tool.toolNumber, Tool.type, sub_type, sub_option, manufacturer, width, " +
                    "purchase_price, material, weight, power_source, Tool.length, LadderTool.weight_capacity, " +
                    "LadderTool.step_count, Straight.rubber_feet, Step.pail_shelf,Gun.capacity, Gun.gauge_rating, " +
                    "Socket.drive_size, Socket.sae_size, Socket.deep_socket, ScrewDriver.screw_size, " +
                    "Hammer.anti_vibration, Plier.adjustable, Ratchet.drive_size, WheelBarrow.bin_material, " +
                    "WheelBarrow.bin_volume, WheelBarrow.wheel_count, Digging.blade_width, Digging.blade_length, " +
                    "Prunning.blade_material,  Prunning.blade_length, Striking.head_weight, Rake.tine_count, " +
                    "PowerTool.volt_rating, PowerTool.amp_rating, PowerTool.max_rpm_rating, " +
                    "PowerTool.min_rpm_rating, Generator.power_rating, Saw.blade_size, Sander.dust_bag, " +
                    "AirCompressor.tank_size, AirCompressor.pressure_rating, Drill.min_torque_rating, " +
                    "Drill.max_torque_rating, Drill.adjustable_clutch, Mixer.drum_size, Mixer.motor_rating, " +
                    "Accessory.description, Accessory.quantity, GardenTool.handle_material, " +
                    "ROUND((purchase_price * 0.15), 2) AS rental_price, " +
                    "ROUND ((purchase_price * 0.40), 2) AS deposit_price "+
                    "FROM Tool "+
                    "LEFT OUTER JOIN LadderTool ON Tool.toolNumber = LadderTool.toolNumber " +
                    "LEFT OUTER JOIN Straight ON LadderTool.toolNumber = Straight.toolNumber " +
                    "LEFT OUTER JOIN Step ON LadderTool.toolNumber = Step.toolNumber " +
                    "LEFT OUTER JOIN HandTool ON Tool.toolNumber = HandTool.toolNumber " +
                    "LEFT OUTER JOIN Gun ON HandTool.toolNumber = Gun.toolNumber " +
                    "LEFT OUTER JOIN Socket ON HandTool.toolNumber = Socket.toolNumber " +
                    "LEFT OUTER JOIN ScrewDriver ON HandTool.toolNumber = ScrewDriver.toolNumber " +
                    "LEFT OUTER JOIN Hammer ON HandTool.toolNumber = Hammer.toolNumber " +
                    "LEFT OUTER JOIN Plier ON HandTool.toolNumber = Plier.toolNumber " +
                    "LEFT OUTER JOIN Ratchet ON HandTool.toolNumber = Ratchet.toolNumber " +
                    "LEFT OUTER JOIN GardenTool ON Tool.toolNumber = GardenTool.toolNumber " +
                    "LEFT OUTER JOIN WheelBarrow ON GardenTool.toolNumber = WheelBarrow.toolNumber " +
                    "LEFT OUTER JOIN Digging ON GardenTool.toolNumber = Digging.toolNumber " +
                    "LEFT OUTER JOIN Prunning ON GardenTool.toolNumber = Prunning.toolNumber " +
                    "LEFT OUTER JOIN Striking ON GardenTool.toolNumber = Striking.toolNumber " +
                    "LEFT OUTER JOIN Rake ON GardenTool.toolNumber = Rake.toolNumber " +
                    "LEFT OUTER JOIN PowerTool ON Tool.toolNumber = PowerTool.toolNumber " +
                    "LEFT OUTER JOIN Generator ON PowerTool.toolNumber = Generator.toolNumber " +
                    "LEFT OUTER JOIN Saw ON PowerTool.toolNumber = Saw.toolNumber " +
                    "LEFT OUTER JOIN Sander ON PowerTool.toolNumber = Sander.toolNumber " +
                    "LEFT OUTER JOIN AirCompressor ON PowerTool.toolNumber = AirCompressor.toolNumber " +
                    "LEFT OUTER JOIN Drill ON PowerTool.toolNumber = Drill.toolNumber " +
                    "LEFT OUTER JOIN Mixer ON PowerTool.toolNumber = Mixer.toolNumber " +
                    "LEFT OUTER JOIN Accessory ON PowerTool.toolNumber = Accessory.toolNumber " +
                    "WHERE Tool.toolNumber = ?;";

            ps = conn.prepareStatement(sql);
            ps.setInt(1, toolNumber);

            rs = ps.executeQuery();

            int toolNum;
            String toolType;
            String subType;
            BigDecimal rentalPrice;
            BigDecimal depositPrice;

            if (rs.next()) {
                toolNum = rs.getInt("toolNumber");
                toolType = rs.getString("type");
                rentalPrice = rs.getBigDecimal("rental_price");
                depositPrice = rs.getBigDecimal("deposit_price");
                switch(toolType){
                    //region Ladder Tools
                    case "Ladder Tool":
                        subType = rs.getString("sub_type");
                        switch (subType){
                            case "Straight":
                                tool = new Tool.LadderTool.Straight(
                                        toolNum,
                                        toolType,
                                        rs.getString("power_source"),
                                        rs.getString("sub_option"),
                                        subType,
                                        rs.getString("material"),
                                        rs.getBigDecimal("length"),
                                        rs.getBigDecimal( "width"),
                                        rs.getBigDecimal("weight"),
                                        rs.getString("manufacturer"),
                                        rs.getBigDecimal("purchase_price"),
                                        rentalPrice,
                                        depositPrice,
                                        rs.getBigDecimal("weight_capacity"),
                                        rs.getInt("step_count"),
                                        rs.getBoolean("rubber_feet"));
                                break;
                            case "Step":
                                tool = new Tool.LadderTool.Step(
                                        toolNum,
                                        toolType,
                                        rs.getString("power_source"),
                                        rs.getString("sub_option"),
                                        subType,
                                        rs.getString("material"),
                                        rs.getBigDecimal("length"),
                                        rs.getBigDecimal( "width"),
                                        rs.getBigDecimal("weight"),
                                        rs.getString("manufacturer"),
                                        rs.getBigDecimal("purchase_price"),
                                        rentalPrice,
                                        depositPrice,
                                        rs.getBigDecimal("weight_capacity"),
                                        rs.getInt("step_count"),
                                        rs.getBoolean("pail_shelf"));
                                break;
                            default: break;
                        }
                        break;
                        //endregions
                    //region Hand Tools
                    case "Hand Tool":
                        subType = rs.getString("sub_type");
                        switch (subType){
                            case "Gun" :
                                tool = new Tool.HandTool.Gun(
                                        toolNum,
                                        toolType,
                                        rs.getString("power_source"),
                                        rs.getString("sub_option"),
                                        subType,
                                        rs.getString("material"),
                                        rs.getBigDecimal("length"),
                                        rs.getBigDecimal( "width"),
                                        rs.getBigDecimal("weight"),
                                        rs.getString("manufacturer"),
                                        rs.getBigDecimal("purchase_price"),
                                        rentalPrice,
                                        depositPrice,
                                        rs.getInt("capacity"),
                                        rs.getInt("gauge_rating"));
                                break;
                            case "Socket" :
                                tool = new Tool.HandTool.Socket(
                                        toolNum,
                                        toolType,
                                        rs.getString("power_source"),
                                        rs.getString("sub_option"),
                                        subType,
                                        rs.getString("material"),
                                        rs.getBigDecimal("length"),
                                        rs.getBigDecimal( "width"),
                                        rs.getBigDecimal("weight"),
                                        rs.getString("manufacturer"),
                                        rs.getBigDecimal("purchase_price"),
                                        rentalPrice,
                                        depositPrice,
                                        rs.getBigDecimal("drive_size"),
                                        rs.getBigDecimal("sae_size"),
                                        rs.getBoolean("deep_socket"));
                                break;
                            case "Screwdriver" :
                                tool = new Tool.HandTool.Screwdriver(
                                        toolNum,
                                        toolType,
                                        rs.getString("power_source"),
                                        rs.getString("sub_option"),
                                        subType,
                                        rs.getString("material"),
                                        rs.getBigDecimal("length"),
                                        rs.getBigDecimal( "width"),
                                        rs.getBigDecimal("weight"),
                                        rs.getString("manufacturer"),
                                        rs.getBigDecimal("purchase_price"),
                                        rentalPrice,
                                        depositPrice,
                                        rs.getInt("screw_size"));
                                break;
                            case "Hammer" :
                                tool = new Tool.HandTool.Hammer(
                                        toolNum,
                                        toolType,
                                        rs.getString("power_source"),
                                        rs.getString("sub_option"),
                                        subType,
                                        rs.getString("material"),
                                        rs.getBigDecimal("length"),
                                        rs.getBigDecimal( "width"),
                                        rs.getBigDecimal("weight"),
                                        rs.getString("manufacturer"),
                                        rs.getBigDecimal("purchase_price"),
                                        rentalPrice,
                                        depositPrice,
                                        rs.getBoolean("anti_vibration"));
                                break;
                            case "Pliers" :
                            case "Plier" :
                                tool = new Tool.HandTool.Pliers(
                                        toolNum,
                                        toolType,
                                        rs.getString("power_source"),
                                        rs.getString("sub_option"),
                                        subType,
                                        rs.getString("material"),
                                        rs.getBigDecimal("length"),
                                        rs.getBigDecimal( "width"),
                                        rs.getBigDecimal("weight"),
                                        rs.getString("manufacturer"),
                                        rs.getBigDecimal("purchase_price"),
                                        rentalPrice,
                                        depositPrice,
                                        rs.getBoolean("adjustable"));
                                break;
                            case "Ratchet" :
                                tool = new Tool.HandTool.Ratchet(
                                        toolNum,
                                        toolType,
                                        rs.getString("power_source"),
                                        rs.getString("sub_option"),
                                        subType,
                                        rs.getString("material"),
                                        rs.getBigDecimal("length"),
                                        rs.getBigDecimal( "width"),
                                        rs.getBigDecimal("weight"),
                                        rs.getString("manufacturer"),
                                        rs.getBigDecimal("purchase_price"),
                                        rentalPrice,
                                        depositPrice,
                                        rs.getBigDecimal("drive_size"));
                                break;
                            case "Wrench" :
                                tool = new Tool.HandTool.Wrench(
                                        toolNum,
                                        toolType,
                                        rs.getString("power_source"),
                                        rs.getString("sub_option"),
                                        subType,
                                        rs.getString("material"),
                                        rs.getBigDecimal("length"),
                                        rs.getBigDecimal( "width"),
                                        rs.getBigDecimal("weight"),
                                        rs.getString("manufacturer"),
                                        rs.getBigDecimal("purchase_price"),
                                        rentalPrice,
                                        depositPrice,
                                        rs.getBigDecimal("drive_size"));
                                break;
                            default : break;
                        }
                        break;
                        //endregion
                    //region Garden Tools
                    case "Garden Tool":
                        subType = rs.getString("sub_type");
                        switch (subType){
                            case "Wheelbarrow" :
                            case "Wheelbarrows" :
                                tool = new Tool.GardenTool.Wheelbarrow(
                                        toolNum,
                                        toolType,
                                        rs.getString("power_source"),
                                        rs.getString("sub_option"),
                                        subType,
                                        rs.getString("material"),
                                        rs.getBigDecimal("length"),
                                        rs.getBigDecimal( "width"),
                                        rs.getBigDecimal("weight"),
                                        rs.getString("manufacturer"),
                                        rs.getBigDecimal("purchase_price"),
                                        rentalPrice,
                                        depositPrice,
                                        rs.getString("handle_material"),
                                        rs.getString("bin_material"),
                                        rs.getBigDecimal("bin_volume"),
                                        rs.getInt("wheel_count"));
                                break;
                            case "Digging" :
                            case "Digger" :
                                tool = new Tool.GardenTool.Digging(
                                        toolNum,
                                        toolType,
                                        rs.getString("power_source"),
                                        rs.getString("sub_option"),
                                        subType,
                                        rs.getString("material"),
                                        rs.getBigDecimal("length"),
                                        rs.getBigDecimal( "width"),
                                        rs.getBigDecimal("weight"),
                                        rs.getString("manufacturer"),
                                        rs.getBigDecimal("purchase_price"),
                                        rentalPrice,
                                        depositPrice,
                                        rs.getString("handle_material"),
                                        rs.getBigDecimal("blade_width"),
                                        rs.getBigDecimal("Digging.blade_length"));
                                break;
                            case "Pruning" :
                            case "Prunning" :
                            case "Pruner" :
                                tool = new Tool.GardenTool.Prunning(
                                        toolNum,
                                        toolType,
                                        rs.getString("power_source"),
                                        rs.getString("sub_option"),
                                        subType,
                                        rs.getString("material"),
                                        rs.getBigDecimal("length"),
                                        rs.getBigDecimal( "width"),
                                        rs.getBigDecimal("weight"),
                                        rs.getString("manufacturer"),
                                        rs.getBigDecimal("purchase_price"),
                                        rentalPrice,
                                        depositPrice,
                                        rs.getString("handle_material"),
                                        rs.getString("blade_material"),
                                        rs.getBigDecimal("Prunning.blade_length"));
                                break;
                            case "Striking" :
                                tool = new Tool.GardenTool.Striking(
                                        toolNum,
                                        toolType,
                                        rs.getString("power_source"),
                                        rs.getString("sub_option"),
                                        subType,
                                        rs.getString("material"),
                                        rs.getBigDecimal("length"),
                                        rs.getBigDecimal( "width"),
                                        rs.getBigDecimal("weight"),
                                        rs.getString("manufacturer"),
                                        rs.getBigDecimal("purchase_price"),
                                        rentalPrice,
                                        depositPrice,
                                        rs.getString("handle_material"),
                                        rs.getBigDecimal("head_weight"));
                                break;
                            case "Rake" :
                            case "Rakes" :
                                tool = new Tool.GardenTool.Rake(
                                        toolNum,
                                        toolType,
                                        rs.getString("power_source"),
                                        rs.getString("sub_option"),
                                        subType,
                                        rs.getString("material"),
                                        rs.getBigDecimal("length"),
                                        rs.getBigDecimal( "width"),
                                        rs.getBigDecimal("weight"),
                                        rs.getString("manufacturer"),
                                        rs.getBigDecimal("purchase_price"),
                                        rentalPrice,
                                        depositPrice,
                                        rs.getString("handle_material"),
                                        rs.getInt("tine_count"));
                                break;
                            default : break;
                        }
                        break;
                        //endregion
                    //region Power Tools
                    case "Power Tool":

                        sql = "SELECT Tool.toolNumber,Accessory.description, Accessory.quantity " +
                                "FROM Tool " +
                                "INNER JOIN Accessory ON Tool.toolNumber = Accessory.toolNumber " +
                                "WHERE Tool.toolNumber = ?;";

                        ps = conn.prepareStatement(sql);
                        ps.setInt(1, toolNumber);
                        ResultSet rs2 = ps.executeQuery();
                        accList = new ArrayList<Accessory>(){};

                        while(rs2.next()){
                            String desc = rs2.getString("description");
                            int quant = rs2.getInt("quantity");

                            if(desc != null && desc.contains("Battery")){
                                BigDecimal Vrating = rs.getBigDecimal("volt_rating");
                                BigDecimal Arating = rs.getBigDecimal("amp_rating");
                                String batt = rs.getString("battery_type");
                                accList.add(new Accessory(Vrating, Arating, desc, quant, batt));
                            }
                            else{accList.add(new Accessory(desc, quant));}
                        }

                        subType = rs.getString("sub_type");
                        switch (subType){
                            case "Generator" :
                                tool = new Tool.PowerTool.Generator(
                                        toolNum,
                                        toolType,
                                        rs.getString("power_source"),
                                        rs.getString("sub_option"),
                                        subType,
                                        rs.getString("material"),
                                        rs.getBigDecimal("length"),
                                        rs.getBigDecimal( "width"),
                                        rs.getBigDecimal("weight"),
                                        rs.getString("manufacturer"),
                                        rs.getBigDecimal("purchase_price"),
                                        rentalPrice,
                                        depositPrice,
                                        rs.getBigDecimal("volt_rating"),
                                        rs.getBigDecimal("amp_rating"),
                                        rs.getBigDecimal("min_rpm_rating"),
                                        rs.getBigDecimal("max_rpm_rating"),
                                        rs.getBigDecimal("power_rating"),
                                        accList);
                                break;
                            case "Saw" :
                                tool = new Tool.PowerTool.Saw(
                                        toolNum,
                                        toolType,
                                        rs.getString("power_source"),
                                        rs.getString("sub_option"),
                                        subType,
                                        rs.getString("material"),
                                        rs.getBigDecimal("length"),
                                        rs.getBigDecimal( "width"),
                                        rs.getBigDecimal("weight"),
                                        rs.getString("manufacturer"),
                                        rs.getBigDecimal("purchase_price"),
                                        rentalPrice,
                                        depositPrice,
                                        rs.getBigDecimal("volt_rating"),
                                        rs.getBigDecimal("amp_rating"),
                                        rs.getBigDecimal("min_rpm_rating"),
                                        rs.getBigDecimal("max_rpm_rating"),
                                        rs.getBigDecimal("blade_size"),
                                        accList);
                                break;
                            case "Sander" :
                                tool = new Tool.PowerTool.Sander(
                                        toolNum,
                                        toolType,
                                        rs.getString("power_source"),
                                        rs.getString("sub_option"),
                                        subType,
                                        rs.getString("material"),
                                        rs.getBigDecimal("length"),
                                        rs.getBigDecimal( "width"),
                                        rs.getBigDecimal("weight"),
                                        rs.getString("manufacturer"),
                                        rs.getBigDecimal("purchase_price"),
                                        rentalPrice,
                                        depositPrice,
                                        rs.getBigDecimal("volt_rating"),
                                        rs.getBigDecimal("amp_rating"),
                                        rs.getBigDecimal("min_rpm_rating"),
                                        rs.getBigDecimal("max_rpm_rating"),
                                        rs.getBoolean("dust_bag"),
                                        accList);
                                break;
                            case "AirCompressor" :
                            case "Air-Compressor" :
                                tool = new Tool.PowerTool.AirCompressor(
                                        toolNum,
                                        toolType,
                                        rs.getString("power_source"),
                                        rs.getString("sub_option"),
                                        subType,
                                        rs.getString("material"),
                                        rs.getBigDecimal("length"),
                                        rs.getBigDecimal( "width"),
                                        rs.getBigDecimal("weight"),
                                        rs.getString("manufacturer"),
                                        rs.getBigDecimal("purchase_price"),
                                        rentalPrice,
                                        depositPrice,
                                        rs.getBigDecimal("volt_rating"),
                                        rs.getBigDecimal("amp_rating"),
                                        rs.getBigDecimal("min_rpm_rating"),
                                        rs.getBigDecimal("max_rpm_rating"),
                                        rs.getBigDecimal("tank_size"),
                                        rs.getBigDecimal("pressure_rating"),
                                        accList);
                                break;
                            case "Drill" :
                                tool = new Tool.PowerTool.Drill(
                                        toolNum,
                                        toolType,
                                        rs.getString("power_source"),
                                        rs.getString("sub_option"),
                                        subType,
                                        rs.getString("material"),
                                        rs.getBigDecimal("length"),
                                        rs.getBigDecimal( "width"),
                                        rs.getBigDecimal("weight"),
                                        rs.getString("manufacturer"),
                                        rs.getBigDecimal("purchase_price"),
                                        rentalPrice,
                                        depositPrice,
                                        rs.getBigDecimal("volt_rating"),
                                        rs.getBigDecimal("amp_rating"),
                                        rs.getBigDecimal("min_rpm_rating"),
                                        rs.getBigDecimal("max_rpm_rating"),
                                        rs.getBoolean("adjustable_clutch"),
                                        rs.getBigDecimal("min_torque_rating"),
                                        rs.getBigDecimal("max_torque_rating"),
                                        accList);
                                break;
                            case "Mixer" :
                                tool = new Tool.PowerTool.Mixer(
                                        toolNum,
                                        toolType,
                                        rs.getString("power_source"),
                                        rs.getString("sub_option"),
                                        subType,
                                        rs.getString("material"),
                                        rs.getBigDecimal("length"),
                                        rs.getBigDecimal( "width"),
                                        rs.getBigDecimal("weight"),
                                        rs.getString("manufacturer"),
                                        rs.getBigDecimal("purchase_price"),
                                        rentalPrice,
                                        depositPrice,
                                        rs.getBigDecimal("volt_rating"),
                                        rs.getBigDecimal("amp_rating"),
                                        rs.getBigDecimal("min_rpm_rating"),
                                        rs.getBigDecimal("max_rpm_rating"),
                                        rs.getBigDecimal("motor_rating"),
                                        rs.getBigDecimal("drum_size"),
                                        accList);
                                break;
                            default : break;
                        }
                        break;
                        //endregion
                    default: break;
                }
            }

            if (status.errorCode > 0) { status.errorMessage = "Success"; }
            else { status.errorMessage = "Error updating reservation pick up."; }
            return tool;
        }

        catch (SQLException e) {
            e.printStackTrace();
            status.errorCode = -1;
            status.errorMessage = e.getMessage();
            return tool;
        }

        finally {
            try { if (rs != null) rs.close(); } catch (Exception e) {}
            try { if (ps != null) ps.close(); } catch (Exception e) {}
            try { if (conn != null) conn.close(); } catch (Exception e) {}
        }
    }

    public Status addNewTool(Tool tool) {
        Connection conn = null;
        PreparedStatement ps = null;
        Status status = new Status();

        try {
            conn = getConnection();
            conn.setAutoCommit(false);

            String sql = "INSERT INTO Tool (type, sub_type, sub_option, power_source, manufacturer, material, width, " +
                    "weight, length, purchase_price) " +
                    "VALUES( ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)";

            ps = conn.prepareStatement(sql);
            ps.setString(1, tool.getType());
            ps.setString(2, tool.getSubType());
            ps.setString(3, tool.getSubOption());
            ps.setString(4, tool.getPowerSource());
            ps.setString(5, tool.getManufacturer());
            ps.setString(6, tool.getMaterial());
            ps.setBigDecimal(7, tool.getWidth());
            ps.setBigDecimal(8, tool.getWeight());
            ps.setBigDecimal(9, tool.getLength());
            ps.setBigDecimal(10, tool.getPurchasePrice());

            status.errorCode = ps.executeUpdate();
            if (status.errorCode > 0) {
                switch (tool.getType().toUpperCase()) {
                    case "LADDER TOOL":
                        status = addNewLadderTool((Tool.LadderTool) tool);
                        break;

                    case "HAND TOOL":
                        status = addNewHandTool((Tool.HandTool)tool);
                        break;

                    case "GARDEN TOOL":
                        status = addNewGardenTool((Tool.GardenTool)tool);
                        break;

                    case "POWER TOOL":
                        status = addNewPowerTool((Tool.PowerTool)tool);
                        break;

                    default:
                        status.errorCode = -3;
                        status.errorMessage = "Tool not added, unknown tool type: " + tool.getType();
                        break;
                }

            }
            else {
                status.errorMessage = "Error adding new tool.";
            }

            if (status.errorCode > 0) {
                conn.commit();
            }
            return status;
        }
        catch (SQLException e) {
            e.printStackTrace();
            status.errorCode = -1;
            status.errorMessage = e.getMessage();
            return status;
        }
        finally {
            try { if (ps != null) ps.close(); } catch (Exception e) {}
            try { if (conn != null) conn.close(); } catch (Exception e) {}
        }
    }

    //region Add New Ladder Tool

    private Status addNewLadderTool(Tool.LadderTool tool) {
        Connection conn = null;
        PreparedStatement ps = null;
        Status status = new Status();

        try {
            conn = getConnection();
            conn.setAutoCommit(false);

            String sql = "INSERT INTO LadderTool (toolNumber, weight_capacity, step_count) " +
                    "VALUES (LAST_INSERT_ID(), ?, ?)";

            ps = conn.prepareStatement(sql);
            ps.setObject(1, tool.getWeightCapacity());
            ps.setObject(2, tool.getStepCount());

            status.errorCode = ps.executeUpdate();
            if (status.errorCode > 0) {

                switch (tool.getSubType().toUpperCase()) {
                    case "STRAIGHT":
                        status = addNewStraightLadderTool((Tool.LadderTool.Straight)tool);
                        break;

                    case "STEP":
                        status = addNewStepLadderTool((Tool.LadderTool.Step)tool);
                        break;

                    default:
                        status.errorCode = -3;
                        status.errorMessage = "Ladder tool not added, unknown tool subtype: " + tool.getSubType();
                        break;
                }
            }
            else {
                status.errorMessage = "Error adding new ladder tool.";
            }

            return status;
        }
        catch (SQLException e) {
            e.printStackTrace();
            status.errorCode = -1;
            status.errorMessage = e.getMessage();
            return status;
        }
        finally {
            try { if (ps != null) ps.close(); } catch (Exception e) {}
        }
    }

    private Status addNewStraightLadderTool(Tool.LadderTool.Straight tool) {
        Connection conn = null;
        PreparedStatement ps = null;
        Status status = new Status();

        try {
            conn = getConnection();
            conn.setAutoCommit(false);

            String sql = "INSERT INTO Straight (toolNumber, rubber_feet) " +
                    "VALUES (LAST_INSERT_ID(), ?)";

            ps = conn.prepareStatement(sql);
            ps.setObject(1, tool.getRubberFeet());

            status.errorCode = ps.executeUpdate();
            if (status.errorCode > 0) {
                status.errorMessage = "Success";
            }
            else {
                status.errorMessage = "Error adding new straight ladder tool.";
            }

            return status;
        }
        catch (SQLException e) {
            e.printStackTrace();
            status.errorCode = -1;
            status.errorMessage = e.getMessage();
            return status;
        }
        finally {
            try { if (ps != null) ps.close(); } catch (Exception e) {}
        }
    }

    private Status addNewStepLadderTool(Tool.LadderTool.Step tool) {
        Connection conn = null;
        PreparedStatement ps = null;
        Status status = new Status();

        try {
            conn = getConnection();
            conn.setAutoCommit(false);

            String sql = "INSERT INTO Step (toolNumber, pail_shelf) " +
                    "VALUES (LAST_INSERT_ID(), ?)";

            ps = conn.prepareStatement(sql);
            ps.setObject(1, tool.getPailShelf());

            status.errorCode = ps.executeUpdate();
            if (status.errorCode > 0) {
                status.errorMessage = "Success";
            }
            else {
                status.errorMessage = "Error adding new step ladder tool.";
            }

            return status;
        }
        catch (SQLException e) {
            e.printStackTrace();
            status.errorCode = -1;
            status.errorMessage = e.getMessage();
            return status;
        }
        finally {
            try { if (ps != null) ps.close(); } catch (Exception e) {}
        }
    }

    //endregion

    //region Add New Hand Tool

    private Status addNewHandTool(Tool.HandTool tool) {
        Connection conn = null;
        PreparedStatement ps = null;
        Status status = new Status();

        try {
            conn = getConnection();
            conn.setAutoCommit(false);

            String sql = "INSERT INTO HandTool (toolNumber) " +
                    "VALUES (LAST_INSERT_ID())";

            ps = conn.prepareStatement(sql);

            status.errorCode = ps.executeUpdate();
            if (status.errorCode > 0) {

                switch (tool.getSubType().toUpperCase()) {
                    case "SCREWDRIVER":
                        status = addNewScrewdriverHandTool((Tool.HandTool.Screwdriver)tool);
                        break;

                    case "SOCKET":
                        status = addNewSocketHandTool((Tool.HandTool.Socket)tool);
                        break;

                    case "RATCHET":
                        status = addNewRatchetHandTool((Tool.HandTool.Ratchet)tool);
                        break;

                    case "WRENCH":
                        status = addNewWrenchHandTool((Tool.HandTool.Wrench)tool);
                        break;

                    case "PLIERS":
                        status = addNewPliersHandTool((Tool.HandTool.Pliers)tool);
                        break;

                    case "GUN":
                        status = addNewGunHandTool((Tool.HandTool.Gun)tool);
                        break;

                    case "HAMMER":
                        status = addNewHammerHandTool((Tool.HandTool.Hammer)tool);
                        break;

                    default:
                        status.errorCode = -3;
                        status.errorMessage = "Hand tool not added, unknown tool subtype: " + tool.getSubType();
                        break;
                }
            }
            else {
                status.errorMessage = "Error adding new hand tool.";
            }

            return status;
        }
        catch (SQLException e) {
            e.printStackTrace();
            status.errorCode = -1;
            status.errorMessage = e.getMessage();
            return status;
        }
        finally {
            try { if (ps != null) ps.close(); } catch (Exception e) {}
        }
    }

    private Status addNewScrewdriverHandTool(Tool.HandTool.Screwdriver tool) {
        Connection conn = null;
        PreparedStatement ps = null;
        Status status = new Status();

        try {
            conn = getConnection();
            conn.setAutoCommit(false);

            String sql = "INSERT INTO ScrewDriver (toolNumber, screw_size) " +
                    "VALUES (LAST_INSERT_ID(), ?)";

            ps = conn.prepareStatement(sql);
            ps.setObject(1, tool.getScrewSize());

            status.errorCode = ps.executeUpdate();
            if (status.errorCode > 0) {
                status.errorMessage = "Success";
            }
            else {
                status.errorMessage = "Error adding new screwdriver hand tool.";
            }

            return status;
        }
        catch (SQLException e) {
            e.printStackTrace();
            status.errorCode = -1;
            status.errorMessage = e.getMessage();
            return status;
        }
        finally {
            try { if (ps != null) ps.close(); } catch (Exception e) {}
        }
    }

    private Status addNewSocketHandTool(Tool.HandTool.Socket tool) {
        Connection conn = null;
        PreparedStatement ps = null;
        Status status = new Status();

        try {
            conn = getConnection();
            conn.setAutoCommit(false);

            String sql = "INSERT INTO Socket (toolNumber, drive_size, sae_size, deep_socket) " +
                    "VALUES (LAST_INSERT_ID(), ?, ?, ?)";

            ps = conn.prepareStatement(sql);
            ps.setBigDecimal(1, tool.getDriveSize());
            ps.setBigDecimal(2, tool.getSaeSize());
            ps.setObject(3, tool.getDeepSocket());

            status.errorCode = ps.executeUpdate();
            if (status.errorCode > 0) {
                status.errorMessage = "Success";
            }
            else {
                status.errorMessage = "Error adding new socket hand tool.";
            }

            return status;
        }
        catch (SQLException e) {
            e.printStackTrace();
            status.errorCode = -1;
            status.errorMessage = e.getMessage();
            return status;
        }
        finally {
            try { if (ps != null) ps.close(); } catch (Exception e) {}
        }
    }

    private Status addNewRatchetHandTool(Tool.HandTool.Ratchet tool) {
        Connection conn = null;
        PreparedStatement ps = null;
        Status status = new Status();

        try {
            conn = getConnection();
            conn.setAutoCommit(false);

            String sql = "INSERT INTO Ratchet (toolNumber, drive_size) " +
                    "VALUES (LAST_INSERT_ID(), ?)";

            ps = conn.prepareStatement(sql);
            ps.setBigDecimal(1, tool.getDriveSize());

            status.errorCode = ps.executeUpdate();
            if (status.errorCode > 0) {
                status.errorMessage = "Success";
            }
            else {
                status.errorMessage = "Error adding new ratchet hand tool.";
            }

            return status;
        }
        catch (SQLException e) {
            e.printStackTrace();
            status.errorCode = -1;
            status.errorMessage = e.getMessage();
            return status;
        }
        finally {
            try { if (ps != null) ps.close(); } catch (Exception e) {}
        }
    }

    private Status addNewWrenchHandTool(Tool.HandTool.Wrench tool) {
        Connection conn = null;
        PreparedStatement ps = null;
        Status status = new Status();

        try {
            conn = getConnection();
            conn.setAutoCommit(false);

            String sql = "INSERT INTO Wrench (toolNumber, drive_size) " +
                    "VALUES (LAST_INSERT_ID(), ?)";

            ps = conn.prepareStatement(sql);
            ps.setBigDecimal(1, tool.getDriveSize());

            status.errorCode = ps.executeUpdate();
            if (status.errorCode > 0) {
                status.errorMessage = "Success";
            }
            else {
                status.errorMessage = "Error adding new wrench hand tool.";
            }

            return status;
        }
        catch (SQLException e) {
            e.printStackTrace();
            status.errorCode = -1;
            status.errorMessage = e.getMessage();
            return status;
        }
        finally {
            try { if (ps != null) ps.close(); } catch (Exception e) {}
        }
    }

    private Status addNewPliersHandTool(Tool.HandTool.Pliers tool) {
        Connection conn = null;
        PreparedStatement ps = null;
        Status status = new Status();

        try {
            conn = getConnection();
            conn.setAutoCommit(false);

            String sql = "INSERT INTO Plier (toolNumber, adjustable) " +
                    "VALUES (LAST_INSERT_ID(), ?)";

            ps = conn.prepareStatement(sql);
            ps.setObject(1, tool.getAdjustable());

            status.errorCode = ps.executeUpdate();
            if (status.errorCode > 0) {
                status.errorMessage = "Success";
            }
            else {
                status.errorMessage = "Error adding new pliers hand tool.";
            }

            return status;
        }
        catch (SQLException e) {
            e.printStackTrace();
            status.errorCode = -1;
            status.errorMessage = e.getMessage();
            return status;
        }
        finally {
            try { if (ps != null) ps.close(); } catch (Exception e) {}
        }
    }

    private Status addNewGunHandTool(Tool.HandTool.Gun tool) {
        Connection conn = null;
        PreparedStatement ps = null;
        Status status = new Status();

        try {
            conn = getConnection();
            conn.setAutoCommit(false);

            String sql = "INSERT INTO Gun (toolNumber, capacity, gauge_rating) " +
                    "VALUES (LAST_INSERT_ID(), ?, ?)";

            ps = conn.prepareStatement(sql);
            ps.setObject(1, tool.getCapacity());
            ps.setObject(2, tool.getGaugeRating());

            status.errorCode = ps.executeUpdate();
            if (status.errorCode > 0) {
                status.errorMessage = "Success";
            }
            else {
                status.errorMessage = "Error adding new gun hand tool.";
            }

            return status;
        }
        catch (SQLException e) {
            e.printStackTrace();
            status.errorCode = -1;
            status.errorMessage = e.getMessage();
            return status;
        }
        finally {
            try { if (ps != null) ps.close(); } catch (Exception e) {}
        }
    }

    private Status addNewHammerHandTool(Tool.HandTool.Hammer tool) {
        Connection conn = null;
        PreparedStatement ps = null;
        Status status = new Status();

        try {
            conn = getConnection();
            conn.setAutoCommit(false);

            String sql = "INSERT INTO Hammer (toolNumber, anti_vibration) " +
                    "VALUES (LAST_INSERT_ID(), ?)";

            ps = conn.prepareStatement(sql);
            ps.setObject(1, tool.getAntiVibration());

            status.errorCode = ps.executeUpdate();
            if (status.errorCode > 0) {
                status.errorMessage = "Success";
            }
            else {
                status.errorMessage = "Error adding new hammer hand tool.";
            }

            return status;
        }
        catch (SQLException e) {
            e.printStackTrace();
            status.errorCode = -1;
            status.errorMessage = e.getMessage();
            return status;
        }
        finally {
            try { if (ps != null) ps.close(); } catch (Exception e) {}
        }
    }

    //endregion

    //region Add New Garden Tool

    private Status addNewGardenTool(Tool.GardenTool tool) {
        Connection conn = null;
        PreparedStatement ps = null;
        Status status = new Status();

        try {
            conn = getConnection();
            conn.setAutoCommit(false);

            String sql = "INSERT INTO GardenTool (toolNumber, handle_material) " +
                    "VALUES (LAST_INSERT_ID(), ?)";

            ps = conn.prepareStatement(sql);
            ps.setString(1, tool.getHandleMaterial());

            status.errorCode = ps.executeUpdate();
            if (status.errorCode > 0) {

                switch (tool.getSubType().toUpperCase()) {
                    case "DIGGER":
                        status = addNewDiggerGardenTool((Tool.GardenTool.Digging)tool);
                        break;

                    case "PRUNER":
                        status = addNewPrunerGardenTool((Tool.GardenTool.Prunning)tool);
                        break;

                    case "RAKES":
                        status = addNewRakeGardenTool((Tool.GardenTool.Rake)tool);
                        break;

                    case "WHEELBARROWS":
                        status = addNewWheelBarrowGardenTool((Tool.GardenTool.Wheelbarrow)tool);
                        break;

                    case "STRIKING":
                        status = addNewStrikingGardenTool((Tool.GardenTool.Striking)tool);
                        break;

                    default:
                        status.errorCode = -3;
                        status.errorMessage = "Garden tool not added, unknown tool subtype: " + tool.getSubType();
                        break;
                }
            }
            else {
                status.errorMessage = "Error adding new garden tool.";
            }

            return status;
        }
        catch (SQLException e) {
            e.printStackTrace();
            status.errorCode = -1;
            status.errorMessage = e.getMessage();
            return status;
        }
        finally {
            try { if (ps != null) ps.close(); } catch (Exception e) {}
        }
    }

    private Status addNewDiggerGardenTool(Tool.GardenTool.Digging tool) {
        Connection conn = null;
        PreparedStatement ps = null;
        Status status = new Status();

        try {
            conn = getConnection();
            conn.setAutoCommit(false);

            String sql = "INSERT INTO Digging (toolNumber, blade_width, blade_length) " +
                    "VALUES (LAST_INSERT_ID(), ?, ?)";

            ps = conn.prepareStatement(sql);
            ps.setBigDecimal(1, tool.getBladeWidth());
            ps.setBigDecimal(2, tool.getBladeLength());

            status.errorCode = ps.executeUpdate();
            if (status.errorCode > 0) {
                status.errorMessage = "Success";
            }
            else {
                status.errorMessage = "Error adding new digger garden tool.";
            }

            return status;
        }
        catch (SQLException e) {
            e.printStackTrace();
            status.errorCode = -1;
            status.errorMessage = e.getMessage();
            return status;
        }
        finally {
            try { if (ps != null) ps.close(); } catch (Exception e) {}
        }
    }

    private Status addNewPrunerGardenTool(Tool.GardenTool.Prunning tool) {
        Connection conn = null;
        PreparedStatement ps = null;
        Status status = new Status();

        try {
            conn = getConnection();
            conn.setAutoCommit(false);

            String sql = "INSERT INTO Prunning (toolNumber, blade_material, blade_length) " +
                    "VALUES (LAST_INSERT_ID(), ?, ?)";

            ps = conn.prepareStatement(sql);
            ps.setString(1, tool.getBladeMaterial());
            ps.setBigDecimal(2, tool.getBladeLength());

            status.errorCode = ps.executeUpdate();
            if (status.errorCode > 0) {
                status.errorMessage = "Success";
            }
            else {
                status.errorMessage = "Error adding new pruner garden tool.";
            }

            return status;
        }
        catch (SQLException e) {
            e.printStackTrace();
            status.errorCode = -1;
            status.errorMessage = e.getMessage();
            return status;
        }
        finally {
            try { if (ps != null) ps.close(); } catch (Exception e) {}
        }
    }

    private Status addNewRakeGardenTool(Tool.GardenTool.Rake tool) {
        Connection conn = null;
        PreparedStatement ps = null;
        Status status = new Status();

        try {
            conn = getConnection();
            conn.setAutoCommit(false);

            String sql = "INSERT INTO Rake (toolNumber, tine_count) " +
                    "VALUES (LAST_INSERT_ID(), ?)";

            ps = conn.prepareStatement(sql);
            ps.setObject(1, tool.getTineCount());

            status.errorCode = ps.executeUpdate();
            if (status.errorCode > 0) {
                status.errorMessage = "Success";
            }
            else {
                status.errorMessage = "Error adding new rake garden tool.";
            }

            return status;
        }
        catch (SQLException e) {
            e.printStackTrace();
            status.errorCode = -1;
            status.errorMessage = e.getMessage();
            return status;
        }
        finally {
            try { if (ps != null) ps.close(); } catch (Exception e) {}
        }
    }

    private Status addNewWheelBarrowGardenTool(Tool.GardenTool.Wheelbarrow tool) {
        Connection conn = null;
        PreparedStatement ps = null;
        Status status = new Status();

        try {
            conn = getConnection();
            conn.setAutoCommit(false);

            String sql = "INSERT INTO WheelBarrow (toolNumber, bin_material, bin_volume, wheel_count) " +
                    "VALUES (LAST_INSERT_ID(), ?, ?, ?)";

            ps = conn.prepareStatement(sql);
            ps.setString(1, tool.getBinMaterial());
            ps.setBigDecimal(2, tool.getBinVolume());
            ps.setObject(3, tool.getWheelCount());

            status.errorCode = ps.executeUpdate();
            if (status.errorCode > 0) {
                status.errorMessage = "Success";
            }
            else {
                status.errorMessage = "Error adding new wheelbarrow garden tool.";
            }

            return status;
        }
        catch (SQLException e) {
            e.printStackTrace();
            status.errorCode = -1;
            status.errorMessage = e.getMessage();
            return status;
        }
        finally {
            try { if (ps != null) ps.close(); } catch (Exception e) {}
        }
    }

    private Status addNewStrikingGardenTool(Tool.GardenTool.Striking tool) {
        Connection conn = null;
        PreparedStatement ps = null;
        Status status = new Status();

        try {
            conn = getConnection();
            conn.setAutoCommit(false);

            String sql = "INSERT INTO Striking (toolNumber, head_weight) " +
                    "VALUES (LAST_INSERT_ID(), ?)";

            ps = conn.prepareStatement(sql);
            ps.setBigDecimal(1, tool.getHeadWeight());

            status.errorCode = ps.executeUpdate();
            if (status.errorCode > 0) {
                status.errorMessage = "Success";
            }
            else {
                status.errorMessage = "Error adding new striking garden tool.";
            }

            return status;
        }
        catch (SQLException e) {
            e.printStackTrace();
            status.errorCode = -1;
            status.errorMessage = e.getMessage();
            return status;
        }
        finally {
            try { if (ps != null) ps.close(); } catch (Exception e) {}
        }
    }

    //endregion

    //region Add New Power Tool

    private Status addNewPowerTool(Tool.PowerTool tool) {
        Connection conn = null;
        PreparedStatement ps = null;
        Status status = new Status();

        try {
            conn = getConnection();
            conn.setAutoCommit(false);

            String sql = "INSERT INTO PowerTool (toolNumber, volt_rating, amp_rating, max_rpm_rating, min_rpm_rating) " +
                    "VALUES (LAST_INSERT_ID(), ?, ?, ?, ?)";

            ps = conn.prepareStatement(sql);
            ps.setBigDecimal(1, tool.getVoltRating());
            ps.setBigDecimal(2, tool.getAmpRating());
            ps.setBigDecimal(3, tool.getMaxRpmRating());
            ps.setBigDecimal(4, tool.getMinRpmRating());

            status.errorCode = ps.executeUpdate();
            if (status.errorCode > 0) {
                status = addAccessories(tool.accList);
            }

            if (status.errorCode >= 0) {
                switch (tool.getSubType().toUpperCase()) {
                    case "DRILL":
                        status = addNewDrillPowerTool((Tool.PowerTool.Drill)tool);
                        break;

                    case "SAW":
                        status = addNewSawPowerTool((Tool.PowerTool.Saw)tool);
                        break;

                    case "SANDER":
                        status = addNewSanderPowerTool((Tool.PowerTool.Sander)tool);
                        break;

                    case "AIR-COMPRESSOR":
                        status = addNewAirCompressorPowerTool((Tool.PowerTool.AirCompressor)tool);
                        break;

                    case "MIXER":
                        status = addNewMixerPowerTool((Tool.PowerTool.Mixer)tool);
                        break;

                    case "GENERATOR":
                        status = addNewGeneratorPowerTool((Tool.PowerTool.Generator)tool);
                        break;

                    default:
                        status.errorCode = -3;
                        status.errorMessage = "Power tool not added, unknown tool subtype: " + tool.getSubType();
                        break;
                }
            }
            else {
                status.errorMessage = "Error adding new power tool.";
            }

            return status;
        }
        catch (SQLException e) {
            e.printStackTrace();
            status.errorCode = -1;
            status.errorMessage = e.getMessage();
            return status;
        }
        finally {
            try { if (ps != null) ps.close(); } catch (Exception e) {}
        }
    }

    private Status addAccessories(List<Accessory> accessories) {
        Connection conn = null;
        PreparedStatement ps = null;
        Status status = new Status();

        try {
            conn = getConnection();
            conn.setAutoCommit(false);

            String sql = "INSERT INTO Accessory (toolNumber, description, quantity) " +
                    "VALUES (LAST_INSERT_ID(), ?, ?)";

            ps = conn.prepareStatement(sql);

            if (accessories != null) {
                for (Accessory a : accessories) {
                    ps.setString(1, a.getDescription());
                    ps.setInt(2, a.getQuantity());

                    status.errorCode = ps.executeUpdate();
                    if (status.errorCode > 0) {
                        status.errorMessage = "Success";
                    } else {
                        status.errorMessage = "Error adding new accessory for power tool.";
                        break;
                    }
                }
            }

            return status;
        }
        catch (SQLException e) {
            e.printStackTrace();
            status.errorCode = -1;
            status.errorMessage = e.getMessage();
            return status;
        }
        finally {
            try { if (ps != null) ps.close(); } catch (Exception e) {}
        }
    }

    private Status addNewDrillPowerTool(Tool.PowerTool.Drill tool) {
        Connection conn = null;
        PreparedStatement ps = null;
        Status status = new Status();

        try {
            conn = getConnection();
            conn.setAutoCommit(false);

            String sql = "INSERT INTO Drill (toolNumber, adjustable_clutch, min_torque_rating, max_torque_rating) " +
                    "VALUES (LAST_INSERT_ID(), ?, ?, ?)";

            ps = conn.prepareStatement(sql);
            ps.setObject(1, tool.getAdjustableClutch());
            ps.setBigDecimal(2, tool.getMinTorqueRating());
            ps.setBigDecimal(3, tool.getMaxTorqueRating());

            status.errorCode = ps.executeUpdate();
            if (status.errorCode > 0) {
                status.errorMessage = "Success";
            }
            else {
                status.errorMessage = "Error adding new drill power tool.";
            }

            return status;
        }
        catch (SQLException e) {
            e.printStackTrace();
            status.errorCode = -1;
            status.errorMessage = e.getMessage();
            return status;
        }
        finally {
            try { if (ps != null) ps.close(); } catch (Exception e) {}
        }
    }

    private Status addNewSawPowerTool(Tool.PowerTool.Saw tool) {
        Connection conn = null;
        PreparedStatement ps = null;
        Status status = new Status();

        try {
            conn = getConnection();
            conn.setAutoCommit(false);

            String sql = "INSERT INTO Saw (toolNumber, blade_size) " +
                    "VALUES (LAST_INSERT_ID(), ?)";

            ps = conn.prepareStatement(sql);
            ps.setBigDecimal(1, tool.getBladeSize());

            status.errorCode = ps.executeUpdate();
            if (status.errorCode > 0) {
                status.errorMessage = "Success";
            }
            else {
                status.errorMessage = "Error adding new saw power tool.";
            }

            return status;
        }
        catch (SQLException e) {
            e.printStackTrace();
            status.errorCode = -1;
            status.errorMessage = e.getMessage();
            return status;
        }
        finally {
            try { if (ps != null) ps.close(); } catch (Exception e) {}
        }
    }

    private Status addNewSanderPowerTool(Tool.PowerTool.Sander tool) {
        Connection conn = null;
        PreparedStatement ps = null;
        Status status = new Status();

        try {
            conn = getConnection();
            conn.setAutoCommit(false);

            String sql = "INSERT INTO Sander (toolNumber, dust_bag) " +
                    "VALUES (LAST_INSERT_ID(), ?)";

            ps = conn.prepareStatement(sql);
            ps.setObject(1, tool.getDustBag());

            status.errorCode = ps.executeUpdate();
            if (status.errorCode > 0) {
                status.errorMessage = "Success";
            }
            else {
                status.errorMessage = "Error adding new sander power tool.";
            }

            return status;
        }
        catch (SQLException e) {
            e.printStackTrace();
            status.errorCode = -1;
            status.errorMessage = e.getMessage();
            return status;
        }
        finally {
            try { if (ps != null) ps.close(); } catch (Exception e) {}
        }
    }

    private Status addNewAirCompressorPowerTool(Tool.PowerTool.AirCompressor tool) {
        Connection conn = null;
        PreparedStatement ps = null;
        Status status = new Status();

        try {
            conn = getConnection();
            conn.setAutoCommit(false);

            String sql = "INSERT INTO AirCompressor (toolNumber, tank_size, pressure_rating) " +
                    "VALUES (LAST_INSERT_ID(), ?, ?)";

            ps = conn.prepareStatement(sql);
            ps.setBigDecimal(1, tool.getTankSize());
            ps.setBigDecimal(2, tool.getPressureRating());

            status.errorCode = ps.executeUpdate();
            if (status.errorCode > 0) {
                status.errorMessage = "Success";
            }
            else {
                status.errorMessage = "Error adding new air-compressor power tool.";
            }

            return status;
        }
        catch (SQLException e) {
            e.printStackTrace();
            status.errorCode = -1;
            status.errorMessage = e.getMessage();
            return status;
        }
        finally {
            try { if (ps != null) ps.close(); } catch (Exception e) {}
        }
    }

    private Status addNewMixerPowerTool(Tool.PowerTool.Mixer tool) {
        Connection conn = null;
        PreparedStatement ps = null;
        Status status = new Status();

        try {
            conn = getConnection();
            conn.setAutoCommit(false);

            String sql = "INSERT INTO Mixer (toolNumber, motor_rating, drum_size) " +
                    "VALUES (LAST_INSERT_ID(), ?, ?)";

            ps = conn.prepareStatement(sql);
            ps.setBigDecimal(1, tool.getMotorRating());
            ps.setBigDecimal(2, tool.getDrumSize());

            status.errorCode = ps.executeUpdate();
            if (status.errorCode > 0) {
                status.errorMessage = "Success";
            }
            else {
                status.errorMessage = "Error adding new mixer power tool.";
            }

            return status;
        }
        catch (SQLException e) {
            e.printStackTrace();
            status.errorCode = -1;
            status.errorMessage = e.getMessage();
            return status;
        }
        finally {
            try { if (ps != null) ps.close(); } catch (Exception e) {}
        }
    }

    private Status addNewGeneratorPowerTool(Tool.PowerTool.Generator tool) {
        Connection conn = null;
        PreparedStatement ps = null;
        Status status = new Status();

        try {
            conn = getConnection();
            conn.setAutoCommit(false);

            String sql = "INSERT INTO Generator (toolNumber, power_rating) " +
                    "VALUES (LAST_INSERT_ID(), ?)";

            ps = conn.prepareStatement(sql);
            ps.setBigDecimal(1, tool.getPowerRating());

            status.errorCode = ps.executeUpdate();
            if (status.errorCode > 0) {
                status.errorMessage = "Success";
            }
            else {
                status.errorMessage = "Error adding new generator power tool.";
            }

            return status;
        }
        catch (SQLException e) {
            e.printStackTrace();
            status.errorCode = -1;
            status.errorMessage = e.getMessage();
            return status;
        }
        finally {
            try { if (ps != null) ps.close(); } catch (Exception e) {}
        }
    }

    //endregion


    //region Generate Clerk Report

    public List<ClerkReport> getClerkReport() {
        Connection conn = null;
        PreparedStatement ps = null;
        ResultSet rs = null;
        ArrayList<ClerkReport> report = new ArrayList<>();

        try {
            conn = getConnection();
            String sql = "SELECT c.userID, u.first_name, u.middle_name, u.last_name, " +
                    "u.email, c.date_hired " +
                    "FROM Clerk c NATURAL JOIN User u";


            ps = conn.prepareStatement(sql);

            rs = ps.executeQuery();

            int clerkID;
            while (rs.next()) {
                clerkID = rs.getInt("c.UserID");
                report.add(
                        new ClerkReport(
                        clerkID,
                        rs.getString("u.first_name"),
                        rs.getString("u.middle_name"),
                        rs.getString("u.last_name"),
                        rs.getString("u.email"),
                        rs.getTimestamp("c.date_hired"),
                        getTotalMonthClerkPickUp(clerkID),
                        getTotalMonthClerkDropOff(clerkID)));
            }

            Collections.sort(report);
            return report;
        }
        catch (SQLException e) {
            e.printStackTrace();
            return report;
        }
        finally {
            try { if (rs != null) rs.close(); } catch (Exception e) {}
            try { if (ps != null) ps.close(); } catch (Exception e) {}
            try { if (conn != null) conn.close(); } catch (Exception e) {}
        }
    }

    private int getTotalMonthClerkPickUp(int clerkID) {
        Connection conn = null;
        PreparedStatement ps = null;
        ResultSet rs = null;
        int pickups = 0;

        try {
            conn = getConnection();
            String sql = "SELECT COUNT(*) AS num_pickups " +
                    "FROM Clerk c " +
                    "INNER JOIN Reservation r " +
                    "ON r.pickUpUserID = c.userID " +
                    "WHERE MONTH(r.start_date) = ? " +
                    "AND YEAR(r.start_date) = ? " +
                    "AND c.userID = ?";


            ps = conn.prepareStatement(sql);
            ps.setInt(1, LocalDateTime.now().getMonthValue());
            ps.setInt(2, LocalDateTime.now().getYear());
            ps.setInt(3, clerkID);

            rs = ps.executeQuery();

            if (rs.next()) {
                pickups = rs.getInt("num_pickups");
            }

            return pickups;
        }
        catch (SQLException e) {
            e.printStackTrace();
            return pickups;
        }
        finally {
            try { if (rs != null) rs.close(); } catch (Exception e) {}
            try { if (ps != null) ps.close(); } catch (Exception e) {}
        }
    }

    private int getTotalMonthClerkDropOff(int clerkID) {
        Connection conn = null;
        PreparedStatement ps = null;
        ResultSet rs = null;
        int dropoffs = 0;

        try {
            conn = getConnection();
            String sql = "SELECT COUNT(*) AS num_dropoffs " +
                    "FROM Clerk c " +
                    "INNER JOIN Reservation r " +
                    "ON r.dropOffUserID = c.userID " +
                    "WHERE MONTH(r.end_date) = ? " +
                    "AND YEAR(r.end_date) = ? " +
                    "AND c.userID = ?";


            ps = conn.prepareStatement(sql);
            ps.setInt(1, LocalDateTime.now().getMonthValue());
            ps.setInt(2, LocalDateTime.now().getYear());
            ps.setInt(3, clerkID);

            rs = ps.executeQuery();

            if (rs.next()) {
                dropoffs = rs.getInt("num_dropoffs");
            }

            return dropoffs;
        }
        catch (SQLException e) {
            e.printStackTrace();
            return dropoffs;
        }
        finally {
            try { if (rs != null) rs.close(); } catch (Exception e) {}
            try { if (ps != null) ps.close(); } catch (Exception e) {}
        }
    }

    //endregion

    //region Generate Customer Report

    public List<CustomerReport> getCustomerReport() {
        Connection conn = null;
        PreparedStatement ps = null;
        ResultSet rs = null;
        ArrayList<CustomerReport> report = new ArrayList<>();

        try {
            conn = getConnection();
            String sql = "SELECT c.userID, u.first_name, u.middle_name, u.last_name, u.email, " +
                    "p.area_code, p.phone_number, p.extension, " +
                    "COUNT(DISTINCT r.reservationID) AS total_reservations, " +
                    "COUNT(i.toolNumber) AS total_tools_rented " +
                    "FROM Customer c " +
                    "INNER JOIN User u ON u.userID = c.userID " +
                    "INNER JOIN PrimaryPhone p ON p.userID = c.userID " +
                    "INNER JOIN Reservation r ON r.customerUserID = c.userID " +
                    "INNER JOIN IsOf i ON r.reservationID = i.reservationID " +
                    "WHERE MONTH(r.start_date) = ? AND YEAR(r.start_date) = ? " +
                    "GROUP BY c.userID " +
                    "ORDER BY total_tools_rented, u.last_name DESC";

            ps = conn.prepareStatement(sql);
            ps.setInt(1, LocalDateTime.now().getMonthValue());
            ps.setInt(2, LocalDateTime.now().getYear());

            rs = ps.executeQuery();

            while (rs.next()) {
                report.add(
                        new CustomerReport(
                                rs.getInt("c.userID"),
                                rs.getString("u.first_name"),
                                rs.getString("u.middle_name"),
                                rs.getString("u.last_name"),
                                rs.getString("u.email"),
                                rs.getString("p.area_code"),
                                rs.getString("p.phone_number"),
                                rs.getString("p.extension"),
                                rs.getInt("total_reservations"),
                                rs.getInt("total_tools_rented")));
            }

            return report;
        }
        catch (SQLException e) {
            e.printStackTrace();
            return report;
        }
        finally {
            try { if (rs != null) rs.close(); } catch (Exception e) {}
            try { if (ps != null) ps.close(); } catch (Exception e) {}
            try { if (conn != null) conn.close(); } catch (Exception e) {}
        }
    }

    //endregion

    public List<Tool> getToolReport(String type){
        Connection conn = null;
        PreparedStatement ps = null;
        ResultSet rs = null;
        Hashtable<Integer,Tool> toolHash = new Hashtable<>();

        Timestamp now = new Timestamp(System.currentTimeMillis());

        try {
            conn = getConnection();
            String sql =
                    "SELECT DISTINCT Tool.toolNumber, sub_type, sub_option, power_source, Reservation.end_date, "+
                            "ROUND((Tool.purchase_price * 0.15), 2) AS rental_price, " +
                            "ROUND((Tool.purchase_price * 0.40), 2) AS deposit_price " +

                    "FROM Tool " +
                    "LEFT JOIN IsOf " +
                    "ON IsOf.toolNumber = Tool.toolNumber " +
                    "LEFT JOIN Reservation " +
                    "ON Reservation.reservationID = IsOf.reservationID " +
                    "WHERE Tool.type LIKE ? " +
                    "AND Tool.toolNumber NOT IN " +
                    "( " +
                        "SELECT Tool.toolNumber " +
                        "FROM Tool " +
                        "LEFT JOIN IsOf " +
                        "ON IsOf.toolNumber = Tool.toolNumber " +
                        "LEFT JOIN Reservation " +
                        "ON Reservation.reservationID = IsOf.reservationID " +
                        "WHERE " +
                        "( " +
                            "(STR_TO_DATE(?, '%Y-%m-%d %H:%i:%s') < Reservation.end_date) " +
                            "AND " +
                            "(STR_TO_DATE(?, '%Y-%m-%d %H:%i:%s') > Reservation.start_date) " +
                        ") " +
                    "); ";

            ps = conn.prepareStatement(sql);
            ps.setString(1, type.isEmpty() ? "%" : type);
            ps.setTimestamp(2, now);
            ps.setTimestamp(3, now);
            rs = ps.executeQuery();

            while (rs.next()) {

                Tool tool = new Tool(
                        rs.getInt("toolNumber"),
                        rs.getString("power_source"),
                        rs.getString("sub_type"),
                        rs.getString("sub_option"),
                        rs.getBigDecimal("rental_price"),
                        rs.getBigDecimal("deposit_price"),
                        "Available",
                        null,
                        null,null);

                toolHash.put(rs.getInt("toolNumber"), tool);
            }

            sql = "SELECT Tool.toolNumber, type, sub_type, sub_option, power_source, Reservation.end_date, "+
                    "ROUND((Tool.purchase_price * 0.15), 2) AS rental_price, " +
                    "ROUND((Tool.purchase_price * 0.40), 2) AS deposit_price " +
                    "FROM Tool " +
                    "INNER JOIN IsOf " +
                    "ON IsOf.toolNumber = Tool.toolNumber " +
                    "INNER JOIN Reservation " +
                    "ON Reservation.reservationID = IsOf.reservationID " +
                    "WHERE Tool.type LIKE ? " +
                    "AND " +
                    "( " +
                        "(STR_TO_DATE(?, '%Y-%m-%d %H:%i:%s') < Reservation.end_date) " +
                        "AND " +
                        "(STR_TO_DATE(?, '%Y-%m-%d %H:%i:%s') > Reservation.start_date) " +
                    ");";

            ps = conn.prepareStatement(sql);
            ps.setString(1, type.isEmpty() ? "%" : type);
            ps.setTimestamp(2, now);
            ps.setTimestamp(3, now);
            rs = ps.executeQuery();

            while (rs.next()) {

                Tool tool = new Tool(
                        rs.getInt("toolNumber"),
                        rs.getString("power_source"),
                        rs.getString("sub_type"),
                        rs.getString("sub_option"),
                        rs.getBigDecimal("rental_price"),
                        rs.getBigDecimal("deposit_price"),
                        "Rented",
                        rs.getTimestamp("end_date"),
                        null,null);

                toolHash.put(rs.getInt("Tool.toolNumber"), tool);
            }

            sql = "SELECT Tool.toolNumber, " +
                    "(SUM(COALESCE(ServiceOrderRequest.cost,0)) + Tool.purchase_price) AS total_cost " +
                    "FROM Tool " +
                    "LEFT OUTER JOIN ServiceOrderRequest " +
                    "ON ServiceOrderRequest.toolNumber = Tool.toolNumber " +
                    "WHERE Tool.type LIKE ? " +
                    "GROUP BY Tool.toolNumber";

            ps = conn.prepareStatement(sql);
            ps.setString(1, type.isEmpty() ? "%" : type);
            rs = ps.executeQuery();

            while(rs.next()){
                if(toolHash.containsKey(rs.getInt("toolNumber"))){
                    toolHash.get(rs.getInt("toolNumber")).setTotalCost(rs.getBigDecimal("total_cost"));
                }
            }

            sql = "SELECT Tool.toolNumber, " +
                    "(SUM(DATEDIFF(Reservation.end_date, Reservation.start_date)) * ROUND((purchase_price * 0.15), 2)) AS rental_profit " +
                    "FROM Tool " +
                    "INNER JOIN IsOf " +
                    "ON IsOf.toolNumber = Tool.toolNumber " +
                    "INNER JOIN Reservation " +
                    "ON Reservation.reservationID = IsOf.reservationID " +
                    "WHERE Tool.type LIKE ? " +
                    "GROUP BY Tool.toolNumber";

            ps = conn.prepareStatement(sql);
            ps.setString(1, type.isEmpty() ? "%" : type);
            rs = ps.executeQuery();

            while(rs.next()){
                if(toolHash.containsKey(rs.getInt("toolNumber"))){
                    toolHash.get(rs.getInt("toolNumber")).setRentalProfit(rs.getBigDecimal("rental_profit"));
                }
            }

            List<Tool> report = new ArrayList<Tool>(toolHash.values());
            Collections.sort(report);
            return report;
        }
        catch (SQLException e) {
            e.printStackTrace();
            return null;
        }
        finally {
            try { if (rs != null) rs.close(); } catch (Exception e) {}
            try { if (ps != null) ps.close(); } catch (Exception e) {}
            try { if (conn != null) conn.close(); } catch (Exception e) {}
        }
    }
}
