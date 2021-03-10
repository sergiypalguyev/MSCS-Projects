package edu.gatech.dbclass.t4r;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import org.junit.Assert;
import org.junit.Test;

import java.io.PrintWriter;
import java.io.Writer;
import java.sql.Timestamp;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

public class JSONConverterTest {

    private JSONConverter jsonConverter = JSONConverter.getInstance();

    @Test
    public void UserToJSONTest()
    {
        User user = new User();
        user.setUserID(1);
        user.setUsername("rviera6");
        user.setPassword("123456");
        user.setEmail("rviera6@gatech.edu");
        user.setFirstName("Raul");
        user.setLastName("Viera");

        String actualJSON = this.jsonConverter.getGson().toJson(user);
        String expectedJSON = "{\n" +
                "  \"userID\": 1,\n" +
                "  \"username\": \"rviera6\",\n" +
                "  \"password\": \"123456\",\n" +
                "  \"email\": \"rviera6@gatech.edu\",\n" +
                "  \"firstName\": \"Raul\",\n" +
                "  \"middleName\": \"\",\n" +
                "  \"lastName\": \"Viera\"\n" +
                "}";

        Assert.assertEquals(expectedJSON, actualJSON);
    }

    @Test
    public void JSONToUserTest()
    {
        String json = "{\n" +
                "  \"userID\": 1,\n" +
                "  \"username\": \"rviera6\",\n" +
                "  \"password\": \"123456\",\n" +
                "  \"email\": \"rviera6@gatech.edu\",\n" +
                "  \"firstName\": \"Raul\",\n" +
                "  \"middleName\": \"\",\n" +
                "  \"lastName\": \"Viera\"\n" +
                "}";

        User expectedUser = new User();
        expectedUser.setUserID(1);
        expectedUser.setUsername("rviera6");
        expectedUser.setPassword("123456");
        expectedUser.setEmail("rviera6@gatech.edu");
        expectedUser.setFirstName("Raul");
        expectedUser.setLastName("Viera");

        User actualUser = this.jsonConverter.getGson().fromJson(json, User.class);

        Assert.assertEquals(expectedUser, actualUser);
    }

    @Test
    public void IntToJSON()
    {
        String json = this.jsonConverter.getGson().toJson(1);
        String expected = "1";

        Assert.assertEquals(expected, json);
    }

    @Test
    public void JsonObjecToToJSON()
    {
        JsonObject jo = new JsonObject();
        jo.addProperty("name", "Raul");
        jo.addProperty("value", 1);
        jo.addProperty("isTrue", true);

        String json = this.jsonConverter.getGson().toJson(jo);
        String expected = "{\n" +
                "  \"name\": \"Raul\",\n" +
                "  \"value\": 1,\n" +
                "  \"isTrue\": true\n" +
                "}";

        Assert.assertEquals(expected, json);
    }

    @Test
    public void JsonUserProfileTest()
    {
        int userID = 3;
        UserInfo userInfo = Database.getInstance().ViewProfile(userID);
        List<Reservation> reservations = Database.getInstance().getCustomerReservations(userID);
        UserProfile userProfile = new UserProfile(userInfo, reservations);

        String jsonUserProfile = this.jsonConverter.getGson().toJson(userProfile);

        System.out.println(jsonUserProfile);
        JsonParser jp = new JsonParser();
        JsonElement je = jp.parse(jsonUserProfile);
    }

    @Test
    public void JsonMakeReservationRequest()
    {
        int userID = 3;
        Timestamp startDate = new Timestamp(117, 11, 25, 11, 30, 00, 00);
        Timestamp endDate = new Timestamp(117, 11, 28, 11, 30, 00, 00);
        JsonArray toolsIDs = new JsonArray();
        toolsIDs.add(5);
        toolsIDs.add(6);

        JsonObject jo = new JsonObject();
        jo.addProperty("userID", userID);
        jo.addProperty("startDate", startDate.toString());
        jo.addProperty("endDate", endDate.toString());
        jo.add("tools", toolsIDs);

        String jsonReservation = this.jsonConverter.getGson().toJson(jo);
        System.out.println(jsonReservation);
    }

}
