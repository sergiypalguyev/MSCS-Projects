package edu.gatech.dbclass.web;

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import edu.gatech.dbclass.t4r.Database;
import edu.gatech.dbclass.t4r.JSONConverter;
import edu.gatech.dbclass.t4r.Status;

import javax.servlet.ServletException;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.IOException;
import java.io.PrintWriter;


public class RegistrationServlet extends HttpServlet {

    @Override
    protected void doOptions(HttpServletRequest request, HttpServletResponse response)
            throws ServletException, IOException {
        Options.setAccessControlHeaders(response);
        response.setStatus(HttpServletResponse.SC_OK);
    }

    @Override
    protected void doPost(HttpServletRequest request, HttpServletResponse response)
            throws ServletException, IOException {

        Options.setAccessControlHeaders(response);
        response.setStatus(HttpServletResponse.SC_UNAUTHORIZED);

        try {
            StringBuffer sb = Options.getRequestData(request);
            JsonParser jp = new JsonParser();


            /*{
                card_number:"12345678",
                card_username:"tuan dang",
                city:"fremont",
                cphone:"5103334444",
                cvc:"333",
                email:"tdang@gmail.com",
                exp_month:"1",
                exp_year:"2017",
                fname:"tuan",
                hphone:"5101112222",
                id:"732f8aac-ed04-4fa0-ac93-46ebfd085d0e",
                lname:"tuan",
                mname:"le",
                password:"12345",
                prim_phone:"Cell Phone",
                state:"CA",
                streetAddr:"1111 tom street",
                username:"tdang",
                wphone:"5102223333",
                zipcode:"91123"
            }*/


            JsonObject joReq = (JsonObject) jp.parse(sb.toString());
            String username = joReq.get("username").getAsString();
            String password = joReq.get("password").getAsString();
            String email = joReq.get("email").getAsString();
            String firstName = joReq.get("fname").getAsString();
            String middleName = joReq.get("mname").getAsString();
            String lastName = joReq.get("lname").getAsString();
            String street = joReq.get("streetAddr").getAsString();
            String city = joReq.get("city").getAsString();
            String state = joReq.get("state").getAsString();
            String zipCode = joReq.get("zipcode").getAsString();


            //Home Phone
            String hAreaCode = "";
            String hPhoneNumber = "";
            String hExtension = "";
            String homePhone = joReq.get("hphone").getAsString();
            if (homePhone.length() >= 10) {
                hAreaCode = homePhone.substring(0, 3);
                hPhoneNumber = homePhone.substring(3, 10);
                hExtension = null;
                if (homePhone.length() > 10) {
                    hExtension = homePhone.substring(10, homePhone.length());
                }
            }

            //Cell Phone
            String cAreaCode = "";
            String cPhoneNumber = "";
            String cExtension = "";
            String cellPhone = joReq.get("cphone").getAsString();
            if (cellPhone.length() >= 10) {
                cAreaCode = cellPhone.substring(0, 3);
                cPhoneNumber = cellPhone.substring(3, 10);
                cExtension = null;
                if (cellPhone.length() > 10) {
                    cExtension = cellPhone.substring(10, cellPhone.length());
                }
            }

            //Work Phone
            String wAreaCode = "";
            String wPhoneNumber = "";
            String wExtension = "";
            String workPhone = joReq.get("wphone").getAsString();
            if (workPhone.length() >= 10) {
                wAreaCode = workPhone.substring(0, 3);
                wPhoneNumber = workPhone.substring(3, 10);
                wExtension = null;
                if (workPhone.length() > 10) {
                    wExtension = workPhone.substring(10, workPhone.length());
                }
            }

            String primaryPhone = joReq.get("prim_phone").getAsString();

            //Credit Card
            String ccName = joReq.get("card_username").getAsString();
            String credNumber = joReq.get("card_number").getAsString();
            int expMonth = joReq.get("exp_month").getAsInt();
            int expYear = joReq.get("exp_year").getAsInt();
            int cvc = joReq.get("cvc").getAsInt();

            Status status = Database.getInstance().RegisterNewUser(username, password, email, firstName, middleName, lastName,
                    zipCode, street, city, state, hAreaCode, hPhoneNumber, hExtension,
                    wAreaCode, wPhoneNumber, wExtension, cAreaCode, cPhoneNumber, cExtension,
                    primaryPhone, ccName, credNumber, expMonth, expYear, cvc);

            JsonObject joResp = new JsonObject();
            joResp.addProperty("status", status.errorCode);
            joResp.addProperty("reason", status.errorMessage);

            if (status.errorCode == 0) {
                response.setStatus(HttpServletResponse.SC_OK);
            }

            response.setContentType("application/json");

            PrintWriter pw = response.getWriter();
            pw.println(JSONConverter.getInstance().getGson().toJson(joResp));

        } catch (Exception e) {
            // crash and burn
            throw new IOException("Error parsing JSON request string");
        }
    }

}
