package edu.gatech.dbclass.t4r;

public class UserInfo{

    String email;
    String first_name;
    String middle_name;
    String last_name;
    String zip_code;
    String street;
    String city;
    String state;
    String homePhone_areaCode;
    String homePhone_phoneNumber;
    String homePhone_extension;
    String cellPhone_areaCode;
    String cellPhone_phoneNumber;
    String cellPhone_extension;
    String workPhone_areaCode;
    String workPhone_phoneNumber;
    String workPhone_extension;
    Status status;


    public UserInfo (){
        email = "";
        first_name = "";
        middle_name = "";
        last_name = "";
        zip_code = "";
        street = "";
        city = "";
        state = "";
        status = null;
    }
    public UserInfo (String Email, String First_name, String Middle_name, String Last_name, String Zip_code, String Street, String City, String State, Status Status){
        email = Email;
        first_name = First_name;
        middle_name = Middle_name;
        last_name = Last_name;
        zip_code = Zip_code;
        street = Street;
        city = City;
        state = State;
        status = Status;
    }

    public String getEmail() {
        return email;
    }

    public String getFirst_name() {
        return first_name;
    }

    public String getMiddle_name() {
        return middle_name;
    }

    public String getLast_name() {
        return last_name;
    }

    public String getZip_code() {
        return zip_code;
    }

    public String getStreet() {
        return street;
    }

    public String getCity() {
        return city;
    }

    public String getState() {
        return state;
    }

    public Status getStatus() {
        return status;
    }

    public String getHomePhone_areaCode() {
        return homePhone_areaCode;
    }

    public String getHomePhone_phoneNumber() {
        return homePhone_phoneNumber;
    }

    public String getHomePhone_extension() {
        return homePhone_extension;
    }

    public String getCellPhone_areaCode() {
        return cellPhone_areaCode;
    }

    public String getCellPhone_phoneNumber() {
        return cellPhone_phoneNumber;
    }

    public String getCellPhone_extension() {
        return cellPhone_extension;
    }

    public String getWorkPhone_areaCode() {
        return workPhone_areaCode;
    }

    public String getWorkPhone_phoneNumber() {
        return workPhone_phoneNumber;
    }

    public String getWorkPhone_extension() {
        return workPhone_extension;
    }
}
