package edu.gatech.dbclass.t4r;

public class Status{
    public int errorCode;
    public String errorMessage;
    public Status (){
        errorCode = 0;
        errorMessage = "";
    }
    public Status (int code, String message){
        errorCode = code;
        errorMessage = message;
    }
}
