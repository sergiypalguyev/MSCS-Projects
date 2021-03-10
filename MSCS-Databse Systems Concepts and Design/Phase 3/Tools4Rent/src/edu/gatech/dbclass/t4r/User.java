package edu.gatech.dbclass.t4r;

public class User {

    private int userID;
    private String username;
    private String password;
    private String email;
    private String firstName;
    private String middleName;
    private String lastName;

    public User() {
        this.userID = -1;
        this.username = "";
        this.password = "";
        this.email = "";
        this.firstName = "";
        this.middleName = "";
        this.lastName = "";
    }

    public int getUserID() {
        return userID;
    }

    public void setUserID(int userID) {
        this.userID = userID;
    }

    public String getUsername() {
        return username;
    }

    public void setUsername(String username) {
        this.username = username;
    }

    public String getPassword() {
        return password;
    }

    public void setPassword(String password) {
        this.password = password;
    }

    public String getEmail() {
        return email;
    }

    public void setEmail(String email) {
        this.email = email;
    }

    public String getFirstName() {
        return firstName;
    }

    public void setFirstName(String firstName) {
        this.firstName = firstName;
    }

    public String getMiddleName() {
        return middleName;
    }

    public void setMiddleName(String middleName) {
        this.middleName = middleName;
    }

    public String getLastName() {
        return lastName;
    }

    public void setLastName(String lastName) {
        this.lastName = lastName;
    }

    @Override
    public boolean equals(Object obj)
    {
        if (!(obj instanceof User)) {
            return false;
        }

        User user = (User)obj;

        return  user.userID == this.userID &&
                user.email.equals(this.email) &&
                user.username.equals(this.username) &&
                user.password.equals(this.password) &&
                user.firstName.equals(this.firstName) &&
                user.middleName.equals(this.middleName) &&
                user.lastName.equals(this.lastName);
    }

    @Override
    public String toString()
    {
        return new StringBuilder()
                .append("userID=" + this.userID)
                .append(",username=" + this.username)
                .append(",password=" + this.password)
                .append(",email=" + this.email)
                .append(",firstName=" + this.firstName)
                .append(",middleName=" + this.middleName)
                .append(",lastName=" + this.lastName)
                .toString();
    }

}
