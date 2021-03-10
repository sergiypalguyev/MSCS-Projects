import React from 'react';
import axios from 'axios';
import TextField from 'material-ui/TextField';
import MenuItem from 'material-ui/MenuItem';
import SelectField from 'material-ui/SelectField';
import MuiThemeProvider from 'material-ui/styles/MuiThemeProvider'
import {RadioButton, RadioButtonGroup} from 'material-ui/RadioButton';
import ActionFavorite from 'material-ui/svg-icons/action/favorite';
import ActionFavoriteBorder from 'material-ui/svg-icons/action/favorite-border';
import {Table} from 'material-ui/Table';
import {Link} from "react-router";
import NavBar from "./nav/NavBar";


export default  class ViewProfile extends React.Component {

  constructor (props) {
  super(props);
  }

  state = {
     "userInfo": {
       "email": "",
       "first_name": "",
       "middle_name": "",
       "last_name": "",
       "homePhone_areaCode": "",
       "homePhone_phoneNumber": "",
       "cellPhone_areaCode": "",
       "cellPhone_phoneNumber": "",
       "workPhone_areaCode": "",
       "workPhone_phoneNumber": "",
       "workPhone_extension": "",
       "zip_code": "",
       "street": "",
       "city": "",
       "state": "",
       "status": {
         "errorCode": 0,
         "errorMessage": "Success"
       }
     },
     "reservations": [
       {
         "reservationID": null,
         "tools": [
           {
             "toolID": null,
             "shortDesc": "",
             "powerSource": "",
             "subOption": "",
             "subType": "",
             "rentalPrice": null,
             "depositPrice": null
           }
         ],
         "startDate": "",
         "endDate": "",
         "numberOfDays": null,
         "pickupUserID": null,
         "dropoffUserID": null,
         "totalRentalPrice": null,
         "totalDepositPrice": null
       }
     ]
  }

  componentDidMount(){
    console.log("In view profile!");
    var getURL;

    if (localStorage.getItem('userType') == "clerk") {
      console.log(localStorage.getItem('customerID'));
      getURL = 'http://localhost:8080/profile?userID=' +  localStorage.getItem('customerID');
    }
    else {
      getURL = 'http://localhost:8080/profile?userID=' +  localStorage.getItem('userId');
    }

    console.log(getURL);
    axios.get(getURL)
      .then(res => {
        this.setState(res['data']);
      })
      .catch(function (error) {
        console.log(error);
      });
  }

  render() {
    return (
      <div>
         <NavBar/>
          <div className="viewprofile">
            <h2>Customer Info </h2>
            <b>Email:</b> {this.state.userInfo.email}
            <br></br><br></br>
            <b>Full Name:</b> {this.state.userInfo.first_name} {this.state.userInfo.middle_name} {this.state.userInfo.last_name}
            <br></br><br></br>
            <b>Home Phone:</b> {this.state.userInfo.homePhone_areaCode} - {this.state.userInfo.homePhone_phoneNumber}
            <br></br><br></br>
            <b>Work Phone:</b> {this.state.userInfo.workPhone_areaCode} - {this.state.userInfo.workPhone_phoneNumber} x {this.state.userInfo.workPhone_extension}
            <br></br><br></br>
            <b>Cell Phone:</b> {this.state.userInfo.cellPhone_areaCode} - {this.state.userInfo.cellPhone_phoneNumber}
            <br></br><br></br>
            <b>Address:</b> {this.state.userInfo.street}, {this.state.userInfo.city}, {this.state.userInfo.state} {this.state.userInfo.zip_code}
          </div>
          <br></br>
          <hr></hr>
          <div className="viewprofile">
            <h2> Reservations </h2>
            <table>
               <thead>
                 <tr>
                   <th>ReservationId</th>
                   <th>Tools</th>
                   <th>Start Date</th>
                   <th>End Date</th>
                   <th>Pick-up Clerk</th>
                   <th>Drop-off Clerk</th>
                   <th>Number of Days</th>
                   <th>Total Deposit Price</th>
                   <th>Total Rental Price</th>
                 </tr>
               </thead>
               <tbody>
                  {this.state.reservations.map((res) => <TableRow data = {res} />)}
               </tbody>
            </table>
          </div>
      </div>
    )
  }
}

class TableRow extends React.Component {
   render() {
      return (
         <tr>
            <td>{this.props.data.reservationID}</td>
            <td>{this.props.data.tools.map((res) => res.shortDesc + '     ')}</td>
            <td>{this.props.data.startDate == undefined ? this.props.data.startDate : this.props.data.startDate.substring(0, 10)}</td>
            <td>{this.props.data.endDate == undefined ? this.props.data.endDate : this.props.data.endDate.substring(0, 10)}</td>
            <td>{this.props.data.pickupUserID}</td>
            <td>{this.props.data.dropoffUserID}</td>
            <td>{this.props.data.numberOfDays}</td>
            <td>${parseFloat(this.props.data.totalDepositPrice).toFixed(2)}</td>
            <td>${parseFloat(this.props.data.totalRentalPrice).toFixed(2)}</td>
         </tr>
      );
   }
}
