import React from 'react';
import axios from 'axios';
import TextField from 'material-ui/TextField';
import MenuItem from 'material-ui/MenuItem';
import SelectField from 'material-ui/SelectField';
import DatePicker from 'material-ui/DatePicker';

import MuiThemeProvider from 'material-ui/styles/MuiThemeProvider'
import {RadioButton, RadioButtonGroup} from 'material-ui/RadioButton';
import ActionFavorite from 'material-ui/svg-icons/action/favorite';
import ActionFavoriteBorder from 'material-ui/svg-icons/action/favorite-border';
import {Link} from "react-router";
import Dialog from 'material-ui/Dialog';
import RaisedButton from 'material-ui/RaisedButton';
import FlatButton from 'material-ui/FlatButton';
import NavBar from "./nav/NavBar";

import {BrowserRouter, Route, NavLink, withRouter} from "react-router-dom";

export default  class ResConfirm extends React.Component {

  state = {
    allMonths: ['1',
    '2',
    '3',
    '4',
    '5',
    '6',
    '7',
    '8',
    '9',
    '10',
    '11',
    '12'],
    allYears:['2017','2018','2019','2020','2021','2022','2023','2024','2025','2026','2027'],
    allStates : [	'AL','MT',
      'AK','NE',
      'AZ','NV',
      'AR','NH',
      'CA','NJ',
      'CO','NM',
      'CT','NY',
      'DE','NC',
      'FL','ND',
      'GA','OH',
      'HI','OK',
      'ID','OR',
      'IL','PA',
      'IN','RI',
      'IA','SC',
      'KS','SD',
      'KY','TN',
      'LA','TX',
      'ME','UT',
      'MD','VT',
      'MA','VA',
      'MI','WA',
      'MN','WV',
      'MS','WI',
      'MO','WY'],
   customerID: '',
   selectedResId: null,
   firstTimeLoad: true,
   firstName: '',
   lastName: '',
   totalDepositPrice: '',
   totalRentalPrice: '',
   cred_type:'',
   card_username: '',
   card_number: '',
   exp_month: '',
   exp_year: '',
   cvc: ''
  }

  change = e => {
    this.setState({
      [e.target.name]: e.target.value
    });
  };

handleChange = (event, index, exp_month) => this.setState({exp_month});
handleMonthChange = (event, index, exp_month) => this.setState({exp_month});
handleYearChange = (event, index, exp_year) => this.setState({exp_year});
  onSubmit = (e) => {
        console.log("submitting");
        e.preventDefault();
        this.state.userId = localStorage.getItem("userId");
        console.log(this.state.cred_type);
        console.log(this.state.customerID);


        if (this.state.cred_type == 'new'){
          // update the credit card
          axios.post('http://localhost:8080/reservation/pickup', {
              "customerID": this.state.customerID,
              "creditCard":{
                "name": this.state.card_username,
                "number": this.state.card_number,
                "exp_month": this.state.exp_month,
                "exp_year": this.state.exp_year,
                "cvc": this.state.cvc
              }
            })
            .then(res => {
                console.log("cred update was successful. Now sending the reservation in...");
                var pickupURL = "http://localhost:8080/reservation/pickup?reservationID=" + this.state.selectedResId + "&clerkID=" + localStorage.getItem("userId");
                //var pickupURL = 'http://localhost:8080/reservation/pickup?reservationID=42&clerkID=2';
                axios.get(pickupURL)
                  .then(res => {
                      console.log("success!");
                      console.log(res);
                      localStorage.setItem("credNumber", this.state.card_number);
                      this.props.history.push('/printconfirmpickup');
                  })
                  .catch(function (error) {
                    console.log(error);
                  });

              }).catch(function(reject){
              console.log(reject);
            });
        }
        else if (this.state.cred_type == 'existing'){
          // sending the pick up to the DB
          var pickupURL = "http://localhost:8080/reservation/pickup?reservationID=" + this.state.selectedResId + "&clerkID=" + localStorage.getItem("userId");
          //var pickupURL = 'http://localhost:8080/reservation/pickup?reservationID=42&clerkID=2';
          axios.get(pickupURL)
            .then(res => {
                console.log("success from existing!");
                console.log(res);
                localStorage.setItem("credNumber", this.state.card_number);
                this.props.history.push('/printconfirmpickup');
            })
            .catch(function (error) {
              console.log(error);
            });
        }

  }


render() {
  if (this.state.firstTimeLoad == true){
    if(localStorage.getItem("selectedResId") != null){
        //this.setState({selectedResId: localStorage.getItem("selectedResId")});
        this.state.selectedResId = localStorage.getItem("selectedResId");
        console.log(this.state.selectedResId);
    }

    console.log("hello moto");
    var getURL = 'http://localhost:8080/reservation/pickup?reservationID=' + this.state.selectedResId;
    axios.get(getURL)
      .then(res => {
        console.log("response is");
        console.log(res);
        this.setState({firstName: res['data']['firstName']});
        this.setState({lastName: res['data']['lastName']});
        this.setState({totalDepositPrice: res['data']['totalDepositPrice']});
        this.setState({totalRentalPrice: res['data']['totalRentalPrice']});
        this.setState({customerID: res['data']['customerID']});
        console.log('i set customer id');
        console.log(this.state.customerID);

      })
      .catch(function (error) {
        console.log(error);
      });


    this.state.firstTimeLoad = false;
  }


  return (
    <div>
          <NavBar/>
          <MuiThemeProvider>
              <h2> Pickup Reservation </h2>
              <br></br><br></br><b>Reservation ID: #{this.state.selectedResId}</b>
              <br></br><br></br><b>Customer Name: </b>
              {this.state.firstName} {this.state.lastName}
              <br></br><br></br><b>Total Deposit: </b>
              ${parseFloat(this.state.totalDepositPrice).toFixed(2)}
              <br></br><br></br><b>Total Rental Price: </b>
              ${parseFloat(this.state.totalRentalPrice).toFixed(2)}

              <br></br><br></br><b>Credit Card: </b>
              <input
                type="radio"
                name="cred_type"
                id="existingCredRadioButton"
                value="existing"
                onChange={e => this.change(e)}
              />
              <label htmlFor="existingCredRadioButton"> Existing  </label>
              <input
                 type="radio"
                 name="cred_type"
                 id="newCredRadioButton"
                 value="new"
                 onChange={e => this.change(e)}
              />
              <label htmlFor="newCredRadioButton"> New  </label>
              <br></br><br></br>
              Enter Updated Credit Card Information
              <br></br>
              ** THIS WILL OVERWRITE THE PRIOR CUSTOMERS CREDIT CARD INFORMATION **

              <br></br>
              <TextField
                  name="card_username"
                  hintText="Name on Credit Card"
                  value={this.state.card_username}
                  onChange={e => this.change(e)}
              />

              <br></br>
              <TextField
                  name="card_number"
                  hintText="Credit Card #"
                  value={this.state.card_number}
                  onChange={e => this.change(e)}
              />

              <br></br>
              <SelectField
              floatingLabelText="Expiration Month"
                name="exp_month"
                value={this.state.exp_month}
                onChange={this.handleMonthChange}
              >
              {this.state.allMonths.map((ps) => (
                  <MenuItem value={ps} primaryText={ps} />
                ))}
              </SelectField>


              <SelectField
              floatingLabelText="Expiration Year"
                name="exp_year"
                value={this.state.exp_year}
                onChange={this.handleYearChange}
              >
              {this.state.allYears.map((ps) => (
                  <MenuItem value={ps} primaryText={ps} />
                ))}
              </SelectField>

              <br></br>
              <TextField
                  name="cvc"
                  hintText="CVC"
                  value={this.state.cvc}
                  onChange={e => this.change(e)}
              />

              <br></br>
              <RaisedButton label="Confirm Pickup" primary={true}  onClick= {(e) => this.onSubmit(e)} />


          </MuiThemeProvider>
    </div>
  )
}

}
