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
import RaisedButton from 'material-ui/RaisedButton';
import FlatButton from 'material-ui/FlatButton';
import ResDetails from './ResDetails';

import NavBar from "./nav/NavBar";


import {BrowserRouter, Route, NavLink, withRouter} from "react-router-dom";

import {
  Table,
  TableBody,
  TableFooter,
  TableHeader,
  TableHeaderColumn,
  TableRow,
  TableRowColumn,
} from 'material-ui/Table';


let DateTimeFormat = global.Intl.DateTimeFormat;


export default  class PickUpReservations extends React.Component {


  constructor() {
      super();

      this.handler = this.handler.bind(this)
    }

  handler(e) {
    e.preventDefault()
      var newModalData = {
        modalIsOpen: false,
        reservationID: this.state.modalData.reservationID,
        customerID: this.state.modalData.customerID,
        firstName: this.state.modalData.firstName,
        lastName: this.state.modalData.lastName,
        startDate: this.state.modalData.startDate,
        endDate: this.state.modalData.endDate,
        totalDepositPrice: this.state.modalData.totalDepositPrice,
        totalRentalPrice: this.state.modalData.totalRentalPrice,
        tools: this.state.modalData.tools
      }

      this.setState({modalData: newModalData});
  }

  state = {
    "reservations": [
      {
        "reservationID": null,
        "customerID": null,
        "customerUsername": null,
        "startDate": null,
        "endDate": null
      }
    ],
   selectedResId: null,
   firstTimeLoad: true,
   fixedHeader: true,
   fixedFooter: true,
   stripedRows: false,
   showRowHover: false,
   selectable: true,
   multiSelectable: false,
   enableSelectAll: false,
   deselectOnClickaway: true,
   showCheckboxes: false,
   height: '200px',
   modalData : {
     modalIsOpen: false,
     reservationID: false,
     customerID: '',
     firstName: '',
     lastName: '',
     startDate: '',
     endDate: '',
     totalDepositPrice: null,
     totalRentalPrice: null,
     tools: [""]
   }
  }

  pushToConfirm = function(){
    console.log("hello");
    console.log(this.state.selectedResId);
    localStorage.setItem("selectedResId", this.state.selectedResId);
    this.props.history.push('/resconfirm');
  }



  handleClick(e, rId) {
    console.log('The link was clicked.');

    axios.get('http://localhost:8080/reservation/pickup?reservationID=' + rId)
      .then(res => {
        console.log(res);
        var newModalData = {
          modalIsOpen: true,
          reservationID: res['data']['reservationID'],
          customerID: res['data']['customerID'],
          firstName: res['data']['firstName'],
          lastName: res['data']['lastName'],
          startDate: res['data']['startDate'],
          endDate: res['data']['endDate'],
          totalDepositPrice: parseFloat(res['data']['totalDepositPrice']).toFixed(2),
          totalRentalPrice: parseFloat(res['data']['totalRentalPrice']).toFixed(2),
        };
        if ('tools' in res['data'])
          newModalData.tools = res['data']['tools']
        else {
          newModalData.tools = [""]
        }
        this.setState({modalData: newModalData});
        console.log(this.state.modalData);
      })
      .catch(function (error) {
        console.log(error);
      });
  }


  render() {
    if (this.state.firstTimeLoad == true){
      axios.get('http://localhost:8080/reservation/pickup')
        .then(res => {
          this.setState({reservations : res['data']});
        })
        .catch(function (error) {
          console.log(error);
        });
      this.state.firstTimeLoad = false;
    }
    return (
      <div>
          <div class="viewprofile">
            <h2> Pickup Reservation </h2>

            <MuiThemeProvider>
            <NavBar/>
            <ResDetails resData={this.state.modalData} handler = {this.handler}/>

            <Table
           height={this.state.height}
           fixedHeader={this.state.fixedHeader}
           fixedFooter={this.state.fixedFooter}
           selectable={this.state.selectable}
           multiSelectable={this.state.multiSelectable}
         >
                <TableHeader
                      enableSelectAll={this.state.enableSelectAll}
                      adjustForCheckbox={this.state.showCheckboxes}
                      displaySelectAll={this.state.showCheckboxes}
                >
                 <TableRow>
                   <TableHeaderColumn colSpan="3" tooltip="Reservations" style={{textAlign: 'Left'}}>
                     Reservations
                   </TableHeaderColumn>
                 </TableRow>
                 <TableRow>
                   <TableHeaderColumn tooltip="Reservation ID"> Reservation ID</TableHeaderColumn>
                   <TableHeaderColumn tooltip="Customer">CustomerID</TableHeaderColumn>
                   <TableHeaderColumn tooltip="CustomerID"> Customer</TableHeaderColumn>
                   <TableHeaderColumn tooltip="Start Date"> Start Date</TableHeaderColumn>
                   <TableHeaderColumn tooltip="End Date"> End Date</TableHeaderColumn>

                 </TableRow>
                 </TableHeader>
                 <TableBody
                     displayRowCheckbox={this.state.showCheckboxes}
                     deselectOnClickaway={this.state.deselectOnClickaway}
                     showRowHover={this.state.showRowHover}
                     stripedRows={this.state.stripedRows}
                   >
                  {this.state.reservations.map( (res) => {
                    return res.startDate == null ?


                  (
                      <TableRow key={res.reservationID}>
                      <TableRowColumn><a href="#" onClick={(e, v) => this.handleClick(e, res.reservationID)}> {res.reservationID}</a></TableRowColumn>
                        <TableRowColumn>{res.customerID}</TableRowColumn>
                        <TableRowColumn>{res.customerUsername}</TableRowColumn>
                        <TableRowColumn>{res.startDate}</TableRowColumn>
                        <TableRowColumn>{res.endDate}</TableRowColumn>
                      </TableRow>
                  )
                  :
                  (
                      <TableRow key={res.reservationID}>
                      <TableRowColumn><a href="#" onClick={(e, v) => this.handleClick(e, res.reservationID)}> {res.reservationID}</a></TableRowColumn>
                        <TableRowColumn>{res.customerID}</TableRowColumn>
                        <TableRowColumn>{res.customerUsername}</TableRowColumn>
                        <TableRowColumn>{res.startDate.substring(0,10)}</TableRowColumn>
                        <TableRowColumn>{res.endDate.substring(0,10)}</TableRowColumn>
                      </TableRow>
                  )

                }

                  )}
                   </TableBody>
                   </Table>
                   <br></br><br></br>
                   <TextField
                       name="selectedResId"
                       hintText="Enter Reservation ID"
                       value={this.state.selectedResId}
                       onChange={e => this.setState({'selectedResId': e.target.value})}
                   />

                   <br></br><br></br>
                   <button onClick= {(e) => this.pushToConfirm(e)} >Pick Up </button>


                   </MuiThemeProvider>

          </div>
      </div>
    )
  }
}
