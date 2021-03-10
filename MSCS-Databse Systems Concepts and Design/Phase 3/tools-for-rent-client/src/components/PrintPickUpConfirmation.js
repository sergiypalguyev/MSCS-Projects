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
import ToolDetails from './ToolDetails';

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


export default  class PrintPickUpConfirmation extends React.Component {


  constructor() {
      super();

      this.handler = this.handler.bind(this)
    }

    handler(e) {
      e.preventDefault()
        var newModalData = {
          modalIsOpen: false,
          modalToolID: this.state.modalData.modalToolID,
          modalToolType: this.state.modalData.modalToolType,
          modalShortDesc: this.state.modalData.modalShortDesc,
          modalFullDesc: this.state.modalData.modalFullDesc,
          modalDepPrice: this.state.modalData.modalDepPrice,
          modalRentPrice: this.state.modalData.modalRentPrice,
          modalAccList: this.state.modalData.modalAccList
        };
        this.setState({modalData: newModalData});
        //this.state.modalData.modalIsOpen = false;


    }

  state = {
    customerID: '',
    selectedResId: null,
    firstTimeLoad: true,
    firstName: '',
    lastName: '',
    totalDepositPrice: '',
    totalRentalPrice: '',
    startDate: '',
    endDate: '',
    tools: [""],

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
     modalToolID: '',
     modalToolType: '',
     modalShortDesc: '',
     modalFullDesc: '',
     modalDepPrice: '',
     modalRentPrice: '',
     modalAccList: [""]
   }
  }

  onSubmit = (e) => {
        e.preventDefault();
        window.print();


      }



  handleClick(e, tId) {
    console.log('The link was clicked.');

    axios.get('http://localhost:8080/tool/details?toolID=' + tId)
      .then(res => {
        var newModalData = {
          modalIsOpen: true,
          modalToolID: res['data']['toolID'],
          modalToolType: res['data']['type'],
          modalShortDesc: res['data']['shortDesc'],
          modalFullDesc: res['data']['longDesc'],
          modalDepPrice: parseFloat(res['data']['depositPrice']).toFixed(2),
          modalRentPrice: parseFloat(res['data']['rentalPrice']).toFixed(2),
          modalAccList: res['data']['accList']
        }
        if ('accList' in res['data'])
          newModalData.modalAccList = res['data']['accList']
        else {
          newModalData.modalAccList = [""]
        }
        this.setState({modalData: newModalData});
      })
      .catch(function (error) {
        console.log(error);
      });
  }

  render() {
    if (this.state.firstTimeLoad == true){
      axios.get('http://localhost:8080/reservation/pickup?reservationID=' + localStorage.getItem("selectedResId"))
        .then(res => {
          this.setState({customerID : res['data']['customerID']});
          this.setState({selectedResId : localStorage.getItem("selectedResId")});
          this.setState({firstName : res['data']['firstName']});
          this.setState({lastName : res['data']['lastName']});
          this.setState({totalDepositPrice : res['data']['totalDepositPrice']});
          this.setState({totalRentalPrice : res['data']['totalRentalPrice']});
          this.setState({startDate : res['data']['startDate']});
          this.setState({endDate : res['data']['endDate']});
          this.setState({tools : res['data']['tools']});
        })
        .catch(function (error) {
          console.log(error);
        });
      this.state.firstTimeLoad = false;
    }
    return (
      <div>
            <NavBar/>
          <div class="viewprofile">
            <h2> Pickup Reservation </h2>
            <h2> Rental Contract </h2>
            <MuiThemeProvider>

            <ToolDetails toolData={this.state.modalData} handler = {this.handler}/>

            <Table
           height={this.state.height}
           fixedHeader={this.state.fixedHeader}
           fixedFooter={this.state.fixedFooter}
           selectable={this.state.selectable}
           multiSelectable={this.state.multiSelectable}
         >
          <br></br><br></br>
          Pick-up Clerk: {localStorage.getItem("name")}
          <br></br><br></br>
          Customer Name: {this.state.firstName} {this.state.lastName}
          <br></br><br></br>
          Credit Card #: {localStorage.getItem("credNumber")}
          <br></br><br></br>
          Start Date: {this.state.startDate}
          <br></br><br></br>
          End Date: {this.state.endDate}

                <TableHeader
                  enableSelectAll={this.state.enableSelectAll}
                  adjustForCheckbox={this.state.showCheckboxes}
                  displaySelectAll={this.state.showCheckboxes}
                >
                 <TableRow>
                   <TableHeaderColumn tooltip="Tool ID"> Tool ID</TableHeaderColumn>
                   <TableHeaderColumn tooltip="Tool Name">Tool Name</TableHeaderColumn>
                   <TableHeaderColumn tooltip="Deposit Price"> Deposit Price</TableHeaderColumn>
                   <TableHeaderColumn tooltip="Rental Price"> Rental Price</TableHeaderColumn>

                 </TableRow>
                 </TableHeader>
                 <TableBody
                     displayRowCheckbox={this.state.showCheckboxes}
                     deselectOnClickaway={this.state.deselectOnClickaway}
                     showRowHover={this.state.showRowHover}
                     stripedRows={this.state.stripedRows}
                   >
                  {this.state.tools.map( (res) => (
                      <TableRow key={res.toolID}>
                        <TableRowColumn>{res.toolID}</TableRowColumn>
                        <TableRowColumn><a href="#" onClick={(e, v) => this.handleClick(e, res.toolID)}> {res.shortDesc}</a></TableRowColumn>

                        <TableRowColumn>{res.shortDesc}</TableRowColumn>
                        <TableRowColumn>${parseFloat(res.depositPrice).toFixed(2)}</TableRowColumn>
                        <TableRowColumn>${parseFloat(res.rentalPrice).toFixed(2)}</TableRowColumn>
                      </TableRow>
                  ))}
                   </TableBody>

                   <TableFooter>
                    <TableRow>
                      <TableHeaderColumn tooltip="Tool ID"> Totals</TableHeaderColumn>
                      <TableHeaderColumn tooltip="Tool Name"></TableHeaderColumn>
                      <TableHeaderColumn tooltip="Total Deposit Price"> ${parseFloat(this.state.totalDepositPrice).toFixed(2)}</TableHeaderColumn>
                      <TableHeaderColumn tooltip="Total Rental Price"> ${parseFloat(this.state.totalRentalPrice).toFixed(2)}</TableHeaderColumn>

                    </TableRow>
                    </TableFooter>


                   </Table>

                   <br></br><br></br>
                   Signatures

                   <br></br><br></br>
                   Pickup Clerk - {localStorage.getItem("name")}
                   <hr width="50%"></hr>

                   <br></br><br></br>
                   Dates
                   <hr width="50%"></hr>

                   <br></br><br></br>
                   Customer - {this.state.firstName} {this.state.lastName}
                   <hr width="50%"></hr>

                   <br></br><br></br>
                   Dates
                   <hr width="50%"></hr>

                   <RaisedButton label="Print Contract" primary={true} onClick= {(e) => this.onSubmit(e)} />

                   </MuiThemeProvider>

          </div>
      </div>
    )
  }
}
