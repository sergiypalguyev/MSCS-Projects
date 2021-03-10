import axios from 'axios';
import React, {Component} from 'react';
import {
  Table,
  TableBody,
  TableFooter,
  TableHeader,
  TableHeaderColumn,
  TableRow,
  TableRowColumn,
} from 'material-ui/Table';
import TextField from 'material-ui/TextField';
import Toggle from 'material-ui/Toggle';

import RaisedButton from 'material-ui/RaisedButton';

import MuiThemeProvider from 'material-ui/styles/MuiThemeProvider'
import {Link} from "react-router";
import NavBar from "./nav/NavBar";
import UserID from './messages/UserID';


const styles = {
  propContainer: {
    width: 200,
    overflow: 'hidden',
    margin: '20px auto 0',
  },
  propToggleHeader: {
    margin: '20px auto 10px',
  },
};

export default  class CustomerSummaryReport extends React.Component {

constructor(){
  super();
}

 state = {
    customers:[],
		selectable: false,
		displaySelectAll: false,
		fixedHeader: false,
		adjustForCheckbox: false,
   }
 customer = {
     "customerID":"",
     "firstName": "",
     "middleName": "",
     "lastName": "",
     "email": "",
     "areaCode": "",
     "phoneNumber": "",
     "extension": "",
     "total_reservations":"",
     "total_tools_rented":""
   }



   	fetchReport () {
   		axios.get('http://localhost:8080/report/customer')
   			 .then(res => {
   			 	console.log("Data received!");
   		 		this.setState(res['data']);
   		 		console.log(this.state);
   			 })
   			 .catch(function (error) {
   			 	console.log(error);
   			 });
   	}

componentDidMount(){
  this.fetchReport();
}


  render() {
    return (
      <div>
         <NavBar/>
         <MuiThemeProvider>
              <h2>Customer Report</h2>
              <b>The list of customers and reservations with tools for the last month.</b>

              <br></br>
              <hr></hr>

              <div className="CustomerSummaryReport">
						<Table
							style={{ tableLayout: 'auto' }}
							fixedHeader={this.state.fixedHeader}
							selectable={this.state.selectable}
						>
							<TableHeader
								adjustForCheckbox={this.state.adjustForCheckbox}
								displaySelectAll={this.state.displaySelectAll}
							>
								<TableRow>
                       <TableHeaderColumn>CustomerID</TableHeaderColumn>
                       <TableHeaderColumn>View Profile?</TableHeaderColumn>
                       <TableHeaderColumn>First Name</TableHeaderColumn>
                       <TableHeaderColumn>Middle Name</TableHeaderColumn>
                       <TableHeaderColumn>Last Name</TableHeaderColumn>
                       <TableHeaderColumn>Email</TableHeaderColumn>
                       <TableHeaderColumn>Phone Number</TableHeaderColumn>
                       <TableHeaderColumn>Total # Reservations</TableHeaderColumn>
                       <TableHeaderColumn>Total # Tools Rented</TableHeaderColumn>
                       </TableRow>
  							</TableHeader>
  							<TableBody>
                      {this.state.customers.map((res) => <TableRowData data = {res} />)}
                </TableBody>
   						</Table>
  					</div>
  					<div style={{ margin: '10px' }}>
  						<RaisedButton
  							style={{ margin: '10px' }}
  							label="Back to Report Menu"
  							href="/reports"
  						/>
  						<RaisedButton
  							style={{ margin: '10px' }}
  							label="Reload Results"
  							onClick={(e) => this.fetchReport(e)}
  						/>
  					</div>
          </MuiThemeProvider>
      </div>
    )
  }
}

class TableRowData extends React.Component {

  handleClick(e, userID) {
    console.log('The link was clicked.');
    console.log(userID);
    localStorage.setItem('customerID', userID);
    window.open("/viewProfile");
  }

   render() {
      return (
         <TableRow>
            <TableRowColumn>{this.props.data.customerID}</TableRowColumn>
            <TableRowColumn><a href="#" onClick={(e, v) => this.handleClick(e, this.props.data.customerID)}> View Profile</a></TableRowColumn>
            <TableRowColumn>{this.props.data.firstName}</TableRowColumn>
            <TableRowColumn>{this.props.data.middleName}</TableRowColumn>
            <TableRowColumn>{this.props.data.lastName}</TableRowColumn>
            <TableRowColumn>{this.props.data.email}</TableRowColumn>
            <TableRowColumn>{this.props.data.areaCode}{this.props.data.phoneNumber}{this.props.data.extension}</TableRowColumn>
            <TableRowColumn>{this.props.data.total_reservations}</TableRowColumn>
            <TableRowColumn>{this.props.data.total_tools_rented}</TableRowColumn>
         </TableRow>
      );
   }
}
