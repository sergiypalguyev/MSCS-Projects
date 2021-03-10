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


export default class GenerateClerkReport extends React.Component {

	constructor() {
		super();
	}

	state = {
		clerks: [], 

		selectable: false,
		displaySelectAll: false,
		fixedHeader: false,
		adjustForCheckbox: false,
	}

	clerk = {
		clerkID: "",
		firstName: "",
		middleName: "",
		lastName: "",
		email: "",
		dateHired: "",
		numPickUps: "",
		numDropOffs: "",
		combinedTotal: ""
	}

	fetchReport () {
		axios.get('http://localhost:8080/report/clerk')
			 .then(res => {
			 	console.log("Data received!");
		 		this.setState(res['data']);
		 		console.log(this.state);
			 })
			 .catch(function (error) {
			 	console.log(error);
			 });
	}

	componentDidMount() {
		this.fetchReport();
	}

  	render() {
		return (
			<div>
			<NavBar/>
				<MuiThemeProvider>
					<h1>Clerk Report</h1>
					<h5>The list of clerks where their total pickups and dropoffs are shown for the past month. </h5>
					<div>
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
									<TableHeaderColumn>Clerk ID</TableHeaderColumn>
									<TableHeaderColumn>First Name</TableHeaderColumn>
									<TableHeaderColumn>Middle Name</TableHeaderColumn>
									<TableHeaderColumn>Last Name</TableHeaderColumn>
									<TableHeaderColumn>Email</TableHeaderColumn>
									<TableHeaderColumn>Hire Date</TableHeaderColumn>
									<TableHeaderColumn>Number of Pickups</TableHeaderColumn>
									<TableHeaderColumn>Number of Dropoffs</TableHeaderColumn>
									<TableHeaderColumn>Combined Total</TableHeaderColumn>
								</TableRow>
							</TableHeader>
							<TableBody>
								{this.state.clerks.map((res) => <TableRowData data = {res} />)}
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
	render() {
		return (
			<TableRow>
				<TableRowColumn>{this.props.data.clerkID}</TableRowColumn>
				<TableRowColumn>{this.props.data.firstName}</TableRowColumn>
				<TableRowColumn>{this.props.data.middleName}</TableRowColumn>
				<TableRowColumn>{this.props.data.lastName}</TableRowColumn>
				<TableRowColumn>{this.props.data.email}</TableRowColumn>
				<TableRowColumn>{this.props.data.dateHired}</TableRowColumn>
				<TableRowColumn>{this.props.data.numPickUps}</TableRowColumn>
				<TableRowColumn>{this.props.data.numDropOffs}</TableRowColumn>
				<TableRowColumn>{this.props.data.combinedTotal}</TableRowColumn>
			</TableRow>
		)
	}
}