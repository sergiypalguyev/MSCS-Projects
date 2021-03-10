import axios from 'axios';
import React, {Component} from 'react';
import RaisedButton from 'material-ui/RaisedButton';

import MuiThemeProvider from 'material-ui/styles/MuiThemeProvider'
import {Link} from "react-router";
import NavBar from "./nav/NavBar";

export default class GenerateClerkReport extends React.Component {

	constructor() {
		super();
	}

	render() {
		return (
			<div>
			<NavBar/>
				<MuiThemeProvider>
					<h1>Select a Report</h1>
					<div>
						<RaisedButton 
						label="Clerk Report" 
							href="/reports/clerk"
							style={{ margin: '10px' }} 
							fullWidth={true} 
						/>
						<RaisedButton 
							label="Customer Report" 
							href="/reports/customer"
							style={{ margin: '10px' }} 
							fullWidth={true} 
						/>
						<RaisedButton 
							label="Tool Inventory Report" 
							href="/reports/tool"
							style={{ margin: '10px' }} 
							fullWidth={true} 
						/>
					</div>
				</MuiThemeProvider>
			</div>
		)
	}
}
