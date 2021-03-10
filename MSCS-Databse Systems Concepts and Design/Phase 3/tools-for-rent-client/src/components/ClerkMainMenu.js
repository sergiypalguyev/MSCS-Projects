import axios from 'axios';
import React, {Component} from 'react';
import RaisedButton from 'material-ui/RaisedButton';

import MuiThemeProvider from 'material-ui/styles/MuiThemeProvider'
import {Link} from "react-router";
import NavBar from "./nav/NavBar";


export default class ClerkMainMenu extends React.Component {

	constructor(props) {
		super(props);
		this.performLogout = this.performLogout.bind(this);
	}

	performLogout(event){
        localStorage.clear();
    }

	render() {
		return (
			<div>
			<NavBar/>
				<MuiThemeProvider>
					<h1>Main Menu</h1>
					<div>
						<RaisedButton 
							label="Pick-Up Reservation" 
							href="/pickUpReservation"
							style={{ margin: '10px' }} 
							fullWidth={true} 
						/>
						<RaisedButton 
							label="Drop-Off Reservation" 
							href="/dropOffReservation"
							style={{ margin: '10px' }} 
							fullWidth={true} 
						/>
						<RaisedButton 
							label="Add New Tool" 
							href="/addNewTool"
							style={{ margin: '10px' }} 
							fullWidth={true} 
						/>
						<RaisedButton 
							label="Generate Reports" 
							href="/reports"
							style={{ margin: '10px' }} 
							fullWidth={true} 
						/>
						<RaisedButton 
							label="Exit" 
							onClick={this.performLogout}
							href="/login"
							style={{ margin: '10px' }} 
							fullWidth={true} 
						/>
					</div>
				</MuiThemeProvider>
			</div>
		)
	}
}
