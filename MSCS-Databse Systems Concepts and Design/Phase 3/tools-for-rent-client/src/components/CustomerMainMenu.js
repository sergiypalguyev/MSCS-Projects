import axios from 'axios';
import React, {Component} from 'react';
import RaisedButton from 'material-ui/RaisedButton';

import MuiThemeProvider from 'material-ui/styles/MuiThemeProvider'
import {Link} from "react-router";
import NavBar from "./nav/NavBar";


export default class CustomerMainMenu extends React.Component {

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
							label="View Profile" 
							href="/viewProfile"
							style={{ margin: '10px' }} 
							fullWidth={true} 
						/>
						<RaisedButton 
							label="Check Tool Availability" 
							href="/checkToolAvailability"
							style={{ margin: '10px' }} 
							fullWidth={true} 
						/>
						<RaisedButton 
							label="Make Reservation" 
							href="/makeReservation"
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
