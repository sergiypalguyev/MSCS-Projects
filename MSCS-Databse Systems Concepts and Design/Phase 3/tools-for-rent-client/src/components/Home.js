import React from 'react';
import {BrowserRouter, Route, NavLink} from "react-router-dom";
import RaisedButton from 'material-ui/RaisedButton';
import MuiThemeProvider from 'material-ui/styles/MuiThemeProvider';
import Modal from 'react-modal';
import '../css/modal.css';
import {
  Table,
  TableBody,
  TableFooter,
  TableHeader,
  TableHeaderColumn,
  TableRow,
  TableRowColumn,
} from 'material-ui/Table';
var Link = require('react-router-dom').Link;

const style = {
  margin: 12,
};

const customStyles = {
  content : {
    top                   : '50%',
    left                  : '50%',
    right                 : 'auto',
    bottom                : 'auto',
    marginRight           : '-50%',
    transform             : 'translate(-50%, -50%)'
  }
};
class Home extends React.Component {

    constructor() {
        super();

        this.state = {
          modalIsOpen: false,
          fixedHeader: true,
           fixedFooter: true,
           stripedRows: false,
           showRowHover: false,
           selectable: true,
           multiSelectable: true,
           enableSelectAll: false,
           deselectOnClickaway: true,
           showCheckboxes: true,
           height: '200px'
        };

        this.addToLocalStorage = this.addToLocalStorage.bind(this);
        localStorage.removeItem("reservedTools");
      }

      rowEntry = {
          id:'',
          description:'',
          depositPrice:'',
          rentalPrice:''
      }



         addToLocalStorage(id, description, depositPrice, rentalPrice) {
             let existingEntries = JSON.parse(localStorage.getItem("reservedTools"));
            if(existingEntries == null) existingEntries = [];
            let entry = {
                "id": id,
                "description": description,
                "depositPrice":depositPrice,
                "rentalPrice":rentalPrice
            };
            localStorage.setItem("entry", JSON.stringify(entry));
            existingEntries.push(entry);
            localStorage.setItem("reservedTools",JSON.stringify(existingEntries));
        };


render() {
    {this.addToLocalStorage("temp1", "temp1", "temp1", "temp1")}
    {this.addToLocalStorage("temp2", "temp2", "temp2", "temp2")}

    const reservedTools = JSON.parse(localStorage.getItem("reservedTools"));
    return(
        <MuiThemeProvider>
          <div>
            <RaisedButton label="Tools For Rent" fullWidth={true} />
            <br />
            <br />
            <NavLink exact to="/signup" >
              <RaisedButton label="SignUp" primary={true} style={style} />
            </NavLink>
        <Link  to="/login">
            <RaisedButton label="Log in" secondary={true} style={style} />
            </Link>
          </div>
        </MuiThemeProvider>
    );
}
}

export default Home;
