import React from 'react';
import MuiThemeProvider from 'material-ui/styles/MuiThemeProvider';
import {
  Table,
  TableBody,
  TableFooter,
  TableHeader,
  TableHeaderColumn,
  TableRow,
  TableRowColumn,
} from 'material-ui/Table';
import RaisedButton from 'material-ui/RaisedButton';
import {BrowserRouter, Route, NavLink, withRouter} from "react-router-dom";
import NavBar from "./nav/NavBar";


const style = {
  margin: 12,
};


export default class Confirmation extends React.Component{
    constructor(props){
        super(props);
        this.selectedTools = [];
        this.handleSelection = this.handleSelection.bind(this);
        this.resetLocalStorage = this.resetLocalStorage.bind(this);
    }

    state = {
      fixedHeader: true,
       fixedFooter: true,
       stripedRows: false,
       showRowHover: false,
       selectable: false,
       multiSelectable: false,
       enableSelectAll: false,
       deselectOnClickaway: false,
       showCheckboxes: false,
       height: 'auto',
       selectedItems: [],
       isCartEmpty: true
    };


    componentWillMount(){
        if(localStorage.getItem("toolsAddedToCart") != null){
            this.setState({isCartEmpty: false});
        }
    }
    componentDidMount(){
        localStorage.removeItem("toolsAddedToCart");
    }

    calculateDepositPrice(){
        let items = JSON.parse(localStorage.getItem("toolsAddedToCart"));
        if(items == null){
            return 0;
        }
        let total = 0;
        items.forEach(function(item){
            total += item.depositPrice;
        });
        return total;
    }

    calculateRentalPrice(){
        let items = JSON.parse(localStorage.getItem("toolsAddedToCart"));
        let total = 0;
        if(items == null){
            return 0;
        }
        items.forEach(function(item){
            total += item.rentalPrice;
        });
        return total;
    }

    handleSelection(arr){
        this.state.selectedItems = arr;
    }

    getDates(){

        let start = new Date(localStorage.getItem("startDate"));
        let end =  new Date(localStorage.getItem("endDate"));
        let startDate = start.getFullYear() + '-' +
                             (parseInt(start.getMonth()) + 1) + '-' +
                             start.getDate() ;
     let endDate = end.getFullYear() + '-' +
                          (parseInt(end.getMonth()) + 1) + '-' +
                          end.getDate() ;
        return (startDate + "   -   " + endDate);
    }

    getDaysRented(){
        let start = new Date(localStorage.getItem("startDate"));
        let end =  new Date(localStorage.getItem("endDate"));
        let timeDiff = Math.abs(end.getTime() - start.getTime());
        return Math.ceil(timeDiff / (1000 * 3600 * 24));
    }
    getReservationID(){
        return localStorage.getItem("reservationID");
    }

    resetLocalStorage(){
        localStorage.removeItem("toolsAddedToCart");
        localStorage.removeItem("reservationID");
        this.props.history.push('/viewProfile');
    }

    render(){
        const totalDepositPrice = this.calculateDepositPrice();
        const totalRentalPrice = this.calculateRentalPrice();
        const {isCartEmpty} = this.state;

        return(
            <MuiThemeProvider>
            {isCartEmpty && this.resetLocalStorage()}
            {!isCartEmpty &&
             <div>
             <NavBar/>
            <h2 ref={subtitle => this.subtitle = subtitle}>Reservation Confirmation</h2>
            <div><b><p><span className="info">Reservation ID:</span>{this.getReservationID()}</p></b></div>
             <div><p><span className="info">Reservation Dates:</span>{this.getDates()}</p></div>
             <div><p><span className="info">Number of Days Rented:</span> {this.getDaysRented()}</p></div>
             <div><p><span className="info">Total Deposit Price: $</span>{parseFloat(totalDepositPrice).toFixed(2)}</p></div>
             <div><p><span className="info">Total Rental Price: $</span>{parseFloat(totalRentalPrice).toFixed(2) * this.getDaysRented()}</p></div>
             <hr size="10"/>
                 <Table
                height={this.state.height}
                fixedHeader={this.state.fixedHeader}
                fixedFooter={this.state.fixedFooter}
                selectable={this.state.selectable}
                multiSelectable={this.state.multiSelectable}
                onRowSelection={this.handleSelection}
              >
              <TableHeader
                     enableSelectAll={this.state.enableSelectAll}
                   >
                   <TableRow>
                        <TableHeaderColumn colSpan="3" tooltip="Tools" style={{textAlign: 'Left'}}>
                        </TableHeaderColumn>
                      </TableRow>
                      <TableRow>
                        <TableHeaderColumn tooltip="Tool ID">Tool ID</TableHeaderColumn>
                        <TableHeaderColumn tooltip="Description">Description</TableHeaderColumn>
                        <TableHeaderColumn tooltip="Deposit Price">Deposit Price</TableHeaderColumn>
                        <TableHeaderColumn tooltip="Rental Price">Rental Price</TableHeaderColumn>
                      </TableRow>
                      </TableHeader>
                      <TableBody
                          displayRowCheckbox={this.state.showCheckboxes}
                          deselectOnClickaway={this.state.deselectOnClickaway}
                          showRowHover={this.state.showRowHover}
                          stripedRows={this.state.stripedRows}
                        >
                        {JSON.parse(localStorage.getItem("toolsAddedToCart")) .map( (tool) => (
                            <TableRow key={tool.id}>
                              <TableRowColumn>{tool.id}</TableRowColumn>
                              <TableRowColumn>{tool.description}</TableRowColumn>
                              <TableRowColumn>${parseFloat(tool.depositPrice).toFixed(2)}</TableRowColumn>
                              <TableRowColumn>${parseFloat(tool.rentalPrice).toFixed(2)}</TableRowColumn>
                            </TableRow>
                        ))}
                        <TableRow>
                            <TableRowColumn><b>Tools</b></TableRowColumn>
                            <TableRowColumn></TableRowColumn>
                            <TableRowColumn>${parseFloat(totalDepositPrice).toFixed(2)}</TableRowColumn>
                            <TableRowColumn>${parseFloat(totalRentalPrice).toFixed(2) * this.getDaysRented()}</TableRowColumn>
                        </TableRow>
                        </TableBody>
                        </Table>
            </div>
        }
            </MuiThemeProvider>
        )
    }
}
