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
import MenuItem from 'material-ui/MenuItem';
import SelectField from 'material-ui/SelectField';
import RaisedButton from 'material-ui/RaisedButton';

import MuiThemeProvider from 'material-ui/styles/MuiThemeProvider'
import {Link} from "react-router";
import NavBar from "./nav/NavBar";
import ToolDetails from './ToolDetails';


export default class ToolSummaryReport extends React.Component {

	constructor() {
		super();
    this.handler = this.handler.bind(this);
    this.handleClick = this.handleClick.bind(this);
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
		tools: [],
    type: "All Tools",

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

	tool = {
		ToolID: "",
		CurrentStatus: "",
		Date: "",
		Description: "",
    RentalProfit: "",
		TotalCost: "",
    TotalProfit:""
	}

  changeType = function(e){
    this.setState({
      [e.target.name]: e.target.value});
  };

  onSubmit = (e) => {
        e.preventDefault();

        var getURL;
        if (this.state.type == 'All Tools')
          getURL = 'http://localhost:8080/report/tool?type=';
        else if (this.state.type == 'Garden Tool')
          getURL = 'http://localhost:8080/report/tool?type=Garden%20Tool';
        else if (this.state.type == 'Hand Tool')
          getURL = 'http://localhost:8080/report/tool?type=Hand%20Tool';
        else if (this.state.type == 'Power Tool')
          getURL = 'http://localhost:8080/report/tool?type=Power%20Tool';
        else if (this.state.type == 'Ladder Tool')
          getURL = 'http://localhost:8080/report/tool?type=Ladder%20Tool';

        console.log(getURL);

        this.fetchReport(getURL)
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
          modalDepPrice: res['data']['depositPrice'],
          modalRentPrice: res['data']['rentalPrice'],
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
  this.setState({});
  }

	fetchReport = function (URL) {
		axios.get(URL)
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
		this.fetchReport('http://localhost:8080/report/tool?type=');
	}

 render() {
		return (
			<div>
			<NavBar/>
				<MuiThemeProvider>
					<h1>Tool Inventory Report</h1>
					<h5>The list of tools where their total profit and cost are shown for all time. </h5>
					<div>
          <ToolDetails toolData={this.state.modalData} handler = {this.handler}/>

          <br></br><br></br>
          Type:&nbsp;&nbsp;&nbsp;&nbsp;
          <input
            type="radio"
            name="type"
            id="allTools"
            value="All Tools"
            checked={this.state.type === 'All Tools'}
            onChange={e => this.changeType(e)}
          />
          <label htmlFor="allTools"> All Tools  </label>&nbsp;&nbsp;&nbsp;&nbsp;

          <input
             type="radio"
             name="type"
             id="handTools"
             value="Hand Tool"
             onChange={e => this.changeType(e)}
          />
          <label htmlFor="handTools"> Hand Tool  </label>&nbsp;&nbsp;&nbsp;&nbsp;

          <input
             type="radio"
             name="type"
             id="gardenTools"
             value="Garden Tool"
             onChange={e => this.changeType(e)}
          />
          <label htmlFor="gardenTools"> Garden Tool  </label>&nbsp;&nbsp;&nbsp;&nbsp;

          <input
             type="radio"
             name="type"
             id="ladderTools"
             value="Ladder Tool"
             onChange={e => this.changeType(e)}
          />
          <label htmlFor="ladderTools"> Ladder Tool  </label>&nbsp;&nbsp;&nbsp;&nbsp;

          <input
             type="radio"
             name="type"
             id="powerTools"
             value="Power Tool"
             onChange={e => this.changeType(e)}
          />
          <label htmlFor="powerTools"> Power Tool  </label>


          <br></br>

          <div style={{ margin: '10px' }}>
            <RaisedButton
              style={{ margin: '10px' }}
              label="Search"
              onClick={(e) => this.onSubmit(e)}
            />
          </div>

          <br></br>

						<Table
							fixedHeader={this.state.fixedHeader}
							selectable={this.state.selectable}
						>
							<TableHeader
                enableSelectAll={this.state.enableSelectAll}
                adjustForCheckbox={this.state.showCheckboxes}
                displaySelectAll={this.state.showCheckboxes}
							>
								<TableRow>
									<TableHeaderColumn>Tool ID</TableHeaderColumn>
									<TableHeaderColumn>Current Status</TableHeaderColumn>
									<TableHeaderColumn>Date</TableHeaderColumn>
									<TableHeaderColumn>Description</TableHeaderColumn>
									<TableHeaderColumn>Rental Profit</TableHeaderColumn>
									<TableHeaderColumn>Total Cost</TableHeaderColumn>
									<TableHeaderColumn>Total Profit</TableHeaderColumn>
								</TableRow>
							</TableHeader>
							<TableBody
                  displayRowCheckbox={this.state.showCheckboxes}
                  deselectOnClickaway={this.state.deselectOnClickaway}
                  showRowHover={this.state.showRowHover}
                  stripedRows={this.state.stripedRows}>
								{this.state.tools.map((res) =>(
                  <TableRow key={res.toolID}>
            				<TableRowColumn>{res.toolID}</TableRowColumn>
            				<TableRowColumn>{res.status}</TableRowColumn>
            				<TableRowColumn>{res.date = (res.date == undefined ? res.date : res.date.substring(0,10))}</TableRowColumn>
                    <TableRowColumn><a href="#" onClick={(e, v) => this.handleClick(e, res.toolID)}> {res.shortDesc}</a></TableRowColumn>
            				<TableRowColumn>${res.rentalProfit = isNaN(parseFloat(res.rentalProfit).toFixed(2))?parseFloat(0.00).toFixed(2):parseFloat(res.rentalProfit).toFixed(2)}</TableRowColumn>
            				<TableRowColumn>${parseFloat(res.totalCost).toFixed(2)}</TableRowColumn>
            				<TableRowColumn>${res.totalProfit = (parseFloat(res.rentalProfit).toFixed(2) - parseFloat(res.totalCost).toFixed(2)).toFixed(2)}</TableRowColumn>
            	   </TableRow>
               ))}
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
              onClick={(e) => this.onSubmit(e)}
						/>
					</div>
				</MuiThemeProvider>
			</div>
		)
	}
}

// class TableRowData extends React.Component {
// 	render() {
// 		return (
// 			<TableRow key={this.props.data.toolID}>
// 				<TableRowColumn>{this.props.data.toolID}</TableRowColumn>
// 				<TableRowColumn>{this.props.data.status}</TableRowColumn>
// 				<TableRowColumn>{this.props.data.date}</TableRowColumn>
//         <TableRowColumn><a href="#" onClick={(e, v) => this.handleClick(e, this.props.data.toolID)}> {this.props.data.shortDesc}</a></TableRowColumn>
// 				<TableRowColumn>${this.props.data.rentalProfit = isNaN(parseFloat(this.props.data.rentalProfit))?Number(0):parseFloat(this.props.data.rentalProfit)}</TableRowColumn>
// 				<TableRowColumn>${this.props.data.totalCost}</TableRowColumn>
// 				<TableRowColumn>${parseFloat(this.props.data.rentalProfit) - parseFloat(this.props.data.totalCost)}</TableRowColumn>
// 	   </TableRow>
//    );
// 	}
// }
