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
import NavBar from "./nav/NavBar";
import Dialog from 'material-ui/Dialog';
import RaisedButton from 'material-ui/RaisedButton';
import FlatButton from 'material-ui/FlatButton';
import ToolDetails from './ToolDetails';
import ErrorMsg from './messages/LoginAndRegistrationErrorMsg';
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


export default  class CheckToolAvailability extends React.Component {


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
    errorTextStartDate: "",
    errorTextEndDate: "",
    startDate: "",
    endDate: "",
    tools: [],
    type: "All Tools",
    subType: "",
    subOption:"",
    powerSource: "",
    allSubTypes: [
      "Digger",
      "Pruner",
      "Rakes",
      "Striking",
      "Wheelbarrows",
      "Gun",
      "Hammer",
      "Pliers",
      "Ratchet",
      "Screwdriver",
      "Socket",
      "Wrench",
      "Step",
      "Straight",
      "Air-Compressor",
      "Drill",
      "Generator",
      "Mixer",
      "Sander",
      "Saw"],
      allPowerSources: ["Manual", "A/C", "Gas", "D/C"],

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

  changeType = function(e){
    this.setState({
      [e.target.name]: e.target.value});
    var getURL = '';
    if (e.target.value == 'All Tools')
      getURL = 'http://localhost:8080/tool/availability';
    else if (e.target.value == 'Garden Tool')
      getURL = 'http://localhost:8080/tool/availability?type=Garden%20Tool';
    else if (e.target.value == 'Hand Tool')
      getURL = 'http://localhost:8080/tool/availability?type=Hand%20Tool';
    else if (e.target.value == 'Power Tool')
      getURL = 'http://localhost:8080/tool/availability?type=Power%20Tool';
    else if (e.target.value == 'Ladder Tool')
      getURL = 'http://localhost:8080/tool/availability?type=Ladder%20Tool';

    console.log(getURL);
    axios.get(getURL)
      .then(res => {
        if ('subTypes' in res['data'])
          this.setState({allSubTypes: res['data']['subTypes']});
        if ('powerSources' in res['data'])
          this.setState({allPowerSources: res['data']['powerSources']});
      })
      .catch(function (error) {
        console.log(error);
      });


  };

  changePowerSource = function(event, powerSource){
    this.setState({powerSource});
    var typeArr = this.state.type.split(" ");
    var getPSURL = "http://localhost:8080/tool/availability?type=" + typeArr[0] + '%20' + typeArr[1] + "&powerSource=" + powerSource;
    console.log(getPSURL);
    axios.get(getPSURL)
      .then(res => {
        this.setState({allSubTypes: res['data']['subTypes']});
      })
      .catch(function (error) {
        console.log(error);
      });
  }





  onSubmit = (e) => {
        e.preventDefault();
        let cur = this;
        if (this.state.startDate == "" ){
          this.setState({errorTextStartDate: "required field"});
        }
        else{
          this.setState({errorTextStartDate: ""});
        }
        if (this.state.endDate == "" ){
          this.setState({errorTextEndDate: "required field"});
        }
        else{
          this.setState({errorTextEndDate: ""});
        }

        var sDate = new Date(this.state.startDate).getTime();
        var eDate = new Date(this.state.endDate).getTime();
        if (eDate < sDate){
          this.setState({errorTextEndDate: "End date should be after or equal to start date."});
          this.setState({error: "End date should be after start date."});
      }else{
        axios.post('http://localhost:8080/tool/availability', {
            "startDate": this.state.startDate,
            "endDate": this.state.endDate,
            "type": this.state.type,
            "subType": this.state.subType,
            "subOption": this.state.subOption,
            "powerSource": this.state.powerSource
          })
          .then(res => {
              this.setState({error:""});
              this.setState({tools: res['data']});
            }).catch(function(reject){
            console.log(reject);
            cur.setState({error: reject.response.data.errorMessage});
          });
      }
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

        this.setState({});
      }

  render() {
      const {error} = this.state;
    return (
      <div>
      <NavBar/>
          <form>
            <MuiThemeProvider>

            <ToolDetails toolData={this.state.modalData} handler = {this.handler}/>


              <div class="viewprofile">
                <h2>Check Tool Availability </h2>
                <br></br>
                Start Date:
                <DatePicker
                  hintText="Pick a start date"
                  errorText={this.state.errorTextStartDate}
                  firstDayOfWeek={0}
                  formatDate={new DateTimeFormat('en-US', {
                    day: 'numeric',
                    month: 'long',
                    year: 'numeric',
                  }).format}
                  onChange={(e, date) => {
                    this.state.startDate = date.getFullYear() + '-' +
                                         (parseInt(date .getMonth()) + 1) + '-' +
                                         date.getDate() + ' 12:00:00'
                  }}
                />
                <br></br>
                End Date:
                <DatePicker
                  hintText="Pick an end date"
                  errorText={this.state.errorTextEndDate}
                  firstDayOfWeek={0}
                  formatDate={new DateTimeFormat('en-US', {
                    day: 'numeric',
                    month: 'long',
                    year: 'numeric',
                  }).format}
                  onChange={(e, date) => {
                    this.state.endDate = date.getFullYear() + '-' +
                                         (parseInt(date .getMonth()) + 1) + '-' +
                                         date.getDate() + ' 12:00:00'
                  }}
                />



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

                <br></br><br></br>

                <br></br>
                Power Source:
                <br></br>
                <SelectField
                  name="powerSource"
                  value={this.state.powerSource}
                  onChange={(event, index, powerSource) => this.changePowerSource(event, powerSource)}
                >
                  {this.state.allPowerSources.map((ps) => (
                      <MenuItem value={ps} primaryText={ps} />
                    ))}
                </SelectField>
                <br></br><br></br>

                <br></br>
                Sub Type:
                <br></br>
                <SelectField
                  name="subType"
                  value={this.state.subType}
                  onChange={(event, index, subType) => this.setState({subType})}
                >
                  {this.state.allSubTypes.map((st) => (
                      <MenuItem value={st} primaryText={st} />
                    ))}
                </SelectField>
                <br></br><br></br>




                Custom Search:
                <br></br>
                <TextField
                    name="subOption"
                    value={this.state.subOption}
                    onChange={ e => this.setState({'subOption': e.target.value})}
                />

                <br></br><br></br>
                <button onClick= {(e) => this.onSubmit(e)} >Search </button>
                <br></br>
              </div>
            </MuiThemeProvider>
          </form>
          <br></br>

          <hr></hr>
          <div>
          {error && <ErrorMsg msg={error} />}
          </div>

          {(!error) && <div class="checkToolAvailability">
            <h2> Tools </h2>

            <MuiThemeProvider>
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
                  {this.state.tools.map( (tool) => (
                      <TableRow key={tool.toolID}>
                        <TableRowColumn>{tool.toolID}</TableRowColumn>
                        <TableRowColumn><a href="#" onClick={(e, v) => this.handleClick(e, tool.toolID)}> {tool.shortDesc}</a></TableRowColumn>
                        <TableRowColumn>${parseFloat(tool.depositPrice).toFixed(2)}</TableRowColumn>
                        <TableRowColumn>${parseFloat(tool.rentalPrice).toFixed(2)}</TableRowColumn>
                      </TableRow>
                  ))}
                   </TableBody>
                   </Table>
                   </MuiThemeProvider>

          </div>}
      </div>
    )
  }
}
