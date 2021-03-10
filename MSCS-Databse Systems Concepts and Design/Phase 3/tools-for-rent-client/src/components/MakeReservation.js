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
import BootstrappedTable from './BootstrappedTable';
import Row from './Row';
import ErrorMsg from './messages/LoginAndRegistrationErrorMsg';
import SuccessfulRegistrationMsg from './messages/SuccessfulRegistrationMsg.js';
import ReactDOM from 'react-dom';

let DateTimeFormat = global.Intl.DateTimeFormat;
export default  class MakeReservation extends React.Component {

constructor() {
        super();
        this.onRowSelection = this.onRowSelection.bind(this);
        this.onSubmit = this.onSubmit.bind(this);
        this.addedToolsSuccessfully = this.addedToolsSuccessfully.bind(this);
}
  state = {
    startDate: "",
    endDate: "",
    tools: [],
    type: "",
    subType: "",
    subOption:"",
    powerSource: "",
    allSubTypes: [""],
    allPowerSources: [""],
    //
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
     height: '200px',
     localStorageCount:0,
     tempStartDate:"",
     tempEndDate:"",
     error:"",
     toolsAdded:false
  }

  addToCart =[];


 onRowSelection(rowNumber){
     var cart = this.state.tools;
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


    axios.get(getURL)
      .then(res => {
        if ('subTypes' in res['data'])
          this.setState({allSubTypes: res['data']['subTypes']});
        this.setState({allPowerSources: res['data']['powerSources']});
      })
      .catch(function (error) {
        console.log(error);
      });


  };

  componentDidMount(){
      this.scrollToBottom();
  }
  componentDidUpdate() {
   this.scrollToBottom();
 }

 scrollToBottom = () => {
  const node = ReactDOM.findDOMNode(this.messagesEnd);
  node.scrollIntoView({ behavior: "smooth" });
}

  changePowerSource = function(event, powerSource){
    this.setState({powerSource});
    var typeArr = this.state.type.split(" ");
    var getPSURL = "http://localhost:8080/tool/availability?type=" + typeArr[0] + '%20' + typeArr[1] + "&powerSource=" + powerSource;

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


      addedToolsSuccessfully(){
          this.setState({toolsAdded: true});
          let cur = this;
          setTimeout(function(){
              cur.setState({toolsAdded: false});
          }, 3000);
      }
  render() {
      const {tools} = this.state;
      const {error} = this.state;
      const{toolsAdded} = this.state;
    return (
      <div>
      <NavBar/>
          <form>
            <MuiThemeProvider>
              <div className="viewprofile">
                <h2> Make Reservation </h2>
                <br></br>
                Start Date:
                <DatePicker
                  hintText="Pick a start date"
                  firstDayOfWeek={0}
                  formatDate={new DateTimeFormat('en-US', {
                    day: 'numeric',
                    month: 'long',
                    year: 'numeric',
                  }).format}
                  onChange={(e, date) => {
                    this.state.startDate = date.getFullYear() + '-' +
                                         (parseInt(date .getMonth()) + 1) + '-' +
                                         date.getDate() + ' 12:00:00';
                     this.state.tempStartDate = date;
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
                                         date.getDate() + ' 12:00:00';
                     this.state.tempEndDate = date;
                  }}
                />



                <br></br><br></br>
                Type:
                <input
                  type="radio"
                  name="type"
                  id="allTools"
                  value="All Tools"
                  onChange={e => this.changeType(e)}
                />
                <label htmlFor="allTools"> All Tools  </label>

                <input
                   type="radio"
                   name="type"
                   id="handTools"
                   value="Hand Tool"
                   onChange={e => this.changeType(e)}
                />
                <label htmlFor="handTools"> Hand Tool  </label>

                <input
                   type="radio"
                   name="type"
                   id="gardenTools"
                   value="Garden Tool"
                   onChange={e => this.changeType(e)}
                />
                <label htmlFor="gardenTools"> Garden Tool  </label>

                <input
                   type="radio"
                   name="type"
                   id="ladderTools"
                   value="Ladder Tool"
                   onChange={e => this.changeType(e)}
                />
                <label htmlFor="ladderTools"> Ladder Tool  </label>

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

          <div class="viewprofile">
            <h2> Available Tools For Rent </h2>
          </div>
          <div>

            {!error && (tools.length > 0) &&  <BootstrappedTable data={tools} startDate={this.state.tempStartDate} endDate={this.state.tempEndDate} onUpdateCart={i => this.setState({localStorageCount: i})} setToolsAdded={i => this.addedToolsSuccessfully()}/>}
            {toolsAdded && !error && <SuccessfulRegistrationMsg msg={"Tools successfully added to Cart! Please view cart above!"} />}
            {error && <ErrorMsg msg={error} />}
          </div>
          <div style={{ float:"left", clear: "both" }}
             ref={(el) => { this.messagesEnd = el; }}>
        </div>
      </div>
    )
  }
}
