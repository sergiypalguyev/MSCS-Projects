import React from 'react';
import axios from 'axios';
import TextField from 'material-ui/TextField';
import MenuItem from 'material-ui/MenuItem';
import SelectField from 'material-ui/SelectField';
import MuiThemeProvider from 'material-ui/styles/MuiThemeProvider'
import {RadioButton, RadioButtonGroup} from 'material-ui/RadioButton';
import ActionFavorite from 'material-ui/svg-icons/action/favorite';
import ActionFavoriteBorder from 'material-ui/svg-icons/action/favorite-border';
import {Table} from 'material-ui/Table';
import {Link} from "react-router";
import {BrowserRouter, Route, NavLink, withRouter} from "react-router-dom";
import RaisedButton from 'material-ui/RaisedButton';
import SuccessfulRegistrationMsg from './messages/SuccessfulRegistrationMsg.js';
import ErrorMsg from './messages/LoginAndRegistrationErrorMsg';


const style = {
  margin: 12,
};



class SignupForm extends React.Component {
  state = {
    fname: '',
    mname: '',
    lname: '',
    hphone: '',
    wphone: '',
    cphone: '',



    allMonths: ['1',
    '2',
    '3',
    '4',
    '5',
    '6',
    '7',
    '8',
    '9',
    '10',
    '11',
    '12'],
    allYears:['2017','2018','2019','2020','2021','2022','2023','2024','2025','2026','2027'],
    allStates : [	'AL','MT',
    	'AK','NE',
    	'AZ','NV',
    	'AR','NH',
    	'CA','NJ',
    	'CO','NM',
    	'CT','NY',
    	'DE','NC',
    	'FL','ND',
    	'GA','OH',
    	'HI','OK',
    	'ID','OR',
    	'IL','PA',
    	'IN','RI',
    	'IA','SC',
    	'KS','SD',
    	'KY','TN',
    	'LA','TX',
    	'ME','UT',
    	'MD','VT',
    	'MA','VA',
    	'MI','WA',
    	'MN','WV',
    	'MS','WI',
    	'MO','WY'],

    prim_phone: '',

    username: '',
    email: '',
    password: '',
    streetAddr: '',
    city: '',
    state: '',
    zipcode: '',

    card_username: '',
    card_number: '',
    exp_month: '',
    exp_year: '',
    cvc: '',
    registeredSucessfully: false,
    error: ''
  }


componentDidUpdate(prevProps, prevState) {
    if(prevState.registeredSucessfully != this.state.registeredSucessfully && this.state.registeredSucessfully == true){
        let current = this;
        setTimeout(function () {
                current.props.history.push('/login');
        }, 3000);
    }

}

  change = e => {
    this.setState({
      [e.target.name]: e.target.value
    });
  };
  onSubmit = (e) => {
        e.preventDefault();
        let current = this;
        // get our form data out of state
      //  const { firstName, lastName, username, email, password } = this.state;
        const {
          fname,
          mname,
          lname,
          hphone,
          wphone,
          cphone,
          prim_phone,
          username,
          email,
          password,
          streetAddr,
          city,
          state,
          zipcode,
          card_username,
          card_number,
          exp_month,
          exp_year,
          cvc
        } = this.state;
        // console.log("username is: ", username);
        // console.log("password is: ", password);
        // console.log("state is: ", state);
        console.log("user is: ", prim_phone);

        //console.log("firstName is ", fname);
        axios.post('http://localhost:8080/register', {
          fname,
          mname,
          lname,
          hphone,
          wphone,
          cphone,
          prim_phone,
          username,
          email,
          password,
          streetAddr,
          city,
          state,
          zipcode,
          card_username,
          card_number,
          exp_month,
          exp_year,
          cvc })
          .then((result) => {
            //access the results here....
            console.log(result);
            this.setState({registeredSucessfully: true});
          }).catch(function(reject){
            console.log(reject);
            //current.setState({registeredSucessfully: false});
            const error = reject.response.data.reason;
            if(error){
                current.setState({error: error});
            }else{
                current.setState({error: "Something went wrong. Please check the database."});
            }
          });
      }
  handleChange = (event, index, state) => this.setState({state});
  handleMonthChange = (event, index, exp_month) => this.setState({exp_month});
  handleYearChange = (event, index, exp_year) => this.setState({exp_year});
  render() {

    const {error, registeredSucessfully }= this.state;

    return (

      <div>
        <h1>Customer Registration Form</h1>
        <form>
          <MuiThemeProvider>
            <TextField
                name="fname"
                hintText="First Name"
                floatingLabelText="First Name"
                floatingLabelFixed={true}
                value={this.state.fname}
                onChange={e => this.change(e)}
            />
            <TextField
                name="mname"
                hintText="Middle Name"
                floatingLabelText="Middle Name"
                floatingLabelFixed={true}
                value={this.state.mname}
                onChange={e => this.change(e)}
            />
            <TextField
                name="lname"
                hintText="Last Name"
                floatingLabelText="Last Name"
                floatingLabelFixed={true}
                value={this.state.lname}
                onChange={e => this.change(e)}
            />
            <div></div>
            <TextField
                name="hphone"
                hintText="Home Phone"
                floatingLabelText="Home Phone"
                floatingLabelFixed={true}
                value={this.state.hphone}
                onChange={e => this.change(e)}
            />
            <TextField
                name="wphone"
                hintText="Work Phone"
                floatingLabelText="Work Phone"
                floatingLabelFixed={true}
                value={this.state.wphone}
                onChange={e => this.change(e)}
            />
            <TextField
                name="cphone"
                hintText="Cell Phone"
                floatingLabelText="Cell Phone"
                floatingLabelFixed={true}
                value={this.state.cphone}
                onChange={e => this.change(e)}
            />



            <br></br>
            <br></br>
            <br></br>
            Primary Phone:
            <input
              type="radio"
              name="prim_phone"
              id="homePhoneRadioButton"
              value="home"
              checked={this.state.prim_phone === 'home'}
              onChange={e => this.change(e)}
            />
            <label htmlFor="homePhoneRadioButton"> Home Phone  </label>
            <input
               type="radio"
               name="prim_phone"
               id="workPhoneRadioButton"
               value="work"
               checked={this.state.prim_phone === 'work'}
               onChange={e => this.change(e)}
            />
            <label htmlFor="workPhoneRadioButton"> Work Phone  </label>
            <input
               type="radio"
               name="prim_phone"
               id="cellPhoneRadioButton"
               value="cell"
               checked={this.state.prim_phone === 'cell'}
               onChange={e => this.change(e)}
            />
            <label htmlFor="cellPhoneRadioButton"> Cell Phone  </label>
            <div></div>
            <br></br>
            <br></br>


            <TextField
                name="username"
                hintText="Username"
                floatingLabelText="Username"
                floatingLabelFixed={true}
                value={this.state.username}
                onChange={e => this.change(e)}
            />
            <TextField
                name="email"
                hintText="Email"
                floatingLabelText="Email"
                floatingLabelFixed={true}
                value={this.state.email}
                onChange={e => this.change(e)}
            />
            <div></div>
            <TextField
                name="password"
                hintText="Password"
                floatingLabelText="Password"
                floatingLabelFixed={true}
                value={this.state.password}
                onChange={e => this.change(e)}
            />
            <TextField
                name="password2"
                hintText="Re-type Password"
                floatingLabelText="Password"
                floatingLabelFixed={true}
                //value={this.state.password2}
                onChange={e => this.change(e)}
            />
            <div></div>
            <TextField
                name="streetAddr"
                hintText="Street Address"
                floatingLabelText="Street Address"
                floatingLabelFixed={true}
                value={this.state.streetAddr}
                onChange={e => this.change(e)}
            />
            <div></div>
            <TextField
                name="city"
                hintText="City"
                floatingLabelText="City"
                floatingLabelFixed={true}
                value={this.state.city}
                onChange={e => this.change(e)}
            />
            <SelectField
              floatingLabelText="State"
              name="state"
              value={this.state.state}
              onChange={this.handleChange}
            >
            {this.state.allStates.map((ps) => (
                <MenuItem value={ps} primaryText={ps} />
              ))}
            </SelectField>

            <TextField
                name="zipcode"
                hintText="Zipcode"
                floatingLabelText="Zipcode"
                floatingLabelFixed={true}
                value={this.state.zipcode}
                onChange={e => this.change(e)}
            />

            <h2>Credit Card</h2>

            <TextField
                name="card_username"
                hintText="Name on Credit Card"
                floatingLabelText="Name on Credit Card"
                floatingLabelFixed={true}
                value={this.state.card_username}
                onChange={e => this.change(e)}
            />
            <TextField
                name="card_number"
                hintText="Credit Card #"
                floatingLabelText="Credit Card #"
                floatingLabelFixed={true}
                value={this.state.card_number}
                onChange={e => this.change(e)}
            />
            <div></div>

            <SelectField
            floatingLabelText="Expiration Month"
              name="exp_month"
              value={this.state.exp_month}
              onChange={this.handleMonthChange}
            >
            {this.state.allMonths.map((ps) => (
                <MenuItem value={ps} primaryText={ps} />
              ))}
            </SelectField>


            <SelectField
            floatingLabelText="Expiration Year"
              name="exp_year"
              value={this.state.exp_year}
              onChange={this.handleYearChange}
            >
            {this.state.allYears.map((ps) => (
                <MenuItem value={ps} primaryText={ps} />
              ))}
            </SelectField>

            <TextField
                name="cvc"
                hintText="CVC"
                floatingLabelText="CVC"
                floatingLabelFixed={true}
                value={this.state.cvc}
                onChange={e => this.change(e)}
            />
            <div>
            {registeredSucessfully && <SuccessfulRegistrationMsg msg={"You have been successfully registered as: " + this.state.username + ". Please login!"} />}
            </div>
            <div>
            {error && <ErrorMsg msg={error} />}
            </div>
            <RaisedButton label="Register" primary={true}  onClick= {(e) => this.onSubmit(e)} />
            <div>
            <NavLink exact to="/login" >
                <RaisedButton label="Already have an account? Login!" secondary={true} style={style} />
            </NavLink>
            </div>
          </MuiThemeProvider>
        </form>
      </div>
    )
  }
}

export default withRouter(SignupForm);
