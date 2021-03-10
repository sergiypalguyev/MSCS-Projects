import React from 'react';
import axios from 'axios';
import TextField from 'material-ui/TextField';
import MuiThemeProvider from 'material-ui/styles/MuiThemeProvider'
import {RadioButton, RadioButtonGroup} from 'material-ui/RadioButton';
import ActionFavorite from 'material-ui/svg-icons/action/favorite';
import ActionFavoriteBorder from 'material-ui/svg-icons/action/favorite-border';
import {Table} from 'material-ui/Table';
import RaisedButton from 'material-ui/RaisedButton';
import ErrorMsg from './messages/LoginAndRegistrationErrorMsg';
import {BrowserRouter, Route, NavLink, withRouter} from "react-router-dom";

const style = {
  margin: 12,
};
class LoginForm extends React.Component {

constructor (props) {
super(props);
this.onSubmit = this.onSubmit.bind(this);
}

// This is a comment.

state = {
  username: '',
  password: '',
  user: '',
  error: ''
}

  change = e => {
    this.setState({
      [e.target.name]: e.target.value
    });
  };


  onSubmit = (e) => {
        e.preventDefault();
        const { username, password, user } = this.state;
        var current = this;
        localStorage.clear();
        axios.post('http://localhost:8080/login', {  username, password, user })
          .then((result) => {
            //access the results here....
            console.log("result", result);
          localStorage.setItem('userId', result.data.userID);
          localStorage.setItem('username', username);
          localStorage.setItem('userType', user);
          localStorage.setItem('name', result.data.name);
          if(localStorage.getItem("userType") == "customer"){
              this.props.history.push('/customerMainMenu');
          }else{
              if(result.data.has_logged_in_before == false){
                  this.props.history.push('/resetClerkPassword');
              }else{
                  this.props.history.push('/clerkMainMenu');
              }
          }

          }).catch(function(reject){
            console.log("reject",reject.response.data.reason);
            const error = reject.response.data.reason;
            current.setState({error: error});
          });

      }

  render() {
      const {error}= this.state;
    return (
    <div>
      <div id="login-form">
        <h1>Login</h1>
      <form>
      <MuiThemeProvider>
      <div id="loginForm">
      <TextField
          name="username"
          hintText="User Name"
          floatingLabelText="User Name"
          floatingLabelFixed={true}
          value={this.state.username}
          onChange={e => this.change(e)}
        />
        <br/>
        <br/>
        <TextField
            type="password"
            name="password"
            hintText="Password"
            floatingLabelText="Password"
            floatingLabelFixed={true}
            value={this.state.password}
            onChange={e => this.change(e)}
          />
          <br/>
          <br/>
               <input type="radio"
               name="user"
               id="customerRadioButton"
               value="customer"
               checked={this.state.user === 'customer'}
               onChange={e => this.change(e)}/>
        <label htmlFor="customerRadioButton">Customer  </label>
        <input type="radio"
               name="user"
               id="clerkRadioButton"
               value="clerk"
               checked={this.state.user === 'clerk'}
               onChange={e => this.change(e)}/>
        <label htmlFor="clerkRadioButton">Clerk</label>
        <div>
        {error && <ErrorMsg msg={error} />}
        </div>
        <div>
        <RaisedButton label="Submit" primary={true} style={style} onClick= {(e) => this.onSubmit(e)} />
        </div>
        <div>
        <NavLink exact to="/signup" >
            <RaisedButton label="Don't have an account? Sign up!" primary={true} style={style} />
        </NavLink>
        </div>
        </div>
        </MuiThemeProvider>
      </form>
      </div>
      </div>
    )
  }
}
export default  withRouter(LoginForm);
