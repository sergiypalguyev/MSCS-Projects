import React from 'react';
import axios from 'axios';
import TextField from 'material-ui/TextField';
import MuiThemeProvider from 'material-ui/styles/MuiThemeProvider';
import RaisedButton from 'material-ui/RaisedButton';
import ErrorMsg from './messages/LoginAndRegistrationErrorMsg';
import {BrowserRouter, Route, NavLink, withRouter} from "react-router-dom";
const style = {
  margin: 12,
};
class ResetClerkPassword extends React.Component {

    constructor (props) {
        super(props);
        this.onSubmit = this.onSubmit.bind(this);
    }

    state = {
      pass1: '',
      pass2: '',
      error:''
  }


  changePass1 = e => {
    this.setState({
      pass1: e.target.value
    });
  };
  changePass2 = e => {
    this.setState({
      pass2: e.target.value
    });
  };

  onSubmit = (e) => {
        e.preventDefault();
        const { pass1, pass2} = this.state;
        var current = this;
        const newPassword = pass1;
        if(pass1!= pass2){
            this.setState({error:"Password mismatch: Please check to make sure the passwords are input correctly"});
            return;
        }else{
            this.setState({error: ""});
        }
        const userID = localStorage.getItem('userId');
        axios.post('http://localhost:8080/login/update/password', { userID,  newPassword})
          .then((result) => {
            this.props.history.push('/clerkMainMenu');
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
          <div id="reset-form">
          <h1>Reset Password</h1>
        <form>
        <MuiThemeProvider>
         <div id="loginForm">
        <br/>
        <br/>
        <TextField
            type="password"
            name="password"
            floatingLabelText="Password"
            floatingLabelFixed={true}
            value={this.state.password}
            onChange={e => this.changePass1(e)}
          />
          <br/>
          <br/>
          <TextField
              type="password"
              name="password2"
              floatingLabelText="Re-type Password"
              floatingLabelFixed={true}
              value={this.state.password}
              onChange={e => this.changePass2(e)}
            />
            <br/>
            <br/>
            <div>
            {error && <ErrorMsg msg={error} />}
            </div>
            <div>
            <RaisedButton label="Reset Password" primary={true} style={style} onClick= {(e) => this.onSubmit(e)} />
            </div>
            </div>
            </MuiThemeProvider>
          </form>

          </div>
          </div>
    )
}
}

export default  withRouter(ResetClerkPassword);
