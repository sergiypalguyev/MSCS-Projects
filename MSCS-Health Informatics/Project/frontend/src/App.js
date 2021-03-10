import React, { Component } from 'react';
import logo from './logo.svg';
import './App.css';
import MainPage from './MainPage.js'
import CheckIns from './CheckIns.js'
import TestingForm from './TestingForm.js'
import TitleBar from './TitleBar.js'
const PAGEOPTIONS = ["CHECKIN", "MAIN", "TESTINGFORM"]

class App extends Component {
  constructor(props) {
    super(props);

    this.getUser = this.getUser.bind(this);
    this.changePage = this.changePage.bind(this);


    this.state = {page : "MAIN"}
  }




  getUser(){
    //axioscallgoeshere

    this.setState({user : "testguy"})
  }

  changePage(page) {

    this.setState({page : page})

 }


  componentDidMount(){
    this.getUser()

  }

  render() {

    return (
      <div className="App">
        <TitleBar {...this.state} changePage = {this.changePage}/>
        {this.state.page == "MAIN" && <MainPage {...this.state} changePage = {this.changePage}/>}
        {this.state.page == "CHECKIN" && <CheckIns {...this.state} changePage = {this.changePage}/>}
        {this.state.page == "TESTINGFORM" && <TestingForm {...this.state} changePage = {this.changePage}/> }
      </div>
    );
  }
}

export default App;
