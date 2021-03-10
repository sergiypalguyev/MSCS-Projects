import React, { Component } from 'react';
import TestingFormLayout from './TestingFormLayout.js'
import Container from '@material-ui/core/Container';
const axios = require('axios');

class TestingForm extends Component {
  constructor(props) {
    super(props);

    this.state = {formVals : {sex: "",
                                 test1Assay : "",
                                 test1Result : "",
                                 test2Assay : "",
                                 test2Result : "",
                                 test3Assay : "",
                                 test3Result : "",
                                 resultReceivedByClinic : "",
                                 age : ""}}

    this.autofillAge = this.autofillAge.bind(this);
    this.changeFieldValue = this.changeFieldValue.bind(this);
    this.submitForm = this.submitForm.bind(this);


  }

  submitForm (){

    console.log(this.state.formVals)

    axios.post('/formSubmitUrl', {
        formVals: this.state.formVals,
    })
    .then(function (response) {
        console.log(response);
    })
    .catch(function (error) {
        console.log(error);
    });
  }


  componentDidMount(){

  }

  yearsBetweenDates(d1, d2){

    let date1 = new Date(d1);
    let date2 = new Date(d2);
    let yearsDiff =  date2.getFullYear() - date1.getFullYear();
    return yearsDiff;

  }

  changeFieldValue(e){

    if (e.target.id == "dateOfBirth"){
      this.autofillAge(e)
    }

    console.log(e.target.name, e.target.value)

    var newformVals = this.state.formVals
    newformVals[e.target.name] = e.target.value

    this.setState({formVals : newformVals})


  }

  autofillAge(e){

    var today = new Date();
    var todaydate = today.getFullYear()+'-'+(today.getMonth()+1)+'-'+today.getDate();
    var age = this.yearsBetweenDates(e.target.value, todaydate)

    var newformVals = this.state.formVals
    newformVals["age"] = age

    this.setState({formVals : newformVals})


  }

  render() {

    return (
      <Container maxWidth="sm">

           <TestingFormLayout submit = {this.submitForm}
           changeFieldValue = {this.changeFieldValue}
           formVals = {this.state.formVals}
           autofillAge = {this.autofillAge}/>

      </Container>




    );
  }
}

export default TestingForm;
