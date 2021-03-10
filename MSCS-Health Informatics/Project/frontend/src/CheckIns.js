import React, { Component } from 'react';
import Container from '@material-ui/core/Container';
import CheckinTable from './CheckinTable.js'


class CheckIns extends Component {
  constructor(props) {
    super(props);



    this.state = {}
  }


  componentDidMount(){

  }

  render() {

    return (
      <Container maxWidth="m">

           <CheckinTable/>

      </Container>

    );
  }
}

export default CheckIns;
