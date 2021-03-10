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
import Dialog from 'material-ui/Dialog';
import RaisedButton from 'material-ui/RaisedButton';
import FlatButton from 'material-ui/FlatButton';

export default  class ResDetails extends React.Component {

handleClose = () => {
  this.props.resData.modalIsOpen = false;
};

render() {
  const isListNull = this.props.resData.tools == null;
  const isListEmpty = (this.props.resData.tools != null && this.props.resData.tools[0] == "");


  return (
    <div>

          <MuiThemeProvider>
            <Dialog
              title=""
              actions={[<FlatButton
                          label="Close"
                          primary={true}
                          onClick={this.props.handler}
                        />]}
              modal={false}
              open={this.props.resData.modalIsOpen}
              onRequestClose={this.props.handler}
              autoScrollBodyContent={true}
            >
              <br></br><br></br><b>Reservation ID: #{this.props.resData.reservationID}</b>
              <br></br><br></br><b>Customer Name: </b>
              {this.props.resData.firstName} {this.props.resData.lastName}
              <br></br><br></br><b>Total Deposit: $</b>
              {parseFloat(this.props.resData.totalDepositPrice).toFixed(2)}
              <br></br><br></br><b>Total Rental Price: $</b>
              {parseFloat(this.props.resData.totalRentalPrice).toFixed(2)}
              {!isListEmpty && !isListNull && <div id='accList'>
              <h4> Tool Name</h4>
              <ul>
                  {this.props.resData.tools.map(function(tool, index){
                      return <li key={ index }>{tool["shortDesc"]}</li>;
                    })}
              </ul>
              </div>}
            </Dialog>
          </MuiThemeProvider>
    </div>
  )
}

}
