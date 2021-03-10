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

export default  class ToolDetails extends React.Component {

handleClose = () => {
  this.props.toolData.modalIsOpen = false;
};



render() {
  const isListNull = this.props.toolData.modalAccList == null;
  const isListEmpty = (this.props.toolData.modalAccList[0] == "");

  return (
    <div>

          <MuiThemeProvider>
            <Dialog
              title="Tool Details"
              actions={[<FlatButton
                          label="Close"
                          primary={true}
                          onClick={this.props.handler}
                        />]}
              modal={false}
              open={this.props.toolData.modalIsOpen}
              onRequestClose={this.props.handler}
              autoScrollBodyContent={true}
            >
              <h4> Tool ID:</h4>
              {this.props.toolData.modalToolID}
              <h4> Tool Type:</h4>
              {this.props.toolData.modalToolType}
              <h4> Short Description:</h4>
              {this.props.toolData.modalShortDesc}
              <h4> Full Description:</h4>
              {this.props.toolData.modalFullDesc}
              <h4> Deposit Price:</h4>
              ${parseFloat(this.props.toolData.modalDepPrice).toFixed(2)}
              <h4> Rental Price:</h4>
              ${parseFloat(this.props.toolData.modalRentPrice).toFixed(2)}
              {!isListEmpty && !isListNull && <div id='accList'>
                <h4> Accesories:</h4>
                <ol>
                    {this.props.toolData.modalAccList.map(function(acc, index){
                        return <li key={ index }>{acc["description"]}</li>;
                      })}
                </ol>
              </div>}
            </Dialog>
          </MuiThemeProvider>
    </div>
  )
}

}
