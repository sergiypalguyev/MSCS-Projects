import React from "react";
import {
  TableRow,
  TableRowColumn,
} from 'material-ui/Table';

class Row extends React.Component {
    render() {
        return (<TableRow onClick={()=> {alert('Click event on row')}}><TableRowColumn>text here</TableRowColumn></TableRow>)
    }
}

export default Row;
