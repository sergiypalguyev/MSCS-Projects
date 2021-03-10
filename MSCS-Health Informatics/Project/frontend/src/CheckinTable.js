import React from 'react';
import { makeStyles } from '@material-ui/core/styles';
import Table from '@material-ui/core/Table';
import TableBody from '@material-ui/core/TableBody';
import TableCell from '@material-ui/core/TableCell';
import TableContainer from '@material-ui/core/TableContainer';
import TableHead from '@material-ui/core/TableHead';
import TableRow from '@material-ui/core/TableRow';
import Paper from '@material-ui/core/Paper';
import Typography from '@material-ui/core/Typography';
import Checkbox from '@material-ui/core/Checkbox';

const useStyles = makeStyles({
  table: {
    minWidth: 650,
  },
});

function createData(visited, id, fname, lname, address, age) {
  return {visited, id, fname, lname, address, age};
}

const rows = [
  createData(true, 115, "test", "guy", "123 w street", 23),
  createData(false, 13, "other", "guy", "123 s drive", 45),
  createData(false, 17, "that", "guy", "123 e lane", 32),
  createData(false, 144, "some", "dude", "123 w street", 63),

];

export default function CheckinTable() {
  const classes = useStyles();

  return (
    <React.Fragment>
      <TableContainer component={Paper}>
        <Table className={classes.table} aria-label="simple table">
          <TableHead>
            <TableRow>
              <TableCell><b>Visited</b></TableCell>
              <TableCell align="right" ><b>Patient ID</b></TableCell>
              <TableCell align="right"><b>First Name</b></TableCell>
              <TableCell align="right"><b>Last Name</b></TableCell>
              <TableCell align="right"><b>Address</b></TableCell>
              <TableCell align="right"><b>Age</b></TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {rows.map(row => (
              <TableRow key={row.id}>
                <TableCell component="th" scope="row">
                <Checkbox color="secondary" id={"visited" + row.id}
                name="visited" value={row.visited} />
                </TableCell>
                <TableCell align="right">{row.id}</TableCell>
                <TableCell align="right">{row.fname}</TableCell>
                <TableCell align="right">{row.lname}</TableCell>
                <TableCell align="right">{row.address}</TableCell>
                <TableCell align="right">{row.age}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
    </React.Fragment>
  );
}
