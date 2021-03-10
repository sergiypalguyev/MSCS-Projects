import React from 'react';
import Grid from '@material-ui/core/Grid';
import Typography from '@material-ui/core/Typography';
import TextField from '@material-ui/core/TextField';
import FormControlLabel from '@material-ui/core/FormControlLabel';
import Checkbox from '@material-ui/core/Checkbox';
import InputLabel from '@material-ui/core/InputLabel';
import MenuItem from '@material-ui/core/MenuItem';
import Select from '@material-ui/core/Select';
import Button from '@material-ui/core/Button';


export default function AddressForm(props) {
  return (
    <React.Fragment>
      <Grid container spacing={3}>
        <Grid item xs={12}>
          <TextField
            required
            id="idNumber"
            name="idNumber"
            label="ID Number"
            fullWidth
            autoComplete="id"
            onChange = {props.changeFieldValue}
            value = {props.formVals.id}

          />
          </Grid>
          <Grid item xs={12} sm={6}>
            <TextField
              required
              id="firstName"
              name="firstName"
              label="First name"
              fullWidth
              autoComplete="fname"
              onChange = {props.changeFieldValue}
              value = {props.formVals.fname}
            />
          </Grid>
          <Grid item xs={12} sm={6}>
            <TextField
              required
              id="lastName"
              name="lastName"
              label="Last name"
              fullWidth
              autoComplete="lname"
              onChange = {props.changeFieldValue}
              value = {props.formVals.lname}
            />
          </Grid>
          <Grid item xs={12} sm={6}>
            <FormControlLabel
              control={<Checkbox color="secondary" id="historicallyTested"
              name="historicallyTested" name="saveAddress" value="yes" />}
              label="Tested for HIV in the past?"
              onChange = {props.changeFieldValue}
              value = {props.formVals.historicallyTested}
            />
          </Grid>
          <Grid item xs={12} sm={6}>
            <TextField
              id="histTestDate"
              name="histTestDate"
              label="If yes, past HIV Test Date."
              type="date"
              InputLabelProps={{shrink: true,}}
              fullWidth
              autoComplete=""
              onChange = {props.changeFieldValue}
              value = {props.formVals.histTestDate}
            />
          </Grid>
          <Grid item xs={12} sm={4}>
            <TextField
              required
              id="dateOfBirth"
              name="dateOfBirth"
              label="Date of Birth"
              type="date"
              InputLabelProps={{shrink: true,}}
              fullWidth
              onChange = {props.autofillAge}
              autoComplete=""
              onChange = {props.changeFieldValue}
              value = {props.formVals.dateOfBirth}
            />
          </Grid>
          <Grid item xs={12} sm={4}>
            <TextField
              required
              id="age"
              name="age"
              label="Age"
              fullWidth
              type="number"
              autoComplete=""
              onChange = {props.changeFieldValue}
              value = {props.formVals.age}
            />
          </Grid>
          <Grid item xs={12} sm={4}>
            <InputLabel id="sex-label">Sex</InputLabel>
            <Select
              labelId="sex-label"
              id="sex"
              name="sex"
              onChange = {props.changeFieldValue}
              value = {props.formVals.sex}
              >
              <MenuItem value="">
              <em></em>
              </MenuItem>
              <MenuItem value="Male">Male</MenuItem>
              <MenuItem value="Female">Female</MenuItem>
              </Select>
          </Grid>
          <Grid item xs={12} sm={6}>
            <FormControlLabel
              control={<Checkbox color="secondary" id="acceptedHIVTest"
              name="acceptedHIVTest" name="saveAddress" value="yes" />}
              label="Accepted HIV Test?"
              onChange = {props.changeFieldValue}
              value = {props.formVals.acceptedHIVTest}
            />
          </Grid>
          <Grid item xs={12} sm={6}>
            <TextField
              required
              id="testDate"
              name="testDate"
              label="Test Date"
              type="date"
              InputLabelProps={{shrink: true,}}
              fullWidth
              autoComplete=""
              onChange = {props.changeFieldValue}
              value = {props.formVals.testDate}
            />
          </Grid>
          <Grid item xs={12} sm={6}>
            <InputLabel id="test1Assay-label">Test 1 Assay</InputLabel>
            <Select
              labelId="test1Assay-label"
              id="test1Assay"
              name="test1Assay"
              onChange = {props.changeFieldValue}
              value = {props.formVals.test1Assay}
              >
              <MenuItem value="">
              <em></em>
              </MenuItem>
              <MenuItem value="Determine">Determine</MenuItem>
              <MenuItem value="Uni-Gold">Uni-Gold</MenuItem>
              <MenuItem value="SD Bioline">SD Bioline</MenuItem>
              </Select>
          </Grid>
          <Grid item xs={12} sm={6}>
          <InputLabel id="test1Result-label">Test 1 Result</InputLabel>
          <Select
            labelId="test1Result-label"
            id="test1Result"
            name="test1Result"
            onChange = {props.changeFieldValue}
            value = {props.formVals.test1Result}
            >
            <MenuItem value="">
            <em></em>
            </MenuItem>
            <MenuItem value="Negative">Negative</MenuItem>
            <MenuItem value="Positive">Positive</MenuItem>
            <MenuItem value="Indeterminate">Indeterminate</MenuItem>
            </Select>
          </Grid>
          <Grid item xs={12} sm={6}>
            <InputLabel id="test2Assay-label">Test 2 Assay</InputLabel>
            <Select
              labelId="test2Assay-label"
              id="test2Assay"
              name="test2Assay"
              onChange = {props.changeFieldValue}
              value = {props.formVals.test2Assay}
              >
              <MenuItem value="">
              <em></em>
              </MenuItem>
              <MenuItem value="Determine">Determine</MenuItem>
              <MenuItem value="Uni-Gold">Uni-Gold</MenuItem>
              <MenuItem value="SD Bioline">SD Bioline</MenuItem>
              </Select>
          </Grid>
          <Grid item xs={12} sm={6}>
          <InputLabel id="test2Result-label">Test 2 Result</InputLabel>
          <Select
            labelId="test2Result-label"
            id="test2Result"
            name="test2Result"
            onChange = {props.changeFieldValue}
            value = {props.formVals.test2Result}
            >
            <MenuItem value="">
            <em></em>
            </MenuItem>
            <MenuItem value="Negative">Negative</MenuItem>
            <MenuItem value="Positive">Positive</MenuItem>
            <MenuItem value="Indeterminate">Indeterminate</MenuItem>
            </Select>
          </Grid>
          <Grid item xs={12} sm={6}>
            <InputLabel id="test3Assay-label">Test 3 Assay</InputLabel>
            <Select
              labelId="test3Assay-label"
              id="test3Assay"
              name="test3Assay"
              onChange = {props.changeFieldValue}
              value = {props.formVals.test3Assay}
              >
              <MenuItem value="">
              <em></em>
              </MenuItem>
              <MenuItem value="Determine">Determine</MenuItem>
              <MenuItem value="Uni-Gold">Uni-Gold</MenuItem>
              <MenuItem value="SD Bioline">SD Bioline</MenuItem>
              </Select>
          </Grid>
          <Grid item xs={12} sm={6}>

            <InputLabel id="test3Result-label">Test 3 Result</InputLabel>
            <Select
              labelId="test3Result-label"
              id="test3Result"
              name="test3Result"
              onChange = {props.changeFieldValue}
              value = {props.formVals.test3Result}
              >
              <MenuItem value="">
              <em></em>
              </MenuItem>
              <MenuItem value="Negative">Negative</MenuItem>
              <MenuItem value="Positive">Positive</MenuItem>
              <MenuItem value="Indeterminate">Indeterminate</MenuItem>
              </Select>
          </Grid>
          <Grid item xs={12} sm={6}>
            <InputLabel id="sex-label">Result Recieved By Clinic.</InputLabel>
            <Select
              labelId="resultReceivedByClinic-label"
              id="resultReceivedByClinic"
              name="resultReceivedByClinic"
              onChange = {props.changeFieldValue}
              value = {props.formVals.resultReceivedByClinic}
              >
              <MenuItem value="">
              <em></em>
              </MenuItem>
              <MenuItem value="Negative">Negative</MenuItem>
              <MenuItem value="Positive">Positive</MenuItem>
              <MenuItem value="Indeterminate">Indeterminate</MenuItem>
              </Select>
          </Grid>
          <Grid item xs={12} sm={6}>
            <TextField
              required
              id="appointmentDate"
              name="appointmentDate"
              label="Appointment Date"
              type="date"
              InputLabelProps={{shrink: true,}}
              fullWidth
              autoComplete=""
              onChange = {props.changeFieldValue}
              value = {props.formVals.appointmentDate}
            />
          </Grid>
          <Grid item xs={12}>
            <TextField
              required
              id="appointmentLocation"
              name="appointmentLocation"
              label="Appointment Location"
              fullWidth
              autoComplete=""
              onChange = {props.changeFieldValue}
              value = {props.formVals.appointmentLocation}
            />
          </Grid>
            <Grid item xs={12}>
            <Button onClick = {props.submit} variant="contained" color="primary">
              Submit
            </Button>
            </Grid>
       </Grid>

    </React.Fragment>
  );
}
