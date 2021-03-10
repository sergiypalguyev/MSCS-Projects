import axios from 'axios';
import React, {Component} from 'react';
import {RadioButton, RadioButtonGroup} from 'material-ui/RadioButton';
import DropDownMenu from 'material-ui/DropDownMenu';
import SelectField from 'material-ui/SelectField';
import MenuItem from 'material-ui/MenuItem';
import TextField from 'material-ui/TextField';
import RaisedButton from 'material-ui/RaisedButton';
import {orange500, blue500} from 'material-ui/styles/colors';
import {
  Table,
  TableBody,
  TableFooter,
  TableHeader,
  TableHeaderColumn,
  TableRow,
  TableRowColumn,
} from 'material-ui/Table';

import MuiThemeProvider from 'material-ui/styles/MuiThemeProvider'
import ErrorMsg from './messages/LoginAndRegistrationErrorMsg';
import SuccessfulRegistrationMsg from './messages/SuccessfulRegistrationMsg.js';
import {Link} from "react-router";
import NavBar from "./nav/NavBar";

const styles = {
  block: {
    maxWidth: 250,
  },
  radioButton: {
    marginBottom: 16,
    textAlign: 'left'
  },
  radioButtonGroup: {
  	display: 'flex',
  	width: 'auto'
  },
  errorStyle: {
    color: orange500,
  },
  underlineStyle: {
    borderColor: orange500,
  },
  floatingLabelStyle: {
    color: orange500,
  },
  floatingLabelFocusStyle: {
    color: blue500,
  },
  textFieldStyle: {
  	margin: 15
  },
  fractionStyle: {
  	width: 150,
  	margin: 20,
  	top: 20
  },
  accessoriesStyle: {
  	width: 250,
  	margin: 20,
  	top: 20,
  },
  errorStyle: {
    color: orange500,
    textAlign: 'left'
  },
};

const fractions = [
	{
		fraction: "0",
		value: 0,
	},
	{	
		fraction: "1/8",
		value: 0.125,
	},
	{	
		fraction: "1/4",
		value: 0.25,
	},
	{	
		fraction: "3/8",
		value: 0.375,
	},
	{	
		fraction: "1/2",
		value: 0.5,
	},
	{	
		fraction: "5/8",
		value: 0.625,
	},
	{	
		fraction: "3/4",
		value: 0.75,
	},
	{	
		fraction: "7/8",
		value: 0.875,
	},
]

const units = [
	"inches",
	"feet"
]

const booleans = [
	"false",
	"true"
]

const volts = [
	{
		label: "110V",
		value: 110,
	},
	{
		label: "120V",
		value: 120,
	},
	{
		label: "220V",
		value: 220,
	},
	{
		label: "240V",
		value: 240,
	},
]

const gaugeRatings = [
	{
		label: "18G",
		value: 18,
	},
	{
		label: "20G",
		value: 20,
	},
	{
		label: "22G",
		value: 22,
	},
	{
		label: "24G",
		value: 24,
	}
]

const accessories = [
	"Drill Bits",
	"Saw Blade",
	"Soft Case",
	"Hard Case",
	"Batteries",
	"Battery Charger",
	"Safety",
	"Hose",
	"Gas Tank"
]

const batteryTypes = [
	"Li-Ion",
	"NiCd",
	"NiMH"
]

const safetyAccessories = [
	"hat",
	"pants",
	"googles",
	"vest"
]

export default class AddNewTool extends React.Component {

	constructor(props) {
		super(props);

		this.setManufacturer = this.setManufacturer.bind(this);
		this.setMaterial = this.setMaterial.bind(this);
		this.setPurchasePrice = this.setPurchasePrice.bind(this);
		this.setWeight = this.setWeight.bind(this);
		this.setWidth = this.setWidth.bind(this);
		this.setLength = this.setLength.bind(this);

		this.setScrewSize = this.setScrewSize.bind(this);
		this.setDriveSize = this.setDriveSize.bind(this);
		this.setSaeSize = this.setSaeSize.bind(this);
		this.setAdjustable = this.setAdjustable.bind(this);
		this.setGaugeRating = this.setGaugeRating.bind(this);
		this.setCapacity = this.setCapacity.bind(this);
		this.setAntiVibration = this.setAntiVibration.bind(this);

		this.setStepCount = this.setStepCount.bind(this);
		this.setWeightCapacity = this.setWeightCapacity.bind(this);
		this.setRubberFeet = this.setRubberFeet.bind(this);
		this.setPailShelf = this.setPailShelf.bind(this);

		this.setHandleMaterial = this.setHandleMaterial.bind(this);
		this.setBladeMaterial = this.setBladeMaterial.bind(this);
		this.setBladeLength = this.setBladeLength.bind(this);
		this.setBladeLengthFraction = this.setBladeLengthFraction.bind(this);
		this.setBladeWidth = this.setBladeWidth.bind(this);
		this.setBladeWidthFraction = this.setBladeWidthFraction.bind(this);
		this.setHeadWeight = this.setHeadWeight.bind(this);
		this.setTineCount = this.setTineCount.bind(this);
		this.setBinMaterial = this.setBinMaterial.bind(this);
		this.setBinVolume = this.setBinVolume.bind(this);
		this.setWheelCount = this.setWheelCount.bind(this);

		this.setVoltRating = this.setVoltRating.bind(this);
		this.setAmpRating = this.setAmpRating.bind(this);
		this.setMinRpmRating = this.setMinRpmRating.bind(this);
		this.setMaxRpmRating = this.setMaxRpmRating.bind(this);
		this.setAdjustableClutch = this.setAdjustableClutch.bind(this);
		this.setMinTorqueRating = this.setMinTorqueRating.bind(this);
		this.setMaxTorqueRating = this.setMaxTorqueRating.bind(this);
		this.setBladeSize = this.setBladeSize.bind(this);
		this.setBladeSizeFraction = this.setBladeSizeFraction.bind(this);
		this.setDustBag = this.setDustBag.bind(this);
		this.setTankSize = this.setTankSize.bind(this);
		this.setPressureRating = this.setPressureRating.bind(this);
		this.setMotorRating = this.setMotorRating.bind(this);
		this.setDrumSize = this.setDrumSize.bind(this);
		this.setPowerRating = this.setPowerRating.bind(this);

		this.setAccessoryQuantity = this.setAccessoryQuantity.bind(this);
		this.setAccessoryValue = this.setAccessoryValue.bind(this);
		this.setSafetyAccessory = this.setSafetyAccessory.bind(this);
		this.setBatteryType = this.setBatteryType.bind(this);
}

	state = { 
		powerSources: [],
		subTypes: [],
		subOptions: [],
		powerSourceValue: "",
		subTypeValue: "",
		subOptionValue: "",
		widthValue: 0,
		widthFractionValue: 0,
		widthUnitValue: units[0],
		lengthValue: 0,
		lengthFractionValue: 0,
		lengthUnitValue: units[0],	

		driveSizeValue: null,
		saeSizeValue: null,
		adjustableValue: null,
		antiVibrationValue: null,
		rubberFeetValue: null,
		pailShelfValue: null,
		gaugeRatingValue: null,

		bladeLengthValue: null,
		bladeLengthFractionValue: null,
		bladeWidthValue: null,
		bladeWidthFractionValue: null,
		bladeSizeValue: null,
		bladeSizeFractionValue: null,


		voltRatingValue: null,
		adjustableClutchValue: null,
		dustBagValue: null,
		motorRatingValue: null,

		accessoryQuantity: 0,
		accessoryValue: accessories[0],
		safetyAccessoryValue: safetyAccessories[0],
		batteryTypeValue: batteryTypes[0],

		displayNotification: false,
		responseMessage: "",
		isError: false,

		selectable: false,
		displaySelectAll: false,
		fixedHeader: false,
		adjustForCheckbox: false,
	}

	tool = {
		type: null,
		subType: null,
		subOption: null,
		powerSource: null,
		purchasePrice: null,
		manufacturer: null,
		material: null,
		weight: null,
		width: null,
		length: null,

		screwSize: null,
		driveSize: null,
		saeSize: null,
		adjustable: null,
		gaugeRating: null,
		capacity: null,
		antiVibration: null,

		stepCount: null,
		weightCapacity: null,
		rubberFeet: null,
		pailShelf: null,

		handleMaterial: null,
		bladeMaterial: null,
		bladeLength: null,
		bladeWidth: null,
		headWeight: null,
		tineCount: null,
		binMaterial: null,
		binVolume: null,
		wheelCount: null,

		voltRating: null,
		ampRating: null,
		minRpmRating: null,
		maxRpmRating: null,
		adjustableClutch: null,
		minTorqueRating: null,
		maxTorqueRating: null,
		bladeSize: null,
		dustBag: null,
		tankSize: null,
		pressureRating: null,
		motorRating: null,
		drumSize: null,
		powerRating: null,

		accList: [],
	}

	response = {
		errorCode: "",
		errorMessage: ""
	}

	componentDidMount() {
		this.hideAllSpecificToolAttributes();
	}

	fetchSubTypes(event, index, value) {
		this.hideAllSpecificToolAttributes();
	 	this.tool.powerSource = value;

		var getURL = "http://localhost:8080/tool/availability?type=" + this.tool.type.replace(" ", "%20") + "&powerSource=" + this.tool.powerSource;
		console.log(getURL);

	  	axios.get(getURL)
	      .then(res => {
	        if ('subTypes' in res['data'])
	          this.setState({
	          	powerSourceValue: value,
	          	subTypes: res['data']['subTypes'],
	          	subTypeValue: res['data']['subTypes'][0],
	          	subOptions: []
	          });
	      		this.fetchSubOptions(event, index, res['data']['subTypes'][0]);
	      })
	      .catch(function (error) {
	        console.log(error);
	      });
	}

	fetchSubOptions(event, index, value) {
	 	this.tool.subType = value;
	 	this.displayAppropiateToolAttributes();

		var getURL = "http://localhost:8080/tool/availability?type=" + this.tool.type.replace(" ", "%20") + 
					 "&powerSource=" + this.tool.powerSource +
					 "&subType=" + this.tool.subType;
		console.log(getURL);

	  	axios.get(getURL)
	      .then(res => {
	        if ('subOptions' in res['data'])
	          this.setState({
	          	subTypeValue: value,
	          	subOptions: res['data']['subOptions'],
	          	subOptionValue: res['data']['subOptions'][0]
	          });
	      	this.setSubOption(event, index, res['data']['subOptions'][0]);
	      })
	      .catch(function (error) {
	        console.log(error);
	      });
	}

	hideAllSpecificToolAttributes() {
		this.hideHandToolAttributes();
		this.hideLadderToolAttributes();
		this.hideGardenToolAttributes();
		this.hidePowerToolAttributes();
	}

	displayAppropiateToolAttributes() {
		switch (this.tool.subType) {
			case "Screwdriver":
				this.displayHTScrewdriver();
				break;

			case "Socket":
				this.displayHTSocket();
				break;

			case "Ratchet":
				this.displayHTRatchet();
				break;

			case "Wrench":
				this.displayHTWrench();
				break;

			case "Pliers":
				this.displayHTPliers();
				break;

			case "Gun":
				this.displayHTGun();
				break;

			case "Hammer":
				this.displayHTHammer();
				break;

			case "Straight":
				this.displayLTStraight();
				break;

			case "Step":
				this.displayLTStep();
				break;

			case "Pruner":
				this.displayGTPruning();
				break;

			case "Striking":
				this.displayGTStriking();
				break;

			case "Digger":
				this.displayGTDigging();
				break;

			case "Rakes":
				this.displayGTRake();
				break;

			case "Wheelbarrows":
				this.displayGTWheelbarrow();
				break;

			case "Drill":
				this.displayPTDrill();
				break;

			case "Saw":
				this.displayPTSaw();
				break;

			case "Sander":
				this.displayPTSander();
				break;

			case "Air-Compressor":
				this.displayPTAirCompressor();
				break;

			case "Mixer":
				this.displayPTMixer();
				break;

			case "Generator":
				this.displayPTGenerator();
				break;

			default:
				this.hideAllSpecificToolAttributes();
				break;
		}
	}

	setSubOption(event, index, value) {
		this.tool.subOption = value;
		this.setState({subOptionValue: value});
	}

	setWidthFraction(event, index, value) {
		this.setState({widthFractionValue: value});
	}

	setWidthUnit(event, index, value) {
		this.setState({widthUnitValue: value});
	}

	setLengthFraction(event, index, value) {
		this.setState({lengthFractionValue: value});
	}

	setLengthUnit(event, index, value) {
		this.setState({lengthUnitValue: value});
	}

	setManufacturer(event) {
		this.tool.manufacturer = event.target.value;
	}

	setMaterial(event) {
		this.tool.material = event.target.value;
	}

	setPurchasePrice(event) {
		this.tool.purchasePrice = event.target.value;
	}

	setWeight(event) {
		this.tool.weight = event.target.value;
	}

	setWidth(event) {
		this.setState({widthValue: event.target.value});
	}

	setLength(event) {
		this.setState({lengthValue: event.target.value});
	}

	setScrewSize(event) {
		this.tool.screwSize = event.target.value;
	}

	setAdjustable(event, index, value) {
		this.tool.adjustable = value;
		this.setState({adjustableValue: value});
	}

	setAntiVibration(event, index, value) {
		this.tool.antiVibration = value;
		this.setState({antiVibrationValue: value});
	}

	setDriveSize(event, index, value) {
		this.tool.driveSize = value;
		this.setState({driveSizeValue: value});
	}

	setSaeSize(event, index, value) {
		this.tool.saeSize = value;
		this.setState({saeSizeValue: value});
	}

	setGaugeRating(event, index, value) {
		this.tool.gaugeRating = value;
		this.setState({gaugeRatingValue: value});
	}

	setCapacity(event) {
		this.tool.capacity = event.target.value;
	}

	setWeightCapacity(event) {
		this.tool.weightCapacity = event.target.value;
	}

	setStepCount(event) {
		this.tool.stepCount = event.target.value;
	}

	setRubberFeet(event, index, value) {
		this.tool.rubberFeet = value;
		this.setState({rubberFeetValue: value});
	}

	setPailShelf(event, index, value) {
		this.tool.pailShelf = value;
		this.setState({pailShelfValue: value})
	}

	setHandleMaterial(event) {
		this.tool.handleMaterial = event.target.value;
	}

	setBladeMaterial(event) {
		this.tool.bladeMaterial = event.target.value;
	}

	setBladeLength(event) {
		this.setState({bladeLengthValue: event.target.value});
	}

	setBladeLengthFraction(event, index, value) {
		this.setState({bladeLengthFractionValue: value});
	}

	setBladeWidth(event) {
		this.setState({bladeWidthValue: event.target.value});
	}

	setBladeWidthFraction(event, index, value) {
		this.setState({bladeWidthFractionValue: value});
	}

	setHeadWeight(event) {
		this.tool.headWeight = event.target.value;
	}

	setTineCount(event) {
		this.tool.tineCount = event.target.value;
	}

	setBinMaterial(event) {
		this.tool.binMaterial = event.target.value;
	}

	setBinVolume(event) {
		this.tool.binVolume = event.target.value;
	}

	setWheelCount(event) {
		this.tool.wheelCount = event.target.value;
	}

	setVoltRating(event, index, value) {
		this.tool.voltRating = value;
		this.setState({voltRatingValue: value});
	}

	setAmpRating(event) {
		this.tool.ampRating = event.target.value;
	}

	setMinRpmRating(event) {
		this.tool.minRpmRating = event.target.value;
	}

	setMaxRpmRating(event) {
		this.tool.maxRpmRating = event.target.value;
	}

	setAdjustableClutch(event, index, value) {
		this.tool.adjustableClutch = value;
		this.setState({adjustableClutchValue: value});
	}

	setMinTorqueRating(event) {
		this.tool.minTorqueRating = event.target.value;
	}

	setMaxTorqueRating(event) {
		this.tool.maxTorqueRating = event.target.value;
	}

	setBladeSize(event) {
		this.setState({bladeSizeValue: event.target.value});
	}

	setBladeSizeFraction(event, index, value) {
		this.setState({bladeSizeFractionValue: value});
	}

	setDustBag(event, index, value) {
		this.tool.dustBag = value;
		this.setState({dustBagValue: value});
	}

	setTankSize(event) {
		this.tool.tankSize = event.target.value;
	}

	setPressureRating(event) {
		this.tool.pressureRating = event.target.value;
	}

	setMotorRating(event, index, value) {
		this.tool.motorRating = value;
		this.setState({motorRatingValue: value});
	}

	setDrumSize(event) {
		this.tool.drumSize = event.target.value;
	}

	setPowerRating(event) {
		this.tool.powerRating = event.target.value;
	}

	setAccessoryQuantity(event) {
		this.setState({accessoryQuantity: event.target.value});
	}

	setAccessoryValue(event, index, value) {

		document.getElementById("batteryTypes").style.display="none";
		document.getElementById("safetyAccessories").style.display="none";

		if (value == "Batteries") {
			document.getElementById("batteryTypes").style.display="inline-block";	
		}
		else if (value == "Safety") {
			document.getElementById("safetyAccessories").style.display="inline-block";
		}

		this.setState({accessoryValue: value});
	}

	setSafetyAccessory(event, index, value) {
		this.setState({safetyAccessoryValue: value});
	}

	setBatteryType(event, index, value) {
		this.setState({batteryTypeValue: value});
	}

	addAccessory() {
		var accQuantity = this.state.accessoryQuantity;
		var accDescription;

		if (this.state.accessoryValue == "Batteries") {
			accDescription = this.state.batteryTypeValue + " " + this.state.accessoryValue;
		}
		else if (this.state.accessoryValue == "Safety") {
			accDescription = this.state.accessoryValue + " " + this.state.safetyAccessoryValue;
		}
		else {
			accDescription = this.state.accessoryValue;
		}

		this.tool.accList.push({
			quantity: accQuantity,
			description: accDescription,
		})

		console.log(this.tool.accList);
		this.forceUpdate();
	}

	addNewTool() {
		console.log("Adding new tool!!!");
		console.log(this.tool);

		this.tool.width = Number(this.state.widthValue) + Number(this.state.widthFractionValue);
		if (this.state.widthUnitValue == "feet") {
			this.tool.width = this.tool.width * 12;
		}

		this.tool.length = Number(this.state.lengthValue) + Number(this.state.lengthFractionValue);
		if (this.state.lengthUnitValue == "feet") {
			this.tool.length = this.tool.length * 12;
		}

		this.tool.bladeLength = Number(this.state.bladeLengthValue) + Number(this.state.bladeLengthFractionValue);
		this.tool.bladeWidth = Number(this.state.bladeWidthValue) + Number(this.state.bladeWidthFractionValue);
		this.tool.bladeSize = Number(this.state.bladeSizeValue) + Number(this.state.bladeSizeFractionValue);


		if (this.tool.width == 0) {
			this.tool.width = null;
		}
		if (this.tool.length == 0) {
			this.tool.length = null;
		}
		if (this.tool.bladeLength == 0) {
			this.tool.bladeLength = null;
		}
		if (this.tool.bladeWidth == 0) {
			this.tool.bladeWidth = null;
		}
		if (this.tool.bladeSize == 0) {
			this.tool.bladeSize = null;
		}

		console.log(JSON.stringify(this.tool));
		let cur = this;

		axios.post('http://localhost:8080/tool/add', JSON.stringify(this.tool))
          .then(res => {
              this.response = res['data'];
              this.setState({
              	displayNotification: true,
              	responseMessage: "Tool added successfully!"
              });
              console.log(this.response);

              setTimeout(function(){
              		cur.setState({
              			displayNotification: false, 
              			isError: false
              		});
          		}, 3000);

            }).catch(reject => {
	            console.log(reject);
	            const error = reject.response.data.errorMessage;
	            if(error){
	                this.setState({
                		displayNotification: true,
	                	responseMessage: error,
	                	isError: true
	                });
	            }else{
	                this.setState({
						displayNotification: true,
	                	responseMessage: "Something went wrong. Check connection to database!",
	                	isError: true
	                });
	            }

	            setTimeout(function(){
              		cur.setState({
              			displayNotification: false, 
              			isError: false
              		});
          		}, 3000);
          });
	}

	displayHTScrewdriver() {
		this.displayHandToolAttributes();
		document.getElementById("htScrewdriver").style.display="inline";
	}

	displayHTSocket() {
		this.displayHandToolAttributes();
		document.getElementById("htSocket").style.display="inline";	
	}

	displayHTRatchet() {
		this.displayHandToolAttributes();
		document.getElementById("htRatchet").style.display="inline";	
	}

	displayHTWrench() {
		this.displayHandToolAttributes();
		document.getElementById("htWrench").style.display="inline";	
	}

	displayHTPliers() {
		this.displayHandToolAttributes();
		document.getElementById("htPliers").style.display="inline";	
	}

	displayHTGun() {
		this.displayHandToolAttributes();
		document.getElementById("htGun").style.display="inline";	
	}

	displayHTHammer() {
		this.displayHandToolAttributes();
		document.getElementById("htHammer").style.display="inline";	
	}

	displayHandToolAttributes() {
		document.getElementById("HandTools").style.display="inline";
		document.getElementById("htScrewdriver").style.display="none";
		document.getElementById("htSocket").style.display="none";
		document.getElementById("htRatchet").style.display="none";
		document.getElementById("htWrench").style.display="none";
		document.getElementById("htPliers").style.display="none";
		document.getElementById("htGun").style.display="none";
		document.getElementById("htHammer").style.display="none";
	}

	hideHandToolAttributes() {
		document.getElementById("HandTools").style.display="none";
	}

	displayLadderToolsAttributes() {
		document.getElementById("LadderTools").style.display="inline";
		document.getElementById("ltStraight").style.display="none";
		document.getElementById("ltStep").style.display="none";
	}

	hideLadderToolAttributes() {
		document.getElementById("LadderTools").style.display="none";
	}

	displayLTStraight() {
		this.displayLadderToolsAttributes();
		document.getElementById("ltStraight").style.display="inline";
	}

	displayLTStep() {
		this.displayLadderToolsAttributes();
		document.getElementById("ltStep").style.display="inline";
	}

	displayGardenToolsAttributes() {
		document.getElementById("GardenTools").style.display="inline";
		document.getElementById("gtPruning").style.display="none";
		document.getElementById("gtStriking").style.display="none";
		document.getElementById("gtDigging").style.display="none";
		document.getElementById("gtRake").style.display="none";
		document.getElementById("gtWheelbarrow").style.display="none";
	}

	hideGardenToolAttributes() {
		document.getElementById("GardenTools").style.display="none";
	}

	displayGTPruning() {
		this.displayGardenToolsAttributes();
		document.getElementById("gtPruning").style.display="inline";
	}

	displayGTStriking() {
		this.displayGardenToolsAttributes();
		document.getElementById("gtStriking").style.display="inline";
	}
	
	displayGTDigging() {
		this.displayGardenToolsAttributes();
		document.getElementById("gtDigging").style.display="inline";
	}

	displayGTRake() {
		this.displayGardenToolsAttributes();
		document.getElementById("gtRake").style.display="inline";
	}

	displayGTWheelbarrow() {
		this.displayGardenToolsAttributes();
		document.getElementById("gtWheelbarrow").style.display="inline";
	}

	displayPowerToolsAttributes() {
		document.getElementById("PowerTools").style.display="inline";
		document.getElementById("ptDrill").style.display="none";
		document.getElementById("ptSaw").style.display="none";
		document.getElementById("ptSander").style.display="none";
		document.getElementById("ptAirCompressor").style.display="none";
		document.getElementById("ptMixer").style.display="none";
		document.getElementById("ptGenerator").style.display="none";
		document.getElementById("batteryTypes").style.display="none";
		document.getElementById("safetyAccessories").style.display="none";
	}

	hidePowerToolAttributes() {
		document.getElementById("PowerTools").style.display="none";
	}

	displayPTDrill() {
		this.displayPowerToolsAttributes();
		document.getElementById("ptDrill").style.display="inline";
	}

	displayPTSaw() {
		this.displayPowerToolsAttributes();
		document.getElementById("ptSaw").style.display="inline";
	}

	displayPTSander() {
		this.displayPowerToolsAttributes();
		document.getElementById("ptSander").style.display="inline";
	}

	displayPTAirCompressor() {
		this.displayPowerToolsAttributes();
		document.getElementById("ptAirCompressor").style.display="inline";
	}

	displayPTMixer() {
		this.displayPowerToolsAttributes();
		document.getElementById("ptMixer").style.display="inline";
	}

	displayPTGenerator() {
		this.displayPowerToolsAttributes();
		document.getElementById("ptGenerator").style.display="inline";
	}

	changeType = function(e, value){
		this.hideAllSpecificToolAttributes();
	 	this.tool.type = value
	 	console.log(this.tool.type);

		var getURL = "http://localhost:8080/tool/availability?type=" + value.replace(" ", "%20");
	 	console.log(getURL);

	 	axios.get(getURL)
	      .then(res => {
	        if ('powerSources' in res['data'])
	          this.setState({
	          	powerSources: res['data']['powerSources'],
	          	powerSourceValue: res['data']['powerSources'][0],
	          	subTypes: [],
	          	subOptions: []
	          });
	      		this.fetchSubTypes(e, null, res['data']['powerSources'][0]);
	      })
	      .catch(function (error) {
	        console.log(error);
	      });
	 }

	handleWidthFractionChange = (event, index, value) => this.setWidthFraction(event, index, value);
	handleWidthUnitChange = (event, index, value) => this.setWidthUnit(event, index, value);
	handleLengthFractionChange = (event, index, value) => this.setLengthFraction(event, index, value);
	handleLengthUnitChange = (event, index, value) => this.setLengthUnit(event, index, value);
	handlePowerSourceChange = (event, index, value) => this.fetchSubTypes(event, index, value);
	handleSubTypeChange = (event, index, value) => this.fetchSubOptions(event, index, value);
	handleSubOptionChange = (event, index, value) => this.setSubOption(event, index, value);

	handleDriveSizeChange = (event, index, value) => this.setDriveSize(event, index, value);
	handleSaeSizeChange = (event, index, value) => this.setSaeSize(event, index, value);

	handleBladeLengthFractionChange = (event, index, value) => this.setBladeLengthFraction(event, index, value);
	handleBladeWidthFractionChange = (event, index, value) => this.setBladeWidthFraction(event, index, value);

	render() {
		const {displayNotification} = this.state;
		const {responseMessage} = this.state;
		const {isError} = this.state;

		return (
			<div>
				<NavBar/>
				<MuiThemeProvider>
					<h1>Add New Tool</h1>
					<div class="toolTypes">
						<h3>Type: </h3>
						<RadioButtonGroup name="ToolTypes" style={styles.radioButtonGroup} onChange={(e, value) => this.changeType(e, value)} >
							<RadioButton
								value="Hand Tool"
								label="Hand Tool"
								style={styles.radioButton}
							/>
							<RadioButton
								value="Garden Tool"
								label="Garden Tool"
								style={styles.radioButton}
							/>
							<RadioButton
								value="Ladder Tool"
								label="Ladder Tool"
								style={styles.radioButton}
							/>
							<RadioButton
								value="Power Tool"
								label="Power Tool"
								style={styles.radioButton}
							/>
						</RadioButtonGroup>
					</div>
					<div class="powerSources">
						<h3>Power Source:</h3>
						<DropDownMenu value={this.state.powerSourceValue} onChange={this.handlePowerSourceChange}>
							{this.state.powerSources.map((ps, index) => (<MenuItem key={index} value={ps} primaryText={ps} />))}
						</DropDownMenu>
					</div>
					<div class="subTypes">
						<h3>Sub Type:</h3>
						<DropDownMenu value={this.state.subTypeValue} onChange={this.handleSubTypeChange}>
							{this.state.subTypes.map((st, index) => (<MenuItem key={index} value={st} primaryText={st} />))}
						</DropDownMenu>
					</div>
					<div class="subOptions">
						<h3>Sub Option:</h3>
						<DropDownMenu value={this.state.subOptionValue} onChange={this.handleSubOptionChange}>
							{this.state.subOptions.map((so, index) => (<MenuItem key={index} value={so} primaryText={so} />))}
						</DropDownMenu>
					</div>
					<div class="toolAttributes">
						<TextField floatingLabelText="Purchase Price (in $)" type="number" min="0"
								floatingLabelStyle={styles.floatingLabelStyle}
      							floatingLabelFocusStyle={styles.floatingLabelFocusStyle}
      							onChange={this.setPurchasePrice}
      							errorText="This field is required"
      							style={styles.textFieldStyle}
      							errorStyle={styles.errorStyle}/>
						<TextField floatingLabelText="Manufacturer" type="text"
								floatingLabelStyle={styles.floatingLabelStyle}
      							floatingLabelFocusStyle={styles.floatingLabelFocusStyle}
      							onChange={this.setManufacturer}
      							errorText="This field is required"
      							style={styles.textFieldStyle}
      							errorStyle={styles.errorStyle}/>
						<TextField floatingLabelText="Material" type="text"
								floatingLabelStyle={styles.floatingLabelStyle}
      							floatingLabelFocusStyle={styles.floatingLabelFocusStyle}
      							onChange={this.setMaterial}
      							errorText="Optional"
      							errorStyle={styles.errorStyle}
      							style={styles.textFieldStyle}/>
						<TextField floatingLabelText="Weight" type="number" min="0"
								floatingLabelStyle={styles.floatingLabelStyle}
      							floatingLabelFocusStyle={styles.floatingLabelFocusStyle}
      							onChange={this.setWeight}
      							errorText="This field is required"
      							style={styles.textFieldStyle}
      							errorStyle={styles.errorStyle}/>
						<div class="ToolWidth">
							<TextField floatingLabelText="Width" type="number" min="0" step="1"
									floatingLabelStyle={styles.floatingLabelStyle}
	      							floatingLabelFocusStyle={styles.floatingLabelFocusStyle}
	      							onChange={this.setWidth}
	      							errorText="This field is required"
	      							style={styles.textFieldStyle}
	      							errorStyle={styles.errorStyle}/>
							<SelectField 
								floatingLabelText="Width Fraction"
								floatingLabelFixed={true}
								errorText="This field is required"
								errorStyle={styles.errorStyle}
								value={this.state.widthFractionValue} 
								onChange={this.handleWidthFractionChange} 
								style={styles.fractionStyle}>
								{fractions.map((e, index) => (<MenuItem key={index} value={e.value} primaryText={e.fraction} />))}
							</SelectField>
							<SelectField 
								floatingLabelText="Width Unit"
								floatingLabelFixed={true}
								errorText="This field is required"
								errorStyle={styles.errorStyle}
								value={this.state.widthUnitValue} 
								onChange={this.handleWidthUnitChange} 
								style={styles.fractionStyle}>
								{units.map((e, index) => (<MenuItem key={index} value={e} primaryText={e} />))}
							</SelectField>
						</div>
						<div class="ToolLength">
							<TextField floatingLabelText="Length" type="number" min="0" step="1"
									floatingLabelStyle={styles.floatingLabelStyle}
	      							floatingLabelFocusStyle={styles.floatingLabelFocusStyle}
	      							onChange={this.setLength}
	      							errorText="This field is required"
	      							style={styles.textFieldStyle}
	      							errorStyle={styles.errorStyle}/>
							<SelectField
								floatingLabelText="Length Fraction"
								floatingLabelFixed={true}
								errorText="This field is required"
								errorStyle={styles.errorStyle}
								value={this.state.lengthFractionValue} 
								onChange={this.handleLengthFractionChange} 
								style={styles.fractionStyle}>
								{fractions.map((e, index) => (<MenuItem key={index} value={e.value} primaryText={e.fraction} />))}
							</SelectField>
							<SelectField 
								floatingLabelText="Length Unit"
								floatingLabelFixed={true}
								errorText="This field is required"
								errorStyle={styles.errorStyle}
								value={this.state.lengthUnitValue} 
								onChange={this.handleLengthUnitChange} 
								style={styles.fractionStyle}>
								{units.map((e, index) => (<MenuItem key={index} value={e} primaryText={e} />))}
							</SelectField>
						</div>
					</div>
					<div id="HandTools">
						<div id="htScrewdriver">
						<TextField 
								floatingLabelText="Screw Size (in #)" type="number" min="0" step="1"
								floatingLabelStyle={styles.floatingLabelStyle}
      							floatingLabelFocusStyle={styles.floatingLabelFocusStyle}
      							onChange={this.setScrewSize}
      							errorText="This field is required"
      							errorStyle={styles.errorStyle}
      							style={styles.textFieldStyle}/>
						</div>
						<div id="htPliers">
							<SelectField 
								floatingLabelText="Adjustable"
								floatingLabelFixed={true}
								errorText="This field is required"
								errorStyle={styles.errorStyle}
								value={this.state.adjustableValue} 
								onChange={this.setAdjustable} 
								style={styles.fractionStyle}>
								{booleans.map((e, index) => (<MenuItem key={index} value={e} primaryText={e} />))}
							</SelectField>
						</div>
						<div id="htSocket">
							<SelectField 
								floatingLabelText="Drive Size (in inches)"
								floatingLabelFixed={true}
								errorText="This field is required"
								errorStyle={styles.errorStyle}
								value={this.state.driveSizeValue} 
								onChange={this.handleDriveSizeChange}
								style={styles.fractionStyle}>
								{fractions.map((e, index) => (<MenuItem key={index} value={e.value} primaryText={e.fraction} />))}
							</SelectField>
							<SelectField 
								floatingLabelText="SAE Size (in inches)"
								floatingLabelFixed={true}
								errorText="This field is required"
								errorStyle={styles.errorStyle}
								value={this.state.saeSizeValue} 
								onChange={this.handleSaeSizeChange}
								style={styles.fractionStyle}>
								{fractions.map((e, index) => (<MenuItem key={index} value={e.value} primaryText={e.fraction} />))}
							</SelectField>
						</div>
						<div id="htRatchet">
							<SelectField 
								floatingLabelText="Drive Size (in inches)"
								floatingLabelFixed={true}
								errorText="This field is required"
								errorStyle={styles.errorStyle}
								value={this.state.driveSizeValue} 
								onChange={this.handleDriveSizeChange} 
								style={styles.fractionStyle}>
								{fractions.map((e, index) => (<MenuItem key={index} value={e.value} primaryText={e.fraction} />))}
							</SelectField>
						</div>
						<div id="htWrench">
							<SelectField 
								floatingLabelText="Drive Size (in inches)"
								floatingLabelFixed={true}
								errorText="Optional"
								errorStyle={styles.errorStyle}
								value={this.state.driveSizeValue} 
								onChange={this.handleDriveSizeChange} 
								style={styles.fractionStyle}>
								{fractions.map((e, index) => (<MenuItem key={index} value={e.value} primaryText={e.fraction} />))}
							</SelectField>
						</div>
						<div id="htGun">
							<SelectField
									floatingLabelText="Gauge Rating (in G)"
									floatingLabelFixed={true}
									errorText="Optional"
									errorStyle={styles.errorStyle}
									value={this.state.gaugeRatingValue} 
	      							onChange={this.setGaugeRating}
	      							style={styles.textFieldStyle}>
	      							{gaugeRatings.map((e, index) => (<MenuItem key={index} value={e.value} primaryText={e.label} />))}
      						</SelectField>
  							<TextField 
  									floatingLabelText="Capacity" type="number" min="0"
									floatingLabelStyle={styles.floatingLabelStyle}
	      							floatingLabelFocusStyle={styles.floatingLabelFocusStyle}
	      							onChange={this.setCapacity}
	      							errorText="This field is required"
      								errorStyle={styles.errorStyle}
	      							style={styles.textFieldStyle}/>
						</div>
						<div id="htHammer">
							<SelectField 
								floatingLabelText="Anti-Vibration"
								floatingLabelFixed={true}
								errorText="This field is required"
								errorStyle={styles.errorStyle}
								value={this.state.antiVibrationValue} 
								onChange={this.setAntiVibration} 
								style={styles.fractionStyle}>
								{booleans.map((e, index) => (<MenuItem key={index} value={e} primaryText={e} />))}
							</SelectField>
						</div>
					</div>
					<div id="LadderTools">
						<TextField 
							floatingLabelText="Step Count" type="number" min="0" step="1"
							floatingLabelStyle={styles.floatingLabelStyle}
      						floatingLabelFocusStyle={styles.floatingLabelFocusStyle}
							errorText="Optional"
							errorStyle={styles.errorStyle}
      						onChange={this.setStepCount}
      						style={styles.textFieldStyle}/>
						<TextField 
							floatingLabelText="Weight Capacity (in lb.)" type="number" min="0"
							floatingLabelStyle={styles.floatingLabelStyle}
      						floatingLabelFocusStyle={styles.floatingLabelFocusStyle}
							errorText="Optional"
							errorStyle={styles.errorStyle}
      						onChange={this.setWeightCapacity}
      						style={styles.textFieldStyle}/>
						<div id="ltStraight">
							<SelectField 
								floatingLabelText="Rubber Feet"
								floatingLabelFixed={true}
								errorText="Optional"
								errorStyle={styles.errorStyle}
								value={this.state.rubberFeetValue} 
								onChange={this.setRubberFeet} 
								style={styles.fractionStyle}>
								{booleans.map((e, index) => (<MenuItem key={index} value={e} primaryText={e} />))}
							</SelectField>
						</div>
						<div id="ltStep">
							<SelectField 
								floatingLabelText="Pail Shelf"
								floatingLabelFixed={true}
								errorText="Optional"
								errorStyle={styles.errorStyle}
								value={this.state.pailShelfValue} 
								onChange={this.setPailShelf} 
								style={styles.fractionStyle}>
								{booleans.map((e, index) => (<MenuItem key={index} value={e} primaryText={e} />))}
							</SelectField>
						</div>
					</div>
					<div id="GardenTools">
						<TextField 
								floatingLabelText="Handle Material" type="text"
								floatingLabelStyle={styles.floatingLabelStyle}
      							floatingLabelFocusStyle={styles.floatingLabelFocusStyle}
								errorText="This field is required"
								errorStyle={styles.errorStyle}
      							onChange={this.setHandleMaterial}
      							style={styles.textFieldStyle}/>
      					<div id="gtPruning">
							<TextField 
									floatingLabelText="Blade Material" type="text"
									floatingLabelStyle={styles.floatingLabelStyle}
	      							floatingLabelFocusStyle={styles.floatingLabelFocusStyle}
									errorText="Optional"
									errorStyle={styles.errorStyle}
	      							onChange={this.setBladeMaterial}
	      							style={styles.textFieldStyle}/>
	      					<div>
								<TextField 
									floatingLabelText="Blade Length (in inches)" type="number" min="0" step="1"
									floatingLabelStyle={styles.floatingLabelStyle}
			      					floatingLabelFocusStyle={styles.floatingLabelFocusStyle}
									errorText="This field is required"
									errorStyle={styles.errorStyle}
			      					onChange={this.setBladeLength}
			      					style={styles.textFieldStyle}/>
								<SelectField
									floatingLabelText="Blade Length Fraction"
									floatingLabelFixed={true}
									errorText="This field is required"
									errorStyle={styles.errorStyle}
									value={this.state.bladeLengthFractionValue} 
									onChange={this.handleBladeLengthFractionChange} 
									style={styles.fractionStyle}>
									{fractions.map((e, index) => (<MenuItem key={index} value={e.value} primaryText={e.fraction} />))}
								</SelectField>
							</div>
						</div>
						<div id="gtStriking">
							<TextField 
								floatingLabelText="Head Weight (in lb.)" type="number" min="0"
								floatingLabelStyle={styles.floatingLabelStyle}
      							floatingLabelFocusStyle={styles.floatingLabelFocusStyle}
								errorText="This field is required"
								errorStyle={styles.errorStyle}
      							onChange={this.setHeadWeight}
      							style={styles.textFieldStyle}/>
						</div>
						<div id="gtDigging">
							<div>
								<TextField 
									floatingLabelText="Blade Width (in inches)" type="number" min="0" step="1"
									floatingLabelStyle={styles.floatingLabelStyle}
			      					floatingLabelFocusStyle={styles.floatingLabelFocusStyle}
									errorText="Optional"
									errorStyle={styles.errorStyle}
			      					onChange={this.setBladeWidth}
			      					style={styles.textFieldStyle}/>
								<SelectField 
									floatingLabelText="Blade Width Fraction"
									floatingLabelFixed={true}
									errorText="Optional"
									errorStyle={styles.errorStyle}
									value={this.state.bladeWidthFractionValue} 
									onChange={this.handleBladeWidthFractionChange} 
									style={styles.fractionStyle}>
									{fractions.map((e, index) => (<MenuItem key={index} value={e.value} primaryText={e.fraction} />))}
								</SelectField>
							</div>
							<div>
								<TextField 
									floatingLabelText="Blade Length (in inches)" type="number" min="0" step="1"
									floatingLabelStyle={styles.floatingLabelStyle}
			      					floatingLabelFocusStyle={styles.floatingLabelFocusStyle}
									errorText="This field is required"
									errorStyle={styles.errorStyle}
			      					onChange={this.setBladeLength}
			      					style={styles.textFieldStyle}/>
								<SelectField 
									floatingLabelText="Blade Length Fraction"
									floatingLabelFixed={true}
									errorText="This field is required"
									errorStyle={styles.errorStyle}
									value={this.state.bladeLengthFractionValue} 
									onChange={this.handleBladeLengthFractionChange} 
									style={styles.fractionStyle}>
									{fractions.map((e, index) => (<MenuItem key={index} value={e.value} primaryText={e.fraction} />))}
								</SelectField>
							</div>
						</div>
						<div id="gtRake">
							<TextField 
								floatingLabelText="Tine Count" type="number" min="0" step="1"
								floatingLabelStyle={styles.floatingLabelStyle}
			      				floatingLabelFocusStyle={styles.floatingLabelFocusStyle}
								errorText="This field is required"
								errorStyle={styles.errorStyle}
			      				onChange={this.setTineCount}
			      				style={styles.textFieldStyle}/>
						</div>
						<div id="gtWheelbarrow">
							<div>
								<TextField 
									floatingLabelText="Bin Material" type="text"
									floatingLabelStyle={styles.floatingLabelStyle}
	      							floatingLabelFocusStyle={styles.floatingLabelFocusStyle}
									errorText="This field is required"
									errorStyle={styles.errorStyle}
	      							onChange={this.setBinMaterial}
	      							style={styles.textFieldStyle}/>
								<TextField 
									floatingLabelText="Bin Volume (in cu. ft.)" type="number" min="0"
									floatingLabelStyle={styles.floatingLabelStyle}
			      					floatingLabelFocusStyle={styles.floatingLabelFocusStyle}
									errorText="Optional"
									errorStyle={styles.errorStyle}
			      					onChange={this.setBinVolume}
			      					style={styles.textFieldStyle}/>
		      				</div>
		      				<div>
								<TextField 
									floatingLabelText="Wheel Count" type="number" min="0" step="1"
									floatingLabelStyle={styles.floatingLabelStyle}
			      					floatingLabelFocusStyle={styles.floatingLabelFocusStyle}
									errorText="This field is required"
									errorStyle={styles.errorStyle}
			      					onChange={this.setWheelCount}
			      					style={styles.textFieldStyle}/>
			      			</div>
						</div>
					</div>
					<div id="PowerTools">
						<div>
							Volt Rating (in V):
							<SelectField 
								floatingLabelText="Adjustable"
								floatingLabelFixed={true}
								errorText="This field is required"
								errorStyle={styles.errorStyle}
								value={this.state.voltRatingValue} 
								onChange={this.setVoltRating} 
								style={styles.fractionStyle}>
								{volts.map((e, index) => (<MenuItem key={index} value={e.value} primaryText={e.label} />))}
							</SelectField>
							<TextField 
								floatingLabelText="AMP Rating (in A)" type="number" min="0"
								floatingLabelStyle={styles.floatingLabelStyle}
		      					floatingLabelFocusStyle={styles.floatingLabelFocusStyle}
								errorText="This field is required"
								errorStyle={styles.errorStyle}
		      					onChange={this.setAmpRating}
		      					style={styles.textFieldStyle}/>
	      				</div>
	      				<div>
							<TextField 
								floatingLabelText="Min RPM Rating (in RPM)" type="number" min="0" step="1"
								floatingLabelStyle={styles.floatingLabelStyle}
	      						floatingLabelFocusStyle={styles.floatingLabelFocusStyle}
								errorText="This field is required"
								errorStyle={styles.errorStyle}
	      						onChange={this.setMinRpmRating}
	      						style={styles.textFieldStyle}/>
							<TextField 
								floatingLabelText="Max RPM Rating (in RPM)" type="number" min="0" step="1"
								floatingLabelStyle={styles.floatingLabelStyle}
	      						floatingLabelFocusStyle={styles.floatingLabelFocusStyle}
								errorText="Optional"
								errorStyle={styles.errorStyle}
	      						onChange={this.setMaxRpmRating}
	      						style={styles.textFieldStyle}/>
						</div>
      					<div id="ptDrill">
							<SelectField 
								floatingLabelText="Adjustable Clutch"
								floatingLabelFixed={true}
								errorText="This field is required"
								errorStyle={styles.errorStyle}
								value={this.state.adjustableClutchValue} 
								onChange={this.setAdjustableClutch} 
								style={styles.fractionStyle}>
								{booleans.map((e, index) => (<MenuItem key={index} value={e} primaryText={e} />))}
							</SelectField>
							<TextField 
								floatingLabelText="Min Torque Rating (in ft-lb.)" type="number" min="0"
								floatingLabelStyle={styles.floatingLabelStyle}
	      						floatingLabelFocusStyle={styles.floatingLabelFocusStyle}
								errorText="This field is required"
								errorStyle={styles.errorStyle}
	      						onChange={this.setMinTorqueRating}
	      						style={styles.textFieldStyle}/>
  							<TextField 
  								floatingLabelText="Max Torque Rating (in ft-lb.)" type="number" min="0"
								floatingLabelStyle={styles.floatingLabelStyle}
	      						floatingLabelFocusStyle={styles.floatingLabelFocusStyle}
								errorText="Optional"
								errorStyle={styles.errorStyle}
	      						onChange={this.setMaxTorqueRating}
	      						style={styles.textFieldStyle}/>
      					</div>
      					<div id="ptSaw">
      						<TextField 
      							floatingLabelText="Blade Size (in inches)" type="number" min="0" step="1"
								floatingLabelStyle={styles.floatingLabelStyle}
	      						floatingLabelFocusStyle={styles.floatingLabelFocusStyle}
								errorText="This field is required"
								errorStyle={styles.errorStyle}
	      						onChange={this.setBladeSize}
	      						style={styles.textFieldStyle}/>
							<SelectField 
								floatingLabelText="Blade Size Fraction"
								floatingLabelFixed={true}
								errorText="This field is required"
								errorStyle={styles.errorStyle}
								value={this.state.bladeSizeFractionValue} 
								onChange={this.setBladeSizeFraction} 
								style={styles.fractionStyle}>
								{fractions.map((e, index) => (<MenuItem key={index} value={e.value} primaryText={e.fraction} />))}
							</SelectField>
      					</div>
      					<div id="ptSander">
							<SelectField 
								floatingLabelText="Dust Bag"
								floatingLabelFixed={true}
								errorText="This field is required"
								errorStyle={styles.errorStyle}
								value={this.state.dustBagValue} 
								onChange={this.setDustBag} 
								style={styles.fractionStyle}>
								{booleans.map((e, index) => (<MenuItem key={index} value={e} primaryText={e} />))}
							</SelectField>
      					</div>
      					<div id="ptAirCompressor">
      						<TextField 
      							floatingLabelText="Tank Size (in gal.)" type="number" min="0"
								floatingLabelStyle={styles.floatingLabelStyle}
	      						floatingLabelFocusStyle={styles.floatingLabelFocusStyle}
								errorText="This field is required"
								errorStyle={styles.errorStyle}
	      						onChange={this.setTankSize}
	      						style={styles.textFieldStyle}/>
	      					<TextField 
	      						floatingLabelText="Pressure Rating (in psi)" type="number" min="0"
								floatingLabelStyle={styles.floatingLabelStyle}
	      						floatingLabelFocusStyle={styles.floatingLabelFocusStyle}
								errorText="Optional"
								errorStyle={styles.errorStyle}
	      						onChange={this.setPressureRating}
	      						style={styles.textFieldStyle}/>
      					</div>
      					<div id="ptMixer">
							<SelectField
								floatingLabelText="Motor Rating (in HP)"
								floatingLabelFixed={true}
								errorText="This field is required"
								errorStyle={styles.errorStyle}
								value={this.state.motorRatingValue} 
								onChange={this.setMotorRating} 
								style={styles.fractionStyle}>
								{fractions.map((e, index) => (<MenuItem key={index} value={e.value} primaryText={e.fraction} />))}
							</SelectField>
							<TextField 
								floatingLabelText="Drum Size (in cu-ft.)" type="number" min="0"
								floatingLabelStyle={styles.floatingLabelStyle}
	      						floatingLabelFocusStyle={styles.floatingLabelFocusStyle}
								errorText="This field is required"
								errorStyle={styles.errorStyle}
	      						onChange={this.setDrumSize}
	      						style={styles.textFieldStyle}/>
      					</div>
      					<div id="ptGenerator">
      						<TextField 
      							floatingLabelText="Power Rating (in Watt)" type="number" min="0"
								floatingLabelStyle={styles.floatingLabelStyle}
	      						floatingLabelFocusStyle={styles.floatingLabelFocusStyle}
								errorText="This field is required"
								errorStyle={styles.errorStyle}
	      						onChange={this.setPowerRating}
	      						style={styles.textFieldStyle}/>
      					</div>
      					<div id="ptAccessories">
      						<h3>Power Tool Accessories:</h3>
      						<Table
								style={{ tableLayout: 'auto' }}
								fixedHeader={this.state.fixedHeader}
								selectable={this.state.selectable}>
								<TableHeader
									adjustForCheckbox={this.state.adjustForCheckbox}
									displaySelectAll={this.state.displaySelectAll}>
									<TableRow>
										<TableHeaderColumn>Quantity</TableHeaderColumn>
										<TableHeaderColumn>Description</TableHeaderColumn>
									</TableRow>
								</TableHeader>
								<TableBody>
									{this.tool.accList.map((res) => <TableRowData data = {res} />)}
								</TableBody>
							</Table>
      						<TextField 
      							floatingLabelText="Quantity" type="number" min="0" step="1"
								floatingLabelStyle={styles.floatingLabelStyle}
	      						floatingLabelFocusStyle={styles.floatingLabelFocusStyle}
	      						onChange={this.setAccessoryQuantity}
	      						style={styles.textFieldStyle}/>
							<SelectField
								floatingLabelText="Accessory"
								floatingLabelFixed={true}
								value={this.state.accessoryValue} 
								onChange={this.setAccessoryValue} 
								style={styles.accessoriesStyle}>
								{accessories.map((e, index) => (<MenuItem key={index} value={e} primaryText={e} />))}
							</SelectField>
							<SelectField 
								id="safetyAccessories" 
								value={this.state.safetyAccessoryValue} 
								floatingLabelText="Safety Accessory"
								floatingLabelFixed={true}
								onChange={this.setSafetyAccessory}
								style={styles.accessoriesStyle}>
								{safetyAccessories.map((e, index) => (<MenuItem key={index} value={e} primaryText={e} />))}
							</SelectField>
							<SelectField 
								id="batteryTypes" 
								value={this.state.batteryTypeValue} 
								floatingLabelText="Battery Type"
								floatingLabelFixed={true}
								onChange={this.setBatteryType} 
								style={styles.accessoriesStyle}>
								{batteryTypes.map((e, index) => (<MenuItem key={index} value={e} primaryText={e} />))}
							</SelectField>
							<RaisedButton 
								style={{ margin: '10px' }}
								label="Add Accessory"
								onClick={(e) => this.addAccessory()} 
							/>
      					</div>
					</div>
					{displayNotification && !isError && <SuccessfulRegistrationMsg msg={responseMessage} />}
					{displayNotification && isError && <ErrorMsg msg={responseMessage} />}
					<div>
						<RaisedButton 
							style={{ margin: '10px' }}
							label="Add New Tool"
							onClick={(e) => this.addNewTool()} 
						/>
					</div>
				</MuiThemeProvider>
			</div>
		)
	}
}

class TableRowData extends React.Component {
	render() {
		return (
			<TableRow>
				<TableRowColumn>{this.props.data.quantity}</TableRowColumn>
				<TableRowColumn>{this.props.data.description}</TableRowColumn>
			</TableRow>
		)
	}
}