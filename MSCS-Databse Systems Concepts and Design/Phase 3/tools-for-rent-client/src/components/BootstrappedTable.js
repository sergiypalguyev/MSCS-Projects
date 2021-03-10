import React from 'react';
import {BootstrapTable, TableHeaderColumn} from 'react-bootstrap-table'
import RaisedButton from 'material-ui/RaisedButton';
import MuiThemeProvider from 'material-ui/styles/MuiThemeProvider'



const style = {
  margin: 12,
};

function onSelectAll(isSelected, rows) {
  alert(`is select all: ${isSelected}`);
  if (isSelected) {
    alert('Current display and selected data: ');
  } else {
    alert('unselect rows: ');
  }
  for (let i = 0; i < rows.length; i++) {
    alert(rows[i].id);
  }
}

var toolsToAddToCart=[];

export default class BootstrappedTable extends React.Component{

    selectRowProp = {
     mode: 'checkbox',
     clickToSelect: true,
     onSelect: this.onRowSelect,
     onSelectAll: onSelectAll,
   };
    onRowSelect(row, isSelected, e) {
      let rowStr = '';
      var toolId;
      for (const prop in row) {
        rowStr += prop + ': "' + row[prop] + '"';
        if(prop=='toolID'){
            toolId = row[prop];
        }
      }

      if(isSelected){
          toolsToAddToCart.push(toolId);
      }else{
          const index = toolsToAddToCart.indexOf(toolId);
          toolsToAddToCart.splice(index, 1);
      }
    }
    addToolsToCart(){
        let data = this.state.data;
        let cur = this;
        toolsToAddToCart.forEach(function(item){
            for(var i = 0; i < data.length; i++){
                if(data[i].toolID == item){
                    cur.addToLocalStorage(data[i], item);
                }
            }
        });
        localStorage.setItem("startDate", this.props.startDate);
        localStorage.setItem("endDate", this.props.endDate);
    }

    addToLocalStorage(data, item) {
        console.log("going to localstorage");
        let existingEntries = JSON.parse(localStorage.getItem("toolsAddedToCart"));
       if(existingEntries == null){
           existingEntries = [];
       }else{
           //Check if localStorage already has item
           for(var l = 0; l < existingEntries.length; l++){
               if(existingEntries[l].id==item){
                   return;
               }
           }
       }
       let entry = {
           "id": data.toolID,
           "description": data.shortDesc,
           "depositPrice": data.depositPrice,
           "rentalPrice": data.rentalPrice
       };
       localStorage.setItem("entry", JSON.stringify(entry));
       existingEntries.push(entry);
       localStorage.setItem("toolsAddedToCart",JSON.stringify(existingEntries));
       console.log("existing entries length", existingEntries.length);
       this.props.onUpdateCart(existingEntries.length);
       this.props.setToolsAdded();

   };

     priceFormatter(value) {
       return '$' + parseFloat(value).toFixed(2);
     }

    constructor(props){
        super(props);
        this.state.data = this.props.data;
        console.log("data", this.state.data);
        this.addToLocalStorage = this.addToLocalStorage.bind(this);
        this.onRowSelect = this.onRowSelect.bind(this);
        this.addToolsToCart = this.addToolsToCart.bind(this);
    }
    state={
        data:''
    }

render(){
    const {data}= this.state;


    return(
        <div>
        <BootstrapTable data={data} selectRow={ this.selectRowProp } version='4'>
         <TableHeaderColumn dataField='toolID' isKey>Tool ID</TableHeaderColumn>
         <TableHeaderColumn dataField='shortDesc'>Description</TableHeaderColumn>
         <TableHeaderColumn dataField='depositPrice' dataFormat={this.priceFormatter}>Deposit Price</TableHeaderColumn>
         <TableHeaderColumn dataField='rentalPrice' dataFormat={this.priceFormatter}>Rental Price</TableHeaderColumn>
        </BootstrapTable>
        <MuiThemeProvider>
        <RaisedButton label="Add to cart!" secondary={true} style={style} onClick={(e) => this.addToolsToCart(e)}  />
        </MuiThemeProvider>
        </div>
    )
}
}
