import React from 'react';
import PropTypes from 'prop-types';
import {Menu, Dropdown, Image, Icon} from 'semantic-ui-react';
import {Link} from 'react-router-dom';
import UserIcon from '../icons/UserIcon';
import ShoppingCartIcon from '../icons/ShoppingCartIcon';



export default class NavBar extends React.Component{


    constructor(props){
        super(props);
    }

    emptyCart(){
        localStorage.removeItem("toolsAddedToCart");
    }
    performLogout(){
        localStorage.clear();
    }
    render(){
        const isCartEmpty = localStorage.getItem("toolsAddedToCart");
        const userType = localStorage.getItem("userType");
        const isCustomer = (userType == "customer");
        return(
            <Menu secondary pointing>
            <Menu.Menu position="left">
            <Dropdown icon={<UserIcon/>}>
              <Dropdown.Menu position="left">
                {isCustomer &&<Dropdown.Item> <Link to="/customerMainMenu"> Main Menu </Link> </Dropdown.Item>}
                {isCustomer &&<Dropdown.Item> <Link to="/viewProfile"> View Profile </Link> </Dropdown.Item>}
                {isCustomer &&<Dropdown.Item> <Link to="/checkToolAvailability"> Check Tool Availability </Link> </Dropdown.Item>}
                {isCustomer &&  <Dropdown.Item> <Link to="/makeReservation"> Make Reservation </Link> </Dropdown.Item>}

                {!isCustomer &&<Dropdown.Item> <Link to="/clerkMainMenu"> Main Menu </Link> </Dropdown.Item>}
                {!isCustomer &&<Dropdown.Item> <Link to="/pickUpReservation"> Pick-Up Reservation </Link> </Dropdown.Item>}
                {!isCustomer &&<Dropdown.Item> <Link to="/dropOffReservation"> Drop-Off Reservation </Link> </Dropdown.Item>}
                {!isCustomer &&<Dropdown.Item> <Link to="/addNewTool"> Add New Tool </Link> </Dropdown.Item>}
                {!isCustomer &&<Dropdown.Item> <Link to="/reports">Generate Reports </Link> </Dropdown.Item>}
                <Dropdown.Item> <Link to="/login"  onClick={this.performLogout}>({localStorage.getItem('username')}) Logout </Link> </Dropdown.Item>
                </Dropdown.Menu>
            </Dropdown>
            </Menu.Menu>
                {isCustomer && <Menu.Menu position="right">
                <Dropdown icon={<ShoppingCartIcon itemCount= {this.props.localStorageCount}/>}>
                {isCartEmpty&&
                    <Dropdown.Menu>
                    <Dropdown.Item> <Link to="/cart"> View Cart </Link> </Dropdown.Item>
                    <Dropdown.Item> <a onClick={this.emptyCart}>Empty Cart </a> </Dropdown.Item>
                    </Dropdown.Menu>
                }
                </Dropdown>
                </Menu.Menu>}
            </Menu>
        );
    }

}
