import React from 'react';
import {Icon} from 'semantic-ui-react';
import Badge from 'material-ui/Badge';
import IconButton from 'material-ui/IconButton';
import NotificationsIcon from 'material-ui/svg-icons/social/notifications';
import MuiThemeProvider from 'material-ui/styles/MuiThemeProvider'
import ShoppingCart from "material-ui/svg-icons/action/shopping-cart";

class ShoppingCartIcon  extends React.Component {

    render(){
        const itemCount = this.props.itemCount;
        return(
            <MuiThemeProvider>
              <Badge
                badgeContent={localStorage.getItem("toolsAddedToCart") == null ? 0: JSON.parse(localStorage.getItem("toolsAddedToCart")).length}
                secondary={true}
                badgeStyle={{top: 30, right: 5}}
              >
                <IconButton tooltip="Notifications">
                  <ShoppingCart />
                </IconButton>
              </Badge>
              </MuiThemeProvider>
        )
    }

};

export default ShoppingCartIcon;
