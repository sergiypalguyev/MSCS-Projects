import React from 'react';
import '../../css/SuccessfulRegistrationMsg.css';


export default class UserID extends React.Component {
state = {
    userID: ''
}

    constructor(props){
        super(props);
        this.state.userID = this.props.userID;
    }
    render() {
        return(
            <div className="alert success">
            <h3>{this.state.msg}</h3>
            </div>
        )
    }
}
