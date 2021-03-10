import React from 'react';
import '../../css/SuccessfulRegistrationMsg.css';


export default class SuccessfulRegistrationMsg extends React.Component {
state = {
    msg: ''
}

    constructor(props){
        super(props);
        this.state.msg = this.props.msg;
    }
    render() {
        return(
            <div className="alert success">
            <h3>{this.state.msg}</h3>
            </div>
        )
    }
}
