import React from 'react';
import '../../css/ErrMsg.css';


export default class LoginAndRegistrationErrorMsg extends React.Component {

    render() {
        const errorMsg = this.props.msg;
        return(
            <div className="alert fail">
            <h3>{errorMsg}</h3>
            </div>
        )
    }
}
