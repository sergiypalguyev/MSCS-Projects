import React from 'react';
import Login from '../components/LoginForm';
import Signup from '../components/SignupForm';
import Home from '../components/Home';
import ViewProfile from '../components/ViewProfile';
import CheckToolAvailability from  '../components/CheckToolAvailability';
import MakeReservation from '../components/MakeReservation';
import GenerateClerkReport from '../components/GenerateClerkReport';
import CustomerSummaryReport from  '../components/CustomerSummaryReport';
import ToolSummaryReport from  '../components/ToolSummaryReport';
import ReportSelection from '../components/ReportSelection';
import Cart from '../components/Cart';
import Confirmation from '../components/Confirmation';
import ResConfirm from '../components/ReservationConfirmation';
import PrintPickUpConfirm from '../components/PrintPickUpConfirmation';
import CustomerMainMenu from '../components/CustomerMainMenu';
import ClerkMainMenu from '../components/ClerkMainMenu';
import AddNewTool from '../components/AddNewTool';
import PickUpReservation from '../components/PickUpReservation';
import DropOffReservation from '../components/DropOffReservation';
import PrintDropOffConfirmation from '../components/PrintDropOffConfirmation';
import ResetClerkPassword from '../components/ResetClerkPassword';

import {
    BrowserRouter, Route, Switch, HashRouter
} from 'react-router-dom';

 export default () => (
    <BrowserRouter>
    <div>
        <Route exact path="/" component={Home} />
        <Route exact path="/login"  render={(props) => (<Login {...props} />)}/>
        <Route exact path="/signup" render={(props) => (<Signup {...props} />)} />
        <Route exact path="/viewProfile" render={(props) => (<ViewProfile {...props} />)} />
        <Route exact path="/makeReservation" render={(props) => (<MakeReservation {...props} />)} />
        <Route exact path="/checkToolAvailability" render={(props) => (<CheckToolAvailability{...props} />)} />
        <Route exact path="/reports/clerk" render={(props) => (<GenerateClerkReport{...props} />)} />
        <Route exact path="/reports/customer" render={(props) => (<CustomerSummaryReport{...props} />)} />
        <Route exact path="/reports/tool" render={(props) => (<ToolSummaryReport{...props} />)} />
        <Route exact path="/reports" render={(props) => (<ReportSelection{...props} />)} />
        <Route exact path="/cart" render={(props) => (<Cart {...props} />)} />
        <Route exact path="/reservationConfirmation" render={(props) => (<Confirmation {...props} />)} />
        <Route exact path="/addNewTool" render={(props) => (<AddNewTool{...props} />)} />

        <Route exact path="/pickUpReservation" render={(props) => (<PickUpReservation {...props} />)} />
        <Route exact path="/resconfirm" render={(props) => (<ResConfirm {...props} />)} />
        <Route exact path="/printconfirmpickup" render={(props) => (<PrintPickUpConfirm {...props} />)} />

        <Route exact path="/dropOffReservation" render={(props) => (<DropOffReservation {...props} />)} />
        <Route exact path="/dropoffconfirm" render={(props) => (<PrintDropOffConfirmation {...props} />)} />
        <Route exact path="/customerMainMenu" render={(props) => (<CustomerMainMenu {...props} />)} />
        <Route exact path="/clerkMainMenu" render={(props) => (<ClerkMainMenu {...props} />)} />
        <Route exact path="/resetClerkPassword" render={(props) => (<ResetClerkPassword{...props} />)} />


        </div>
    </BrowserRouter>
 );
