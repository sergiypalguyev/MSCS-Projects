import React from 'react';
import { makeStyles } from '@material-ui/core/styles';
import AppBar from '@material-ui/core/AppBar';
import Toolbar from '@material-ui/core/Toolbar';
import Typography from '@material-ui/core/Typography';
import Button from '@material-ui/core/Button';
import IconButton from '@material-ui/core/IconButton';
import MenuIcon from '@material-ui/icons/Menu';
import CssBaseline from '@material-ui/core/CssBaseline';
import Container from '@material-ui/core/Container';

const useStyles = makeStyles(theme => ({
  root: {
    flexGrow: 1,
  },
  menuButton: {
    marginRight: theme.spacing(2),
  },
  title: {
    flexGrow: 1,
  },
  heroContent: {
    padding: theme.spacing(8, 0, 6),
  },
}));




export default function TitleBar(props) {
  const classes = useStyles();

  return (
    <React.Fragment>
      <CssBaseline />

      <AppBar position="static">
      <Toolbar>
        <IconButton edge="start" className={classes.menuButton} color="inherit" aria-label="menu">
          <MenuIcon />
        </IconButton>
        <Typography variant="h6" color="inherit" noWrap className={classes.title}>
          {props.page == "MAIN" && "Welcome, "+ props.user}
          {props.page == "CHECKIN" && "Patients to Revisit"}
          {props.page == "TESTINGFORM" && "Patient Information"}
        </Typography>
        {props.page != "MAIN" && <Button onClick = {() => props.changePage("MAIN")} color="inherit">Back</Button>}

      </Toolbar>
      </AppBar>

      <Container maxWidth="sm" component="main" className={classes.heroContent}>


      </Container>
    </React.Fragment>
  );
}
