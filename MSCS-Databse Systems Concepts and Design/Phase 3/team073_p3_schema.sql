-- Phase 2 | CS 6400 - Fall 2017 | Team 073
-- Schema tested with MySQL Ver 14.14 Distrib 5.7.19, for Linux (x86_64)
-- Server version: 5.7.19-0ubuntu0.17.04.1 (Ubuntu)

-- User 
CREATE USER IF NOT EXISTS gatechUser@'%' IDENTIFIED BY 'gatech123';

-- Database
DROP DATABASE IF EXISTS cs6400_fa17_team073;
SET default_storage_engine=InnoDB;
SET NAMES utf8mb4 COLLATE utf8mb4_unicode_ci;

CREATE DATABASE IF NOT EXISTS cs6400_fa17_team073
	DEFAULT CHARACTER SET utf8mb4
	DEFAULT COLLATE utf8mb4_unicode_ci;
USE cs6400_fa17_team073;

GRANT SELECT, INSERT, UPDATE, DELETE, FILE ON *.* TO gatechUser@'%';
GRANT ALL PRIVILEGES ON gatechUser.* TO gatechUser@'%';
GRANT ALL PRIVILEGES ON cs6400_fa17_team073.* TO gatechUser@'%';
FLUSH PRIVILEGES;

-- Tables
CREATE TABLE cs6400_fa17_team073.User (
	userID int unsigned NOT NULL AUTO_INCREMENT,
	username varchar(50) NOT NULL,
	password varchar(20) NOT NULL, 
	email varchar(250) NOT NULL,
	first_name varchar(50) NOT NULL,
	middle_name varchar(50) DEFAULT NULL,
	last_name varchar(50) NOT NULL,
	PRIMARY KEY (userID),
	UNIQUE KEY (username),
	UNIQUE KEY (email)
);

CREATE TABLE cs6400_fa17_team073.Customer (
	userID int unsigned NOT NULL,
	zip_code varchar(10) NOT NULL,
	street varchar(100) NOT NULL,
	city varchar(50) NOT NULL,
	state varchar(20) NOT NULL,
	PRIMARY KEY (userID),
	CONSTRAINT FK_Customer_userID_User_userID
	FOREIGN KEY (userID) REFERENCES User (userID)
);

CREATE TABLE cs6400_fa17_team073.CreditCard (
	userID int unsigned NOT NULL,	
	cred_number varchar(30) NOT NULL,
	name varchar(150) NOT NULL,
	cvc int unsigned NOT NULL,
	exp_month int unsigned NOT NULL,
	exp_year int unsigned NOT NULL,
	PRIMARY KEY (userID),
	CONSTRAINT FK_CreditCard_userID_Customer_userID
	FOREIGN KEY (userID) REFERENCES Customer (userID)
);


CREATE TABLE cs6400_fa17_team073.HomePhone (
	userID int unsigned NOT NULL, 
	area_code varchar(3) NOT NULL,
	phone_number varchar(8) NOT NULL,
	extension varchar(5) DEFAULT NULL,
	PRIMARY KEY (userID),
	CONSTRAINT FK_HomePhone_userID_Customer_userID
	FOREIGN KEY (userID) REFERENCES Customer (userID)
);

CREATE TABLE cs6400_fa17_team073.CellPhone (
	userID int unsigned NOT NULL, 
	area_code varchar(3) NOT NULL,
	phone_number varchar(8) NOT NULL,
	extension varchar(5) DEFAULT NULL,
	PRIMARY KEY (userID),
	CONSTRAINT FK_CellPhone_userID_Customer_userID
	FOREIGN KEY (userID) REFERENCES Customer (userID)
);

CREATE TABLE cs6400_fa17_team073.WorkPhone (
	userID int unsigned NOT NULL, 
	area_code varchar(3) NOT NULL,
	phone_number varchar(8) NOT NULL,
	extension varchar(5) DEFAULT NULL,
	PRIMARY KEY (userID),
	CONSTRAINT FK_WorkPhone_userID_User_userID
	FOREIGN KEY (userID) REFERENCES User (userID)
);

CREATE TABLE cs6400_fa17_team073.PrimaryPhone (
	userID int unsigned NOT NULL, 
	area_code varchar(3) NOT NULL,
	phone_number varchar(8) NOT NULL,
	extension varchar(5) DEFAULT NULL,
	PRIMARY KEY (userID),
	CONSTRAINT FK_PrimaryPhone_userID_Customer_userID
	FOREIGN KEY (userID) REFERENCES Customer (userID)
);

CREATE TABLE cs6400_fa17_team073.Clerk (
	userID int unsigned NOT NULL, 
	emp_num int unsigned NOT NULL,
	date_hired DATETIME NOT NULL,
	has_logged_in_before boolean NOT NULL DEFAULT false,
	PRIMARY KEY (userID),
	CONSTRAINT FK_Clerk_userID_User_userID
	FOREIGN KEY (userID) REFERENCES User (userID),
	UNIQUE KEY (emp_num)
);


CREATE TABLE cs6400_fa17_team073.Tool (
	toolNumber int unsigned NOT NULL AUTO_INCREMENT,
	type varchar(50) NOT NULL,
	sub_type varchar(50) NOT NULL,
	sub_option varchar(50) NOT NULL,
	power_source varchar(20) NOT NULL,
	material varchar(50),
	length double NOT NULL,
	width double NOT NULL,
	weight double NOT NULL,
	manufacturer varchar(100) NOT NULL,
	purchase_price decimal(15,2) NOT NULL,
	PRIMARY KEY (toolNumber)
);

CREATE TABLE cs6400_fa17_team073.LadderTool (
	toolNumber int unsigned NOT NULL,
	weight_capacity int unsigned DEFAULT NULL,
	step_count int unsigned DEFAULT NULL,
	PRIMARY KEY (toolNumber),
	CONSTRAINT FK_LadderTool_toolNumber_Tool_toolNumber
	FOREIGN KEY (toolNumber) REFERENCES Tool (toolNumber)	
);

CREATE TABLE cs6400_fa17_team073.Straight (
	toolNumber int unsigned NOT NULL,
	rubber_feet boolean DEFAULT NULL,
	PRIMARY KEY (toolNumber),
	CONSTRAINT FK_Straight_toolNumber_LadderTool_toolNumber
	FOREIGN KEY (toolNumber) REFERENCES LadderTool (toolNumber)	
);

CREATE TABLE cs6400_fa17_team073.Step (
	toolNumber int unsigned NOT NULL,
	pail_shelf boolean DEFAULT NULL,
	PRIMARY KEY (toolNumber),
	CONSTRAINT FK_Step_toolNumber_LadderTool_toolNumber
	FOREIGN KEY (toolNumber) REFERENCES LadderTool (toolNumber)	
);

CREATE TABLE cs6400_fa17_team073.PowerTool (
	toolNumber int unsigned NOT NULL,
	volt_rating double unsigned NOT NULL,
	amp_rating double unsigned NOT NULL,
	min_rpm_rating int unsigned NOT NULL,
	max_rpm_rating int unsigned DEFAULT NULL,
	PRIMARY KEY (toolNumber),
	CONSTRAINT FK_PowerTool_toolNumber_Tool_toolNumber
	FOREIGN KEY (toolNumber) REFERENCES Tool (toolNumber)	
);

CREATE TABLE cs6400_fa17_team073.Generator (
	toolNumber int unsigned NOT NULL,
	power_rating double NOT NULL,
	PRIMARY KEY (toolNumber),
	CONSTRAINT FK_Generator_toolNumber_PowerTool_toolNumber
	FOREIGN KEY (toolNumber) REFERENCES PowerTool (toolNumber)	
);

CREATE TABLE cs6400_fa17_team073.Saw (
	toolNumber int unsigned NOT NULL,
	blade_size double NOT NULL,
	PRIMARY KEY (toolNumber),
	CONSTRAINT FK_Saw_toolNumber_PowerTool_toolNumber
	FOREIGN KEY (toolNumber) REFERENCES PowerTool (toolNumber)	
);

CREATE TABLE cs6400_fa17_team073.Sander (
	toolNumber int unsigned NOT NULL,
	dust_bag boolean NOT NULL,
	PRIMARY KEY (toolNumber),
	CONSTRAINT FK_Sander_toolNumber_PowerTool_toolNumber
	FOREIGN KEY (toolNumber) REFERENCES PowerTool (toolNumber)	
);

CREATE TABLE cs6400_fa17_team073.AirCompressor (
	toolNumber int unsigned NOT NULL,
	tank_size double NOT NULL,
	pressure_rating double DEFAULT NULL,
	PRIMARY KEY (toolNumber),
	CONSTRAINT FK_AirCompressor_toolNumber_PowerTool_toolNumber
	FOREIGN KEY (toolNumber) REFERENCES PowerTool (toolNumber)	
);

CREATE TABLE cs6400_fa17_team073.Drill (
	toolNumber int unsigned NOT NULL,
	adjustable_clutch boolean NOT NULL,
	min_torque_rating double NOT NULL,
	max_torque_rating double DEFAULT NULL,
	PRIMARY KEY (toolNumber),
	CONSTRAINT FK_Drill_toolNumber_PowerTool_toolNumber
	FOREIGN KEY (toolNumber) REFERENCES PowerTool (toolNumber)	
);

CREATE TABLE cs6400_fa17_team073.Mixer (
	toolNumber int unsigned NOT NULL,
	motor_rating double NOT NULL,	
	drum_size double NOT NULL,
	PRIMARY KEY (toolNumber),
	CONSTRAINT FK_Mixer_toolNumber_PowerTool_toolNumber
	FOREIGN KEY (toolNumber) REFERENCES PowerTool (toolNumber)	
);

CREATE TABLE cs6400_fa17_team073.Accessory (
	toolNumber int unsigned NOT NULL,
	description varchar(300) NOT NULL,
	quantity int unsigned NOT NULL DEFAULT 1,
	KEY (toolNumber),
	CONSTRAINT FK_Accessory_toolNumber_PowerTool_toolNumber
	FOREIGN KEY (toolNumber) REFERENCES PowerTool (toolNumber)	
);

CREATE TABLE cs6400_fa17_team073.HandTool (
	toolNumber int unsigned NOT NULL,
	PRIMARY KEY (toolNumber),
	CONSTRAINT FK_HandTool_toolNumber_Tool_toolNumber
	FOREIGN KEY (toolNumber) REFERENCES Tool (toolNumber)	
);

CREATE TABLE cs6400_fa17_team073.Gun (
	toolNumber int unsigned NOT NULL,
	capacity int unsigned NOT NULL,
	gauge_rating int DEFAULT NULL,
	PRIMARY KEY (toolNumber),
	CONSTRAINT FK_Gun_toolNumber_HandTool_toolNumber
	FOREIGN KEY (toolNumber) REFERENCES HandTool (toolNumber)	
);

CREATE TABLE cs6400_fa17_team073.Socket (
	toolNumber int unsigned NOT NULL,
	drive_size double unsigned NOT NULL,
	sae_size double unsigned NOT NULL,
	deep_socket boolean DEFAULT NULL,
	PRIMARY KEY (toolNumber),
	CONSTRAINT FK_Socket_toolNumber_HandTool_toolNumber
	FOREIGN KEY (toolNumber) REFERENCES HandTool (toolNumber)	
);

CREATE TABLE cs6400_fa17_team073.ScrewDriver (
	toolNumber int unsigned NOT NULL,
	screw_size int unsigned NOT NULL,
	PRIMARY KEY (toolNumber),
	CONSTRAINT FK_ScrewDriver_toolNumber_HandTool_toolNumber
	FOREIGN KEY (toolNumber) REFERENCES HandTool (toolNumber)	
);

CREATE TABLE cs6400_fa17_team073.Hammer (
	toolNumber int unsigned NOT NULL,
	anti_vibration boolean NOT NULL,
	PRIMARY KEY (toolNumber),
	CONSTRAINT FK_Hammer_toolNumber_HandTool_toolNumber
	FOREIGN KEY (toolNumber) REFERENCES HandTool (toolNumber)	
);

CREATE TABLE cs6400_fa17_team073.Plier (
	toolNumber int unsigned NOT NULL,
	adjustable boolean NOT NULL,
	PRIMARY KEY (toolNumber),
	CONSTRAINT FK_Plier_toolNumber_HandTool_toolNumber
	FOREIGN KEY (toolNumber) REFERENCES HandTool (toolNumber)	
);

CREATE TABLE cs6400_fa17_team073.Ratchet (
	toolNumber int unsigned NOT NULL,
	drive_size double unsigned NOT NULL,
	PRIMARY KEY (toolNumber),
	CONSTRAINT FK_Ratchet_toolNumber_HandTool_toolNumber
	FOREIGN KEY (toolNumber) REFERENCES HandTool (toolNumber)	
);

CREATE TABLE cs6400_fa17_team073.Wrench (
	toolNumber int unsigned NOT NULL,
	drive_size double unsigned DEFAULT NULL,
	PRIMARY KEY (toolNumber),
	CONSTRAINT FK_Wrench_toolNumber_HandTool_toolNumber
	FOREIGN KEY (toolNumber) REFERENCES HandTool (toolNumber)	
);

CREATE TABLE cs6400_fa17_team073.GardenTool (
	toolNumber int unsigned NOT NULL,
	handle_material varchar(50) NOT NULL,
	PRIMARY KEY (toolNumber),
	CONSTRAINT FK_GardenTool_toolNumber_Tool_toolNumber
	FOREIGN KEY (toolNumber) REFERENCES Tool (toolNumber)	
);

CREATE TABLE cs6400_fa17_team073.WheelBarrow (
	toolNumber int unsigned NOT NULL,
	bin_material varchar(50) NOT NULL,
	bin_volume double DEFAULT NULL,
	wheel_count int unsigned NOT NULL,
	PRIMARY KEY (toolNumber),
	CONSTRAINT FK_WheelBarrow_toolNumber_GardenTool_toolNumber
	FOREIGN KEY (toolNumber) REFERENCES GardenTool (toolNumber)	
);

CREATE TABLE cs6400_fa17_team073.Digging (
	toolNumber int unsigned NOT NULL,
	blade_length double NOT NULL,	
	blade_width double DEFAULT NULL,
	PRIMARY KEY (toolNumber),
	CONSTRAINT FK_Digging_toolNumber_GardenTool_toolNumber
	FOREIGN KEY (toolNumber) REFERENCES GardenTool (toolNumber)	
);

CREATE TABLE cs6400_fa17_team073.Prunning (
	toolNumber int unsigned NOT NULL,
	blade_material varchar(50) DEFAULT NULL,
	blade_length double NOT NULL,
	PRIMARY KEY (toolNumber),
	CONSTRAINT FK_Prunning_toolNumber_GardenTool_toolNumber
	FOREIGN KEY (toolNumber) REFERENCES GardenTool (toolNumber)	
);


CREATE TABLE cs6400_fa17_team073.Striking (
	toolNumber int unsigned NOT NULL,
	head_weight double NOT NULL,
	PRIMARY KEY (toolNumber),
	CONSTRAINT FK_Striking_toolNumber_GardenTool_toolNumber
	FOREIGN KEY (toolNumber) REFERENCES GardenTool (toolNumber)	
);

CREATE TABLE cs6400_fa17_team073.Rake (
	toolNumber int unsigned NOT NULL,
	tine_count int NOT NULL,
	PRIMARY KEY (toolNumber),
	CONSTRAINT FK_Rake_toolNumber_GardenTool_toolNumber
	FOREIGN KEY (toolNumber) REFERENCES GardenTool (toolNumber)	
);

-- Add TRIGGER to verify end_date is after start_date
CREATE TABLE cs6400_fa17_team073.ServiceOrderRequest (
	serviceOrderID int unsigned NOT NULL AUTO_INCREMENT,
	userID int unsigned NOT NULL,
	toolNumber int unsigned NOT NULL,
	cost decimal(15,2) unsigned NOT NULL,
	start_date DATETIME NOT NULL,
	end_date DATETIME NOT NULL,
	PRIMARY KEY (serviceOrderID),
	CONSTRAINT FK_ServiceOrderRequest_userID_Clerk_userID
	FOREIGN KEY (userID) REFERENCES Clerk (userID),
	CONSTRAINT FK_ServiceOrderRequest_toolNumber_Tool_toolNumber
	FOREIGN KEY (toolNumber) REFERENCES Tool (toolNumber)	
);


CREATE TABLE cs6400_fa17_team073.SaleOrder (
	saleOrderID int unsigned NOT NULL AUTO_INCREMENT,
	clerkUserID int unsigned NOT NULL,
	customerUserID int unsigned DEFAULT NULL,
	toolNumber int unsigned NOT NULL,
	for_sale_date DATETIME NOT NULL,
	sold_date DATETIME DEFAULT NULL,
	PRIMARY KEY (saleOrderID),
	CONSTRAINT FK_SaleOrder_clerkUserID_Clerk_userID
	FOREIGN KEY (clerkUserID) REFERENCES Clerk (userID),
	CONSTRAINT FK_SaleOrder_customerUserID_Customer_userID
	FOREIGN KEY (customerUserID) REFERENCES Customer (userID),
	CONSTRAINT FK_SaleOrder_toolNumber_Tool_toolNumber
	FOREIGN KEY (toolNumber) REFERENCES Tool (toolNumber)	
);


-- Add TRIGGER to verify end_date is after start_date
CREATE TABLE cs6400_fa17_team073.Reservation (
	reservationID int unsigned NOT NULL AUTO_INCREMENT,
	customerUserID int unsigned NOT NULL,
	start_date DATETIME NOT NULL,
	end_date DATETIME NOT NULL,
	pickUpUserID int unsigned DEFAULT NULL,
	dropOffUserID int unsigned DEFAULT NULL,
	PRIMARY KEY (reservationID),
	CONSTRAINT FK_Reservation_customerUserID_Customer_userID
	FOREIGN KEY (customerUserID) REFERENCES Customer (userID),
	CONSTRAINT FK_Reservation_pickUpUserID_User_userID
	FOREIGN KEY (pickUpUserID) REFERENCES Clerk (userID),
	CONSTRAINT FK_Reservation_dropOffUserID_User_userID
	FOREIGN KEY (dropOffUserID) REFERENCES Clerk (userID)		
);

CREATE TABLE cs6400_fa17_team073.IsOf (
	reservationID int unsigned NOT NULL,
	toolNumber int unsigned NOT NULL,
	PRIMARY KEY (reservationID, toolNumber),
	CONSTRAINT FK_IsOf_reservationID_Reservation_reservationID
	FOREIGN KEY (reservationID) REFERENCES Reservation (reservationID),
	CONSTRAINT FK_IsOf_toolNumber_Tool_toolNumber
	FOREIGN KEY (toolNumber) REFERENCES Tool (toolNumber)
);


CREATE TABLE cs6400_fa17_team073.ToolTypeOption (
	tt_name varchar(100) NOT NULL,
	tst_name varchar(100) NOT NULL,
	tso_name varchar(100) NOT NULL,
	ps_name varchar(100) NOT NULL,
	battery_type varchar(100) DEFAULT NULL,
	UNIQUE (tt_name, tst_name, tso_name, ps_name)
);

