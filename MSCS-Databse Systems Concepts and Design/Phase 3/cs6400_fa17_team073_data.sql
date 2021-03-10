-- phpMyAdmin SQL Dump
-- version 4.6.6deb4
-- https://www.phpmyadmin.net/
--
-- Host: localhost:3306
-- Generation Time: Oct 19, 2017 at 08:00 PM
-- Server version: 5.7.19-0ubuntu0.17.04.1
-- PHP Version: 7.0.22-0ubuntu0.17.04.1

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `cs6400_fa17_team073`
--

--
-- Dumping data for table `CellPhone`
--

INSERT INTO `CellPhone` (`userID`, `area_code`, `phone_number`, `extension`) VALUES
(3, '321', '654-3210', NULL);

--
-- Dumping data for table `Clerk`
--

INSERT INTO `Clerk` (`userID`, `emp_num`, `date_hired`, `has_logged_in_before`) VALUES
(1, 1, '2017-11-10 00:00:00', 1),
(2, 2, '2017-10-15 12:01:37', 1);

--
-- Dumping data for table `CreditCard`
--

INSERT INTO `CreditCard` (`userID`, `cred_number`, `name`, `cvc`, `exp_month`, `exp_year`) VALUES
(3, '0987654321', 'Raul Viera', 123, 1, 2017);

--
-- Dumping data for table `Customer`
--

INSERT INTO `Customer` (`userID`, `zip_code`, `street`, `city`, `state`) VALUES
(3, '12345-1234', '123 TestStreet', 'TestCity', 'TestState');

--
-- Dumping data for table `Digging`
--

INSERT INTO `Digging` (`toolNumber`, `blade_length`, `blade_width`) VALUES
(3, 10, NULL);

--
-- Dumping data for table `Drill`
--

INSERT INTO `Drill` (`toolNumber`, `adjustable_clutch`, `min_torque_rating`, `max_torque_rating`) VALUES
(4, 0, 1, 5);

--
-- Dumping data for table `GardenTool`
--

INSERT INTO `GardenTool` (`toolNumber`, `handle_material`) VALUES
(3, 'steel');

--
-- Dumping data for table `HandTool`
--

INSERT INTO `HandTool` (`toolNumber`) VALUES
(1),
(2);

--
-- Dumping data for table `HomePhone`
--

INSERT INTO `HomePhone` (`userID`, `area_code`, `phone_number`, `extension`) VALUES
(3, '123', '123-4567', NULL);

--
-- Dumping data for table `IsOf`
--

INSERT INTO `IsOf` (`reservationID`, `toolNumber`) VALUES
(32, 2),
(1, 3),
(7, 3),
(12, 3),
(13, 3),
(8, 4),
(13, 4),
(14, 4),
(32, 5),
(35, 7);

--
-- Dumping data for table `LadderTool`
--

INSERT INTO `LadderTool` (`toolNumber`, `weight_capacity`, `step_count`) VALUES
(5, 2501, 4),
(6, 2501, 4);

--
-- Dumping data for table `PowerTool`
--

INSERT INTO `PowerTool` (`toolNumber`, `volt_rating`, `amp_rating`, `min_rpm_rating`, `max_rpm_rating`) VALUES
(4, 5.5, 3.3, 1, 5);

--
-- Dumping data for table `PrimaryPhone`
--

INSERT INTO `PrimaryPhone` (`userID`, `area_code`, `phone_number`, `extension`) VALUES
(3, '123', '123-4567', NULL);

--
-- Dumping data for table `Reservation`
--

INSERT INTO `Reservation` (`reservationID`, `customerUserID`, `start_date`, `end_date`, `pickUpUserID`, `dropOffUserID`) VALUES
(1, 3, '2017-10-16 00:00:00', '2017-10-18 00:00:00', NULL, NULL),
(7, 3, '2017-10-10 00:00:00', '2017-10-17 00:00:00', NULL, NULL),
(8, 3, '2017-10-20 00:00:00', '2017-10-23 00:00:00', NULL, NULL),
(12, 3, '2017-10-22 00:00:00', '2017-10-27 00:00:00', NULL, NULL),
(13, 3, '2017-10-28 00:00:00', '2017-10-30 00:00:00', NULL, NULL),
(14, 3, '2017-12-01 00:00:00', '2017-12-05 00:00:00', NULL, NULL),
(32, 3, '2017-12-01 10:00:00', '2017-12-03 09:59:59', NULL, NULL),
(35, 3, '2017-12-02 10:00:00', '2017-12-10 09:59:59', NULL, NULL);

--
-- Dumping data for table `SaleOrder`
--

INSERT INTO `SaleOrder` (`saleOrderID`, `clerkUserID`, `customerUserID`, `toolNumber`,  `for_sale_date`, `sold_date`) VALUES
(1, 2, NULL, 1, '2017-10-15 00:00:00', NULL);

--
-- Dumping data for table `ScrewDriver`
--

INSERT INTO `ScrewDriver` (`toolNumber`, `screw_size`) VALUES
(1, 5),
(2, 10);

--
-- Dumping data for table `ServiceOrderRequest`
--

INSERT INTO `ServiceOrderRequest` (`serviceOrderID`, `userID`, `toolNumber`, `cost`, `start_date`, `end_date`) VALUES
(1, 2, 2, '10.00', '2017-10-15 00:00:00', '2017-10-18 00:00:00'),
(2, 2, 2, '4.99', '2017-09-01 00:00:00', '2017-09-03 00:00:00'),
(3, 2, 3, '9.99', '2017-09-01 00:00:00', '2017-09-04 00:00:00'),
(4, 2, 2, '19.99', '2017-10-30 00:00:00', '2017-11-02 00:00:00');

--
-- Dumping data for table `Straight`
--

INSERT INTO `Straight` (`toolNumber`, `rubber_feet`) VALUES
(5, NULL),
(6, NULL);

--
-- Dumping data for table `Tool`
--

INSERT INTO `Tool` (`toolNumber`, `type`, `sub_type`, `sub_option`, `power_source`, `material`, `length`, `width`, `weight`, `manufacturer`, `purchase_price`) VALUES
(1, 'Hand Tool', 'Screwdriver', 'Phillips', 'Manual', NULL, 1.5, 2.5, 5, 'Crappy Inc.', '80.00'),
(2, 'Hand Tool', 'Screwdriver', 'Hex', 'Manual', NULL, 4, 3, 90, 'Something Corp.', '39.99'),
(3, 'Garden Tool', 'Digger', 'Edger', 'Manual', NULL, 3, 4, 50, 'Crappy Inc.', '79.99'),
(4, 'Power Tool', 'Drill', 'Driver', 'A/C', NULL, 4, 3, 2, 'Crappy Inc.', '99.99'),
(5, 'Ladder Tool', 'Straight', 'rigid', 'Manual', 'wood', 6.5, 8.5, 6.6, 'home depot', '100.00'),
(6, 'Ladder Tool', 'Straight', 'rigid', 'Manual', 'aluminum', 6.5, 4.5, 6.6, 'home depot', '50.00');

--
-- Dumping data for table `User`
--

INSERT INTO `User` (`userID`, `username`, `password`, `email`, `first_name`, `middle_name`, `last_name`) VALUES
(1, 'jwatson', 'Changeme2', 'jwatson@tools4rent.com', 'Jill', NULL, 'Watson'),
(2, 'clerk1', '1234', 'clerk1@tools4rent.com', 'Test', 'E', 'Clerk'),
(3, 'rviera6', '1234', 'rviera6@gatech.edu', 'Raul', 'E', 'Viera');

--
-- Dumping data for table `WorkPhone`
--

INSERT INTO `WorkPhone` (`userID`, `area_code`, `phone_number`, `extension`) VALUES
(3, '987', '456-7890', NULL);

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;

BEGIN;
INSERT INTO Tool (type, sub_type, sub_option, manufacturer, width, purchase_price, material, weight, power_source, length)
VALUES('Ladder Tool', 'Straight', 'rigid', 'home depot',  8.5, 100 , 'wood', 6.6, 'Manual', 6.5);
INSERT INTO LadderTool (toolNumber, weight_capacity, step_count) VALUES (LAST_INSERT_ID(), 2500.6, 4);
INSERT INTO Straight (toolNumber, rubber_feet) VALUES (LAST_INSERT_ID(), NULL);
COMMIT;


BEGIN;
INSERT INTO Tool (type, sub_type, sub_option, manufacturer, width, purchase_price, material, weight, power_source, length)
VALUES('Ladder Tool', 'Step', 'folding', 'home depot',  8.5, 100 , 'wood', 6.6, 'Manual', 6.5);
INSERT INTO LadderTool (toolNumber, weight_capacity, step_count) VALUES (LAST_INSERT_ID(), 2500.6, 4);
INSERT INTO Step(toolNumber, pail_shelf) VALUES (LAST_INSERT_ID(), NULL);
COMMIT;

BEGIN;
INSERT INTO Tool (type, sub_type, sub_option, manufacturer, width, purchase_price, material, weight, power_source, length)
VALUES('Power Tool', 'Drill', 'driver', 'home depot',  8.5, 100 , 'wood', 6.6, 'A/C', 6.5);
INSERT INTO PowerTool (toolNumber, volt_rating, amp_rating, max_rpm_rating, min_rpm_rating) VALUES(LAST_INSERT_ID(), 110,1.0,NULL, 2000);
INSERT INTO Drill (toolNumber, min_torque_rating, max_torque_rating, adjustable_clutch) VALUES (LAST_INSERT_ID(),500, NULL, false);
INSERT INTO Accessory(toolNumber, description, quantity) VALUES (LAST_INSERT_ID(), 'testing A/C power tool accessory', 1);
COMMIT;


BEGIN;
INSERT INTO Tool (type, sub_type, sub_option, manufacturer, width, purchase_price, material, weight, power_source, length)
VALUES('Power Tool', 'Saw', 'circular', 'home depot',  8.5, 100 , 'wood', 6.6, 'A/C', 6.5);
INSERT INTO PowerTool (toolNumber, volt_rating, amp_rating, max_rpm_rating, min_rpm_rating) VALUES(LAST_INSERT_ID(), 110,1.0,NULL, 2000);
INSERT INTO Saw (toolNumber, blade_size) VALUES (LAST_INSERT_ID(),10);
INSERT INTO Accessory(toolNumber, description, quantity) VALUES (LAST_INSERT_ID(), 'testing A/C power tool accessory', 1);
COMMIT;

BEGIN;
INSERT INTO Tool (type, sub_type, sub_option, manufacturer, width, purchase_price, material, weight, power_source, length)
VALUES('Power Tool', 'Sander', 'finish', 'home depot',  8.5, 100 , 'wood', 6.6, 'A/C', 6.5);
INSERT INTO PowerTool (toolNumber, volt_rating, amp_rating, max_rpm_rating, min_rpm_rating) VALUES(LAST_INSERT_ID(), 110,1.0,NULL, 2000);
INSERT INTO Sander(toolNumber, dust_bag) VALUES(LAST_INSERT_ID(), false);
INSERT INTO Accessory(toolNumber, description, quantity) VALUES (LAST_INSERT_ID(), 'testing A/C power tool accessory', 1);
COMMIT;


BEGIN;
INSERT INTO Tool (type, sub_type, sub_option, manufacturer, width, purchase_price, material, weight, power_source, length)
VALUES('Power Tool', 'Air-Compressor', 'reciprocating', 'home depot',  8.5, 100 , 'wood', 6.6, 'A/C', 6.5);
INSERT INTO PowerTool (toolNumber, volt_rating, amp_rating, max_rpm_rating, min_rpm_rating) VALUES(LAST_INSERT_ID(), 110,1.0,NULL, 2000);
INSERT INTO AirCompressor(toolNumber, tank_size, pressure_rating) VALUES (LAST_INSERT_ID(), 10.0, NULL);
INSERT INTO Accessory(toolNumber, description, quantity) VALUES (LAST_INSERT_ID(), 'testing A/C power tool accessory', 1);
COMMIT;


BEGIN;
INSERT INTO Tool (type, sub_type, sub_option, manufacturer, width, purchase_price, material, weight, power_source, length)
VALUES('Power Tool', 'Mixer', 'concrete', 'home depot',  8.5, 100 , 'wood', 6.6, 'A/C', 6.5);
INSERT INTO PowerTool (toolNumber, volt_rating, amp_rating, max_rpm_rating, min_rpm_rating) VALUES(LAST_INSERT_ID(), 110,1.0,NULL, 2000);
INSERT INTO Mixer(toolNumber, motor_rating, drum_size) VALUES (LAST_INSERT_ID(), 0.5, 3.5);
INSERT INTO Accessory(toolNumber, description, quantity) VALUES (LAST_INSERT_ID(), 'testing A/C power tool accessory', 1);
COMMIT;





BEGIN;
INSERT INTO Tool (type, sub_type, sub_option, manufacturer, width, purchase_price, material, weight, power_source, length)
VALUES('Power Tool', 'Drill', 'driver', 'home depot',  8.5, 100 , 'wood', 6.6, 'D/C', 6.5);
INSERT INTO PowerTool (toolNumber, volt_rating, amp_rating, max_rpm_rating, min_rpm_rating) VALUES(LAST_INSERT_ID(), 110,1.0,NULL, 2000);
INSERT INTO Drill (toolNumber, min_torque_rating, max_torque_rating, adjustable_clutch) VALUES (LAST_INSERT_ID(),500, NULL, false);
INSERT INTO Accessory(toolNumber, description, quantity) VALUES (LAST_INSERT_ID(), 'li-ion', 5);
INSERT INTO Accessory(toolNumber, description, quantity) VALUES (LAST_INSERT_ID(), 'random d/c accessory', 1);
COMMIT;

BEGIN;
INSERT INTO Tool (type, sub_type, sub_option, manufacturer, width, purchase_price, material, weight, power_source, length)
VALUES('Power Tool', 'Saw', 'circular', 'home depot',  8.5, 100 , 'wood', 6.6, 'D/C', 6.5);
INSERT INTO PowerTool (toolNumber, volt_rating, amp_rating, max_rpm_rating, min_rpm_rating) VALUES(LAST_INSERT_ID(), 110,1.0,NULL, 2000);
INSERT INTO Saw (toolNumber, blade_size) VALUES (LAST_INSERT_ID(),10);
INSERT INTO Accessory(toolNumber, description, quantity) VALUES (LAST_INSERT_ID(), 'Li_ion', 5);
INSERT INTO Accessory(toolNumber, description, quantity) VALUES (LAST_INSERT_ID(), 'testing A/C power tool accessory', 1);
COMMIT;

BEGIN;
INSERT INTO Tool (type, sub_type, sub_option, manufacturer, width, purchase_price, material, weight, power_source, length)
VALUES('Power Tool', 'Sander', 'finish', 'home depot',  8.5, 100 , 'wood', 6.6, 'D/C', 6.5);
INSERT INTO PowerTool (toolNumber, volt_rating, amp_rating, max_rpm_rating, min_rpm_rating) VALUES(LAST_INSERT_ID(), 110,1.0,NULL, 2000);
INSERT INTO Sander(toolNumber, dust_bag) VALUES(LAST_INSERT_ID(), false);
INSERT INTO Accessory(toolNumber, description, quantity) VALUES (LAST_INSERT_ID(), 'NiMh', 8);
INSERT INTO Accessory(toolNumber, description, quantity) VALUES (LAST_INSERT_ID(), 'testing A/C power tool accessory', 1);
COMMIT;

BEGIN;
INSERT INTO Tool (type, sub_type, sub_option, manufacturer, width, purchase_price, material, weight, power_source, length)
VALUES('Power Tool', 'Air-Compressor', 'reciprocating', 'home depot',  8.5, 100 , 'wood', 6.6, 'Gas', 6.5);
INSERT INTO PowerTool (toolNumber, volt_rating, amp_rating, max_rpm_rating, min_rpm_rating) VALUES(LAST_INSERT_ID(), 110,1.0,NULL, 2000);
INSERT INTO AirCompressor(toolNumber, tank_size, pressure_rating) VALUES (LAST_INSERT_ID(), 10.0, NULL);
INSERT INTO Accessory(toolNumber, description, quantity) VALUES (LAST_INSERT_ID(), 'testing Gas power tool accessory', 1);
COMMIT;

BEGIN;
INSERT INTO Tool (type, sub_type, sub_option, manufacturer, width, purchase_price, material, weight, power_source, length)
VALUES('Power Tool', 'Mixer', 'concrete', 'home depot',  8.5, 100 , 'wood', 6.6, 'Gas', 6.5);
INSERT INTO PowerTool (toolNumber, volt_rating, amp_rating, max_rpm_rating, min_rpm_rating) VALUES(LAST_INSERT_ID(), 110,1.0,NULL, 2000);
INSERT INTO Mixer(toolNumber, motor_rating, drum_size) VALUES (LAST_INSERT_ID(), 0.5, 3.5);
INSERT INTO Accessory(toolNumber, description, quantity) VALUES (LAST_INSERT_ID(), 'gas tank', 1);
INSERT INTO Accessory(toolNumber, description, quantity) VALUES (LAST_INSERT_ID(), 'testing Gas power tool accessory', 1);
COMMIT;

BEGIN;
INSERT INTO Tool (type, sub_type, sub_option, manufacturer, width, purchase_price, material, weight, power_source, length)
VALUES('Power Tool', 'Generator', 'electric', 'home depot',  8.5, 100 , 'steel', 6.6, 'Gas', 6.5);
INSERT INTO PowerTool (toolNumber, volt_rating, amp_rating, max_rpm_rating, min_rpm_rating) VALUES(LAST_INSERT_ID(), 110,1.0,NULL, 2000);
INSERT INTO Generator(toolNumber, power_rating) VALUES (LAST_INSERT_ID(), 3.5);
INSERT INTO Accessory(toolNumber, description, quantity) VALUES (LAST_INSERT_ID(), 'gas tank', 1);
INSERT INTO Accessory(toolNumber, description, quantity) VALUES (LAST_INSERT_ID(), 'testing Gas power tool accessory', 1);
COMMIT;







BEGIN;
INSERT INTO Tool (type, sub_type, sub_option, manufacturer, width, purchase_price, material, weight, power_source, length)
VALUES('Hand Tool', 'Screwdriver', 'flat', 'home depot',  8.5, 100 , 'wood', 6.6, 'Manual', 6.5);
INSERT INTO HandTool(toolNumber) VALUES (LAST_INSERT_ID());
INSERT INTO ScrewDriver(toolNumber, screw_size) VALUES(LAST_INSERT_ID(),10.8);
COMMIT;

BEGIN;
INSERT INTO Tool (type, sub_type, sub_option, manufacturer, width, purchase_price, material, weight, power_source, length)
VALUES('Hand Tool', 'Socket', 'deep', 'home depot',  8.5, 100 , 'wood', 6.6, 'Manual', 6.5);
INSERT INTO HandTool(toolNumber) VALUES (LAST_INSERT_ID());
INSERT INTO Socket(toolNumber,drive_size, sae_size,deep_socket)
VALUES(LAST_INSERT_ID(),5 ,6.8, true);
COMMIT;

BEGIN;
INSERT INTO Tool (type, sub_type, sub_option, manufacturer, width, purchase_price, material, weight, power_source, length)
VALUES('Hand Tool', 'Ratchett', 'adjustable', 'home depot',  8.5, 100 , 'wood', 6.6, 'Manual', 6.5);
INSERT INTO HandTool(toolNumber) VALUES (LAST_INSERT_ID());
INSERT INTO Ratchet(toolNumber,drive_size)
VALUES(LAST_INSERT_ID(),5);
COMMIT;

BEGIN;
INSERT INTO Tool (type, sub_type, sub_option, manufacturer, width, purchase_price, material, weight, power_source, length)
VALUES('Hand Tool', 'Wrench', 'crescent', 'home depot',  8.5, 100 , 'wood', 6.6, 'Manual', 6.5);
INSERT INTO HandTool(toolNumber) VALUES (LAST_INSERT_ID());
INSERT INTO Wrench(toolNumber,drive_size)
VALUES(LAST_INSERT_ID(),5);
COMMIT;

BEGIN;
INSERT INTO Tool (type, sub_type, sub_option, manufacturer, width, purchase_price, material, weight, power_source, length)
VALUES('Hand Tool', 'Pliers', 'cutting', 'home depot',  8.5, 100 , 'wood', 6.6, 'Manual', 6.5);
INSERT INTO HandTool(toolNumber) VALUES (LAST_INSERT_ID());
INSERT INTO Plier(toolNumber,adjustable)
VALUES(LAST_INSERT_ID(),true);
COMMIT;

BEGIN;
INSERT INTO Tool (type, sub_type, sub_option, manufacturer, width, purchase_price, material, weight, power_source, length)
VALUES('Hand Tool', 'Hammer', 'claw', 'home depot',  8.5, 100 , 'wood', 6.6, 'Manual', 6.5);
INSERT INTO HandTool(toolNumber) VALUES (LAST_INSERT_ID());
INSERT INTO Hammer(toolNumber,anti_vibration) VALUES(LAST_INSERT_ID(), true);
COMMIT;

BEGIN;
INSERT INTO Tool (type, sub_type, sub_option, manufacturer, width, purchase_price, material, weight, power_source, length)
VALUES('Garden Tool', 'Pruner', 'sheer', 'home depot',  8.5, 100 , 'wood', 6.6, 'Manual', 6.5);
INSERT INTO GardenTool(toolNumber, handle_material) VALUES (LAST_INSERT_ID(), 'aluminum');
INSERT INTO Prunning(toolNumber,blade_material, blade_length) VALUES(LAST_INSERT_ID(),'metal', 5.0);
COMMIT;

BEGIN;
INSERT INTO Tool (type, sub_type, sub_option, manufacturer, width, purchase_price, material, weight, power_source, length)
VALUES('Garden Tool', 'Rakes', 'leaf', 'home depot',  8.5, 100 , 'wood', 6.6, 'Manual', 6.5);
INSERT INTO GardenTool(toolNumber, handle_material) VALUES (LAST_INSERT_ID(), 'aluminum');
INSERT INTO Rake(toolNumber, tine_count) VALUES(LAST_INSERT_ID(), 6);
COMMIT;

BEGIN;
INSERT INTO Tool (type, sub_type, sub_option, manufacturer, width, purchase_price, material, weight, power_source, length)
VALUES('Garden Tool', 'Wheelbarrows', '1-wheel', 'home depot',  8.5, 100 , 'wood', 6.6, 'Manual', 6.5);
INSERT INTO GardenTool(toolNumber, handle_material) VALUES (LAST_INSERT_ID(), 'steel');
INSERT INTO WheelBarrow(toolNumber,bin_material, bin_volume, wheel_count) VALUES(LAST_INSERT_ID(), 'metal', 6.5, 9);
COMMIT;


BEGIN;
INSERT INTO Tool (type, sub_type, sub_option, manufacturer, width, purchase_price, material, weight, power_source, length)
VALUES('Garden Tool', 'Striking', 'bar pry', 'home depot',  8.5, 100 , 'wood', 6.6, 'Manual', 6.5);
INSERT INTO GardenTool(toolNumber, handle_material) VALUES (LAST_INSERT_ID(), 'aluminum');
INSERT INTO Striking(toolNumber, head_weight) VALUES(LAST_INSERT_ID(), 1.0);
COMMIT;


INSERT INTO `SaleOrder` (`saleOrderID`, `clerkUserID`, `customerUserID`, `toolNumber`, `for_sale_date`, `sold_date`) VALUES (NULL, '2', NULL, '1', '2017-10-15 00:00:00', NULL);

-- insert service order request ---

INSERT INTO `ServiceOrderRequest` (`serviceOrderID`, `userID`, `toolNumber`, `cost`, `start_date`, `end_date`) VALUES (NULL, '2', '2', '10.00', '2017-10-15 00:00:00', '2017-10-18 00:00:00');
INSERT INTO `ServiceOrderRequest` (`serviceOrderID`, `userID`, `toolNumber`, `cost`, `start_date`, `end_date`) VALUES (NULL, '2', '7', '10.00', '2017-12-15 00:00:00', '2017-12-18 00:00:00');

INSERT INTO ToolTypeOption(tt_name, tst_name, tso_name, ps_name) VALUES ('Hand Tool','Screwdriver','phillips (cross)', 'Manual');
INSERT INTO ToolTypeOption(tt_name, tst_name, tso_name, ps_name) VALUES ('Hand Tool','Screwdriver','hex', 'Manual');
INSERT INTO ToolTypeOption(tt_name, tst_name, tso_name, ps_name) VALUES ('Hand Tool','Screwdriver','torx', 'Manual');
INSERT INTO ToolTypeOption(tt_name, tst_name, tso_name, ps_name) VALUES ('Hand Tool','Screwdriver','slotted (flat)', 'Manual');

INSERT INTO ToolTypeOption(tt_name, tst_name, tso_name, ps_name) VALUES ('Hand Tool','Socket','deep', 'Manual');
INSERT INTO ToolTypeOption(tt_name, tst_name, tso_name, ps_name) VALUES ('Hand Tool','Socket','standard', 'Manual');

INSERT INTO ToolTypeOption(tt_name, tst_name, tso_name, ps_name) VALUES ('Hand Tool','Ratchet','adjustable', 'Manual');
INSERT INTO ToolTypeOption(tt_name, tst_name, tso_name, ps_name) VALUES ('Hand Tool','Ratchet','fixed', 'Manual');

INSERT INTO ToolTypeOption(tt_name, tst_name, tso_name, ps_name) VALUES ('Hand Tool','Wrench','crescent', 'Manual');
INSERT INTO ToolTypeOption(tt_name, tst_name, tso_name, ps_name) VALUES ('Hand Tool','Wrench','torque', 'Manual');
INSERT INTO ToolTypeOption(tt_name, tst_name, tso_name, ps_name) VALUES ('Hand Tool','Wrench','pipe', 'Manual');

INSERT INTO ToolTypeOption(tt_name, tst_name, tso_name, ps_name) VALUES ('Hand Tool','Pliers','needle nose', 'Manual');
INSERT INTO ToolTypeOption(tt_name, tst_name, tso_name, ps_name) VALUES ('Hand Tool','Pliers','cutting', 'Manual');
INSERT INTO ToolTypeOption(tt_name, tst_name, tso_name, ps_name) VALUES ('Hand Tool','Pliers','crimper', 'Manual');

INSERT INTO ToolTypeOption(tt_name, tst_name, tso_name, ps_name) VALUES ('Hand Tool','Gun','nail', 'Manual');
INSERT INTO ToolTypeOption(tt_name, tst_name, tso_name, ps_name) VALUES ('Hand Tool','Gun','staple', 'Manual');

INSERT INTO ToolTypeOption(tt_name, tst_name, tso_name, ps_name) VALUES ('Hand Tool','Hammer','claw', 'Manual');
INSERT INTO ToolTypeOption(tt_name, tst_name, tso_name, ps_name) VALUES ('Hand Tool','Pliers','sledge', 'Manual');
INSERT INTO ToolTypeOption(tt_name, tst_name, tso_name, ps_name) VALUES ('Hand Tool','Pliers','framing', 'Manual');

INSERT INTO ToolTypeOption(tt_name, tst_name, tso_name, ps_name) VALUES ('Garden Tool','Digger','pointed shovel', 'Manual');
INSERT INTO ToolTypeOption(tt_name, tst_name, tso_name, ps_name) VALUES ('Garden Tool','Digger','flat shovel', 'Manual');
INSERT INTO ToolTypeOption(tt_name, tst_name, tso_name, ps_name) VALUES ('Garden Tool','Digger','scoop shovel', 'Manual');
INSERT INTO ToolTypeOption(tt_name, tst_name, tso_name, ps_name) VALUES ('Garden Tool','Digger','edger', 'Manual');

INSERT INTO ToolTypeOption(tt_name, tst_name, tso_name, ps_name) VALUES ('Garden Tool','Pruner','sheer', 'Manual');
INSERT INTO ToolTypeOption(tt_name, tst_name, tso_name, ps_name) VALUES ('Garden Tool','Pruner','loppers', 'Manual');
INSERT INTO ToolTypeOption(tt_name, tst_name, tso_name, ps_name) VALUES ('Garden Tool','Pruner','hedge', 'Manual');

INSERT INTO ToolTypeOption(tt_name, tst_name, tso_name, ps_name) VALUES ('Garden Tool','Rakes','leaf', 'Manual');
INSERT INTO ToolTypeOption(tt_name, tst_name, tso_name, ps_name) VALUES ('Garden Tool','Rakes','landscaping', 'Manual');
INSERT INTO ToolTypeOption(tt_name, tst_name, tso_name, ps_name) VALUES ('Garden Tool','Rakes','rock', 'Manual');

INSERT INTO ToolTypeOption(tt_name, tst_name, tso_name, ps_name) VALUES ('Garden Tool','Wheelbarrows','1-wheel', 'Manual');
INSERT INTO ToolTypeOption(tt_name, tst_name, tso_name, ps_name) VALUES ('Garden Tool','Wheelbarrows','2-wheel', 'Manual');

INSERT INTO ToolTypeOption(tt_name, tst_name, tso_name, ps_name) VALUES ('Garden Tool','Striking','bar pry', 'Manual');
INSERT INTO ToolTypeOption(tt_name, tst_name, tso_name, ps_name) VALUES ('Garden Tool','Striking','rubber mallet', 'Manual');
INSERT INTO ToolTypeOption(tt_name, tst_name, tso_name, ps_name) VALUES ('Garden Tool','Striking','tamper', 'Manual');
INSERT INTO ToolTypeOption(tt_name, tst_name, tso_name, ps_name) VALUES ('Garden Tool','Striking','pick axe', 'Manual');
INSERT INTO ToolTypeOption(tt_name, tst_name, tso_name, ps_name) VALUES ('Garden Tool','Striking','single bit axe', 'Manual');

INSERT INTO ToolTypeOption(tt_name, tst_name, tso_name, ps_name) VALUES ('Ladder Tool','Straight','rigid', 'Manual');
INSERT INTO ToolTypeOption(tt_name, tst_name, tso_name, ps_name) VALUES ('Ladder Tool','Straight','telescoping', 'Manual');

INSERT INTO ToolTypeOption(tt_name, tst_name, tso_name, ps_name) VALUES ('Ladder Tool','Step','folding', 'Manual');
INSERT INTO ToolTypeOption(tt_name, tst_name, tso_name, ps_name) VALUES ('Ladder Tool','Step','multi-position', 'Manual');

INSERT INTO ToolTypeOption(tt_name, tst_name, tso_name, ps_name) VALUES ('Power Tool','Drill','driver', 'A/C');
INSERT INTO ToolTypeOption(tt_name, tst_name, tso_name, ps_name) VALUES ('Power Tool','Drill','driver', 'D/C');
INSERT INTO ToolTypeOption(tt_name, tst_name, tso_name, ps_name) VALUES ('Power Tool','Drill','hammer', 'A/C');
INSERT INTO ToolTypeOption(tt_name, tst_name, tso_name, ps_name) VALUES ('Power Tool','Drill','hammer', 'D/C');

INSERT INTO ToolTypeOption(tt_name, tst_name, tso_name, ps_name) VALUES ('Power Tool','Saw','circular', 'A/C');
INSERT INTO ToolTypeOption(tt_name, tst_name, tso_name, ps_name) VALUES ('Power Tool','Saw','circular', 'D/C');
INSERT INTO ToolTypeOption(tt_name, tst_name, tso_name, ps_name) VALUES ('Power Tool','Saw','reciprocating', 'A/C');
INSERT INTO ToolTypeOption(tt_name, tst_name, tso_name, ps_name) VALUES ('Power Tool','Saw','reciprocating', 'D/C');
INSERT INTO ToolTypeOption(tt_name, tst_name, tso_name, ps_name) VALUES ('Power Tool','Saw','jig', 'A/C');
INSERT INTO ToolTypeOption(tt_name, tst_name, tso_name, ps_name) VALUES ('Power Tool','Saw','jig', 'D/C');

INSERT INTO ToolTypeOption(tt_name, tst_name, tso_name, ps_name) VALUES ('Power Tool','Sander','finish', 'A/C');
INSERT INTO ToolTypeOption(tt_name, tst_name, tso_name, ps_name) VALUES ('Power Tool','Sander','finish', 'D/C');
INSERT INTO ToolTypeOption(tt_name, tst_name, tso_name, ps_name) VALUES ('Power Tool','Sander','sheet', 'A/C');
INSERT INTO ToolTypeOption(tt_name, tst_name, tso_name, ps_name) VALUES ('Power Tool','Sander','sheet', 'D/C');
INSERT INTO ToolTypeOption(tt_name, tst_name, tso_name, ps_name) VALUES ('Power Tool','Sander','belt', 'A/C');
INSERT INTO ToolTypeOption(tt_name, tst_name, tso_name, ps_name) VALUES ('Power Tool','Sander','belt', 'D/C');
INSERT INTO ToolTypeOption(tt_name, tst_name, tso_name, ps_name) VALUES ('Power Tool','Sander','random orbital', 'A/C');
INSERT INTO ToolTypeOption(tt_name, tst_name, tso_name, ps_name) VALUES ('Power Tool','Sander','random orbital', 'D/C');

INSERT INTO ToolTypeOption(tt_name, tst_name, tso_name, ps_name) VALUES ('Power Tool','Air-Compressor','reciprocating', 'A/C');
INSERT INTO ToolTypeOption(tt_name, tst_name, tso_name, ps_name) VALUES ('Power Tool','Air-Compressor','reciprocating', 'Gas');
INSERT INTO ToolTypeOption(tt_name, tst_name, tso_name, ps_name) VALUES ('Power Tool','Mixer','concrete', 'A/C');
INSERT INTO ToolTypeOption(tt_name, tst_name, tso_name, ps_name) VALUES ('Power Tool','Mixer','concrete', 'Gas');
INSERT INTO ToolTypeOption(tt_name, tst_name, tso_name, ps_name) VALUES ('Power Tool','Generator','electric', 'Gas');


BEGIN;
INSERT INTO User(userID, username, password, email, first_name, middle_name, last_name) VALUES (4, 'spalguyev3', '1234', 'spalguyev3@gatech.edu', 'Sergiy', 'I', 'Palguyev');
INSERT INTO Customer(userID, zip_code, street, city, state) VALUES (4, '98765-123', '999 Ga Tech Ave.', 'Atlanta', 'GA');
INSERT INTO CreditCard(userID, cred_number, name, cvc, exp_month, exp_year) VALUES (4, '1234567890', 'Sergiy Palguyev', '098', '4', '2022');
INSERT INTO CellPhone(userID, area_code, phone_number, extension) VALUES (4, '777', '4320000', NULL);
INSERT INTO PrimaryPhone(userID, area_code, phone_number, extension) VALUES (4, '777', '4320000', NULL);
COMMIT;

BEGIN;
INSERT INTO User(userID, username, password, email, first_name, middle_name, last_name) VALUES (5, 'adang33', '1234', 'adang33@gatech.edu', 'Tuan', 'L', 'Dang');
INSERT INTO Customer(userID, zip_code, street, city, state) VALUES (5, '98765-123', '123 Ga Tech Ave.', 'Atlanta', 'GA');
INSERT INTO CreditCard(userID, cred_number, name, cvc, exp_month, exp_year) VALUES (5, '0987654321', 'Tuan Dang', '543', '2', '2019');
INSERT INTO HomePhone(userID, area_code, phone_number, extension) VALUES (5, '777', '9874321', '3214');
INSERT INTO PrimaryPhone(userID, area_code, phone_number, extension) VALUES (5, '777', '9874321', '3214');
COMMIT;

BEGIN;
INSERT INTO User(userID, username, password, email, first_name, middle_name, last_name) VALUES (6, 'nanandan3', '1234', 'nanandan3@gatech.edu', 'Nitesh', NULL, 'Anandan');
INSERT INTO Customer(userID, zip_code, street, city, state) VALUES (6, '98765-123', '543 Ga Tech Ave.', 'Atlanta', 'GA');
INSERT INTO CreditCard(userID, cred_number, name, cvc, exp_month, exp_year) VALUES (6, '9328746812', 'Nitesh Anandan', '765', '8', '2025');
INSERT INTO WorkPhone(userID, area_code, phone_number, extension) VALUES (6, '777', '8761234', '6734');
INSERT INTO PrimaryPhone(userID, area_code, phone_number, extension) VALUES (6, '777', '8761234', '6734');
COMMIT;

BEGIN;
INSERT INTO User(userID, username, password, email, first_name, middle_name, last_name) VALUES (10, 'bacon1', '1234', 'bacon1@hungry.com', 'Chris', 'P', 'Bacon');
INSERT INTO Customer(userID, zip_code, street, city, state) VALUES (10, '65438-999', '333 Food Truck St.', 'San Francisco', 'CA');
INSERT INTO CreditCard(userID, cred_number, name, cvc, exp_month, exp_year) VALUES (10, '675123456709', 'Chris P. Bacon', '765', '8', '2025');
INSERT INTO WorkPhone(userID, area_code, phone_number, extension) VALUES (10, '777', '5551234', '6734');
INSERT INTO PrimaryPhone(userID, area_code, phone_number, extension) VALUES (10, '777', '5551234', '6734');
COMMIT;

BEGIN;
INSERT INTO User(userID, username, password, email, first_name, middle_name, last_name) VALUES (7, 'tjohnson306', '1234', 'tjohnson306@tools4rent.com', 'Terry', 'W', 'Johnson');
INSERT INTO Clerk(userID, emp_num, date_hired, has_logged_in_before) VALUES (7, 7, '2017-11-01 00:00:00', 0);
COMMIT;

BEGIN;
INSERT INTO User(userID, username, password, email, first_name, middle_name, last_name) VALUES (8, 'mark1', '9876', 'mark1@tools4rent.com', 'John', 'T', 'Mark');
INSERT INTO Clerk(userID, emp_num, date_hired, has_logged_in_before) VALUES (8, 8, '2017-11-01 00:00:00', 0);
COMMIT;


BEGIN;
INSERT INTO Reservation(customerUserID, start_date, end_date, pickUpUserID, dropOffUserID) VALUES (4, '2017-11-01 10:30:00', '2017-11-03 10:30:00', 7, 8);
INSERT INTO IsOf(reservationID, toolNumber) VALUES (LAST_INSERT_ID(), 10);
INSERT INTO IsOf(reservationID, toolNumber) VALUES (LAST_INSERT_ID(), 11);
INSERT INTO IsOf(reservationID, toolNumber) VALUES (LAST_INSERT_ID(), 12);
COMMIT;

BEGIN;
INSERT INTO Reservation(customerUserID, start_date, end_date, pickUpUserID, dropOffUserID) VALUES (5, '2017-11-01 10:30:00', '2017-11-05 10:30:00', 8, 7);
INSERT INTO IsOf(reservationID, toolNumber) VALUES (LAST_INSERT_ID(), 22);
INSERT INTO IsOf(reservationID, toolNumber) VALUES (LAST_INSERT_ID(), 25);
INSERT INTO IsOf(reservationID, toolNumber) VALUES (LAST_INSERT_ID(), 18);
COMMIT;

BEGIN;
INSERT INTO Reservation(customerUserID, start_date, end_date, pickUpUserID, dropOffUserID) VALUES (6, '2017-11-03 10:31:00', '2017-11-08 10:31:00', 8, 8);
INSERT INTO IsOf(reservationID, toolNumber) VALUES (LAST_INSERT_ID(), 10);
INSERT INTO IsOf(reservationID, toolNumber) VALUES (LAST_INSERT_ID(), 11);
INSERT INTO IsOf(reservationID, toolNumber) VALUES (LAST_INSERT_ID(), 12);
COMMIT;

BEGIN;
INSERT INTO Reservation(customerUserID, start_date, end_date, pickUpUserID, dropOffUserID) VALUES (4, '2017-11-08 10:30:00', '2017-11-13 10:30:00', 7, NULL);
INSERT INTO IsOf(reservationID, toolNumber) VALUES (LAST_INSERT_ID(), 20);
COMMIT;

BEGIN;
INSERT INTO Reservation(customerUserID, start_date, end_date, pickUpUserID, dropOffUserID) VALUES (4, '2017-11-11 10:30:00', '2017-11-13 10:30:00', 7, NULL);
INSERT INTO IsOf(reservationID, toolNumber) VALUES (LAST_INSERT_ID(), 21);
INSERT INTO IsOf(reservationID, toolNumber) VALUES (LAST_INSERT_ID(), 22);
INSERT INTO IsOf(reservationID, toolNumber) VALUES (LAST_INSERT_ID(), 29);
INSERT INTO IsOf(reservationID, toolNumber) VALUES (LAST_INSERT_ID(), 12);
COMMIT;

BEGIN;
INSERT INTO Reservation(customerUserID, start_date, end_date, pickUpUserID, dropOffUserID) VALUES (5, '2017-11-11 10:30:00', '2017-11-15 10:30:00', 7, 2);
INSERT INTO IsOf(reservationID, toolNumber) VALUES (LAST_INSERT_ID(), 19);
INSERT INTO IsOf(reservationID, toolNumber) VALUES (LAST_INSERT_ID(), 17);
COMMIT;

BEGIN;
INSERT INTO Reservation(customerUserID, start_date, end_date, pickUpUserID, dropOffUserID) VALUES (6, '2017-11-13 10:30:00', '2017-11-17 10:30:00', NULL, NULL);
INSERT INTO IsOf(reservationID, toolNumber) VALUES (LAST_INSERT_ID(), 23);
COMMIT;

BEGIN;
INSERT INTO Reservation(customerUserID, start_date, end_date, pickUpUserID, dropOffUserID) VALUES (3, '2017-11-20 01:30:00', '2017-11-25 01:30:00', NULL, NULL);
INSERT INTO IsOf(reservationID, toolNumber) VALUES (LAST_INSERT_ID(), 21);
INSERT INTO IsOf(reservationID, toolNumber) VALUES (LAST_INSERT_ID(), 25);
INSERT INTO IsOf(reservationID, toolNumber) VALUES (LAST_INSERT_ID(), 18);
INSERT INTO IsOf(reservationID, toolNumber) VALUES (LAST_INSERT_ID(), 26);
INSERT INTO IsOf(reservationID, toolNumber) VALUES (LAST_INSERT_ID(), 23);
COMMIT;
