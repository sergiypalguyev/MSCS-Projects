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

INSERT INTO `GardenTool` (`toolNumber`) VALUES
(3);

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
(32, 5);

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
(14, 3, '2017-11-01 00:00:00', '2017-11-05 00:00:00', NULL, NULL),
(32, 3, '2017-11-01 10:00:00', '2017-11-03 09:59:59', NULL, NULL);

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
(3, 'Garden', 'Digger', 'Edger', 'Manual', NULL, 3, 4, 50, 'Crappy Inc.', '79.99'),
(4, 'Power Source', 'Drill', 'Driver', 'A/C', NULL, 4, 3, 2, 'Crappy Inc.', '99.99'),
(5, 'Ladder Tool', 'Straight', 'rigid', 'Manual', 'wood', 6.5, 8.5, 6.6, 'home depot', '100.00'),
(6, 'Ladder Tool', 'Straight', 'rigid', 'Manual', 'aluminum', 6.5, 4.5, 6.6, 'home depot', '50.00');

--
-- Dumping data for table `User`
--

INSERT INTO `User` (`userID`, `username`, `password`, `email`, `first_name`, `middle_name`, `last_name`) VALUES
(1, 'jwatson', 'Changeme2', 'jwatson@tools4rent.com', 'Jill', NULL, 'Watson'),
(2, 'clerk1', '1234', 'clerk1@gatech.edu', 'Test', 'E', 'Clerk'),
(3, 'rviera6', '1234', 'rviera6@gatech.edu', 'Raul', 'E', 'Viera');

--
-- Dumping data for table `WorkPhone`
--

INSERT INTO `WorkPhone` (`userID`, `area_code`, `phone_number`, `extension`) VALUES
(3, '987', '456-7890', NULL);

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
