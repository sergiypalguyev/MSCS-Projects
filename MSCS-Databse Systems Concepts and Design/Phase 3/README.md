# VM Configuration Changes
Follow this steps to allow remote connections in MySQL. This will allow to develop on the local machine rather than on the VM.


### Change the Network settings on the VM configuration:
	Go to Select VM -> Settings -> Network
	Attached to: Bridged Adapter
	Name: the name of your NIC


### Allow MySQL to allow remote connections:
	sudo vim /etc/mysql/mysql.conf.d/mysqld.cnf
	
	Change the bind-address to look like this:
	bind-address            = 0.0.0.0


### Restart MySQL
	systemctl restart mysql.service
	
	
### Import DB Schema
	Import the DB schema from Phase 3 into MySQL
	Schema filename: team073_p3_schema.sql
	
### Import DB Seed Data
	Import the DB seed data from Phase 3 into MySQL
	Data filename: cs6400_fa17_team073_data.sql
	* When importing make sure to uncheck the "Enable foreign key checks"


	

