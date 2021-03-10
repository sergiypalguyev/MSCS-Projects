1.A grocery list consists of items the users want to buy at a grocery store. The application must allow users to add items to a list, delete items from a list, and change the quantity of items in the list (e.g., change from one to two pounds of apples).
~ Two classes must be created: GroceryList and Item. Grocery store is not considered.
~ GroceryList has three operations: addItem, deleteItem, updateQuantity
2.The application must contain a database (DB) of items and corresponding item types.
~ Database is modeled as another class
~ Database has an attribute Item with an internal attribute ItemType
3.Users must be able to add items to a list by picking them from a hierarchical list, where the first level is the item type (e.g., cereal), and the second level is the name of the actual item (e.g., shredded wheat). After adding an item, users must be able to specify a quantity for that item.
~ Searching interface from user to a hierarchal list composed of item names and item types.
~ I assume Heirarchal List gets the item information from the database.
4.Users must also be able to specify an item by typing its name. In this case, the application must look in its DB for items with similar names and ask the users, for each of them, whether that is the item they intended to add. If a match cannot be found, the application must ask the user to select a type for the item and then save the new item, together with its type, in its DB.
~ DB can add items
~ DB contains items abel to be queried by user, same as Heirarchal list.
5.Lists must be saved automatically and immediately after they are modified.
~ Operation of the list only. Does not state where to save the list to.
~ “automatically and immediately” is not considered as the save will be triggered within the implementation by every GroceryList operation or associated function.
6.Users must be able to check off items in a list (without deleting them).
~ Association between user and GroceryList.
7.Users must also be able to clear all the check-off marks in a list at once.
~ Special use of the check-off association. 
~ “at once” is not considered as the update will trigger automatically as soon as the association is invoked.
8.Check-off marks for a list are persistent and must also be saved immediately.
~ Checked should be an attribute of the item in order to be saved with the list.
9.The application must present the items in a list grouped by type, so as to allow users to shop for a specific type of products at once (i.e., without having to go back and forth between aisles).
~ Not considered because it does not affect design directly. This is stylistic GUI design.
10.The application must support multiple lists at a time (e.g., “weekly grocery list”, “monthly farmer’s market list”). Therefore, the application must provide the users with the ability to create, (re)name, select, and delete lists.
~ Functions of the User only. 
11.The User Interface (UI) must be intuitive and responsive.
~ Not considered because it does not affect design directly.