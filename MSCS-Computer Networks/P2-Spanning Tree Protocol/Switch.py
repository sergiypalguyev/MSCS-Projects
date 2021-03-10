# Project 2 for OMS6250
#
# This defines a Switch that can can send and receive spanning tree 
# messages to converge on a final loop free forwarding topology.  This
# class is a child class (specialization) of the StpSwitch class.  To 
# remain within the spirit of the project, the only inherited members
# functions the student is permitted to use are:
#
# self.switchID                   (the ID number of this switch object)
# self.links                      (the list of swtich IDs connected to this switch object)
# self.send_message(Message msg)  (Sends a Message object to another switch)
#
# Student code MUST use the send_message function to implement the algorithm - 
# a non-distributed algorithm will not receive credit.
#
# Student code should NOT access the following members, otherwise they may violate
# the spirit of the project:
#
# topolink (parameter passed to initialization function)
# self.topology (link to the greater topology structure used for message passing)
#
# Copyright 2016 Michael Brown, updated by Kelly Parks
#           Based on prior work by Sean Donovan, 2015
			    												

from Message import *
from StpSwitch import *

class Switch(StpSwitch):

    # Contain all necessary data in a "struct" or object of spanningTreeLinks.

    # Create the variables and list per project requirements.
    # These are kept global for now so can be accessed in any function.

    i_rootSwitchID = 0
    i_distanceToSwitchRoot = 0
    # ActiveLinks list should be a dictionary type.
    # Lists are good, but Dictionary cna contain the switchId and an T/F Active state
    activeLinkDict = {}
    i_throughNeighbor = None # This is NULL so we do not assume 0 is a through-switch

    def __init__(self, idNum, topolink, neighbors):    
        # Invoke the super class constructor, which makes available to this object the following members:
        # -self.switchID                   (the ID number of this switch object)
        # -self.links                      (the list of swtich IDs connected to this switch object)
        super(Switch, self).__init__(idNum, topolink, neighbors)
        
        #TODO: Define a data structure to keep track of which links are part of / not part of the spanning tree.

        self.i_rootSwitchID = idNum
        self.activeLinkDict = {}
        # Initally, set all links to assume they are active -> TRUE
        for i in neighbors:
            self.activeLinkDict[i] = True
        # Do not want to do anything with topoLink
        # Leave the rest of the params as default.

    def send_initial_messages(self):
        #TODO: This function needs to create and send the initial messages from this switch.
        #      Messages are sent via the superclass method send_message(Message msg) - see Message.py.
        #      Use self.send_message(msg) to send this.  DO NOT use self.topology.send_message(msg)

        # Initial message is broadcasted to all links
        # From Message.py, Message root is the rootId
        # Distance for initial message is 0, for all links.
        # Origin is this switch
        # Destination is the link in the links list.
        # pathThrough is False for initial message, no idea which neighbor is pathThrough

        [self.send_message(Message(self.i_rootSwitchID, 0, self.switchID, i, False)) for i in self.links]

    def process_message(self, message):
        #TODO: This function needs to accept an incoming message and process it accordingly.
        #      This function is called every time the switch receives a new message.

        # Check for any irregular or obviously problematic message types here.
        if message.verify_message() == False:
            #print ("Message_verify FAILED.")
            return
        if (message.root > self.i_rootSwitchID):
            #print ("Message root is greater than root switch ID")
            return
        # a. Determine whether an update to the switchs root information is necessary and update accordingly.
        ## i. The switch should update the root stored in its data structure if it receives a message with
        ##      a lower claimedRoot.
        # root updated so broadcast to neighboars.
        if (message.root < self.i_rootSwitchID):
            self.i_rootSwitchID = message.root
            # The switch should update the distance stored in its data structure if
            # a) the switch updates the root
            self.i_distanceToSwitchRoot = message.distance+1
            self.broadcastToNeighbors(message, True)
            return
        ## ii.b. The switch should update the distance stored in its data structure if
        ##      b) there is a shorter path to the same root.
        # update everything but root if root is the same & distance is shorter
        # distance updated so broadcast to neighboars.
        if (message.distance + 1 < self.i_distanceToSwitchRoot) :
            self.i_distanceToSwitchRoot = message.distance + 1
            self.broadcastToNeighbors(message, True)
            return
        # message distance greater than current distanceToRoot
        elif message.distance + 1 > self.i_distanceToSwitchRoot:
            if message.pathThrough :
                self.activeLinkDict[message.origin] = True
            else :
                self.activeLinkDict[message.origin] = False
            return
        # add to activeLinks if through path and same root
        # i. The switch finds a new path to the root (through a different neighbor). In this
        # case, the switch should add the new link to activeLinks and (potentially) remove
        # the old link from activeLinks
        # New active links so broadcast to neighbors.
        if message.origin < self.i_throughNeighbor :
            temp  = self.i_throughNeighbor
            # Record new active link
            self.i_throughNeighbor = message.origin
            self.activeLinkDict[self.i_throughNeighbor] = True
            #Delete the old Active link
            self.activeLinkDict[temp] = False
            self.activeLinkDict.pop(temp)
            self.broadcastToNeighbors(message, True)
        # The switch found a longer path, deactivate Link.
        elif message.origin > self.i_throughNeighbor:
            self.activeLinkDict[message.origin] = False
            self.broadcastToNeighbors(message, False)

    def broadcastToNeighbors(self, message, activateLink):
        if activateLink:
            self.i_throughNeighbor = message.origin
            self.activeLinkDict[message.origin] = True
        else:
            self.activeLinkDict[message.origin] = False
        [self.send_message(Message(self.i_rootSwitchID, self.i_distanceToSwitchRoot, self.switchID, i, i == self.i_throughNeighbor)) for i in self.links]

    def generate_logstring(self):
        #TODO: This function needs to return a logstring for this particular switch.  The
        #      string represents the active forwarding links for this switch and is invoked 
        #      only after the simulaton is complete.  Output the links included in the 
        #      spanning tree by increasing destination switch ID on a single line. 
        #      Print links as '(source switch id) - (destination switch id)', separating links 
        #      with a comma - ','.  
        #
        #      For example, given a spanning tree (1 ----- 2 ----- 3), a correct output string 
        #      for switch 2 would have the following text:
        #      2 - 1, 2 - 3
        #      A full example of a valid output file is included (sample_output.txt) with the project skeleton.


        temp = sorted(self.activeLinkDict.keys())
        sortedActive = []
        for i in temp:
            if self.activeLinkDict[i]:
                sortedActive.append(i)

        log = ""
        addComma = 0
        for i in sortedActive:
            if 0 < addComma < len(sortedActive):
                log+= ", "
            log += "{} - {}".format(self.switchID, i)
            addComma += 1

        #print (log)
        return log