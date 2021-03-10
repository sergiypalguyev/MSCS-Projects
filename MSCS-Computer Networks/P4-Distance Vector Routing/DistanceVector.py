# Project 4 for CS 6250: Computer Networks
#
# This defines a DistanceVector (specialization of the Node class)
# that can run the Bellman-Ford algorithm. The TODOs are all related
# to implementing BF. Students should modify this file as necessary,
# guided by the TODO comments and the assignment instructions. This
# is the only file that needs to be modified to complete the project.
#
# Student code should NOT access the following members, otherwise they may violate
# the spirit of the project:
#
# topolink (parameter passed to initialization function)
# self.topology (link to the greater topology structure used for message passing)
#
# Copyright 2017 Michael D. Brown
# Based on prior work by Dave Lillethun, Sean Donovan, and Jeffrey Randow.

from Node import *
from helpers import *


class DistanceVector(Node):

    def __init__(self, name, topolink, outgoing_links, incoming_links):
        ''' Constructor. This is run once when the DistanceVector object is
        created at the beginning of the simulation. Initializing data structure(s)
        specific to a DV node is done here.'''

        super(DistanceVector, self).__init__(name, topolink, outgoing_links, incoming_links)

        # TODO: Create any necessary data structure(s) to contain the Node's internal state / distance vector data

        self.sender = name
        self.inNodes = incoming_links
        self.outNodes = outgoing_links
        self.dist = {}
        self.dist[self.sender] = 0

    def send_initial_messages(self):
        ''' This is run once at the beginning of the simulation, after all
        DistanceVector objects are created and their links to each other are
        established, but before any of the rest of the simulation begins. You
        can have nodes send out their initial DV advertisements here.

        Remember that links points to a list of Neighbor data structure.  Access
        the elements with .name or .weight '''

        # TODO - Each node needs to build a message and send it to each of its neighbors
        # HINT: Take a look at the skeleton methods provided for you in Node.py

        [self.send_msg((self.sender, self.dist), node.name) for node in self.inNodes]

    def ProcessRootToNewNodeDistance(self, node, msg):
        for link in self.outNodes:
            # Adjust weight depending if path is direct or indirect from root to node.

            weight = int(self.get_outgoing_neighbor_weight(node)) \
                if node == link.name \
                else int(self.get_outgoing_neighbor_weight(msg[0])) + int(msg[1][node])

            self.dist[node] = weight
            return True
        return False

    def ProcessRootToKnownNodeDistanceUpdate(self, node, msg):
        #Update distances and deal with infinite tun-time

        RootToNode = int(self.get_outgoing_neighbor_weight(msg[0]))
        NodeToDest = int(msg[1][node])
        totalDistance = RootToNode + NodeToDest

        CONST_NEGATIVE_INFINITY = -99
        if (RootToNode <= CONST_NEGATIVE_INFINITY \
            or NodeToDest <= CONST_NEGATIVE_INFINITY \
            or totalDistance <= CONST_NEGATIVE_INFINITY) \
                and self.dist[node] != CONST_NEGATIVE_INFINITY:
            self.dist[node] = CONST_NEGATIVE_INFINITY
            return True
        elif totalDistance < self.dist[node]:
            self.dist[node] = totalDistance \
                if totalDistance > CONST_NEGATIVE_INFINITY \
                else CONST_NEGATIVE_INFINITY
            return True
        return False

    def ProcessMessageForNode(self, keepSending, destNode, msg):
        return (keepSending | self.ProcessRootToNewNodeDistance(destNode, msg) \
                if destNode not in self.dist \
                else keepSending | self.ProcessRootToKnownNodeDistanceUpdate(destNode, msg))

    def process_BF(self):
        ''' This is run continuously (repeatedly) during the simulation. DV
        messages from other nodes are received here, processed, and any new DV
        messages that need to be sent to other nodes as a result are sent. '''

        keepSending = False
        for msg in self.messages:
            for destNode in msg[1].keys():
                if destNode == self.name:
                    continue
                keepSending = self.ProcessMessageForNode(keepSending, destNode, msg)

        # Empty queue
        self.messages = []

        if keepSending:
            [self.send_msg((self.sender, self.dist), node.name) for node in self.inNodes]


    def log_distances(self):
        ''' This function is called immedately after process_BF each round.  It
        prints distances to the console and the log file in the following format (no whitespace either end):

        A:A0,B1,C2

        Where:
        A is the node currently doing the logging (self),
        B and C are neighbors, with vector weights 1 and 2 respectively
        NOTE: A0 shows that the distance to self is 0 '''

        delimiter = ','
        log = []
        [log.append(node + str(self.dist[node])) for node in self.dist.keys()]
        add_entry(self.name, delimiter.join(log))