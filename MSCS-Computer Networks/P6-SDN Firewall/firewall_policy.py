#!/usr/bin/python
# CS 6250 Spring 2020 - Project 6 - SDN Firewall
# build argyle-v12

from pyretic.lib.corelib import *
from pyretic.lib.std import *
from pyretic.lib.query import packets
from pyretic.core import packet

SKIP = '-'

def make_firewall_policy(config):

    # You may place any user-defined functions in this space.
    # You are not required to use this space - it is available if needed.

    # feel free to remove the following "print config" line once you no longer need it
    # it will not affect the performance of the autograder
    
   #print config

    # The rules list contains all of the individual rule entries.
    rules = []

    for entry in config:
        rules.append(create_firewall(entry))   

    # Think about the following line.  What is it doing?
    allowed = ~(union(rules))

    return allowed


def create_firewall(entry):
    
    # Process TCP_Protocol
    if entry['protocol'] == 'T':
        return policy_match(entry, match(ethtype=packet.IPV4, protocol=packet.TCP_PROTO))

    # Process UDP_Protocol
    if entry['protocol'] == 'U':
        return policy_match(entry, match(ethtype=packet.IPV4, protocol=packet.UDP_PROTO))

    # Process ICMP_Protocol
    if entry['protocol'] == 'I':
        return policy_match(entry, match(ethtype=packet.IPV4, protocol=packet.ICMP_PROTO))

    # Process Both
    if entry['protocol'] == 'B':
        tcp = match(ethtype=packet.IPV4, protocol=packet.TCP_PROTO) 
        udp = match(ethtype=packet.IPV4, protocol=packet.UDP_PROTO)
        return policy_match(entry, tcp & udp)

    if entry['protocol'] == 'O':
        if entry['ipproto'] == '6': # TCP Port specified in policy
            return policy_match(entry, match(ethtype=packet.IPV4, protocol=packet.TCP_PROTO))
        if entry['ipproto'] == '17': # UDP Port specified in policy
            return policy_match(entry, match(ethtype=packet.IPV4, protocol=packet.UDP_PROTO))
        return policy_match(entry, match(ethtype=packet.IPV4, protocol=int(entry['ipproto'])))

    if entry['protocol'] == SKIP\
        and entry['port_dst'] == SKIP\
        and entry['port_src'] == SKIP\
        and entry['ipaddr_dst'] == SKIP\
        and entry['ipaddr_src'] == SKIP\
        and entry['macaddr_src'] == SKIP\
        and entry['macaddr_dst'] == SKIP:
       return match(ethtype=packet.IPV4, protocol=packet.TCP_PROTO)

def policy_match(entry, protocol_match):
	
    # Process MAC Address
    if entry['macaddr_src'] != SKIP:
        protocol_match &= match (srcmac = EthAddr(entry['macaddr_src']))
    if entry['macaddr_dst'] != SKIP:
        protocol_match &= match (dstmac = EthAddr(entry['macaddr_dst']))
    # Process IP Address
    if entry['ipaddr_src'] != SKIP:
        protocol_match &= match (srcip = IPAddr(entry['ipaddr_src']))
    if entry['ipaddr_dst'] != SKIP:
        protocol_match &= match (dstip = IPAddr(entry['ipaddr_dst']))
    # Process Ports
    if entry['port_src'] != SKIP:
        protocol_match &= match (srcport = int(entry['port_src']))
    if entry['port_dst'] != SKIP:
        protocol_match &= match (dstport = int(entry['port_dst']))
   
    return protocol_match