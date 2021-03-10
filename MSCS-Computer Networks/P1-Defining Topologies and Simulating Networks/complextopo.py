from mininet.topo import Topo
from mininet.net import Mininet
from mininet.link import TCLink
from mininet.util import custom
																			
# Topology to be instantiated in Mininet
class ComplexTopo(Topo):
    "Mininet Complex Topology"

    def __init__(self, cpu=.1, max_queue_size=None, **params):

        # Initialize topo
        Topo.__init__(self, **params)

        #TODO: Create your Mininet Topology here!

        hostConfig = {'cpu': cpu}

        # Ethernet configured from project
        link1Config = {'bw': 25, 'delay': '2ms', 'loss': 0,
                   'max_queue_size': max_queue_size }
        # WiFi configured from project
        link2Config = {'bw': 10, 'delay': '6ms', 'loss': 3,
            'max_queue_size': max_queue_size }
        # 3G configured from project
        link3Config = {'bw': 3, 'delay': '10ms', 'loss': 8,
            'max_queue_size': max_queue_size }

        # Hosts
        h1 = self.addHost("h1", **hostConfig)
        h2 = self.addHost("h2", **hostConfig)
        h3 = self.addHost("h3", **hostConfig)

        # Switches
        s1 = self.addSwitch("s1")
        s2 = self.addSwitch("s2")
        s3 = self.addSwitch("s3")
        s4 = self.addSwitch("s4")

        # Connectivity
        self.addLink(h1, s1, **link1Config) # Ethernet link
        self.addLink(s1, s2, **link1Config) # Ethernet link
        self.addLink(s2, s3, **link1Config) # Ethernet link
        self.addLink(s2, s4, **link3Config) # Ethernet link
        self.addLink(h2, s3, **link2Config) # WiFi link
        self.addLink(h3, s4, **link3Config) # 3G link