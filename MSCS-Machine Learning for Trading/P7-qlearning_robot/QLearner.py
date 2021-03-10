"""  		   	  			    		  		  		    	 		 		   		 		  
Template for implementing QLearner  (c) 2015 Tucker Balch  		   	  			    		  		  		    	 		 		   		 		  
  		   	  			    		  		  		    	 		 		   		 		  
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		   	  			    		  		  		    	 		 		   		 		  
Atlanta, Georgia 30332  		   	  			    		  		  		    	 		 		   		 		  
All Rights Reserved  		   	  			    		  		  		    	 		 		   		 		  
  		   	  			    		  		  		    	 		 		   		 		  
Template code for CS 4646/7646  		   	  			    		  		  		    	 		 		   		 		  
  		   	  			    		  		  		    	 		 		   		 		  
Georgia Tech asserts copyright ownership of this template and all derivative  		   	  			    		  		  		    	 		 		   		 		  
works, including solutions to the projects assigned in this course. Students  		   	  			    		  		  		    	 		 		   		 		  
and other users of this template code are advised not to share it with others  		   	  			    		  		  		    	 		 		   		 		  
or to make it available on publicly viewable websites including repositories  		   	  			    		  		  		    	 		 		   		 		  
such as github and gitlab.  This copyright statement should not be removed  		   	  			    		  		  		    	 		 		   		 		  
or edited.  		   	  			    		  		  		    	 		 		   		 		  
  		   	  			    		  		  		    	 		 		   		 		  
We do grant permission to share solutions privately with non-students such  		   	  			    		  		  		    	 		 		   		 		  
as potential employers. However, sharing with other current or future  		   	  			    		  		  		    	 		 		   		 		  
students of CS 7646 is prohibited and subject to being investigated as a  		   	  			    		  		  		    	 		 		   		 		  
GT honor code violation.  		   	  			    		  		  		    	 		 		   		 		  
  		   	  			    		  		  		    	 		 		   		 		  
-----do not edit anything above this line---  		   	  			    		  		  		    	 		 		   		 		  
  		   	  			    		  		  		    	 		 		   		 		  
Student Name: Sergiy Palguyev		   	  			    		  		  		    	 		 		   		 		  
GT User ID: spalguyev3  		   	  			    		  		  		    	 		 		   		 		  
GT ID: 903272028  		   	  			    		  		  		    	 		 		   		 		  
"""  		   	  			    		  		  		    	 		 		   		 		  
  		   	  			    		  		  		    	 		 		   		 		  
import numpy as np  		   	  			    		  		  		    	 		 		   		 		  
import random as rand  		   	  			    		  		  		    	 		 		   		 		  
  		   	  			    		  		  		    	 		 		   		 		  
class QLearner(object):  		   	  			    		  		  		    	 		 		   		 		  
  		   	  
    def author(self):
	    return 'spalguyev3'		

    def __init__(self, \
        num_states=100, \
        num_actions = 4, \
        alpha = 0.2, \
        gamma = 0.9, \
        rar = 0.5, \
        radr = 0.99, \
        dyna = 0, \
        verbose = False):  					    		  		  		    	 		 		   		 		  
  		   	  			    		  		  		    	 		 		   		 		  
        self.verbose = verbose  		   	  			    		  		
        self.num_states = num_states    # the number of states to consider	  		    	 		 		   		 		  
        self.num_actions = num_actions  # the number of actions available   	  			    		  		  		    	 		 		   		 		  
        self.state = 0  		   	  			    		  		  		    	 		 		   		 		  
        self.action = 0  
        self.alpha = alpha              # learning rate used in the update rule. Should range between 0.0 and 1.0 with 0.2 as a typical value.
        self.gamma = gamma              # discount rate used in the update rule. Should range between 0.0 and 1.0 with 0.9 as a typical value.
        self.rar = rar                  # random action rate: the probability of selecting a random action at each step. Should range between 0.0 (no random actions) to 1.0 (always random action) with 0.5 as a typical value.
        self.radr = radr                # random action decay rate, after each update, rar = rar * radr. Ranges between 0.0 (immediate decay to 0) and 1.0 (no decay). Typically 0.99.
        self.dyna = dyna                # conduct this number of dyna updates for each regular update. When Dyna is used, 200 is a typical value.
        self.verbose = verbose          # if True, your class is allowed to print debugging statements, if False, all printing is prohibited. 
        
        self.QTable = []
        self.ETable = [] # Experience table
        
        self.instantiate()

    def instantiate(self):
        """
        @summary: Init Q table with random small values
        """
        
        self.QTable = np.random.uniform(-1, 1, size=(self.num_states, self.num_actions))
        # self.ETable is empty and will be appended with new experiences.
        
    def updateQTable(self, *args):
        """
        @summary: Update QTable with values assed int the arguments
                    Used by both non-dyna and dyna updates
        """
        args = args[0]

        if(len(args)!=4):
            return

        state = args[0]
        action = args[1]
        sPrime = args[2]
        reward = args[3]

        oldValue = (1 - self.alpha) * self.QTable[state][action]
        improvedEstimate = self.alpha*(reward + self.gamma * self.QTable[sPrime][np.argmax(self.QTable[sPrime])])
        self.QTable[state][action] = oldValue + improvedEstimate

    def querysetstate(self, s):  		   	  			    		  		  		    	 		 		   		 		  
        """  		   	  			    		  		  		    	 		 		   		 		  
        @summary: Update the state without updating the Q-table 		   	  			    		  		  		    	 		 		   		 		  
        @param s: The new state  		   	  			    		  		  		    	 		 		   		 		  
        @returns: The selected action  		   	  			    		  		  		    	 		 		   		 		  
        """  		   

        self.state = s
        self.action = np.where(np.random.random() < self.rar, np.random.randint(0, self.num_actions - 1), np.argmax(self.QTable[s]))
        self.rar = self.rar * self.radr
        if self.verbose: print "s =", self.state,"a =", self.action
        return self.action    		  		  		    	 		 		   		 		  

    def query(self, s_prime, r):  		   	  			    		  		  		    	 		 		   		 		  
        """  		   	  			    		  		  		    	 		 		   		 		  
        @summary: Update the Q table and return an action  		   	  			    		  		  		    	 		 		   		 		  
        @param s_prime: The new state  		   	  			    		  		  		    	 		 		   		 		  
        @param r: The new state  		   	  			    		  		  		    	 		 		   		 		  
        @returns: The selected action  		   	  			    		  		  		    	 		 		   		 		  
        """  		   	  			    
        self.updateQTable((self.state, self.action, s_prime, r))    
        if(self.dyna>0):
            self.ETable.append((self.state, self.action, s_prime, r))
            [self.updateQTable(rand.choice(self.ETable)) for _ in range(self.dyna)]
        self.state = s_prime
        self.action = self.querysetstate(s_prime)
        if self.verbose: print "Update State :s =", s_prime,"a =", self.action, "r=", r
        return self.action	    		  		  		    	 		 		   		 		  
  		   	  			    		  		  		    	 		 		   		 		  
if __name__=="__main__":  		   	  			    		  		  		    	 		 		   		 		  
    print ""
    k = QLearner(dyna=10,verbose=False)
    k.query(93, -1)