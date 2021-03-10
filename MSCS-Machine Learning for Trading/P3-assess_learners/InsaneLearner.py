import numpy as np
import BagLearner as bl
import LinRegLearner as lrl

class InsaneLearner(object):   	  

    def __init__(self, verbose = False):
        self.verbose = verbose

        lrlKwarg = {"verbose" : self.verbose}
        boost = False
        bags = 20
        blKwargs =  {"learner" : lrl.LinRegLearner, "kwargs" : lrlKwarg, "bags" : bags, "boost": boost}
        self.learner = bl.BagLearner(learner = bl.BagLearner, kwargs = blKwargs, bags = bags, boost = boost, verbose = self.verbose)

    def author(self):  		   	  			    		  		  		    	 		 		   		 		  
        return 'spalguyev3'

    def addEvidence(self, dataX, dataY):
		self.learner.addEvidence(dataX, dataY)

    def query(self, points):
        return self.learner.query(points)