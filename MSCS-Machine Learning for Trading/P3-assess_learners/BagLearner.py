import numpy as np

class BagLearner(object):   	  

    def __init__(self, learner, kwargs, bags, boost, verbose = False):

        learners = []
        for i in range(0, bags):
            learners.append(learner(**kwargs))
        self.learners = learners			    		  		  		    	 		 		   		 		  

    def author(self):  		   	  			    		  		  		    	 		 		   		 		  
        return 'spalguyev3'

    def addEvidence(self, dataX, dataY):
        samples = dataX.shape[0]
        for learner in self.learners:
            randomChoice = np.random.choice(samples, size = samples, replace = True)
            learner.addEvidence(dataX[randomChoice], dataY[randomChoice])

    def query(self, points):
        allResults = []
        for learner in self.learners:
            result = learner.query(points)
            allResults.append(result)
        resultsMean = np.mean(allResults, axis=0)
        return resultsMean
