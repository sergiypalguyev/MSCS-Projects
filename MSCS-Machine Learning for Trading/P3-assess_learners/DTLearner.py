import numpy as np

class DTLearner(object):  		   	  			    		  		  		    	 		 		   		 		  
  		  
    def __init__(self, leaf_size = 1, verbose = False):
        self.leaf_size = leaf_size
        self.tree = [] 

    def author(self):
        return 'spalguyev3'

    def addEvidence(self, dataX, dataY):
        leaf = np.array([[-1, np.mean(dataY),-1,-1]])
        if dataX.shape[0] <= self.leaf_size or np.unique(dataY).size == 1:
            self.tree = leaf
            return leaf
        else:
			feature = self.getFeature(dataX, dataY)
			subX = dataX[:, feature]
			SplitVal = np.median(subX)

			# Avoiding infinite recursion
			if (len(dataX[subX > SplitVal]) == 0 or len(dataX[subX <= SplitVal]) == 0):
				self.tree = leaf
				return self.tree

			lefttree = self.addEvidence(dataX[subX <= SplitVal], dataY[subX <= SplitVal])
			righttree = self.addEvidence(dataX[subX > SplitVal], dataY[subX > SplitVal])
			root = np.array([[feature, SplitVal, 1, lefttree.shape[0] + 1]])
			self.tree = np.vstack((root, lefttree, righttree))
			return self.tree

    def getFeature(self, dataX, dataY):
		corr = np.corrcoef(dataX, dataY, rowvar = False)
		corr = np.absolute(corr)
		corr = corr[:, -1]
		maxLen = corr.size - 1
		maxVal = np.argmax(corr[0:maxLen])
		return maxVal

    def query(self, points):
        predicted = np.empty(len(points))
        idx = 0
        leaf = -1
        for point in points:
            index = 0
            node = self.tree[index]

            factor = node[0]
            splitVal = node[1]
            left = node[2]
            right = node[3]

            while factor != leaf:
                if point[int(factor)] <= splitVal:
                    index += int(left)
                if point[int(factor)] > splitVal:
                    index += int(right)
                node = self.tree[index]

                factor = node[0]
                splitVal = node[1]
                left = node[2]
                right = node[3]

            predicted[idx] = splitVal
            idx += 1
        return predicted
    