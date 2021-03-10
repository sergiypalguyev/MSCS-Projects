import numpy as np

class DTLearner(object):

	def __init__(self, leaf_size = 1, verbose = False):
		self.leaf_size = leaf_size
		self.tree = [] 

	def author(self):
		return 'ksit3'

	def addEvidence(self, dataX, dataY):
		if dataX.shape[0] <= self.leaf_size:
			self.tree = np.array([[-1, np.mean(dataY), -1, -1]])
			return np.array([[-1, np.mean(dataY), -1, -1]])
		if np.unique(dataY).size == 1:
			self.tree = np.array([[-1, np.mean(dataY), -1, -1]])
			return np.array([[-1, np.mean(dataY), -1, -1]])
		else:
			corr = np.absolute(np.corrcoef(dataX, dataY, rowvar = False))
			bestFeature = np.argmax(corr[:, -1][0:corr[:, -1].size - 1] )
			splitVal = np.median(dataX[: , bestFeature])
			if (len(dataX[dataX[: , bestFeature] > splitVal]) == 0):
				self.tree = np.vstack((np.array([[-1, np.mean(dataY), -1, -1]])))
				return self.tree
			lefttree = self.addEvidence(dataX[dataX[:, bestFeature] <= splitVal], dataY[dataX[:, bestFeature] <= splitVal])
			righttree = self.addEvidence(dataX[dataX[:, bestFeature] > splitVal], dataY[dataX[:, bestFeature] > splitVal])
			root = np.array([[bestFeature, splitVal, 1, lefttree.shape[0] + 1]])
			self.tree = np.vstack((root, lefttree, righttree))
			return self.tree

	def query(self, points):
		predVal = np.empty(len(points))
		num = 0
		for point in points:
			index = 0
			curr = self.tree[index]
			while curr[0] != -1:
				if point[int(curr[0])] <= curr[1]:
					index += int(curr[2])
				else:
					index += int(curr[3])
				curr = self.tree[index]
			predVal[num] = curr[1]
			num += 1
		return predVal
