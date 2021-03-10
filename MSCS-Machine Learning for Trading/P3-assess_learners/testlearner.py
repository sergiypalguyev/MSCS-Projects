"""  		   	  			    		  		  		    	 		 		   		 		  
Test a learner.  (c) 2015 Tucker Balch  		   	  			    		  		  		    	 		 		   		 		  
  		   	  			    		  		  		    	 		 		   		 		  
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
"""  		   	  			    		  		  		    	 		 		   		 		  
  		   	  				   	  			    		  		  		    	 		 		   		 		  
import math  		  			    		  		  		    	 		 		   		 		  
import sys  		    		  		  		    	 		 		   		 		  
import numpy as np  	    		  		  		    	 		 		   		 		  
import pandas as pd	  			    		  		  		    	 		 		   		 		  
import matplotlib.pyplot as plt  	   	  	   	  			    		  		  		    	 		 		   		 		  
import LinRegLearner as lrl 
import DTLearner as dt
import RTLearner as rt
import BagLearner as bl
import InsaneLearner as it  
import timeit			    		  		  		    	 		 		   		 		  
  		   	  			    		  		  		    	 		 		   		 		  
if __name__=="__main__":  		   	  			    		  		  		    	 		 		   		 		  
   		 		  
    inf = open("Data/Istanbul.csv")	    		  		  		    	 		 		   		 		  
    data = np.array([map(str, s.strip().split(',')) for s in inf.readlines()])	

    try:
        float(data[0,:].all())
    except:
        data = data[1:,:]

    try:
        float(data[:,0].all())
    except:
        data = data[:,1:]
    
    data = data.astype(float)
      		  		  		    	 		 		   	  		   	  			    		  		  		    	 		 		   		 		  
    train_rows = int(0.6 * data.shape[0])  		   	  			    		  		  		    	 		 		   		 		  
    test_rows = data.shape[0] - train_rows  		   	  			    		  		  		    	 		 		   		 		  
  		   	  			    		  		  		   	  			    		  		  		    	 		 		   		 		  
    trainX = data[:train_rows,0:-1]  		   	  			    		  		  		    	 		 		   		 		  
    trainY = data[:train_rows,-1]  		   	  			    		  		  		    	 		 		   		 		  
    testX = data[train_rows:,0:-1]  		   	  			    		  		  		    	 		 		   		 		  
    testY = data[train_rows:,-1]  		   	  			    		  		  		    	 		 		   		 		     	  			    		  		  		    	 		 		   		 		  
  		   	  	
    leaf_range = 50
    leaf_sizes = [x + 1 for x in range(leaf_range)]
    rmseTrain = []
    rmseTest = []
    
    for leaf_size in leaf_sizes:
        
        learner = dt.DTLearner(leaf_size=leaf_size, verbose=False)

        learner.addEvidence(trainX, trainY)
        predY = learner.query(trainX)
        rmse = math.sqrt(((trainY - predY) ** 2).sum()/trainY.shape[0])
        rmseTrain.append(rmse)

        predY = learner.query(testX)
        rmse = math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0])
        rmseTest.append(rmse)

    plt.clf()	
    plt.xlabel('leaf size')
    plt.ylabel('RMSE')
    plt.title('RMSE for DTLearner')
    plt.xlim(0,leaf_range)
    plt.ylim(0,0.010)
    plt.plot(leaf_sizes, rmseTrain, label='InSample')
    plt.plot(leaf_sizes, rmseTest, label='OutOfSample')
    plt.legend()
    plt.savefig("Q1.png")

    rmseTrain = []
    rmseTest = []
    bags = 4
    while bags<300:
        for leaf_size in leaf_sizes:
            learner = bl.BagLearner(learner=dt.DTLearner, kwargs={"leaf_size": leaf_size}, bags=bags, boost=False, verbose=False)

            learner.addEvidence(trainX, trainY)
            predY = learner.query(trainX)
            rmse = math.sqrt(((trainY - predY) ** 2).sum()/trainY.shape[0])
            rmseTrain.append(rmse)

            predY = learner.query(testX)
            rmse = math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0])
            rmseTest.append(rmse)

        plt.clf()	
        plt.xlabel('leaf size')
        plt.ylabel('RMSE')
        plt.title('RMSE for DTLearner')
        plt.xlim(0,leaf_range)
        plt.ylim(0,0.010)
        plt.plot(leaf_sizes, rmseTrain, label='InSample')
        plt.plot(leaf_sizes, rmseTest, label='OutOfSample')
        plt.legend()
        plt.savefig(str("Q2-"+str(bags)+".png"))
        rmseTrain = []
        rmseTest = []
        bags = bags*4

    leaf_range = 25
    leaf_sizes = [x + 1 for x in range(leaf_range)]
    rmseDTTrain = []
    rmseDTTest = []
    rmseRTTrain = []
    rmseRTTest = []
    for leaf_size in leaf_sizes:
        learnerDT = dt.DTLearner(leaf_size=leaf_size, verbose=False)

        learnerDT.addEvidence(trainX, trainY) 
        predY = learnerDT.query(trainX)
        rmse = math.sqrt(((trainY - predY) ** 2).sum()/trainY.shape[0])
        rmseDTTrain.append(rmse)

        predY = learnerDT.query(testX)
        rmse = math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0])
        rmseDTTest.append(rmse)

        
        learnerRT = rt.RTLearner(leaf_size=leaf_size, verbose=False)

        learnerRT.addEvidence(trainX, trainY)
        predY = learnerRT.query(trainX)
        rmse = math.sqrt(((trainY - predY) ** 2).sum()/trainY.shape[0])
        rmseRTTrain.append(rmse)

        predY = learnerRT.query(testX)
        rmse = math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0])
        rmseRTTest.append(rmse)

    plt.clf()	
    plt.xlabel('leaf size')
    plt.ylabel('RMSE')
    plt.title('RMSE for DTLearner vs RTLearner')
    plt.xlim(0,leaf_range)
    plt.ylim(0,0.010)
    plt.plot(leaf_sizes, rmseDTTrain, label='InSample-DT')
    plt.plot(leaf_sizes, rmseDTTest, label='OutOfSample-DT')
    dtOverfit = 0
    for i in range(leaf_range):
        if (rmseDTTrain[i] > rmseDTTest[i]):
            dtOverfit = i
            plt.plot(leaf_sizes[i],rmseDTTrain[i], 'ro')
            plt.annotate(str("leaf_size="+str(i)), xy=(leaf_sizes[i], rmseDTTrain[i]-0.0005))
            break
    plt.plot(leaf_sizes, rmseRTTrain, label='InSample-RT')
    plt.plot(leaf_sizes, rmseRTTest, label='OutOfSample-RT')
    
    rtOverfit = 0
    for i in range(leaf_range):
        if (rmseRTTrain[i] > rmseRTTest[i]):
            rtOverfit = i
            plt.plot(leaf_sizes[i],rmseRTTrain[i], 'ro')
            plt.annotate(str("leaf_size="+str(i)), xy=(leaf_sizes[i], rmseRTTrain[i]+0.0005))
            break
    plt.legend()
    plt.savefig(str("Q3-1.png"))

    DTTime = []
    RTTime = []
    bags = 5
    for leaf_size in leaf_sizes:
        learnerDT = bl.BagLearner(learner=dt.DTLearner, kwargs={"leaf_size": leaf_size}, bags=bags, boost=False, verbose=False)

        start_time = timeit.default_timer()
        learnerDT.addEvidence(trainX, trainY)
        DTTime.append( timeit.default_timer() - start_time)

        learnerRT = bl.BagLearner(learner=rt.RTLearner, kwargs={"leaf_size": leaf_size}, bags=bags, boost=False, verbose=True)
        
        start_time = timeit.default_timer()
        learnerRT.addEvidence(trainX, trainY)
        RTTime.append( timeit.default_timer() - start_time)

    plt.clf()	
    plt.xlabel('leaf_size')
    plt.ylabel('Time (s)')
    plt.title('Train Time vs. leaf_size')
    plt.plot(leaf_sizes, DTTime, label='DT')
    plt.plot(leaf_sizes, RTTime, label='RT')
    plt.legend()
    plt.savefig(str("Q3-2.png"))