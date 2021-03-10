"""  		   	  			    		  		  		    	 		 		   		 		  
template for generating data to fool learners (c) 2016 Tucker Balch  		   	  			    		  		  		    	 		 		   		 		  
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
  		   	  			    		  		  		    	 		 		   		 		  
Student Name: Sergiy Palguyev (replace with your name)  		   	  			    		  		  		    	 		 		   		 		  
GT User ID: spalguyev3  (replace with your User ID)  		   	  			    		  		  		    	 		 		   		 		  
GT ID: 903272028 (replace with your GT ID)  		   	  			    		  		  		    	 		 		   		 		  
"""  		   	  			    		  		  		    	 		 		   		 		  
  		   	  			    		  		  		    	 		 		   		 		  
import numpy as np  		   	  			    		  		  		    	 		 		   		 		  
import math  	 		 		   		 		  
import matplotlib.pyplot as plt 		   	  			    		  		  		    	 		 		   		 		  
  		   	  			    		  		  		    	 		 		   		 		  
# this function should return a dataset (X and Y) that will work  		   	  			    		  		  		    	 		 		   		 		  
# better for linear regression than decision trees  		   	  			    		  		  		    	 		 		   		 		  
def best4LinReg(seed=1234567891): 	   
    np.random.seed(seed) 	  			    		  		  		    	 		 		   		 		  

    points = 1000		   	  			  
    X_cols = 5

    X = np.random.random((points, X_cols))
    Y = 0

    for i in range(X_cols):
        if i==0:
            Y += np.tan(X[:,i]) 
        elif i % 2 == 0:
            Y+=i * np.sin(X[:,i])**i
        else:
            Y+=i * np.cos(X[:,i])**i


    return X, Y  		   	  			    		  		  		    	 		 		   		 		  

# this function should return a dataset (X and Y) that will work  		   	  			    		  		  		    	 		 		   		 		  
# better for decision trees than linear regression 		   	  			    		  		  		    	 		 		   		 		  
def best4DT(seed=1234567891): 
    np.random.seed(seed) 		   	  			    		  		  		    	 		 		   		 		  
    points = 1000 		   	  			  
    X_cols = 5
    X = np.random.random((points,X_cols))
    Y = np.zeros(points)
    X_Mean = []
    for i in range(X_cols):
        X_Mean.append(np.mean(X[:,i]))
    for j in range(points):
        mean = []
        for k in range(X_cols):
            mean.append(X[j,k] >= X_Mean[k])
        if all(value == True for value in mean):
            Y[j] = 1
        else: 
            Y[j] = 0

    return X, Y  		  
 	  			    		  		  		    	 		 		   		 		  
  		   	  			    		  		  		    	 		 		   		 		  
def author():  		   	  			    		  		  		    	 		 		   		 		  
    return 'spalguyev3'	   	  			    		  		  		    	 		 		   		 		  
  		   	  			    		  		  		    	 		 		   		 		  
if __name__=="__main__":  		
    X1, Y1 = best4LinReg(seed = 5)
    X2, Y2 = best4DT(seed = 5)

