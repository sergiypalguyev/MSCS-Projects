"""Assess a betting strategy.  		   	  			    		  		  		    	 		 		   		 		  
  		   	  			    		  		  		    	 		 		   		 		  
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
  		   	  			    		  		  		    	 		 		   		 		  
Student Name: Tucker Balch (replace with your name)  		   	  			    		  		  		    	 		 		   		 		  
GT User ID: tb34 (replace with your User ID)  		   	  			    		  		  		    	 		 		   		 		  
GT ID: 900897987 (replace with your GT ID)  		   	  			    		  		  		    	 		 		   		 		  
"""  		   	  			    		  		  		    	 		 		   		 		  
  		   	  			    		  		  		    	 		 		   		 		  
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib as mpl
from  matplotlib import pyplot as plt

def author():  		   	  			    		  		  		    	 		 		   		 		  
    return 'spalguyev3' # replace tb34 with your Georgia Tech username.
  		   	  			    		  		  		    	 		 		   		 		  
def gtid():  		   	  			    		  		  		    	 		 		   		 		  
	return 903272028 # replace with your GT ID number
  		   	  			    		  		  		    	 		 		   		 		  
def get_spin_result(win_prob):  		   	  			    		  		  		    	 		 		   		 		  
	result = False		  		    	 		 		   		 		  
	if np.random.random() <= win_prob:  		   	  			    		  		  		    	 		 		   		 		  
		result = True		    		  		  		    	 		 		   		 		  
	return result  		   	  			    		  		  		    	 		 		   		 		  

def test_code():
	win_prob = 0.474 # set appropriately to the probability of a win  		   	  			    		  		  		    	 		 		   		 		  
	np.random.seed(gtid()) # do this only once  		   	  			    		  		  		    	 		 		   		 		  
	print get_spin_result(win_prob) # test the roulette spin  		   	  			    		  		  		    	 		 		   		 		  

	# add your code here to implement the experiments			  		  		    	 		 			   		 		  
	algorithm(10, win_prob, 1, 'Figure1.png', "", False)   		 		  
	algorithm(1000, win_prob, 1, 'Figure2.png','Figure3.png', True)  	 		  
	algorithm(1000, win_prob, 2, 'Figure4.png', 'Figure5.png',True)

def experiment_One(win_prob):
	episode_winnings = 0
	maxBets = 1000
	i = 1

	winnings = np.full(maxBets,80)
	winnings[0]=0
	while(episode_winnings<80):
		won = False
		bet_amount = 1
		while(won==False):
			won = get_spin_result(win_prob)
			if(won):
				episode_winnings += bet_amount
			else:
				episode_winnings -= bet_amount
				bet_amount *= 2
			winnings[i]=episode_winnings
			i=i+1			
			if (i==maxBets):
				break	
		if (i==maxBets):
			break
	return winnings

def experiment_Two(win_prob):
	episode_winnings = 0
	bank_roll = 256
	maxBets = 1000
	i = 1
	continue_bet = True

	winnings = np.full(maxBets,80)
	winnings[0]=0
	while(episode_winnings<80):
		won = False
		bet_amount = 1
		while(won==False):
			if(continue_bet==True):
				won = get_spin_result(win_prob)
				if(bet_amount > bank_roll):
					bet_amount = bank_roll
				if(won):
					episode_winnings += bet_amount
					bank_roll +=bet_amount
				else:
					episode_winnings -= bet_amount
					bank_roll -= bet_amount
					bet_amount *= 2
				if episode_winnings <= -256:
					continue_bet=False	
			winnings[i]=episode_winnings
			i=i+1
			if (i==maxBets):
				break	
		if (i==maxBets):
			break
	return winnings

def algorithm(runs, win_prob, experiment, meanName, medianName, calcMean):
	
	idx = 0	
	plt.xlim(0,300)
	plt.ylim(-256,100)
	
	win_array = np.zeros((runs, 1000))
	mean = np.zeros(runs)
	median = np.zeros(runs)
	stdev = np.zeros(runs)

	while idx<runs:
		if(experiment == 1):
			winnings = experiment_One(win_prob)
		elif(experiment == 2):
			winnings = experiment_Two(win_prob)
		win_array[idx,:] = winnings
		if(calcMean == False):
			plt.plot(winnings)
		idx = idx+1
	
	if( calcMean==True):
		for x in range(len(winnings)):
			mean[x] = np.mean(win_array[:,x])
			stdev[x] = np.std(win_array[:,x])
			median[x] = np.median(win_array[:,x])
		
		#To count probability
		unique, counts = np.unique(win_array[:,len(winnings)-1], return_counts=True)
		dict(zip(unique, counts))
		#To count average
		np.mean(mean)
		
	if(calcMean==True):
		plt.clf()
		plt.xlim(0,300)
		plt.ylim(-256,100)
		plt.plot(mean)
		plt.plot(mean + stdev)
		plt.plot(mean - stdev)
		plt.savefig(meanName)	
		plt.clf()
		plt.xlim(0,300)
		plt.ylim(-256,100)
		plt.plot(median)
		plt.plot(median + stdev)
		plt.plot(median - stdev)
		plt.savefig(medianName)	 
	else:
		plt.savefig(meanName)	
	plt.close()


if __name__ == "__main__": 	
	test_code()	


