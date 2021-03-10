"""MC1-P2: Optimize a portfolio.  		   	  			    		  		  		    	 		 		   		 		  
  		   	  			    		  		  		    	 		 		   		 		  
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
  		   	  			    		  		  		    	 		 		   		 		  
import pandas as pd  		   	  			    		  		  		    	 		 		   		 		  
import matplotlib.pyplot as plt  		   	  			    		  		  		    	 		 		   		 		  
import numpy as np  		   	  			    		  		  		    	 		 		   		 		  
import datetime as dt  		
import scipy.optimize as spo   	  			    		  		  		    	 		 		   		 		  
from util import get_data, plot_data 
  		   	  			    		  		  		    	 		 		   		 		  
# This is the function that will be tested by the autograder  		   	  			    		  		  		    	 		 		   		 		  
# The student must update this code to properly implement the functionality  		   	  			    		  		  		    	 		 		   		 		  
def optimize_portfolio(sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,1,1), \
    syms=['GOOG','AAPL','GLD','XOM'], gen_plot=False):  		   	  			    		  		  		    	 		 		   		 		  
  		   	  			    		  		  		    	 		 		   		 		  
    # Read in adjusted closing prices for given symbols, date range  		   	  			    		  		  		    	 		 		   		 		  
    dates = pd.date_range(sd, ed)  		   	  			    		  		  		    	 		 		   		 		  
    prices_all = get_data(syms, dates)  # automatically adds SPY  		   	  			    		  		  		    	 		 		   		 		  
    prices = prices_all[syms]  # only portfolio symbols  		   	  			    		  		  		    	 		 		   		 		  
    prices_SPY = prices_all['SPY']  # only SPY, for comparison later  		   	  			    		  		  		    	 		 		   		 		  
  		   	  			    		  		  		    	 		 		   		 		  
    # find the allocations for the optimal portfolio  		   	  			    		  		  		    	 		 		   		 		  
    # note that the values here ARE NOT meant to be correct for a test case  		   	  			    		  		  		    	 		 		   		 		  
    
    num_stocks = len(syms)
    default_allocs = num_stocks * [ float(1)/num_stocks ]

    print "Default Allocations", default_allocs

    # return negative Sharp Ratio so the minimizer optimizes for smallest value
    def get_SharpRatio(allocs, prices, prices_SPY):
        return (-1.0) * assess_portfolio(prices, prices_SPY, allocs)[3]

    bounds = num_stocks*[(0,1)] # each dimention is bound between 0 and 1
    cons = ({'type':'eq', 'fun': lambda x: 1.0- np.sum(x)}) # equality constraint where the result is is such that sum of all allocations must equal 0
    mthd ='SLSQP' # Sequential Least Squares Programming method
    opts = ({'disp': False}) # Options
    
    allocs = spo.minimize(get_SharpRatio, default_allocs, args = (prices, prices_SPY), method=mthd, bounds = bounds, constraints = cons, options=opts)
    # cr, adr, sddr, sr = [0.25, 0.001, 0.0005, 2.1] # add code here to compute stats

    # Get daily portfolio value
    # port_val = prices_SPY # add code here to compute daily portfolio values

    cr, adr, sddr, sr, port_val = assess_portfolio(prices, prices_SPY, allocs.x)

    # Compare daily portfolio value with SPY using a normalized plot
    if gen_plot:
        # add code to plot here
        df_temp = pd.concat([port_val, prices_SPY], keys=['Portfolio', 'SPY'], axis=1)
        #plot_data(df_temp / df_temp.iloc[0], title = 'Daily Portfolio Value and SPY')
        
        df = df_temp / df_temp.iloc[0]
        ax = df.plot(title='Daily Portfolio Value and SPY', fontsize=12)  		   	  			    		  		  		    	 		 		   		 		  
        ax.set_xlabel("Date")  		   	  			    		  		  		    	 		 		   		 		  
        ax.set_ylabel("Price")
        plt.savefig("plot.png")
        pass
 
    return allocs.x, cr, adr, sddr, sr  		   	  			    		  		  		    	 		 		   		 		  

def assess_portfolio(prices, prices_SPY, allocs):

    daily_rf = 0.0 # Risk Free Rate of Return according to assignment instructions
    trading_days = 252 # Number of trading days according to assignment instructions
    start_value = 1000000
    normed = prices / prices.iloc[0] 
    alloced = normed * allocs
    pos_vals = alloced * start_value
    port_val = pos_vals.sum(axis=1)

    cummulative_returns = (port_val[-1] - port_val[0]) / port_val[0]  

    daily_returns = (port_val / port_val.shift(1)) - 1
    daily_returns = daily_returns.iloc[1:]

    mean_returns = daily_returns.mean()
    std_returns = daily_returns.std()
    sharpe_ratio = np.sqrt(trading_days)*( mean_returns - daily_rf ) / std_returns

    return cummulative_returns, mean_returns, std_returns, sharpe_ratio, port_val
	    		  		  		    	 		 		   		 		  
def test_code():  		   	  			    		  		  		    	 		 		   		 		  
    # This function WILL NOT be called by the auto grader  		   	  			    		  		  		    	 		 		   		 		  
    # Do not assume that any variables defined here are available to your function/code  		   	  			    		  		  		    	 		 		   		 		  
    # It is only here to help you set up and test your code  		   	  			    		  		  		    	 		 		   		 		  
  		   	  			    		  		  		    	 		 		   		 		  
    # Define input parameters  		   	  			    		  		  		    	 		 		   		 		  
    # Note that ALL of these values will be set to different values by  		   	  			    		  		  		    	 		 		   		 		  
    # the autograder!  		   	  			    		  		  		    	 		 		   		 		  
  		   	  			    		  		  		    	 		 		   		 		  
    # start_date = dt.datetime(2009,1,1)  		   	  			    		  		  		    	 		 		   		 		  
    # end_date = dt.datetime(2010,1,1)  		   	  			    		  		  		    	 		 		   		 		  
    # symbols = ['GOOG', 'AAPL', 'GLD', 'XOM', 'IBM']  		   	  			    		  		  		    	 		 		   		 		  

    # Assignment requirements
    # Start Date: 2008-06-01, End Date: 2009-06-01, Symbols: ['IBM', 'X', 'GLD', 'JPM'] 		    	 		 		   		 		  
    start_date = dt.datetime(2008,6,1)  		   	  			    		  		  		    	 		 		   		 		  
    end_date = dt.datetime(2009,6,1)  		   	  			    		  		  		    	 		 		   		 		  
    symbols = ['IBM', 'X', 'GLD', 'JPM']

    # Assess the portfolio  		   	  			    		  		  		    	 		 		   		 		  
    allocations, cr, adr, sddr, sr = optimize_portfolio(sd = start_date, ed = end_date,\
        syms = symbols, \
        gen_plot = True)  		   	  			    		  		  		    	 		 		   		 		  
  		   	  			    		  		  		    	 		 		   		 		  
    # Print statistics  		   	  			    		  		  		    	 		 		   		 		  
    print "Start Date:", start_date  		   	  			    		  		  		    	 		 		   		 		  
    print "End Date:", end_date  		   	  			    		  		  		    	 		 		   		 		  
    print "Symbols:", symbols  		   	  			    		  		  		    	 		 		   		 		  
    print "Allocations:", allocations 		   	  			    		  		  		    	 		 		   		 		  
    print "Sharpe Ratio:", sr  		   	  			    		  		  		    	 		 		   		 		  
    print "Volatility (stdev of daily returns):", sddr  		   	  			    		  		  		    	 		 		   		 		  
    print "Average Daily Return:", adr  		   	  			    		  		  		    	 		 		   		 		  
    print "Cumulative Return:", cr  		   	  			    		  		  		    	 		 		   		 		  
  		   	  			    		  		  		    	 		 		   		 		  
if __name__ == "__main__":  		   	  			    		  		  		    	 		 		   		 		  
    # This code WILL NOT be called by the auto grader  		   	  			    		  		  		    	 		 		   		 		  
    # Do not assume that it will be called  		   	  			    		  		  		    	 		 		   		 		  
    test_code()  		   	  			    		  		  		    	 		 		   		 		  
