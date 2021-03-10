"""MC2-P1: Market simulator.  		   	  			    		  		  		    	 		 		   		 		  
  		   	  			    		  		  		    	 		 		   		 		  
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
import numpy as np  		   	  			    		  		  		    	 		 		   		 		  
import datetime as dt		    		  		  		    	 		 		   		 		  
from util import get_data, plot_data  		   	  			    		  		  		    	 		 		   		 		  
  		   	  			    		  		  		    	 		 		   		 		  
def compute_portvals(orders_file = "./orders/orders-02.csv", start_val = 1000000, commission=9.95, impact=0.005):  		   	  			    		  		  		    	 		 		   		 		  
    # this is the function the autograder will call to test your code  		   	  			    		  		  		    	 		 		   		 		  
    # NOTE: orders_file may be a string, or it may be a file object. Your  		   	  			    		  		  		    	 		 		   		 		  
    # code should work correctly with either input  		   	  			    		  		  		    	 		 		   		 		  
    # TODO: Your code here

    str_Date = 'Date'
    str_Symbol = 'Symbol'
    str_Cash = 'Cash'
    str_Buy = 'BUY'
    str_Sell = 'SELL'

    # DataFrame of orders file
    df_orders = pd.read_csv(orders_file)

    # Sort DataFrame
    df_orders = df_orders.sort_values([str_Date])
    
    # Get company ticker symbols
    # Get Date Range
    # Get Prices
    tickers = list(df_orders[str_Symbol].unique())
    dates = pd.date_range(df_orders[str_Date].iloc[0],df_orders[str_Date].iloc[-1])
    df_prices = get_data(symbols=tickers, dates=dates, addSPY = True, colname = 'Adj Close')
    
    # Portfolio dataframe
    portfolio = tickers[:]
    portfolio.extend([str_Cash])
    df_portfolio = pd.DataFrame(0.0, index=dates, columns=portfolio)
    df_portfolio = df_portfolio.loc[df_prices.index]
    df_portfolio.iloc[0][str_Cash] = start_val

    print df_orders
    
    # compute change of holdings over time
    print df_orders
    for order in df_orders.iterrows():
        
        order_price = df_prices[order[1][1]].loc[order[1][0]]
        
        if order[1][2] == str_Buy:
            df_portfolio.loc[order[1][0]:, order[1][1]] += order[1][3]

            cur_cash = df_portfolio.loc[order[1][0], str_Cash]
            cur_cash -= commission
            cur_cash -= (order[1][3] * order_price)
            cur_cash -= (order[1][3] * order_price * impact)

            df_portfolio.loc[order[1][0]:, str_Cash] = cur_cash

        if order[1][2] == str_Sell:
            df_portfolio.loc[order[1][0]:, order[1][1]] -= order[1][3]
            
            cur_cash = df_portfolio.loc[order[1][0], str_Cash]
            cur_cash -= commission
            cur_cash += (order[1][3] * order_price)
            cur_cash -= (order[1][3] * order_price * impact)

            df_portfolio.loc[order[1][0]:, str_Cash] = cur_cash
    
    # return portfolio value
    df_prices[str_Cash] = 1
    return (df_prices*df_portfolio).sum(axis=1)  			    		  		  		    	 		 		   		 		  

def author():
    return "spalguyev3"

def test_code():  		   	  			    		  		  		    	 		 		   		 		  
    # this is a helper function you can use to test your code  		   	  			    		  		  		    	 		 		   		 		  
    # note that during autograding his function will not be called.  		   	  			    		  		  		    	 		 		   		 		  
    # Define input parameters  		   	  			    		  		  		    	 		 		   		 		  
  		   	  			    		  		  		    	 		 		   		 		  
    of = "./orders/orders-02.csv"  		   	  			    		  		  		    	 		 		   		 		  
    sv = 1000000  		   	  			    		  		  		    	 		 		   		 		  
  		   	  			    		  		  		    	 		 		   		 		  
    # Process orders  		   	  			    		  		  		    	 		 		   		 		  
    portvals = compute_portvals(orders_file = of, start_val = sv)  		   	  			    		  		  		    	 		 		   		 		  
    if isinstance(portvals, pd.DataFrame):  		   	  			    		  		  		    	 		 		   		 		  
        portvals = portvals[portvals.columns[0]] # just get the first column  		   	  			    		  		  		    	 		 		   		 		  
    else:  		   	  			    		  		  		    	 		 		   		 		  
        "warning, code did not return a DataFrame"  		   	  			    		  		  		    	 		 		   		 		  
  		   	  			    		  		  		    	 		 		   		 		  
    # Get portfolio stats  		   	  			    		  		  		    	 		 		   		 		  
    # Here we just fake the data. you should use your code from previous assignments.

     # Get portfolio stats
    # Here we just fake the data. you should use your code from previous assignments.
    start_date = dt.datetime(2008,1,1)
    end_date = dt.datetime(2008,6,1)
    #cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = [0.2,0.01,0.02,1.5]

    cumulative_returns=(portvals[-1]/portvals[0])-1
    daily_returns=(portvals/portvals.shift(1))-1
    avg_daily_returns=daily_returns.mean()
    std_daily_returns=daily_returns.std()
    sharpe_ratio=np.sqrt(252)*((daily_returns.subtract(0)).mean())/std_daily_returns
    
    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = [0.2,0.01,0.02,1.5]
   	  			    		  		  		    	 		 		   		 		  
    # Compare portfolio against $SPX  		   	  			    		  		  		    	 		 		   		 		  
    print "Date Range: {} to {}".format(start_date, end_date)  		   	  			    		  		  		    	 		 		   		 		  
    print  		   	  			    		  		  		    	 		 		   		 		  
    print "Sharpe Ratio of Fund: {}".format(sharpe_ratio)  		   	  			    		  		  		    	 		 		   		 		  
    print "Sharpe Ratio of SPY : {}".format(sharpe_ratio_SPY)  		   	  			    		  		  		    	 		 		   		 		  
    print  		   	  			    		  		  		    	 		 		   		 		  
    print "Cumulative Return of Fund: {}".format(cumulative_returns)  		   	  			    		  		  		    	 		 		   		 		  
    print "Cumulative Return of SPY : {}".format(cum_ret_SPY)  		   	  			    		  		  		    	 		 		   		 		  
    print  		   	  			    		  		  		    	 		 		   		 		  
    print "Standard Deviation of Fund: {}".format(std_daily_returns)  		   	  			    		  		  		    	 		 		   		 		  
    print "Standard Deviation of SPY : {}".format(std_daily_ret_SPY)  		   	  			    		  		  		    	 		 		   		 		  
    print  		   	  			    		  		  		    	 		 		   		 		  
    print "Average Daily Return of Fund: {}".format(avg_daily_returns)  		   	  			    		  		  		    	 		 		   		 		  
    print "Average Daily Return of SPY : {}".format(avg_daily_ret_SPY)  		   	  			    		  		  		    	 		 		   		 		  
    print  		   	  			    		  		  		    	 		 		   		 		  
    print "Final Portfolio Value: {}".format(portvals[-1])  		   	  			    		  		  		    	 		 		   		 		  
  		   	  			    		  		  		    	 		 		   		 		  
if __name__ == "__main__":  		   	  			    		  		  		    	 		 		   		 		  
    test_code()    	  			    		  		  		    	 		 		   		 		  