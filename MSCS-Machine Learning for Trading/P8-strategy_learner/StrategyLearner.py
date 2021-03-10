"""  		   	  			    		  		  		    	 		 		   		 		  
Template for implementing StrategyLearner  (c) 2016 Tucker Balch  		   	  			    		  		  		    	 		 		   		 		  
  		   	  			    		  		  		    	 		 		   		 		  
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
  		   	  			    		  		  		    	 		 		   		 		  
import datetime as dt  		   	  			    		  		  		    	 		 		   		 		  
import pandas as pd  		   	  			    		  		  		    	 		 		   		 		  
import util as ut  	
import indicators as ind	   	  			    		  		  		    	 		 		   		 		  
import random  as rand	
import numpy as np
import QLearner as ql	
import matplotlib.pyplot as plt
import marketsimcode as mm
from marketsimcode import compute_portvals
from ManualStrategy import ManualStrategy

  		   	  			    		  		  		    	 		 		   		 		  
class StrategyLearner(object):  		   	  			    		  		  		    	 		 		   		 		  
  		   	
    def author(self):
	    return 'spalguyev3'  

    # constructor  		   	  			    		  		  		    	 		 		   		 		  
    def __init__(self, verbose = False, impact=0.0):  		   	  			    		  		  		    	 		 		   		 		  
        self.verbose = verbose  		   	  			    		  		  		    	 		 		   		 		  
        self.impact = impact  
        self.learner = None
        self.inidcators = None

    # Function for discretizing dataframe into steps, making it easier for our QLearner to swallow.
    def discretize(self, df, steps):
        threshold = []
        for i in range(0,steps):
            threshold.append(df.sort_values()[(i+1)*(df.shape[0]/steps)])
        return np.searchsorted(threshold,df)
    
    # this method should create a QLearner, and train it for trading  		   	  			    		  		  		    	 		 		   		 		  
    def addEvidence(self, symbol = "IBM", \
        sd=dt.datetime(2008,1,1), \
        ed=dt.datetime(2009,1,1), \
        sv = 10000):  		   	  	
        self.dates = pd.date_range(sd, ed)		
        prices_all = ut.get_data([symbol], self.dates)
        prices = prices_all[symbol]
        indicatorDF = ind.indicators(prices_all, symbol)

        steps = 10
        indicatorDF["returns"] = (prices[1:] / prices[:-1].values) - 1
        indicatorDF["returns"][0] = np.nan
        indicatorDF["sma_long"]=self.discretize(indicatorDF["price"]/indicatorDF["sma_long"],steps=steps)
        indicatorDF["sma_short"]=self.discretize(indicatorDF["price"]/indicatorDF["sma_short"],steps=steps)
        indicatorDF["bb_up"]=self.discretize(indicatorDF["price"]/indicatorDF["bb_up"],steps=steps)
        indicatorDF["bb_down"]=self.discretize(indicatorDF["price"]/indicatorDF["bb_down"],steps=steps)
        indicatorDF["momentum"]=self.discretize(indicatorDF["price"]/indicatorDF["momentum"],steps=steps)

        States = np.floor((indicatorDF["sma_long"] *10000+ indicatorDF["sma_short"] *1000+ indicatorDF["bb_up"]*100+indicatorDF["bb_down"]*10+indicatorDF["momentum"])/1e3).astype(int)

        self.learner = ql.QLearner(num_states=100, num_actions=3)

        SHARES = 1000
        for iteration in range(100):
            curHold = 0
            profits = 0
            rewards = 0
            for i in range(States.shape[0]-1): 
                if i == 0:
                    action = self.learner.querysetstate(States[i])
                else:
                    action = self.learner.query(States[i], rewards)
                    profits += prices[i - 1] * curHold * SHARES * indicatorDF["returns"][i]

                if curHold == -1 and action == 0: #-1000 Shares and HOLD
                    curHold = -1
                    rewards = -indicatorDF["returns"][i + 1]
                elif curHold == -1 and action == 1: #-1000 Shares and LONG
                    curHold = -1
                    rewards = -indicatorDF["returns"][i + 1]
                elif curHold == -1 and action == 2: #-1000 Shares and SHORT
                    curHold = 1
                    rewards = 2*indicatorDF["returns"][i + 1]
                elif curHold == 0 and action == 0: #0 Shares and HOLD
                    curHold = 1
                    rewards = indicatorDF["returns"][i + 1]
                elif curHold == 0 and action == 1: #0 Shares and LONG
                    curHold = 1
                    rewards = indicatorDF["returns"][i + 1]
                elif curHold == 0 and action == 2: #0 Shares and SHORT
                    curHold = -1
                    rewards = indicatorDF["returns"][i + 1]
                elif curHold == 1 and action == 0: #1000 Shares and HOLD
                    curHold = -1
                    rewards = -indicatorDF["returns"][i + 1] 
                elif curHold == 1 and action == 1: #1000 Shares and LONG
                    curHold = -1
                    rewards = 2*indicatorDF["returns"][i + 1]
                elif curHold == 1 and action == 2: #1000 Shares and SHORT  	
                    curHold = -1
                    rewards = -indicatorDF["returns"][i + 1] 

    # this method should use the existing policy and test it against new data  		   	  			    		  		  		    	 		 		   		 		  
    def testPolicy(self, symbol = "IBM", \
        sd=dt.datetime(2009,1,1), \
        ed=dt.datetime(2010,1,1), \
        sv = 10000):  		

        self.dates = pd.date_range(sd, ed)		
        prices_all = ut.get_data([symbol], self.dates)
        prices = prices_all[symbol]

        steps = 10
        indicatorDF = ind.indicators(prices_all, symbol)
        indicatorDF["returns"] = (prices[1:] / prices[:-1].values) - 1
        indicatorDF["returns"][0] = np.nan
        indicatorDF["sma_long"]=self.discretize(indicatorDF["price"]/indicatorDF["sma_long"],steps=steps)
        indicatorDF["sma_short"]=self.discretize(indicatorDF["price"]/indicatorDF["sma_short"],steps=steps)
        indicatorDF["bb_up"]=self.discretize(indicatorDF["price"]/indicatorDF["bb_up"],steps=steps)
        indicatorDF["bb_down"]=self.discretize(indicatorDF["price"]/indicatorDF["bb_down"],steps=steps)
        indicatorDF["momentum"]=self.discretize(indicatorDF["price"]/indicatorDF["momentum"],steps=steps)

        States = np.floor((indicatorDF["sma_long"] *10000+ indicatorDF["sma_short"] *1000+ indicatorDF["bb_up"]*100+indicatorDF["bb_down"]*10+indicatorDF["momentum"])/1e3).astype(int)

        trades=prices.copy()

        SHARES = 1000
        prevHold = 0
        profits = 0
        curHold = 0
        for i in range(indicatorDF["returns"].shape[0] - 1):
            if (i > 0):
                profits += prices[i - 1] * prevHold * SHARES * indicatorDF["returns"][i]
            state = States[i]
            action = self.learner.querysetstate(state)

            if curHold == -1 and action == 0: #-1000 Shares and HOLD
                curHold = -1
                rewards = -indicatorDF["returns"][i + 1]
            elif curHold == -1 and action == 1: #-1000 Shares and LONG
                curHold = -1
                rewards = -indicatorDF["returns"][i + 1]
            elif curHold == -1 and action == 2: #-1000 Shares and SHORT
                curHold = 1
                rewards = 2*indicatorDF["returns"][i + 1]
            elif curHold == 0 and action == 0: #0 Shares and HOLD
                curHold = 1
                rewards = indicatorDF["returns"][i + 1]
            elif curHold == 0 and action == 1: #0 Shares and LONG
                curHold = 1
                rewards = indicatorDF["returns"][i + 1]
            elif curHold == 0 and action == 2: #0 Shares and SHORT
                curHold = -1
                rewards = indicatorDF["returns"][i + 1]
            elif curHold == 1 and action == 0: #1000 Shares and HOLD
                curHold = -1
                rewards = -indicatorDF["returns"][i + 1] 
            elif curHold == 1 and action == 1: #1000 Shares and LONG
                curHold = -1
                rewards = 2*indicatorDF["returns"][i + 1]
            elif curHold == 1 and action == 2: #1000 Shares and SHORT  	
                curHold = -1
                rewards = -indicatorDF["returns"][i + 1]    	
            trades[i] = (curHold - prevHold) * SHARES
            prevHold = curHold
		   	  			    		  		  		    	 		 		   		 		  
        if self.verbose: print type(trades) # it better be a DataFrame!  		   	  			    		  		  		    	 		 		   		 		  
        if self.verbose: print trades  		   	  			    		  		  		    	 		 		   		 		  
        if self.verbose: print prices_all

        trades_df = pd.DataFrame(columns=['Date', 'Symbol', 'Order', 'Shares'])
        trades_df['Date'] = trades.index
        trades_df['Symbol']= symbol
        trades_df['Order'] = ["BUY" if x > 0 else "SELL" if x < 0 else "HOLD" for x in trades.values]
        trades_df['Shares']=abs(trades.values)

        trades_df.iloc[-1, trades_df.columns.get_loc('Order')] = "HOLD"
        trades_df.iloc[-1, trades_df.columns.get_loc('Shares')] = 0.0	    	 		 		   		 		  
        return trades_df  		   	  			    	

if __name__=="__main__":  		   	  			    		  		  		    	 		 		   		 		  
    print "One does not simply think up a strategy"  		 
            
    def get_portfolio_stats(port_val, daily_rf=0, samples_per_year=252):

        daily_ret = (port_val / port_val.shift(1)) - 1
        cr = (port_val[-1] / port_val[0]) - 1
        adr = daily_ret.mean()
        sddr = daily_ret.std()
        k = np.sqrt(samples_per_year)
        sr = k * np.mean(adr - daily_rf) /sddr
        return cr, adr, sddr, sr

    sl = StrategyLearner()

    symbol = 'JPM'
    sv = 100000
    commission = 0
    impact = 0

    sl.addEvidence(symbol=symbol,sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), \
                sv=sv)
    trades_df=sl.testPolicy(symbol=symbol,sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), \
                sv=sv)
                
    portval_sbl = compute_portvals(trades_df, start_val=sv, commission=commission, impact=impact)

    # normalize the portval_bps
    normed_portval_sbl = portval_sbl / portval_sbl.ix[0]
    cr_sbl, adr_sbl, sdr_sbl, sr_sbl = get_portfolio_stats(normed_portval_sbl)
    print "Cumulative Return of {}: {}".format("QLearner-based  Strategy", cr_sbl)
    print "Standard Deviation of daily return of {}: {}".format("QLearner-based  Strategy", sdr_sbl)
    print "Mean Daily Return of {}: {}".format("QLearner-based  Strategy", adr_sbl)
    print "Portval of {}: {}".format("QLearner-based Strategy", portval_sbl[-1])

    ax = normed_portval_sbl.plot(title="Figure 7. Benchmark vs. Manual Rule-Based Strategy (JPM)", fontsize=12, color="black",
                        label="Manual Rule-Based")
    #portval_benchmark.plot(ax=ax, color="blue", label="Benchmark")
    ax.set_ylabel('Normalized Value')
    ax.set_xlabel('Dates')
    ymin,ymax=ax.get_ylim()
    entries=[]
    entries2=[]

    for i in range(0,len(trades_df),2):
        if trades_df.ix[i,"Order"]=="SELL":
            entries.append(trades_df.index[i])
        elif trades_df.ix[i,"Order"]=="BUY":
            entries2.append(trades_df.index[i])

    for day in entries:
        ax.axvline(x=day,color="r")

    for day in entries2:
        ax.axvline(x=day, color="g")

    plt.grid(True)


    plt.legend(loc=0)
    #plt.savefig("Fig7.png")
    plt.show()



############################################################################################################################




    ms = ManualStrategy()
    df_trades = ms.testPolicy(symbol="JPM",  sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv=100000) 
    df_benchmark = ms.get_benchmark_order(symbol="JPM",  sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv=100000)

    manual_potfolio = mm.compute_portvals(df_trades, 100000)
    benchmark_portfolio = mm.compute_portvals(df_benchmark, 100000)  
    
    plt.figure(figsize=(15,5))
    plt.xlim(dt.datetime(2008,1,1), dt.datetime(2009,12,31))
    plt.title("Manual Strategy In-Sample")
    plt.gca().set_color_cycle(['blue','black', 'green'])
    plt.legend(loc="lower right")
    plt.plot(manual_potfolio)
    plt.plot(benchmark_portfolio)
    plt.plot(portval_sbl)
    plt.legend(['Manual','Benchmark', 'QLearned'], loc="lower right")
    plt.xlabel('Dates')
    plt.ylabel('Normalized Prices')
    plt.show()
    #plt.savefig("Strategy-Manual.png")	
    plt.clf




############################################################################################################################





    plt.clf
    sl.addEvidence(symbol=symbol,sd=dt.datetime(2010,1,1), ed=dt.datetime(2011,12,31), \
                sv=sv)
    sbl_trades=sl.testPolicy(symbol=symbol,sd=dt.datetime(2010,1,1), ed=dt.datetime(2011,12,31), \
                sv=sv)
                
    trades_df = pd.DataFrame(columns=['Date', 'Symbol', 'Order', 'Shares'])
    trades_df['Date'] = sbl_trades.index
    trades_df['Symbol']= symbol
    trades_df['Order'] = ["BUY" if x >= 0 else "SELL" for x in sbl_trades.values]
    trades_df['Shares']=abs(sbl_trades.values)
    portval_sbl = compute_portvals(trades_df, start_val=sv, commission=commission, impact=impact)

    df_trades = ms.testPolicy(symbol="JPM",  sd=dt.datetime(2010,1,1), ed=dt.datetime(2011,12,31), sv=100000) 
    df_benchmark = ms.get_benchmark_order(symbol="JPM",  sd=dt.datetime(2010,1,1), ed=dt.datetime(2011,12,31), sv=100000)

    manual_potfolio = mm.compute_portvals(df_trades, 100000)
    benchmark_portfolio = mm.compute_portvals(df_benchmark, 100000)  
    
    plt.figure(figsize=(15,5))
    plt.xlim(dt.datetime(2010,1,1), dt.datetime(2011,12,31))
    plt.title("Manual Strategy Out-Of-Sample")
    plt.gca().set_color_cycle(['blue','black', 'green'])
    plt.plot(manual_potfolio)
    plt.plot(benchmark_portfolio)
    plt.plot(portval_sbl)
    plt.legend(['Manual','Benchmark', 'QLearning'], loc="lower right")
    plt.xlabel('Dates')
    plt.ylabel('Normalized Prices')
    #plt.savefig("Strategy-Manual-OS.png")	
    plt.show()
    plt.clf