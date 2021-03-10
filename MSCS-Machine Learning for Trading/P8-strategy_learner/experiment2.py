"""	    		  		  		    	 		 		   		 		  
Student Name: Sergiy Palguyev		   	  			    		  		  		    	 		 		   		 		  
GT User ID: spalguyev3  		   	  			    		  		  		    	 		 		   		 		  
GT ID: 903272028		   	  			    		  		  		    	 		 		   		 		  
"""  		   	 

import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import marketsimcode as mm
import ManualStrategy as ms
import StrategyLearner as sl
from ManualStrategy import ManualStrategy
from StrategyLearner import StrategyLearner
from marketsimcode import compute_portvals

def author():
    return'spalguyev3'

def experiment2():

    sl = StrategyLearner()
    ms = ManualStrategy()
    start_date = dt.datetime(2008, 1, 1)
    end_date = dt.datetime(2009, 12, 31)

    symbol = 'JPM'
    starting_value = 100000
    commission = 0
    impact = 0.002

    sl.addEvidence(symbol=symbol,sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=starting_value)
    qlearn_trades=sl.testPolicy(symbol="JPM",sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=starting_value)
    manual_trades = ms.testPolicy(symbol="JPM",  sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv=starting_value)
    benchmark_trades = ms.get_benchmark_order(symbol="JPM",  sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv=starting_value)

    qlearn_portfolio = (mm.compute_portvals(qlearn_trades, starting_value, commission, impact))/starting_value
    manual_potfolio = (mm.compute_portvals(manual_trades, starting_value, commission, impact))/starting_value
    benchmark_portfolio = (mm.compute_portvals(benchmark_trades, starting_value, commission, impact))/starting_value
    
    plt.clf()	
    plt.figure(figsize=(15,5))
    plt.xlim(start_date, end_date)
    plt.ylim(0.5,2.0)

    ax = qlearn_portfolio.plot(title="Figure 1. QLearning Strategy vs. Manual Rule-based vs. Benchmark with 0.002 Impact", fontsize=12, color="green", label="QLearner-based")
    benchmark_portfolio.plot(ax=ax, color="black", label="Benchmark")
    manual_potfolio.plot(ax=ax,color="blue",label="Manual Rule-based")
    ax.set_ylabel('Normalized Value')
    ax.set_xlabel('Dates')
    
    plt.legend(['QLearner', 'Benchmark', "Manual Rule"], loc="lower right")
    plt.savefig("Impact1.png")
    #plt.show()


    #======================================================================================================================================================

    impact_2 = 0.007
    
    sl_2 = StrategyLearner()
    ms_2 = ManualStrategy()
    sl_2.addEvidence(symbol=symbol,sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=starting_value)
    qlearn_trades_2 =sl_2.testPolicy(symbol="JPM",sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=starting_value)
    manual_trades_2 = ms.testPolicy(symbol="JPM",  sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv=starting_value)
    benchmark_trades_2 = ms.get_benchmark_order(symbol="JPM",  sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv=starting_value)

    qlearn_portfolio_2 = (mm.compute_portvals(qlearn_trades_2, starting_value, commission, impact_2))/starting_value
    manual_potfolio_2 = (mm.compute_portvals(manual_trades_2, starting_value, commission, impact_2))/starting_value
    benchmark_portfolio_2 = (mm.compute_portvals(benchmark_trades_2, starting_value, commission, impact_2))/starting_value
    
    plt.clf()	
    plt.figure(figsize=(15,5))
    plt.xlim(start_date, end_date)
    plt.ylim(0.5,2.0)

    ax = qlearn_portfolio_2.plot(title="Figure 1. QLearning Strategy vs. Manual Rule-based vs. Benchmark with 0.007 Impact", fontsize=12, color="green", label="QLearner-based")
    benchmark_portfolio_2.plot(ax=ax, color="black", label="Benchmark")
    manual_potfolio_2.plot(ax=ax,color="blue",label="Manual Rule-based")
    ax.set_ylabel('Normalized Value')
    ax.set_xlabel('Dates')
    
    plt.legend(['QLearner', 'Benchmark', "Manual Rule"], loc="lower right")
    plt.savefig("Impact2.png")
    #plt.show()


if __name__ == '__main__':
    experiment2()