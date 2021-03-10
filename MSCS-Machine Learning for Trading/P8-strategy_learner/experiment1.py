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

def get_portfolio_stats(port_val, daily_rf=0, samples_per_year=252):
    daily_ret = (port_val / port_val.shift(1)) - 1
    cr = (port_val[-1] / port_val[0]) - 1
    adr = daily_ret.mean()
    sddr = daily_ret.std()
    k = np.sqrt(samples_per_year)
    sr = k * np.mean(adr - daily_rf) /sddr
    return cr, adr, sddr, sr

def experiment1():

    sl = StrategyLearner()
    ms = ManualStrategy()
    start_date = dt.datetime(2008, 1, 1)
    end_date = dt.datetime(2009, 12, 31)

    symbol = 'JPM'
    starting_value = 100000
    commission = 0
    impact = 0

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

    ax = qlearn_portfolio.plot(title="Figure 1. QLearning Strategy vs. Manual Rule-based vs. Benchmark", fontsize=12, color="green", label="QLearner-based")
    benchmark_portfolio.plot(ax=ax, color="black", label="Benchmark")
    manual_potfolio.plot(ax=ax,color="blue",label="Manual Rule-based")
    ax.set_ylabel('Normalized Value')
    ax.set_xlabel('Dates')
    
    plt.legend(['QLearner', 'Benchmark', "Manual Rule"], loc="lower right")
    plt.savefig("In-Sample.png")
    #plt.show()

    q_cr, q_adr, q_sddr, q_sr = get_portfolio_stats(qlearn_portfolio)
    m_cr, m_adr, m_sddr, m_sr = get_portfolio_stats(manual_potfolio)
    b_cr, b_adr, b_sddr, b_sr = get_portfolio_stats(benchmark_portfolio)
    
    print "Cumulative Return of {}: {}".format("QLearner", q_cr)
    print "Cumulative Return of {}: {}".format("ManualStrategy", m_cr)
    print "Cumulative Return of {}: {}".format("Benchmark", b_cr)
    print "Mean Daily Return of {}: {}".format("QLearner", q_adr)
    print "Mean Daily Return of {}: {}".format("ManualStrategy", m_adr)
    print "Mean Daily Return of {}: {}".format("Benchmark", b_adr)
    print "Standard Deviation of daily return of {}: {}".format("QLearner", q_sddr)
    print "Standard Deviation of daily return of {}: {}".format("ManualStrategy", m_sddr)
    print "Standard Deviation of daily return of {}: {}".format("Benchmark", b_sddr)


if __name__ == '__main__':
    experiment1()