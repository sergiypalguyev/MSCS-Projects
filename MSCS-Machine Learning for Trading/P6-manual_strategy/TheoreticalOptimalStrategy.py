import pandas as pd
import numpy as np
import datetime as dt
import os
from util import get_data, plot_data
import matplotlib.pyplot as plt
import marketsimcode as mm
from indicators import *

class TheoreticalOptimalStrategy(object):

    def __init__(self,symbol="JPM", sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31),sv=100000):
        self.symbol=symbol
        self.sd=sd
        self.ed=ed
        self.sv=sv

    def author(self):
        return 'spalguyev3'

    def get_optimal_orders(self, symbol, sd, ed, sv):
        symbols=[]
        symbols.append(symbol)
        dates=pd.date_range(sd,ed)
        price=get_data(symbols,dates)
        order_list = []
        curr_holding = 0

        for i in range(len(price)-1):
            if price.ix[i, symbol] > price.ix[i + 1, symbol] and curr_holding > -1000:
                    curr_holding -= 1000
                    order_list.append([price.index[i], symbol, "SELL", 1000])
            elif price.ix[i, symbol] < price.ix[i + 1, symbol] and curr_holding < 1000:
                    curr_holding += 1000
                    order_list.append([price.index[i], symbol, "BUY", 1000])
            else:
                order_list.append([price.index[i],symbol,"HOLD",0])

        trades_df = pd.DataFrame(order_list, columns=["Date", "Symbol", "Order", "Shares"])
        
        return trades_df

    def testPolicy(self, symbol, sd, ed, sv):
        return self.get_optimal_orders(symbol,sd,ed,sv)

    def get_benchmark_order(self, symbol, sd, ed, sv):
        dates = pd.date_range(sd, ed)
        symbols=[]
        symbols.append(symbol)
        # get the benchmark prices, important, is to get the trade dates
        benchmark_prices = get_data(symbols, dates)
        orders = []
        for i in range(len(benchmark_prices)):
            if i==0:
                orders.append([benchmark_prices.index[0], symbol, "BUY", 1000])
            else:
                orders.append([benchmark_prices.index[i], symbol, "HOLD",0])

        benchmark_trades = pd.DataFrame(orders, columns=["Date", "Symbol", "Order", "Shares"])

        return benchmark_trades

if __name__ == "__main__":
    tos = TheoreticalOptimalStrategy()
    df_trades = tos.testPolicy(symbol="JPM",  sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv=100000) 
    df_benchmark = tos.get_benchmark_order(symbol="JPM",  sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv=100000)

    optimal_portfolio = mm.compute_portvals(df_trades, 100000)
    benchmark_portfolio = mm.compute_portvals(df_benchmark, 100000)   

    optimal_portfolio = optimal_portfolio/optimal_portfolio[0]
    benchmark_portfolio = benchmark_portfolio/benchmark_portfolio[0]

    plt.clf
    plt.figure(figsize=(15,5))   
    plt.xlim(dt.datetime(2008,1,1), dt.datetime(2009,12,31))
    plt.title("Theoretical Optimal Strategy")
    plt.gca().set_color_cycle(['black','blue'])
    plt.plot(optimal_portfolio)
    plt.plot(benchmark_portfolio)
    plt.legend(['Optimal','Benchmark'], loc="lower right")
    plt.xlabel('Dates')
    plt.ylabel('Normalized Prices')
    plt.savefig("Strategy-Optimal.png")	
    plt.clf