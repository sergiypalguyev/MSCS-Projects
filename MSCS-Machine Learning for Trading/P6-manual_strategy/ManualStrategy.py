import pandas as pd
import numpy as np
import datetime as dt
import os
from util import get_data, plot_data
import matplotlib.pyplot as plt
import marketsimcode as mm
from indicators import *

class ManualStrategy(object):

    def __init__(self,symbol="JPM", sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31),sv=100000):
        self.symbol=symbol
        self.sd=sd
        self.ed=ed
        self.sv=sv

    def author(self):
        return 'spalguyev3'

    def testPolicy(self, symbol = 'JPM', sd=dt.datetime(2010,1,1), ed=dt.datetime(2011,12,31), sv = 100000):
        return self.get_manual_orders(symbol, sd, ed, sv)

    def get_manual_orders(self, symbol = 'JPM', sd=dt.datetime(2010,1,1), ed=dt.datetime(2011,12,31), sv = 100000):
        orders = []
        lookback = 10
        holdings=0
    
        dates = pd.date_range(sd,ed)
        
        prices = get_data([symbol],dates)
        
        df_indicators = indicators(prices, symbol)
        
        sma_short = df_indicators['sma_short']
        sma_long = df_indicators['sma_long']
        bbu = df_indicators['bb_up']
        bbd = df_indicators['bb_down']
        moment = df_indicators['momentum']
        
        prices = prices[symbol]
        prices = prices/prices[0]

        for day in range(lookback+1,df_indicators.shape[0]): 
            #print "day", day, "price", prices[day],"sma_short", sma_short.ix[day], "sma_long", sma_long.ix[day], "bbu", bbu.ix[day], "bbd", bbd.ix[day], "bbp", bbp.ix[day], "momentum", moment.ix[day], "holding", holdings[sym]

            # Two Moving Averages "Golden Cross" strategy
            # If sma_short crosses down sma_long = SELL
            # If sma_short crosses up sma_long = BUY
            yesterday_sma_diff = round(sma_short.ix[day-1] - sma_long.ix[day-1], 3) 
            today_sma_diff = round(sma_short.ix[day] - sma_long.ix[day], 3) 

            # Standard Bollinger Band strategy
            # If price crosses bb_up, then goes down = SELL
            # If price crosses bb_down, then goes up = BUY
            yesterday_price_bbu_diff = round(prices[day-1] - bbu.ix[day-1], 3) 
            yesterday_price_bbd_diff = round(prices[day-1] - bbd.ix[day-1], 3) 
            today_price_bbu_diff = round(prices[day] - bbu.ix[day] , 3) 
            today_price_bbd_diff = round(prices[day] - bbd.ix[day], 3) 

            # Momentum crossover strategy
            # If momentum crosses below moving average = SELL
            # If momentum crosses above moving average = BUY
            yesterday_momentum_diff = round(moment.ix[day-1] - sma_short.ix[day-1], 3) 
            today_momentum_diff = round(moment.ix[day] - sma_short.ix[day], 3) 

            if (yesterday_sma_diff > 0 and today_sma_diff < 0) or\
            (yesterday_price_bbu_diff > 0 and today_price_bbu_diff < 0) or\
            (yesterday_momentum_diff < 0 and today_momentum_diff > 0):
                if holdings>-1000:
                    holdings -= 1000
                    orders.append([prices.index[day].date(),symbol,'SELL',1000, holdings])
                else:
                    orders.append([prices.index[day].date(),symbol,'HOLD',0, holdings])
            elif (yesterday_sma_diff < 0 and today_sma_diff > 0) or\
            (yesterday_price_bbd_diff < 0 and today_price_bbd_diff > 0) or\
            (yesterday_momentum_diff > 0 and today_momentum_diff < 0):
                if holdings<1000:
                    holdings += 1000
                    orders.append([prices.index[day].date(),symbol,'BUY',1000, holdings])   
                else:      
                    orders.append([prices.index[day].date(),symbol,'HOLD',0, holdings])
            else:
                orders.append([prices.index[day].date(),symbol,'HOLD',0, holdings])

        manual_orders=pd.DataFrame(orders)
        manual_orders.columns=['Date','Symbol','Order','Shares', 'Holdings']
        
        return manual_orders

    def get_benchmark_order(self, symbol,sd,ed,sv):
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
    ms = ManualStrategy()
    df_trades = ms.testPolicy(symbol="JPM",  sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv=100000) 
    df_benchmark = ms.get_benchmark_order(symbol="JPM",  sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv=100000)

    manual_potfolio = mm.compute_portvals(df_trades, 100000)
    benchmark_portfolio = mm.compute_portvals(df_benchmark, 100000)  
    

    plt.figure(figsize=(15,5))
    plt.xlim(dt.datetime(2008,1,1), dt.datetime(2009,12,31))
    plt.title("Manual Strategy In-Sample")
    plt.gca().set_color_cycle(['blue','black', 'green', 'red'])
    plt.legend(loc="lower right")
    
    plt.plot(manual_potfolio)
    plt.plot(benchmark_portfolio)

    #Vertical green lines indicating LONG entry points.
    #Vertical red lines indicating SHORT entry points.
    for i in range(1,len(df_trades)):   
        today = df_trades.ix[i,"Holdings"]
        yesterday = df_trades.ix[i-1,"Holdings"]
        if yesterday > today:
            plt.axvline(x=df_trades['Date'][i],color='r')
        if yesterday < today:
            plt.axvline(x=df_trades['Date'][i],color='g')

        
    plt.legend(['Manual','Benchmark', 'Short', 'Long'], loc="lower right")
    plt.xlabel('Dates')
    plt.ylabel('Normalized Prices')
    #plt.show()
    plt.savefig("Strategy-Manual.png")	
    plt.clf



    plt.clf
    df_trades = ms.testPolicy(symbol="JPM",  sd=dt.datetime(2010,1,1), ed=dt.datetime(2011,12,31), sv=100000) 
    df_benchmark = ms.get_benchmark_order(symbol="JPM",  sd=dt.datetime(2010,1,1), ed=dt.datetime(2011,12,31), sv=100000)

    manual_potfolio = mm.compute_portvals(df_trades, 100000)
    benchmark_portfolio = mm.compute_portvals(df_benchmark, 100000)  
    

    plt.figure(figsize=(15,5))
    plt.xlim(dt.datetime(2010,1,1), dt.datetime(2011,12,31))
    plt.title("Manual Strategy Out-Of-Sample")
    plt.gca().set_color_cycle(['blue','black', 'green', 'red'])
    
    plt.plot(manual_potfolio)
    plt.plot(benchmark_portfolio)

    #Vertical green lines indicating LONG entry points.
    #Vertical red lines indicating SHORT entry points.
    for i in range(1,len(df_trades)):   
        today = df_trades.ix[i,"Holdings"]
        yesterday = df_trades.ix[i-1,"Holdings"]
        if yesterday > today:
            plt.axvline(x=df_trades['Date'][i],color='r')
        if yesterday < today:
            plt.axvline(x=df_trades['Date'][i],color='g')

        
    plt.legend(['Manual','Benchmark', 'Short', 'Long'], loc="lower right")
    plt.xlabel('Dates')
    plt.ylabel('Normalized Prices')
    plt.savefig("Strategy-Manual-OS.png")	
    plt.clf