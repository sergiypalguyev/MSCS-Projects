import pandas as pd
import numpy as np
import datetime as dt
import util as ut
import matplotlib.pyplot as plt

def momentum(data, lookback=10):
    data.iloc[:lookback] = np.nan
    price_a = data.iloc[lookback:]
    price_b = data.iloc[:-lookback]
    data.iloc[lookback:]= price_a/price_b.values - 1
    return  data

def std(data, window=10, center=False):
    return data.rolling(window=window,center=center).std()

def sma(data, window=10, center=False):
    sma_short = data.rolling(window=window, center=center).mean()
    sma_long = data.rolling(window=window*2, center=center).mean()
    return (sma_short, sma_long)

def bollinger(data, window=10, center=False):
    avg, avg_long = sma(data, window, center)
    dev = std(data, window, center)
    
    bb_up = avg + (2*dev)
    bb_down = avg - (2*dev)

    return (bb_up, bb_down)

def indicators(df_prices, sym, window=10, center=False):
    prices=df_prices[sym]
    prices=prices/prices[0]
    data = pd.DataFrame(index=prices.index)

    data['price'] = prices
    data['sma_short'], data['sma_long'] = sma(data=prices, window=window, center=center)
    data['bb_up'], data['bb_down'] = bollinger(data=prices, window=window, center=center)
    data['momentum'] = momentum(data=prices, lookback=window)
    
    return data

def author():
    return 'spalguyev3'

def test_code():
    start_date=dt.datetime(2008,1,1)
    end_date=dt.datetime(2009,12,31)
    dates = pd.date_range(start_date, end_date)
    
    symbol='JPM'
    prices = ut.get_data([symbol], dates)
    window = 10
    center = False

    # Simple Moving Average
    data = indicators(prices, symbol, window, center)
    plt.clf()
    plt.figure(figsize=(15,5))
    plt.title("JPM Simple Moving Average", loc="center")
    plt.xlim(start_date, end_date)
    plt.ylim(0.2,1.4)
    plt.gca().set_color_cycle(['black','blue', 'green'])
    plt.plot(data['price'], linestyle="-", linewidth=1)
    plt.plot(data['sma_short'], linestyle="-", linewidth=2)
    plt.plot(data['sma_long'], linestyle="-", linewidth=2)
    plt.legend(['Price','SMA (Short Window)', 'SMA (Long Window)'], loc="lower right")
    plt.ylabel('Normalized Price')
    plt.xlabel('Dates')
    #plt.show()
    plt.savefig("Indicator-Average.png")
    plt.clf()	

    # Bollinger
    data = indicators(prices, symbol, window, center)
    plt.clf()	
    plt.figure(figsize=(15,5))
    plt.title("JPM Bollinger Bands", loc="center")
    plt.xlim(start_date, end_date)
    plt.ylim(0.2, 1.4)
    plt.gca().set_color_cycle(['black','blue', 'green'])
    plt.plot(data['price'], linestyle="-", linewidth=1)
    plt.plot(data['bb_up'], linestyle="-", linewidth=2)
    plt.plot(data['bb_down'], linestyle="-", linewidth=2)
    plt.legend(['Price','Bollinger Band (upper)', 'Bollinger Band (lower)'], loc="lower right")
    plt.ylabel('Normalized Price')
    plt.xlabel('Dates')
    #plt.show()
    plt.savefig("Indicator-Bollinger.png")
    plt.clf()	

    # Momentum
    data = indicators(prices, symbol, window, center)
    plt.clf()	
    plt.figure(figsize=(15,5))
    plt.title("JPM Momentum", loc="center")
    plt.xlim(start_date, end_date)
    # plt.ylim(-0.4,1.4)
    plt.gca().set_color_cycle(['blue'])
    plt.plot(data['momentum'], linestyle="-", linewidth=2)
    plt.legend(['Momentum'], loc="lower right")
    plt.ylabel('Normalized Price and Momentum')
    plt.xlabel('Dates')
    #plt.show()
    plt.savefig("Indicator-Momentum.png")
    plt.clf()	


if __name__ == "__main__":
    test_code()