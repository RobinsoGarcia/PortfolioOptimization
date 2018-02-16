import portfolio_optimizer.port as prt
import portfolio_optimizer.backtest as bkt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import pandas_datareader as pr
from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd

#%matplotlib inline
plt.style.use('ggplot')
def help():
    print("data_reader(init_port = {'MSFT':5000,'F':950,'CRAY':2000,'VZ':2000,'AAPL':3000,'IBM':1500,'NVDA':1001},data_source = 'yahoo',start_date = '2000-01-01',end_date = '2016-12-31'))")
    pass

def A1():
    '''
    INPUT
    prt.load_data(): loads stock data from a csv file into a dataframe
    OUTPUT
    data: dataframe with stock daily prices information
    returns: dataframe with stock daily returns (percent change) information
    Share balance: number of shares per stock in the portfolio
    Start: date of the initial investment
    V_0: total value of the initial portfolio at start
    '''
    data,returns = prt.load_data()
    start = '2015-01-02'
    init_port = {'MSFT':5000,'F':950,'CRAY':2000,'VZ':2000,'AAPL':3000,'IBM':1500,'NVDA':1001}

    portfolio=[]
    stocks = data.columns
    for x in stocks:
        if x in init_port:
            portfolio.append(init_port[x])
        else:
            portfolio.append(0)

    share_balance = np.array(portfolio)[np.newaxis,:]

    V_0 = [data.loc[start][x]*init_port[x] for x in init_port]
    print("data description:\n\nmeans: \n{}\n\nstds: \n{}".format(data.mean(),data.std()))
    print("data description:\n\nmax: \n{}\n\nmin: \n{}".format(data.max(),data.min()))

    return returns,data,V_0,share_balance,stocks


def data_reader(dic):
    #http://www.learndatasci.com/python-finance-part-yahoo-finance-api-pandas-matplotlib/

    init_port = dic['init_port']
    # Define the instruments to download. We would like to see Apple, Microsoft and the S&P500 index.
    tickers = list(init_port.keys())

    # Define which online source one should use
    data_source = dic['data_source']

    # We would like all available data from 01/01/2000 until 12/31/2016.
    start_date = pd.Timestamp(dic['start_date'])
    end_date = pd.Timestamp(dic['end_date'])

    # User pandas_reader.data.DataReader to load the desired data. As simple as that.
    panel_data = data.DataReader(tickers, data_source, start_date, end_date)
    data_ = panel_data.loc['Close'].sort_index(ascending=True)
    
    count=0
    while start_date not in data_.index:
        start_date = start_date + pd.DateOffset(days=1)
        count +=1
        if count > 15:
            print('date mismatch')
            print(start_date-pd.DateOffset(days=count))
            break


    tickers = data_.columns
    share_balance = []
    for x in tickers:
        share_balance.append(init_port[x])
    share_balance = np.array(share_balance)[np.newaxis,:]

    print("sanity-check, tickers in the right order: {}".format(data_.columns==tickers))

    returns = data_.shift(1)/data_-1
    end = end_date
    start = start_date

    V_0 = [data_.loc[start][x]*init_port[x] for x in init_port]
    print(np.sum(V_0))
    data_.to_csv('yahoo-fin-data.csv')
    if start != start_date:
        print('data_reader wasnt able to retireve all data requested:')
        print('retrieved start_date: {}'.format(start))
        print('retrieved end_date: {}'.format(end))

    print("data description:\n\nmeans: \n{}\n\nstds: \n{}".format(data_.mean(),data_.std()))
    print("data description:\n\nmax: \n{}\n\nmin: \n{}".format(data_.max(),data_.min()))
    return returns,data_,V_0,share_balance,tickers
