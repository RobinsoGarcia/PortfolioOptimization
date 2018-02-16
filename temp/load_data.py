import portfolio_optimizer.port as prt
import portfolio_optimizer.backtest as bkt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

#%matplotlib inline
plt.style.use('ggplot')

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
    start = '1/2/2015'
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
    return returns,data,V_0,share_balance,stocks
