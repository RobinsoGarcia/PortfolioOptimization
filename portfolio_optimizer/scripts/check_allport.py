import pandas as pd
import os
import portfolio_optimizer.portfolio.port as prt
import sys
import argparse
import numpy as np
from pandas.tseries.offsets import *
import matplotlib.pyplot as plt
'''python check_allport.py -s SANB11.SA CSAN3.SA SEER3.SA BBSE3.SA -sb 10 10 10 10 -sf ~/Documents/portfolio-optimizer/portfolio_optimizer/stock_data'''

'''Description
This script computes the efficient frontier simulate a number of portfolios including the ones with minimum
variance, maximum return and maximum sharpe ratio. It takes  a positional argument to the path of a folder containing a csv file named
"Daily_closing_prices.csv'. This file should be a panel with a columns for 'Date'
(time series on the format YYYY-MM-DD) and additional columns for each stock being studied (quotes/prices).
In addition, the user must inform the list of stocks (REPEAT the name of the files without the .csv part),
a series/sequence of integers representing the number of stocks for each stock, eg: for a portfolio_optimizer
with 4 stocks (AAPL YHOO BLAB AKLL) having 1000, 2000, 300, 4000 stocks for each asset correspondly, the series
should simply be 1000 2000 300 4000.
'''
if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-sf","--stock_folder",default="portfolio_optimizer/stock_data",help="folder containing stock history/csv files")

    parser.add_argument("-sb","--share_balance",nargs='+',type=int,help="Quantity of shares for each stock")

    parser.add_argument("-s","--stocks",nargs='+',help="list of stock tickers")

    parser.add_argument("-rf","--risk_free_rate",nargs='+',help="Risk free rate to be considered: influences the max sharpe portfolio")

    args = parser.parse_args()

    try:
        data = pd.read_csv(os.path.join(args.stock_folder,'Daily_closing_prices.csv'),parse_dates=['Date'],index_col='Date')
    except:
        print('first build Daily_closing...')

    try:
        rf = args.rf
    except:
        rf = 0.0001

    '''remove .csv from file name and make sure columns are in the same order as share balance'''
    data.columns = [i[:-4] for i in data.columns]

    data.plot()
    data = data[args.stocks]

    dic = {'shares':args.stocks,'share_balance':args.share_balance}

    returns = data.pct_change()[data.shift(1).notnull()].dropna()
    mu = np.array(returns.mean())

    Q = np.array(prt.cov(returns))

    print("\n#### Return Summary ####")
    print("\nmean:\n{}".format((returns.mean()+1)**252-1))
    print("\nstd:\n{}".format(returns.std()*np.sqrt(252)))

    returns = returns + 1
    returns = returns.cumprod(axis=1)
    returns.plot()

    kwargs = {'stocks':dic['shares'],'share_balance':dic['share_balance'],'rf':rf}

    portf = prt.effFront(**kwargs)
    portf.optimize(Q,mu,plot=1)

    plt.show()
