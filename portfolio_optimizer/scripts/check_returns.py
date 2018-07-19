import pandas as pd
import os
import portfolio_optimizer.portfolio.port as prt
import sys
import argparse
import numpy as np
from pandas.tseries.offsets import *
import matplotlib.pyplot as plt
import matplotlib

'''Description
This script takes a positional argument to the path of a folder containing a csv file named
"Daily_closing_prices.csv'. This file should be a panel with a columns for 'Date'
(time series on the format YYYY-MM-DD) and additional columns for each stock being studied (quotes/prices).
The output is a line plot of the historical prices of each stock over time, the cumulative
return over time and the mean and std of the returs for the time period being evaluated.
'''

if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("stock_folder",default="portfolio_optimizer/stock_data",help="path to a directory that contains a csv file named 'Daily_closing_prices.csv'. This file should be a panel with a column for 'Date' (time series on the format YYYY-MM-DD) and additional columns for each stock being studied (quotes/prices).IMPORTANT: the headers must be the name of the files on the stock_folder and one header should refer to the time series and be names 'Date'")

    args = parser.parse_args()

    try:
        data = pd.read_csv(os.path.join(args.stock_folder,'Daily_closing_prices.csv'),parse_dates=['Date'],index_col='Date')
    except:
        print('first build Daily_closing...')

    #returns = data.shift(1)/data-

    data.plot()

    returns = data.pct_change()[data.shift(1).notnull()].dropna()
    mu = np.array(returns.mean())

    Q = np.array(prt.cov(returns))

    print("#### Summary ####")
    print("\nmean:\n{}".format((returns.mean()+1)**252-1))
    print("\nstd\n{}".format(returns.std()*np.sqrt(252)))

    returns = returns + 1
    returns = returns.cumprod(axis=1)
    returns.plot()

    plt.show()
