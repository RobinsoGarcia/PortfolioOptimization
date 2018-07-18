import pandas as pd
import os
import portfolio_optimizer.portfolio.port as prt
import portfolio_optimizer.backtest as fn
import sys
import argparse
import numpy as np
from pandas.tseries.offsets import *
import matplotlib.pyplot as plt

'''python check_rebal.py -s SANB11.SA CSAN3.SA SEER3.SA BBSE3.SA -sb 10 10 10 10 -st 2016-07-15 -sf ~/Documents/portfolio-optimizer/portfolio_optimizer/stock_data -rf 2 -en 2018-05-12
'''

if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-sf","--stock_folder",required=True,default="portfolio_optimizer/stock_data",help="folder containing stock history/csv files")

    parser.add_argument("-sb","--share_balance",nargs='+',type=int,required=True,help="Quantity of shares for each stock")

    parser.add_argument("-st","--start",help="starting point for the simulation YYYY-MM-DD")

    parser.add_argument("-en","--end",help="ending point for the simulation YYYY-MM-DD")

    parser.add_argument("-s","--stocks",nargs='+',help="list of stock tickers")

    parser.add_argument("-rf","--rebal_freq",type=int,default=2,help="Rebalancing frequency in months")

    args = parser.parse_args()

    print("lalalal")

    try:
        data = pd.read_csv(os.path.join(args.stock_folder,'Daily_closing_prices.csv'),parse_dates=['Date'],index_col='Date')
    except:
        print('first build Daily_closing...')

    '''remove .csv from file name and make sure columns are in the same order as share balance'''
    data.columns = [i[:-4] for i in data.columns]

    data = data[args.stocks]


    try:
        v = data.loc[args.start].T.values*args.share_balance
        V = sum(v) #Portfolio Value start
        print(V)
    except:
        print("couldnt print V")
        pass

    dic = {'shares':args.stocks,'share_balance':args.share_balance,
            'start':pd.to_datetime(args.start),'end':pd.to_datetime(args.end),'V':V,'data':data,'rebal_freq':args.rebal_freq}

    fn.evaluate(dic)
