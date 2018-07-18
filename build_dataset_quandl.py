import quandl
import datetime
import pandas as pd
import argparse
#https://chrisconlan.com/download-historical-stock-data-google-r-python/

quandl.ApiConfig.api_key = 'API key'

def get_from_quandl(symbol, start_date=(2000, 1, 1), end_date=None):
    """
    symbol is a string representing a stock symbol, e.g. 'AAPL'

    start_date and end_date are tuples of integers representing the year, month,
    and day

    end_date defaults to the current date when None
    """
    #if quandl.ApiConfig.api_key == 'use your key':
    #    quandl.ApiConfig.api_key = input("enter here your quandl API key or go get one @ (https://www.quandl.com/tools/api)")

    query_list = ['WIKI' + '/' + symbol + '.' + str(k) for k in range(1, 13)]

    start_date = datetime.date(*start_date)

    if end_date:
        end_date = datetime.date(*end_date)
    else:
        end_date = datetime.date.today()

    return quandl.get(query_list,
            returns='pandas',
            start_date=start_date,
            end_date=end_date,
            collapse='daily',
            order='asc'
            )

import numpy as np

def get_quotes(symbol, start_date=(2000, 1, 1), end_date=None):

    data = get_from_quandl(symbol, start_date, end_date=None)
    close = [x for x in data.columns if 'Close' in x][0]

    return data[close]

if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-s","--stocks",nargs='+',help="list of stock tickers")

    parser.add_argument("-st","--start",help="starting point for the simulation YYYY-MM-DD")

    parser.add_argument("-en","--end",help="ending point for the simulation YYYY-MM-DD")

    args = parser.parse_args()

    if args.start==None:
        args.start = (2000, 1, 1)

    data = []

    for stock in args.stocks:
        temp_data = get_quotes(stock,start_date=args.start, end_date=args.end)
        data.append(temp_data)

    data = pd.concat(data,axis=1,join='inner')

    data.to_csv('Daily.csv')
    print(data)
