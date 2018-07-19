#!/usr/bin/env python
import pandas as pd
import os
import portfolio_optimizer
import sys
import argparse

'''Description:
This script allows you to build a dataframe out of individual csv files
containing historical stock quotes obtained from any source of your choice.
The required positional argument is the path to the folder in which the files where saved.
The output comes in the form of a csv file that will be used for the other scripts.
'''

def build(args):
    #list csv files
    stocks_series = os.listdir(args.stock_folder)
    try:
        stock_series = stocks_series.remove('Daily_closing_prices.csv')
    except:
        pass

    #read csv files
    data = {}
    for stock in stocks_series:
        data[stock]= pd.read_csv(os.path.join(args.stock_folder,stock),parse_dates=['Date'],index_col='Date')['Close']

    #build dataframe
    stock_series = pd.DataFrame()
    columns = []
    for i in data.keys():
        stock_series = pd.concat([stock_series,data[i]],axis=1)
        columns.append(i)

    stock_series.columns = columns
    return stock_series

if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("stock_folder",default="portfolio_optimizer/stock_data",help="path to a directory that contains one csv file with the PRICE history for each stock that needs to be analyzed. The only two headers on the csv files are the 'Date' and 'Ticker'.")

    args = parser.parse_args()

    stock_series = build(args)

    stock_series.to_csv(os.path.join(args.stock_folder,'Daily_closing_prices.csv'))

    print("New file added to: ",args.stock_folder)
    print(os.listdir(args.stock_folder))
