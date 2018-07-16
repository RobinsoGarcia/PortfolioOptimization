# Installation
The package needs to be installed as sudo user in order to add the main scripts to your /usr/local/bin folder to enable calling from any directory in the command line.

# Main scripts:

(1) build_dataset.py

'''Description:
This script allows you to build a dataframe out of individual csv files
containing historical stock quotes obtained from any source of your choice.
The required positional argument is the path to the folder in which the files where saved.
The output comes in the form of a csv file that will be used for the other scripts.
'''
usage: build_dataset.py [-h] stock_folder

positional arguments:
  stock_folder  path to a directory that contains one csv file with the PRICE
                history for each stock that needs to be analyzed. The only two
                headers on the csv files are the 'Date' and 'Ticker'.

optional arguments:
  -h, --help    show this help message and exit

(2) check_returns.py

'''Description
This script takes a positional argument to the path of a folder containing a csv file named
"Daily_closing_prices.csv'. This file should be a panel with a columns for 'Date'
(time series on the format YYYY-MM-DD) and additional columns for each stock being studied (quotes/prices).
The output is a line plot of the historical prices of each stock over time, the cumulative
return over time and the mean and std of the returs for the time period being evaluated.
'''

usage: check_returns.py [-h] stock_folder

positional arguments:
  stock_folder  path to a directory that contains a csv file named
                'Daily_closing_prices.csv'. This file should be a panel with a
                column for 'Date' (time series on the format YYYY-MM-DD) and
                additional columns for each stock being studied
                (quotes/prices).IMPORTANT: the headers must be the name of the
                files on the stock_folder and one header should refer to the
                time series and be names 'Date'

optional arguments:
  -h, --help    show this help message and exit

(3) check_allport.py

usage: check_allport.py [-h] [-sf STOCK_FOLDER]
                        [-sb SHARE_BALANCE [SHARE_BALANCE ...]]
                        [-s STOCKS [STOCKS ...]]

optional arguments:
  -h, --help            show this help message and exit
  -sf STOCK_FOLDER, --stock_folder STOCK_FOLDER
                        folder containing stock history/csv files
  -sb SHARE_BALANCE [SHARE_BALANCE ...], --share_balance SHARE_BALANCE [SHARE_BALANCE ...]
                        Quantity of shares for each stock
  -s STOCKS [STOCKS ...], --stocks STOCKS [STOCKS ...]
                        list of stock tickers


# Sample usage:

python check_rebal.py -s SANB11.SA CSAN3.SA SEER3.SA BBSE3.SA  -sb 10 10 10 10 10 -st 2016-07-15 -sf ~/Documents/portfolio-optimizer/portfolio_optimizer/stock_data -rf 2 -en 2018-05-12

python check_allport.py -s SANB11.SA CSAN3.SA SEER3.SA BBSE3.SA -sb 10 10 10 10 -sf ~/Documents/portfolio-optimizer/portfolio_optimizer/stock_data
