import pickle #https://www.saltycrane.com/blog/2008/01/saving-python-dict-to-file-using-pickle/
from portfolio_optimizer import evaluate_port
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
from pandas_datareader import data

pkl_file = open('init_port.pkl', 'rb')
init_port = pickle.load(pkl_file)
pkl_file.close()

data = {'init_port':init_port,'data_source':'toy',
'start_date':'2015-01-01','end_date': '2016-12-31'}
strategies = ['strat_buy_and_hold','strat_equally_weighted',
'strat_max_Sharpe','strat_max_Sharpe_eff','strat_min_variance']
backtest = {'t0':2,'dt':2,'rf':0.0001,'T':24,'plot_eff':0}

evaluate_port.evaluate(strategies,data)


data = {'init_port':init_port,'data_source':'yahoo',
'start_date':'2015-01-01','end_date': '2016-12-31'}
strategies = ['strat_buy_and_hold','strat_equally_weighted',
'strat_max_Sharpe','strat_max_Sharpe_eff','strat_min_variance']
backtest = {'t0':2,'dt':2,'rf':0.0001,'T':24,'plot_eff':0}

evaluate_port.evaluate(strategies,data)
