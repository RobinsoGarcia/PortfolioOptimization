import port as prt
import backtest as bkt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
%matplotlib inline
'''
Compares the sharpe optimized portfolio with
the max shapre obtained from a  dicrete
efficient get_eff_frontier
'''

def load_info():
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

returns,data,V_0,share_balance,stocks = load_info()

strategies = {'strat_max_Sharpe':prt.strat_max_Sharpe(stocks=stocks,share_balance = share_balance,V = V_0),
                'strat_max_Sharpe_eff':prt.strat_max_Sharpe(stocks=stocks,share_balance = share_balance,V = V_0,use_eff=1)}

value = []
strat = []
daily_returns = {}
summary = {}
for x in strategies:
    print(x)
    port = strategies[x]
    w,log = bkt.backtest_port(port,data,t0=2,dt=2,rf=0.0001,T=24,start= '1/2/2015',plot_eff=1)
    value.append(float((port.V_hist[-1:]-port.V_hist[0])/port.V_hist[0]))
    strat.append(x)
    daily_returns[x]=port.dailyV
    summary[x]=log['value'].map('${:,.2f}'.format)
summary['beg'] = log['beg']
summary['end'] = log['end']
summary = pd.DataFrame(summary)
summary.to_csv('summary.csv')


total_ret = pd.DataFrame(data=value,index=strat,columns=['return'])
total_ret.T.plot(title='portfolio total return',table=True,kind='bar',alpha=0.5,sort_columns=True,use_index=False)
day_ret = pd.DataFrame(daily_returns,columns=daily_returns.keys(),index=port.time)
day_ret.plot()
print(summary)
print(total_ret)
#plt.show()
