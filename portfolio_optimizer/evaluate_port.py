import portfolio_optimizer.port as prt
import portfolio_optimizer.backtest as bkt
from portfolio_optimizer import load_data
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

#%matplotlib inline
plt.style.use('ggplot')
def help():
    print("The default parameters are:\n t0=2,dt=2,rf=0.0001,T=24,start= '1/2/2015',plot_eff=0")
    pass

def evaluate(strategy_list,port_data,backtest,t0=2,dt=2,rf=0.025/252,T=24,plot_eff=0):

    if backtest is not None:
        t0 = backtest['t0']
        T = backtest['T']
        dt = backtest['dt']
        rf = backtest['rf']
        plot_eff = backtest['plot_eff']

    if port_data['data_source']=='yahoo':
        returns,data,V_0,share_balance,stocks = load_data.data_reader(port_data)
        print('loaded data from yahoo!')
    else:
        returns,data,V_0,share_balance,stocks  = load_data.A1()
        print('loaded toy data!')
    print(data.index[0])
    strategies = {'strat_buy_and_hold':prt.strat_buy_and_hold(stocks=stocks,share_balance = share_balance,V = V_0,buy_h=1),
                    'strat_equally_weighted':prt.strat_equally_weighted(stocks=stocks,share_balance = share_balance,V = V_0),
                    'strat_max_Sharpe':prt.strat_max_Sharpe(stocks=stocks,share_balance = share_balance,V = V_0),
                    'strat_max_Sharpe_eff':prt.strat_max_Sharpe(stocks=stocks,share_balance = share_balance,V = V_0,use_eff=1),
                    'strat_min_variance':prt.strat_min_variance(stocks=stocks,share_balance = share_balance,V = V_0)}

    print("This script calls the backtest.py and port.py files\n ")
    value = []
    strat = []
    daily_returns = {}
    summary = {}
    for x in strategy_list:
        print(x)
        port = strategies[x]
        w,log = bkt.backtest_port(port,data,start= port_data['start_date'],end= port_data['end_date'],t0=t0,dt=dt,rf=rf,T=T,plot_eff=plot_eff)
        value.append(float((port.V_hist[-1:]-port.V_hist[0])/port.V_hist[0]))
        strat.append(x)
        daily_returns[x]=port.dailyV
        summary[x]=log['value'].map('${:,.2f}'.format)
    summary['beg'] = log['beg']
    summary['end'] = log['end']
    summary = pd.DataFrame(summary)

    total_ret = pd.DataFrame(data=value,index=strat,columns=['return'])
    total_ret.T.plot(title='portfolio total return',table=True,kind='bar',alpha=0.5,sort_columns=True,use_index=False)
    day_ret = pd.DataFrame(daily_returns,columns=daily_returns.keys(),index=port.time)
    day_ret.plot()
    print(summary)
    print(total_ret)
    pass
