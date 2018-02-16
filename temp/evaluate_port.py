import portfolio_optimizer.port as prt
import portfolio_optimizer.backtest as bkt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

#%matplotlib inline
plt.style.use('ggplot')
def help():
    print("The default parameters are:\n t0=2,dt=2,rf=0.0001,T=24,start= '1/2/2015',plot_eff=0")
    pass

def evaluate(strategies,data,t0=2,dt=2,rf=0.0001,T=24,start= '1/2/2015',plot_eff=0):
    print("This script calls the backtest.py and port.py files\n ")
    value = []
    strat = []
    daily_returns = {}
    summary = {}
    for x in strategies:
        print(x)
        port = strategies[x]
        w,log = bkt.backtest_port(port,data,t0=t0,dt=dt,rf=rf,T=T,start= start,plot_eff=plot_eff)
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
    pass
