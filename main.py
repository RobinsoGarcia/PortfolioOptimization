import port as prt
import backtest as bkt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
%matplotlib inline
plt.style.use('ggplot')

def load_info():
    '''Loading data'''
    '''Share balance = number of shares per stock in the portfolio'''
    '''start = starting point'''
    '''V_0 = value of the initial portfolio at start'''
    data,returns = prt.load_data()
    start = '1/2/2015'
    init_port = {'MSFT':5000,'F':950,'CRAY':2000,'VZ':2000,'AAPL':3000,'IBM':1500,'NVDA':1001}
    portfolio=[]
    for x in data.columns:
        if x in init_port:
            portfolio.append(init_port[x])
        else:
            portfolio.append(0)

    share_balance = np.array(portfolio)[np.newaxis,:]

    V_0 = [data.loc[start][x]*init_port[x] for x in init_port]
    return returns,data,V_0,share_balance

returns,data,V_0,share_balance = load_info()

port1 = prt.strat_buy_and_hold(share_balance = share_balance,V = V_0)
port1.switch_2_buy_and_hold()
bkt.backtest_port(port1,data,t0=2,dt=2,rf=0.0001,T=24,start= '1/2/2015')

port1 = prt.strat_equally_weighted(share_balance = share_balance,V = V_0)
port1.use_eff=0
bkt.backtest_port(port1,data,t0=2,dt=2,rf=0.0001,T=24,start= '1/2/2015')

port1 = prt.strat_max_Sharpe(share_balance = share_balance,V = V_0)
port1.use_eff=0
bkt.backtest_port(port1,data,t0=2,dt=2,rf=0.0001,T=24,start= '1/2/2015')

port2 = prt.strat_max_Sharpe(share_balance = share_balance,V = V_0)
port2.use_eff=1
bkt.backtest_port(port2,data,t0=2,dt=2,rf=0.0001,T=24,start= '1/2/2015')

port3 = prt.strat_min_variance(share_balance = share_balance,V = V_0)
bkt.backtest_port(port3,data,t0=2,dt=2,rf=0.0001,T=24,start= '1/2/2015')
