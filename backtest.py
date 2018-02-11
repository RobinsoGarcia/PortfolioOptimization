
import port as prt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def get_data(t,dt,T,start,data):
    period = pd.date_range(start, periods=T, freq='BMs',format='%Y-%m-%d')
    data = data.loc[:period[t]]
    returns = data.shift(1)/data-1
    Q = np.array(prt.cov(returns))
    mu = np.array(returns.mean())
    p = data[period[t-dt]:]
    return Q,mu,p.values,np.array(returns)

def backtest_port(port1,data,t0=2,dt=2,rf=0.0001,T=24,start= '1/2/2015') :

    portfolio_return = np.empty([1,3])
    w = port1.share_balance/np.sum(port1.V)

    for t in np.arange(t0,T,dt):

        Q,mu,p,ret = get_data(t,dt,24,start,data)
        port1.update_V(p,rf,t)

        w = port1.optimize(Q,mu)

        port1.update_cash_account()

    real_ret = (port1.V_hist[-1:]-port1.V_hist[0])/port1.V_hist[0]
    print("Total portfolio Valorization: {}".format(real_ret))
    #port1.plot_ret()
    plt.figure()
    port1.plot_cash()
    plt.figure()
    port1.plot_dailyV()
