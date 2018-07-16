import pandas as pd
from pandas.tseries.offsets import *
import numpy as np
import matplotlib.pyplot as plt
import portfolio_optimizer.portfolio.port as prt


def get_data(data):
    returns = data.pct_change()[data.shift(1).notnull()].dropna()
    mu = np.array(returns.mean())
    Q = np.array(prt.cov(returns))
    return Q,mu

def backtest(obj,d1,start,end,params,rf=0.001):
    log = {'beg':[],'end':[],'value':[],'cash':[]}
    index=[start]
    back= params['back']
    forward = params['forward']

    offset = BMonthEnd()
    start = offset.rollback(start)

    t0 = start+DateOffset(**back)
    roll_forward = DateOffset(**forward)
    dates = pd.date_range(start+roll_forward,end+roll_forward,freq=roll_forward)

    W = []
    for i in dates:

        data_window = d1.loc[:i].copy()

        Q,mu = get_data(data_window)
        p = d1.loc[i-roll_forward:i].copy()

        obj.update_V(p,rf)

        w = obj.optimize(Q,mu,stats=0)

        obj.update_cash_account()

        log['beg'].append(i-roll_forward)
        log['end'].append(i)
        index.append(i)

    log['value'] = obj.V_hist[1:]

    if obj.buy_h==1:
        log['cash']= np.zeros(len(log['end']))
    else:
        log['cash'] = obj.cash_hist

    log = pd.DataFrame(log,columns=log.keys())
    real_ret = (obj.V_hist[-1:]-obj.V_hist[0])/obj.V_hist[0]
    print("\n Total portfolio Valorization: {}".format(real_ret))
    W = np.vstack(obj.W)

    W = pd.DataFrame(data=W,columns=obj.stocks)

    W.plot(kind='bar',title="Dynamic allocation "+obj.method,stacked=True).legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #plt.show()
    return W,log

def evaluate(dic):
    kwargs = {'stocks':dic['shares'],'share_balance':dic['share_balance'],'V':dic['V']}

    strategies = {'strat_buy_and_hold':prt.strat_buy_and_hold(**kwargs,buy_h=1),
                    'strat_equally_weighted':prt.strat_equally_weighted(**kwargs),
                    #'strat_max_Sharpe':prt.strat_max_Sharpe(**kwargs),
                    'strat_max_Sharpe_eff':prt.strat_max_Sharpe(**kwargs,use_eff=1),
                    'strat_min_variance':prt.strat_min_variance(**kwargs),
                    'strat_equal_risk_contrib':prt.strat_equal_risk_contrib(**kwargs)}

    params= {"forward": {"months": dic['rebal_freq']},"back": {"months": 0}}
    value = []
    strat = []
    daily_returns = {}
    summary = {}
    for x in strategies:
        print(x)
        print('Portfolio initil value: {}'.format(np.sum(dic['V'])))

        port = strategies[x]
        W,log = backtest(port,dic['data'],dic['start'],dic['end'],params,.0001) #.0001 interest

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
    plt.show()
    print(summary)
    print(total_ret)
    pass
