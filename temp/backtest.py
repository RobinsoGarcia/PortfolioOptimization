
import portfolio_optimizer.port as prt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def get_business_day(date,data):
    count=1
    adjusted_date =[]
    for x in date:
        while x not in data.index:
            x = x + pd.DateOffset(days=1)
            count +=1
            if count > 15:
                print('date mismatch')
                print(x-pd.DateOffset(days=count))
                break
        adjusted_date.append(x)
    return adjusted_date

def get_period(start,T,data):
    period1 = pd.date_range(start, periods=T, freq='BMs',format='%Y-%m-%d')
    period2 = pd.date_range(start, periods=T, freq='BM',format='%Y-%m-%d')
    new_period1 = get_business_day(period1,data)
    new_period2 = get_business_day(period2,data)
    period = pd.DataFrame({'beg':new_period1,'end':new_period2})
    return period

def get_data(t,dt,period,data):
    '''
    This function generates a sequence of period intervals starting from start
    and increasing until T. The spot dates are the first business day of the
    month from start until T.
    INPUT
    t: time period to collect data up until.
    dt: interval between time periods.
    T: total number of periods to test the model.
    start: starting date range.
    data: dataframe with stock daily prices.
    OUTPUT
    Q: covariance matrix for the specified time period (adjusted to ensure PSD) [start:t].
    mu: average returns for each stock for the specified time period [start:t].
    p: stock price dataframe from last period up until t [t-1:t].
    returns: 2d array of stock returns for the specified period [start:t].
    '''

    data = data.loc[:period['end'][t-1]]
    returns = data.shift(1)/data-1
    Q = np.array(prt.cov(returns))
    mu = np.array(returns.mean())
    p = data[period['beg'][t-dt]:]
    return Q,mu,p

def backtest_port(port1,data,t0=2,dt=2,rf=0.0001,T=24,start= '1/2/2015',plot_eff=0,report=0) :
    '''
    backtest repeatedly get Q,mu,p,ret for a prespecifier time window.
    Q,mu and ret reflect all values from start until t, while p is
    a snapshot of the stock prices between t and the last period. p is
    used to estimate the daily value of the portolio given the last
    allocation.
    INPUT
    port1: portfolio object specified by the initial share_balance (allocation
    and initial portfolio value V_0.
    data: data: dataframe with stock daily prices (from start until T)
    t0: first portfolio value update. eg: 2, given initial share balance
    after two periods the value of the portfolio will be estimated. from
    that on the backtest function will continue to rebalance and update
    portfolio value and cash balances.
    t: time period to collect data up until.
    dt: interval between time periods.
    T: total number of periods to test the model.
    rf: interest rate used to optimize for sharpe portfolios and calculate
    interest for the amount retained on the cash account.
    start: starting date range.
    OUTPUT
    Total portfolio Valorization after T periods, rebalancing at each t time
    interval.
    Chart showing portfolio daily valorization.
    Chart showing variations in the cash cash_account.
    '''
    period = get_period(start,T,data)
    index = [period['beg'][0]]
    log = {'beg':[],'end':[],'value':[],'cash':[]}


    w = (np.array(data.loc[period['beg'][0]])*port1.share_balance)/np.sum(port1.V) #initial balance

    port1.w_buy_and_hold = w # for buy and hold portfolio, it has no effect on other strategies

    W = w
    print("Portfolio initial value: {}".format(np.sum(port1.V)))

    for t in np.arange(t0,T+dt,dt):

        Q,mu,p= get_data(t,dt,period,data)
        port1.update_V(p,rf,t) #update portfolio value on a daily basis for the period [t-dt:t]

        w = port1.optimize(Q,mu) #optimize, calculate new allocation

        port1.update_cash_account() #update cash account after new allocation is calculates
        if plot_eff==1:
            eff = prt.effFront(share_balance = port1.share_balance,V = port1.V)
            eff.optimize(Q,mu,plot=1)
        W = np.vstack([W,port1.w])
        index.append(period['end'][t-1])

        log['beg'].append(period['beg'][t-dt])
        log['end'].append(period['end'][t-1])

    log['value'] = port1.V_hist[1:]

    if port1.buy_h==1:
        log['cash']= np.zeros(len(log['end']))
    else:
        log['cash'] = port1.cash_hist

    log = pd.DataFrame(log,columns=log.keys())
    if report==1:
        print("\n",log)

    real_ret = (port1.V_hist[-1:]-port1.V_hist[0])/port1.V_hist[0]
    print("\n Total portfolio Valorization: {}".format(real_ret))

    port1.W = pd.DataFrame(W,columns=port1.stocks,index=pd.DatetimeIndex(index).normalize())
    plt.figure()
    port1.W.plot(kind='bar',title="Dynamic allocation "+port1.method,stacked=True).legend(loc='center left', bbox_to_anchor=(1, 0.5))

    return W,log
