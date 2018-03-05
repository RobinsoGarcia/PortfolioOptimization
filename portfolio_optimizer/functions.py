from portfolio_optimizer.widgets_notebook import *
import pandas as pd
from pandas.tseries.offsets import *
import portfolio_optimizer.port as prt
import numpy as np
import matplotlib.pyplot as plt
import portfolio_optimizer


class PortData():
    def __init__(self,dic):
        self.Widgets = dic
        self.init_port = dic['data']['init_port']
        self.share_balance = np.array(list(dic['data']['init_port'].values()))[np.newaxis,:]
        self.shares = dic['data']['init_port'].keys()#[x for x in dic['data']['init_port'].keys() if dic['data']['init_port'][x]>0]
        self.start = pd.to_datetime(dic['data']['start_date'])
        self.end = pd.to_datetime(dic['data']['end_date'])
        self.strategies = dic['strategies']
        self.params = dic['backtest']['params']
        self.load_data()
        self.V0()
        self.rf=0.001
    def V0(self):
        start = self.start
        while (start in self.data.index) == False:
            start += DateOffset(days=1)
            
        
        self.V = [self.data.loc[start][x]*self.init_port[x] for x in self.init_port]
    
    def load_data(self):
        csv = os.path.join(portfolio_optimizer.__path__[0],'Daily_closing_prices.csv')
        data = pd.read_csv(csv,index_col='Date')
        data.index = pd.to_datetime(data.index)
        data = data.drop(['YHOO'],axis=1)
        data = data[list(self.init_port.keys())]
        returns = data.shift(1)/data-1
        self.data = data
        self.returns = returns
        pass



def evaluate(Iport):
    
    kwargs = {'stocks':Iport.shares,'share_balance':Iport.share_balance,'V':Iport.V}
    
    strategies = {'strat_buy_and_hold':prt.strat_buy_and_hold(**kwargs,buy_h=1),
                    'strat_equally_weighted':prt.strat_equally_weighted(**kwargs),
                    'strat_max_Sharpe':prt.strat_max_Sharpe(**kwargs),
                    'strat_max_Sharpe_eff':prt.strat_max_Sharpe(**kwargs,use_eff=1),
                    'strat_min_variance':prt.strat_min_variance(**kwargs)}
    value = []
    strat = []
    daily_returns = {}
    summary = {}
    for x in Iport.strategies:
        print(x)
        print('Portfolio initial value: {}'.format(np.sum(Iport.V)))

        port = strategies[x]
        W,log = backtest(port,Iport.data,Iport.start,Iport.end,Iport.params,Iport.rf)
        
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





def get_data(data):
    returns = data.shift(1)/data-1
    Q = np.array(prt.cov(returns))
    mu = np.array(returns.mean())
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
        
        w = obj.optimize(Q,mu) 

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
    print(W.shape)
    print(len(obj.stocks))
    
    W = pd.DataFrame(data=W,columns=obj.stocks)
    plt.figure()
    W.plot(kind='bar',title="Dynamic allocation "+obj.method,stacked=True).legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()

 
    return W,log
