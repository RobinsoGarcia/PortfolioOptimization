import pandas as pd
import numpy as np
import functions
from cvxopt import matrix
from cvxopt import solvers
import matplotlib.pyplot as plt
import math
import matplotlib.dates as mdates

solvers.options['show_progress'] = False
solvers.options['max_iters'] = 200

def load_data():
    data = pd.read_csv('Daily_closing_prices.csv',index_col='Date')
    data.index = pd.to_datetime(data.index)
    returns = data.shift(1)/data-1
    return data,returns

def cov(returns):
    X = np.array(returns)[1:]
    X -= np.mean(X,axis=0)
    Q = np.dot(X.T,X)/X.shape[0]
    U, s, V = np.linalg.svd(Q)
    if np.min(s)<0:
        s[s<0]=0.001
        new_Q = np.matmul(np.matmul(U,np.diag(s)),V.T)
        diff = np.linalg.norm(Q_new.flatten() - Q.flatten(),ord='forb')
        print("covariance adjusted, forbenius norm: {}".fornat(diff))
        Q = new_Q
    return Q

class portfolio():

    def __init__(self,stocks=None,share_balance=None,p=None,V=0,cash_account=0,cost=0.005,steps=100,use_eff=0,buy_h=0):

        self.p = p
        self.share_balance = share_balance
        self.rebalancing = 1
        self.buy_h = buy_h
        if self.buy_h==1:
            self.switch_2_buy_and_hold()
        self.cost=cost
        self.return_hist = []
        self.cash_account = cash_account
        self.cash_hist = []
        self.performance = []
        self.V = V
        self.V_hist = [np.sum(V)]
        self.rf = 0.0001
        self.use_eff= use_eff
        self.V_gain = []
        self.dailyV = np.array([])
        self.stocks = stocks
        self.time=pd.DatetimeIndex([])
        self.buy_h=buy_h
        self.W = None



    def switch_2_buy_and_hold(self):
        self.rebalancing = 0

    def update_V(self,p,rf=0.0001,t=2):
        self.rf = rf
        self.p = p[-1:].values
        dv = np.sum(self.share_balance*p,axis=1)
        self.dailyV = np.hstack([self.dailyV,dv])
        self.time = np.concatenate((self.time,p.index))
        new_V = self.p*self.share_balance
        total_value = np.sum(new_V) + self.cash_account*(1+rf)**t
        self.V_hist.append(total_value)
        self.V_gain.append(total_value/np.sum(self.V))
        self.V = new_V

    def calculate_performance(self,w,ret):
        self.performance = np.prod(ret.dot(w.T)[1:]+1)-1
        self.return_hist.append(self.performance)
        pass


    def init_opt(self,Q,mu,sharpe=0):
        if sharpe==1:
            self.G = -1*np.diag(np.ones(Q.shape[0]+1))
            self.h = np.zeros(Q.shape[0]+1)
            mu_ = mu-self.rf
            self.A = np.vstack([mu_,np.ones(Q.shape[0])])
            self.A = np.hstack([self.A,np.array([[0],[-1]])])

            self.b = [1.0,0.0]

        else:
            self.G = -1*np.diag(np.ones(Q.shape[0]))
            self.h = np.zeros(Q.shape[0])
            self.A = np.ones(Q.shape[0])
            self.b = [1.0]

    def update_cash_account(self):
        if self.rebalancing==1:
            p = self.p
            w = self.w
            #w[w<0.001]=0
            new_balance = np.array(np.divide(w*np.sum(self.V),p))
            new_balance[new_balance<1]=0
            new_balance = np.round(new_balance)
            transactions = -1*(new_balance-self.share_balance)
            cash = float(np.dot(transactions,p.T))*self.cost

            while cash<0:
                #print("negative cash account: {}".format(cash))
                mask = new_balance>0
                new_balance -= 1*mask
                new_balance = np.round(new_balance)
                transactions = -1*(new_balance-self.share_balance)
                cash = float(np.dot(transactions,p.T))*self.cost
            #print("cash after loop: {}".format(cash))

            self.share_balance = new_balance
            self.w = new_balance/np.sum(new_balance)
            self.cash_hist.append(cash)
        pass

    def metrics(self,mu,Q,w):
        ret = (mu.dot(w)+1)**252-1
        std = np.sqrt(252)*(w.T.dot(Q).dot(w))
        sharpe = ret/std
        print('Expected return: {}'.format(ret))
        print('Expected std: {}'.format(std))
        print('Expected sharpe: {}'.format(sharpe))

    def plot_ret(self):
        plt.title("Portfolio cumulative return - "+self.method)
        plt.xlabel("time interval")
        plt.ylabel("cumulative return")
        plt.plot(np.cumprod(self.V_gain)-1)

    def plot_cash(self,time):
        if self.buy_h==0:
            fig, ax = plt.subplots()
            plt.title("cash account - "+self.method)
            plt.xlabel("time interval")
            plt.ylabel("cash balance")
            plt.plot(time,self.cash_hist)
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%y'))


    def plot_dailyV(self):
        fig, ax = plt.subplots()
        plt.title("Portfolio daily value - "+self.method)
        plt.xlabel("time interval")
        plt.ylabel("Total value")
        plt.plot(self.time,self.dailyV)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%y'))
        #https://matplotlib.org/api/dates_api.html

class strat_max_Sharpe(portfolio):

        def optimize(self,Q,mu,stats=0):
            self.method = 'strat_max_Sharpe'
            if self.use_eff==0:
                self.init_opt(Q,mu,sharpe=1)
                Q=2*Q
                Q_ = np.hstack([Q,np.zeros((Q.shape[0],1))])
                Q_ = np.vstack([Q_,np.zeros(Q_.shape[1])])
                P = matrix(np.array(Q_) ,tc='d')
                q = matrix(np.zeros(Q_.shape[0]),tc='d')
                G = matrix(self.G,tc='d')
                h = matrix(self.h,tc='d')
                A = matrix(self.A,tc='d')
                b = matrix(self.b,tc='d')
                sol = solvers.qp(P,q,G,h,A,b)
                y = np.array(sol['x'])
                w = np.array(y[:-1])/y[-1:]
                self.w = w.flatten()
                if stats==1:
                    self.metrics(mu,Q*0.5,w)
            else:
                eff = effFront()
                w = eff.optimize(Q=Q,mu=mu,plot=0)
                self.w = w.flatten()
                if stats==1:
                    self.metrics(mu,Q*0.5,w)


            return self.w

class strat_min_variance(portfolio):
    def optimize(self,Q,mu,stats=0):
            self.method = 'strat_min_variance'
            self.init_opt(Q,mu,sharpe=0)
            Q=2*Q

            P = matrix(np.array(Q) ,tc='d')
            q = matrix(np.zeros(Q.shape[0]),tc='d')

            G = matrix(self.G,tc='d')
            h = matrix(self.h,tc='d')
            A = matrix(self.A,tc='d').T
            b = matrix(self.b,tc='d')

            sol = solvers.qp(P,q,G,h,A,b)
            w = np.array(sol['x']).flatten()
            self.w = w
            if stats==1:
                self.metrics(mu,Q*0.5,w)


            return w

class strat_max_return(portfolio):
    def optimize(self,Q,mu,stats=0):
        self.method = 'strat_max_return'
        self.init_opt(Q,mu,sharpe=0)
        Q=2*Q

        P = matrix(np.zeros([mu.shape[0],mu.shape[0]]) ,tc='d')
        q = -1*matrix(np.array(mu),tc='d')

        G = matrix(self.G,tc='d')
        h = matrix(self.h,tc='d')
        A = matrix(self.A,tc='d').T
        b = matrix(self.b,tc='d')

        sol = solvers.qp(P,q,G,h,A,b)
        w = np.array(sol['x']).flatten()
        self.w = w
        if stats==1:
            self.metrics(mu,Q*0.5,w)

        return w

class effFront(portfolio):
        #def __init__(self,steps=100,rf = 0.0001):
        steps = 500

        def optimize(self,Q,mu,stats=0,plot=0):
            self.method = 'effFront'
            self.Q = Q
            self.mu = mu
            #self.share_balance = share_balance
            #print("Result with eff frontier:")
            self.get_extremes()
            self.get_return_range()
            self.get_eff_frontier()
            self.max_sharpe()
            #print("Result with optimization:")
            #self.maxSharpe = strat_max_Sharpe().optimize(self.Q,self.mu)
            if plot==1:
                self.cloud()
                self.plot()

            return self.w_sharpe

        def get_return_range(self):
            self.e_maxRet = self.w_maxRet.T.dot(self.mu)
            self.e_minVar = self.w_minVar.T.dot(self.mu)
            self.e = np.linspace(self.e_minVar,self.e_maxRet,self.steps)
            pass

        def get_eff_frontier(self):

            self.w_eff = [self.get_optimal_port(x).flatten() for x in self.e]
            self.stds = [float(x.T.dot(self.Q).dot(x)) for x in self.w_eff]
            pass

        def get_optimal_port(self,e):
            Q=2*self.Q
            mu=self.mu
            P = matrix(np.array(Q) ,tc='d')
            q = matrix(np.zeros(Q.shape[0]),tc='d')
            G = np.diag(np.ones(Q.shape[0]))
            G = -1*matrix(np.vstack([G,mu]),tc='d')
            h = np.zeros(Q.shape[0])
            h = -1*matrix(np.append(h,e),tc='d')
            A = matrix(np.ones(Q.shape[0]),tc='d').T
            b = matrix(1.0)
            sol = solvers.qp(P,q,G,h,A,b)
            w = np.array(sol['x'])
            sol['primal objective']
            return w

        def get_extremes(self):
            self.w_maxRet = strat_max_return(self.share_balance).optimize(self.Q,self.mu,stats=0)
            self.w_minVar = strat_min_variance(self.share_balance).optimize(self.Q,self.mu,stats=0)

        def plot(self):
            plt.figure()
            w_maxRet = self.w_maxRet
            w_minVar = self.w_minVar
            w_sharpe = self.w_sharpe
            ports = [w_maxRet,w_minVar,w_sharpe]

            std = self.stds
            ret = self.e
            std = np.array([x*math.sqrt(252) for x in std])
            ret = (ret+1)**252-1
            plt.title("Efficient frontier")
            plt.xlabel("std")
            plt.ylabel("return")
            plt.scatter(std,ret)
            for x in ports:
                ret = x.T.dot(self.mu)
                std = x.T.dot(self.Q).dot(x)
                std = std*math.sqrt(252)
                ret = (ret+1)**252-1
                plt.scatter(std,ret)

            plt.scatter(self.cloud_std,self.cloud_ret)
            asset_ret = (self.mu+1)**252-1
            asset_std = np.array([x*math.sqrt(252) for x in np.diag(self.Q)])
            plt.scatter(asset_std,asset_ret)
            plt.plot()


        def cloud(self):
            self.cloud_n = 1000
            ret_ = []
            std_ = []
            w = np.random.rand(self.cloud_n,max(self.mu.shape))
            w_ = np.sum(w,axis=1)[np.newaxis,:]
            c = w.T/w_
            w = c.T
            print("sanity check: {}".format(np.sum(np.sum(w,axis=1))/self.cloud_n==1))
            ret = w.dot(self.mu.T)
            std = np.diag(w.dot(self.Q).dot(w.T))

            std = np.array([x*math.sqrt(252) for x in std])
            ret = (ret+1)**252-1

            self.cloud_ret = ret
            self.cloud_std = std



        def max_sharpe(self):
            idx = np.argmax((self.e-self.rf)/self.stds)
            self.w_sharpe = self.w_eff[idx]
            #self.metrics(self.mu,self.Q,self.w_sharpe)
            pass

class strat_equally_weighted(portfolio):
    def optimize(self,Q,mu,stats=0):
        self.method = 'strat_equally_weighted'
        n = max(self.share_balance.shape)
        w = np.ones(n) * (1/n)
        self.w = w.flatten()
        if stats==1:
            self.metrics(mu,Q*0.5,w)
        return self.w

class strat_buy_and_hold(portfolio):

    def optimize(self,Q,mu,stats=0):
        self.method = 'strat_buy_and_hold'
        self.w = self.w_buy_and_hold
        if stats==1:
            self.metrics(mu,Q*0.5,w)
        return self.w
