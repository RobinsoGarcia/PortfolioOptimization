import pandas as pd
import numpy as np
from cvxopt import matrix
from cvxopt import solvers
import matplotlib.pyplot as plt
import math

solvers.options['show_progress'] = False
solvers.options['max_iters'] = 100

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
        Q = Q_new
    return Q



class portfolio():

    def __init__(self,Q=np.zeros([2,2]),mu=np.zeros(2),p=None,V=0,cash_account=0,steps=100):
        self.Q = Q
        self.mu = mu
        self.p = p
        self.V = V
        self.cash_account = cash_account
        self.steps = steps
        self.performance = []
        self.get_max_return_portfolio()
        self.get_min_var_portfolio()
        self.get_return_range()
        self.get_eff_frontier()
        self.get_max_sharpe()
        self.max_sharpe_check()
        self.equalW = np.ones([self.mu.shape[0]])*(1/self.mu.shape[0])
        self.results = {'equalW':self.equalW,'sharpe':self.w_sharpe,'sharpe_check':self.w_sharpe_ch,'minVar':self.w_minVar,'maxReturn':self.w_maxRet}

    def calculate_performance(self,w,ret):
        self.performance = np.prod(ret.dot(w)[1:]+1)-1
        pass

    def select_by(self,method,value=None):
        w = self.results[method].flatten()

        #=print(w)
        #print('Expected return: {}'.format((w.T.dot(self.mu)+1)**252-1))
        #print('Risk: {}'.format(w.T.dot(self.Q).dot(w)*np.sqrt(252)))
        #print('sharpe: {}'.format(((w.T.dot(self.mu)+1)**252-1)/(w.T.dot(self.Q).dot(w)*np.sqrt(252))))
        return w

    def max_sharpe_check(self):
        idx = np.argmax(self.e/self.stds)
        self.w_sharpe_ch = self.w_eff[idx]
        pass

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

    def get_max_sharpe(self):
        Q=2*self.Q
        mu=self.mu

        Q_ = np.hstack([Q,np.zeros((Q.shape[0],1))])
        Q_ = np.vstack([Q_,np.zeros(Q_.shape[1])])

        P = matrix(np.array(Q_) ,tc='d')
        q = matrix(np.zeros(Q_.shape[0]),tc='d')
        '''
        if self.V!= 0:
            G = np.diag(np.ones(Q_.shape[0]))
            add_G = np.append(self.V/self.p,0)
            G = matrix(-1*np.vstack([G,add_G]),tc='d')
            h = matrix(np.hstack([np.zeros(Q_.shape[0]),-1*self.cash_account]),tc='d')
        else:
            G = -1*matrix(np.diag(np.ones(Q_.shape[0])),tc='d')
            h = matrix(np.zeros(Q_.shape[0]),tc='d')
        '''
        G = -1*matrix(np.diag(np.ones(Q_.shape[0])),tc='d')
        h = matrix(np.zeros(Q_.shape[0]),tc='d')

        A = np.vstack([mu,np.ones(Q.shape[0])])
        A = matrix(np.hstack([A,np.array([[0],[-1]])]),tc='d')
        b = matrix([1,0],tc='d')

        sol = solvers.qp(P,q,G,h,A,b)
        y = np.array(sol['x'])

        self.w_sharpe = np.array(y[:-1])/y[-1:]

        pass

    def plot(self):
        plt.scatter(self.stds,self.e)

        for x in self.results:
            w = self.results[x]
            ret = w.T.dot(self.mu)
            std = w.T.dot(self.Q).dot(w)
            plt.scatter(std,ret)

    def get_min_var_portfolio(self):
        Q=2*self.Q
        mu=self.mu
        P = matrix(np.array(Q) ,tc='d')
        q = matrix(np.zeros(Q.shape[0]),tc='d')

        G = np.diag(np.ones(Q.shape[0]))
        G = -1*matrix(np.vstack([G,mu]),tc='d')

        h = -1*matrix(np.zeros(Q.shape[0]+1),tc='d')
        A = matrix(np.ones(Q.shape[0]),tc='d').T
        b = matrix(1.0)
        sol = solvers.qp(P,q,G,h,A,b)
        self.w_minVar = np.array(sol['x'])
        pass

    def get_max_return_portfolio(self):
        Q=2*self.Q
        mu=self.mu
        P = matrix(np.zeros([mu.shape[0],mu.shape[0]]) ,tc='d')
        q = -1*matrix(np.array(mu),tc='d')
        G = -1*matrix(np.diag(np.ones(Q.shape[0])),tc='d')
        h = -1*matrix(np.zeros(Q.shape[0]),tc='d')
        A = matrix(np.ones(Q.shape[0]),tc='d').T
        b = matrix(1.0)
        sol = solvers.qp(P,q,G,h,A,b)
        self.w_maxRet = np.array(sol['x'])
        pass
