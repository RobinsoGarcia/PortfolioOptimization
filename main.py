import portfolio as prt
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use('ggplot')
quotes, returns = prt.load_data()
init_port = {'MSFT':500,'F':950,'CRAY':2000,'VZ':200,'AAPL':3000,'IBM':1500,'NVDA':1001}
Q = np.array(prt.cov(returns))
mu = returns.mean()
U, s, V = np.linalg.svd(Q.T.dot(Q))
s
a = prt.portfolio(Q,mu)

w = a.select_by('sharpe')
w = a.select_by('sharpe_check')

a.plot()
