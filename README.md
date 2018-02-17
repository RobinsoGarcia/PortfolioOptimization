# Installation
The package needs to be installed as sudo user in order to add the portopt script to your /usr/local/bin folder so that it is callable from any folder. If you choose not to install with sudo, just download the __main__.py file and run it where your Json file is.


# Basic usage
One can use this code just by filling a json file and running portopt command using the command line at the directory where the Json file is located.

# Json sample
            {"data":
                {"init_port":   {"AAPL": 3000, "AMD": 0,
                                "AMZN": 0, "BAC": 0, "BK": 0,
                                "CRAY": 2000, "CSCO": 0, "F": 950,
                                "GOOG": 0, "HOG": 0, "HPQ": 0, "IBM": 1500,
                                "INTC": 0, "MS": 0, "MSFT": 5000, "NVDA": 1001,
                                "SNE": 0, "T": 0, "VZ": 2000},
                                "data_source": "toy",
                                "start_date": "2015-01-01",
                                "end_date": "2016-12-31"},
                   "strategies": ["strat_buy_and_hold",
                                  "strat_equally_weighted",
                                  "strat_max_Sharpe",
                                  "strat_max_Sharpe_eff",
                                  "strat_min_variance"],
                    "backtest": {"t0": 2, "dt": 2, "rf": 0.0001,
                                  "T": 24, "plot_eff": 0}
            }
  
# init_port:  
          initial portfolio:
            type() = dictionary
            tickers and allocation (# of shares)
          data_source:
            type(data_source) = string
            'toy' - uses a toy dataset contaning 20 stocks
            'yahoo' - request stock data from yeahoo based on init_port
          start_date:
            type(start_date) = string
            Starting date for the simulation to run from
          end_date:
            type(end_date) = string
            end of the simulation
          PS: the code is set to rebalance the portfolio at the last business day of the month.

##   strategies
        type() =list(string)
        'strat_min_variance' : return the minimum variance portfolio
        'strat_max_Sharpe'  : returns the maximum sharpe portfolio
        'strat_max_return' : returns the maximum return portfolio
        'strat_equally_weighted': returns the equally weighted portfolio
        'strat_buy_and_hold' : keeps the initial portfolio until the end of the simulation

##   backtest
        t0:
          type(t0) = int
          number of periods after start_date before the first rebalancing operation.
        dt:
          type(dt): int
          discrete time interval between rebalancing operations
        T:
          type(T) = int
          total number of periods to consider
        rf:
          type(rf) = float
          average daily interest rate.

 
# Additional data

1. if the data source is yahoo-finance python package, one should be advised that this package
very often sends an error message such as no data was retrieved or something else.

2. The sharpe's portfolio is sensitive to the data used. eg. when performing analysis with the toy dataset, if we include november and december/14, the return for the max sharpe portfolio is severely different. The real returns are:

    max_sharpe = 0.57 | backtest | get_data(); data = data.loc[:period['end'][t-1]]
    equally_weighted = 0.40 | backtest | get_data(); data = data.loc[:period['end'][t-1]]

    max_sharpe = 0.30 | backtest | get_data(); data = data.loc[period['beg'][0]:period['end'][t-1]]
    equally_weighted = 0.0.40 | backtest | get_data(); data = data.loc[period['beg'][0]:period['end'][t-1]]

3.  The evaluate function runs the backtest script on different portfolio strategies to create comparsion metrics and 
    charts.

    `evaluate(strategy_list,port_data,t0=2,dt=2,rf=0.0001,T=24,start= '1/2/2015',plot_eff=0)`

    strategies = ['strat_buy_and_hold',
                  'strat_equally_weighted',
                  'strat_max_Sharpe',
                  'strat_max_Sharpe_eff',
                  'strat_min_variance']
    port_data = {'init_port':init_port,'data_source':'toy',
   'start_date':'2015-01-01','end_date': '2016-12-31'}

  init_port:
    type(init_port) = dictionary with all stock tickers to be considered) and the number of shares in the initial 
    portfolio
    *be aware that yahoo finance package not always find the ticker you asked for. An error will be returned if that happens. Also, for some reason, the package fails to acquire data at some attempts. I suggest shutting down the python kernel and trying at least three times. Otherwise, use a spreadsheet with our data.

  data_source:
    type(data_source) = string
    'toy' - uses a toy dataset contaning 20 stocks
    'yahoo' - request stock data from yeahoo based on init_port
  start_date:
    type(start_date) = string
    Starting date for the simulation to run from
  end_date:
    type(end_date) = string
    end of the simulation
  PS: the code is set to rebalance the portfolio at the last business day of the month.
t0:
  type(t0) = int
  number of periods after start_date before the first rebalancing operation.
dt:
  type(dt): int
  discrete time interval between rebalancing operations
T:
  type(T) = int
  total number of periods to consider
rf:
  type(rf) = float
  average daily interest rate.

OUTPUT:
  Weight distribution plots
  Portfolio value per periods
  Comparative summary

[4] The backtest function will test one oprtfolio strategy.

w,log = backtest_port(port1,data,start,end,t0=2,dt=2,rf=0.0001,T=24,plot_eff=0,report=0)

INPUT:

  port1:
    type(port1): class instance
    port1 is an optimization object, which is a subclasss of the portfolio class (see port.py script).

  data:
    type(data) = pandas dataframe
    dataframe with your stock data
  start:
    type(start) = staring
    date to initiate the simulation from
  t0,dt,rf,T:
    as described in [3]
  plot_eff:
    type(plot_eff): int [ 1 or 0]
    1 - plot the efficient frontier
    2 - don't plot the efficient get_eff_frontier
  report:
    type(report) = int
    1 - print summary DataFrame
    0 - don't print the DataFrame

OUTPUT:
  w - optimized weights per period
  log - intermediate results, period, total value and cash balance

[5] portfolio class:
portfolio(stocks=None,share_balance=None,p=None,V=0,cash_account=0,cost=0.005,steps=100,use_eff=0,buy_h=0):

INPUTS:
  stocks: list of stock ticker labels
    share_balance: number of shares per stock in the asset list
  p: current prices
  V: initial value of the portfolio
  cash_account: initial balance of the cash account
  cost: cost per transaction
  steps: number of iterations in the optimization       process
  use_eff: for the sharpe maximization method, with this variable flaged (1), the choosen portfolio will come from the simulated efficient frontier.
  buy_h: basic buy and hold portfolio

METHODS:
  optimize(Q,mu): finds the allocation (weights) based on a given covariance matrix and return vector. This method is customized for each of the subclasses.
    strat_min_variance(Q,mu)
    strat_max_Sharpe(Q,mu)
    strat_max_return(Q,mu)
    strat_equally_weighted(Q,mu)
    effFront(Q,mu)

  update_cash_account: takes current prices and recalculate weights to keep the cash balance greater than zerp. If the new portfolio imposes a
  negative cash balance, the code will deduct one stock for each of the assests that need positive rebalancing (purchase).

  update_V: updates the portfolio value while keeping a history.
