import quandl
import datetime

try:
    import my_quandl.quandl as mq
except ImportError:
    import derivatives_pricing.load_data.get_from_quandl.quandl as mq

import numpy as np

def get_quotes(symbols, start_date, end_date=None):

    data = mq.get_quotes(symbols, start_date, end_date=None)
    close = [x for x in data.columns if 'Close' in x][0]
    returns = data[close].shift(1)/data[close]-1
    vol = np.std(returns)
    return data[close],returns,vol
