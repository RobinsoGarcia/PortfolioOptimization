#!/usr/local/bin/env python3
#import pickle #https://www.saltycrane.com/blog/2008/01/saving-python-dict-to-file-using-pickle/
#from portfolio_optimizer import evaluate_port
#import matplotlib.pyplot as plt
import portfolio_optimizer as po
import portfolio_optimizer.evaluate_port as ev
import os
#import numpy as np
#from pandas_datareader import data
import json
import sys

def main(args=None):
    """The main routine."""
    if args is None:
        args = sys.argv[1:]
        
    hmdir = str(sys.argv[1])

    f_in = open(hmdir)
    dic = json.load(f_in)

    data = dic['data']
    strategies = dic['strategies']
    backtest = dic['backtest']

    ev.evaluate(strategies,data,backtest)


if __name__ == "__main__":
    main()
