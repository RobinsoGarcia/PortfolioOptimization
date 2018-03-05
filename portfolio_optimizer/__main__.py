#!/usr/local/bin/env python3
#import pickle #https://www.saltycrane.com/blog/2008/01/saving-python-dict-to-file-using-pickle/

import portfolio_optimizer as po
from portfolio_optimizer.functions import *
import os
import json
import sys

def main(args=None):
    """The main routine."""
    if args is None:
        args = sys.argv[1:]
        
    hmdir = str(sys.argv[1])

    f_in = open(hmdir)
    dic = json.load(f_in)

    Iport = PortData(dic)
    evaluate(Iport)


if __name__ == "__main__":
    main()
