
import portfolio_optimizer as po
import portfolio_optimizer.evaluate_port as ev
import os
import json
import sys
from datetime import date
import ipywidgets as widgets



def main(args=None):
    
    dic = args

    data = dic['data']
    strategies = dic['strategies']
    backtest = dic['backtest']

    ev.evaluate(strategies,data,backtest)





class PortWidgets():


    def __init__(self):
        self.all_shares = ['AAPL','AMD','AMZN','BAC','BK','CRAY','CSCO','F','GOOG','HOG','HPQ','IBM','INTC','MS','MSFT','NVDA','SNE','T','VZ']


        self.sel_strat = widgets.SelectMultiple(
            options=["strat_buy_and_hold",
                            "strat_equally_weighted",
                            "strat_max_Sharpe",
                            "strat_max_Sharpe_eff",
                            "strat_min_variance"],
            value=["strat_buy_and_hold",
                            "strat_equally_weighted",
                            "strat_max_Sharpe",
                            "strat_max_Sharpe_eff",
                            "strat_min_variance"],
            rows=5,
            description='Strategies',
            disabled=False
        )

        self.stocks = widgets.SelectMultiple(
            options=self.all_shares,
            value=['AAPL'],
            rows=15,
            description='Stocks',
            disabled=False
        )

        self.start_date = widgets.DatePicker(
            description='Start_date',
            value= date(2015,1,1),
            disabled=False
        )
        self.end_date = widgets.DatePicker(
            description='End_date',
            value=date(2016,12,31),
            disabled=False
        )

        self.dt = widgets.BoundedIntText(
            value=2,
            min=0,
            max=10,
            step=1,
            description='Rebal. Freq:',
            disabled=False
        )

        self.freq_unit = widgets.Dropdown(
            options={'month': 'months', 'Days': 'Days'},
            value='months',
            description='Unit:',
        )

        self.dtF = widgets.BoundedIntText(
            value=2,
            min=0,
            max=10,
            step=1,
            description='InitialWindow frq:',
            disabled=False
        )

        self.freq_unitF = widgets.Dropdown(
            options={'month': 'months', 'Days': 'Days'},
            value='months',
            description='Unit:',
        )

        self.run_botton = widgets.ToggleButton(
            value=False,
            description='run',
            disabled=False,
            button_style='', # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Start Simulation',
            icon='check'
        )

        self.plot_eff = widgets.Checkbox(
                        value=False,
                        description='Check me',
                        disabled=False
                    )
        self.weights=[]
        self.share_balance=[]

        display(self.stocks)
        display(self.sel_strat)
        display(self.start_date)
        display(self.end_date)
        display(widgets.HBox([self.freq_unit,self.dt]))
        display(widgets.HBox([self.freq_unitF,self.dtF]))
        display(self.plot_eff)

        self.button = widgets.Button(description="set weights!");	
        display(self.button);
        self.button.on_click(self.get_weights);
        self.button2 = widgets.Button(description="Build!")	;
        display(self.button2);
        self.button2.on_click(self.solve);

    
    def build_dic(self):
        T = (self.end_date.value.month-self.start_date.value.month)*(1+self.end_date.value.year-self.start_date.value.year)/self.dt.value
      
        dic = {"data":
            {"init_port": self.init_port,
            "data_source": "toy",
            "start_date": str(self.start_date.value),
            "end_date": str(self.end_date.value)},
            "strategies": self.sel_strat.value,
                  "backtest": {'params':{'forward': {self.freq_unit.value:self.dt.value},'back':{self.freq_unitF.value:self.dtF.value}}
                , "rf": 0.0001,
                             "plot_eff": self.plot_eff.value}
        }

        self.dic = dic

        pass
    def solve(self,b):
        w = [x.value for x in self.share_balance]
        self.port_data = dict(zip(self.stocks.value,w))
        dic = {}
        for i in self.all_shares:
            if i in self.stocks.value:
                dic[i] = self.port_data[i]
            else:
                dic[i]=0
        self.init_port = dic
        self.build_dic()
        


    def get_weights(self,b):
        self.share_balance = [widgets.FloatText(value=10,description=i,disabled=False) for i in self.stocks.value]
        [display(i) for i in self.share_balance]
        self.weights = [x.value for x in self.share_balance]
        

        
        

