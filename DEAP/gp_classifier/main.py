# Derek M Tishler - 2018 - dmtishler@gmail.com
# DEAP Genetic Programming Example for Symbolic Regression Classification on Quant Connect

#DEAP Source: https://github.com/DEAP/deap
#DEAP Docs: https://deap.readthedocs.io/en/master/

from clr import AddReference
AddReference("System")
AddReference("QuantConnect.Algorithm")
AddReference("QuantConnect.Common")

from System import *
from QuantConnect import *
from QuantConnect.Algorithm import *
from QuantConnect.Brokerages import BrokerageName
import random
from scipy import stats
import numpy as np
from scipy import stats
from scipy import stats as sstats
import pandas as pd
import operator
import math
import time

from evo import *

# using random math on random inputs can lead to many warnings(ex try in protected div, undefined math, etc). This cleans the logs for reading evo table. 
# Remove when adjusting/testing pset ops
import warnings
warnings.filterwarnings('ignore')


class BasicTemplateAlgorithm(QCAlgorithm):

    def Initialize(self):
        
        self.evo_time = 0.
        self.evo      = Evolution(self)
        
        self.SetStartDate(2016,1,1)  #Set Start Date
        #self.SetEndDate(2018,1,1)    #Set End Date
        
        self.SetCash(100000)         #Set Strategy Cash
        
        self.symbol      = "SPY"
        self.evo.symbol  = self.symbol
        self.granularity = Resolution.Daily
        
        self.SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage, AccountType.Margin)
        
        self.symbol_Symbol = self.AddEquity(self.symbol, Resolution.Minute, extendedMarketHours=False).Symbol
        
        sPlot = Chart('Strategy Equity')
        sPlot.AddSeries(Series('Signal', SeriesType.Scatter, 2))
        self.AddChart(sPlot)
        
        fit1Plot = Chart('Strategy Evolution')
        fit1Plot.AddSeries(Series('Mean_Loss', SeriesType.Line, 0))
        fit1Plot.AddSeries(Series('Max_Loss', SeriesType.Line, 1))
        fit1Plot.AddSeries(Series('Consistency', SeriesType.Line, 2))
        fit1Plot.AddSeries(Series('Size', SeriesType.Line, 3))
        fit1Plot.AddSeries(Series('Size_Max', SeriesType.Line, 3))
        fit1Plot.AddSeries(Series('Size_Min', SeriesType.Line, 3))
        fit1Plot.AddSeries(Series('Size_Mean', SeriesType.Line, 3))
        fit1Plot.AddSeries(Series('Label Dist', SeriesType.Line, 4))
        fit1Plot.AddSeries(Series('N_Long', SeriesType.Line, 4))
        fit1Plot.AddSeries(Series('N_Short', SeriesType.Line, 4))
        self.AddChart(fit1Plot)
        
        sPlot2 = Chart('Strategy Info')
        sPlot2.AddSeries(Series('Leverage',  SeriesType.Line, 0))
        sPlot2.AddSeries(Series('RAM',  SeriesType.Line, 1))
        sPlot2.AddSeries(Series('Evo Time',  SeriesType.Line, 2)) #Label
        self.AddChart(sPlot2)
        
        
        
        self.Schedule.On(self.DateRules.EveryDay(self.symbol),
            self.TimeRules.AfterMarketOpen(self.symbol, 2),
            Action(self.Evolve))
            
        self.Schedule.On(self.DateRules.EveryDay(self.symbol),
            self.TimeRules.AfterMarketOpen(self.symbol, 30),
            Action(self.Checkpoint))
        
        self.Schedule.On(self.DateRules.EveryDay(self.symbol),
            self.TimeRules.BeforeMarketClose(self.symbol, 2),
            Action(self.Liquidate))
            
        # in case you want to add a relative stop, needs to uncomment in OnData
        self.max_loss_frac      = 0.03
        self.asset_best_price   = {}
        
        # trigger large history download one time
        self.do_once  = True
        
        # weight used for SetHoldings
        self.signal   = 0.0


    def Evolve(self):
        
        # update data in smaller batches for speed
        self.evo.current_price = float(self.Securities[self.symbol].Price)
        if not self.do_once:
            new_hist           = self.History([self.symbol], 1, self.granularity, extendedMarketHours=False).astype(np.float32)
            self.evo.hist_data = self.evo.hist_data.append(new_hist).iloc[1:] #append and pop stack   
        # large download, one time only
        else:
            self.evo.hist_data = self.History([self.symbol], self.evo.warmup_count, self.granularity, extendedMarketHours=False).astype(np.float32)
            self.do_once       = False
        
        # perform evolution and get trading signal
        self.signal = self.evo.OnEvolve()
        
        # handle trading signals
        self.SetHoldings(self.symbol, self.signal)#, liquidateExistingHoldings=True)
        
        
    def Checkpoint(self):

        self.Plot("Strategy Equity", 'Signal', self.signal)

        self.Plot("Strategy Evolution",'Mean_Loss', float(self.evo.logbook.chapters["fitness"].select("min")[-1][0]))
        self.Plot("Strategy Evolution",'Max_Loss', float(self.evo.logbook.chapters["fitness"].select("min")[-1][1]))
        self.Plot("Strategy Evolution",'Consistency', float(-self.evo.logbook.chapters["fitness"].select("min")[-1][2]))
        
        self.Plot("Strategy Evolution", 'Size_Max', float(self.evo.logbook.chapters["size"].select("max")[-1]))
        self.Plot("Strategy Evolution", 'Size_Min', float(self.evo.logbook.chapters["size"].select("min")[-1]))
        self.Plot("Strategy Evolution", 'Size_Mean', float(self.evo.logbook.chapters["size"].select("avg")[-1]))
        
        t =  float(self.evo.n_long_labels) + float(self.evo.n_short_labels)
        self.Plot("Strategy Evolution", 'N_Long', float(self.evo.n_long_labels)/t)
        self.Plot("Strategy Evolution", 'N_Short', float(self.evo.n_short_labels)/t)
        
        self.account_leverage = self.Portfolio.TotalHoldingsValue / self.Portfolio.TotalPortfolioValue
        self.Plot('Strategy Info','Leverage', float(self.account_leverage))
        self.Plot('Strategy Info','RAM', float(OS.ApplicationMemoryUsed/1024.))
        self.Plot('Strategy Info','Evo Time', float(self.evo_time))
        
    def OnData(self, data):
        
        # risk managment to limit per position loss to n%
        #map(self.RiskManagement, [self.symbol_Symbol])
        pass

    
    def RiskManagement(self, symbol):
        # https://github.com/QuantConnect/Lean/blob/24fcd239a702c391c26854601a99c514136eba7c/Common/Securities/SecurityHolding.cs#L79https://github.com/QuantConnect/Lean/blob/24fcd239a702c391c26854601a99c514136eba7c/Common/Securities/SecurityHolding.cs#L79
        if self.Portfolio[symbol].Quantity != 0:
            
            # init the avg price as our current best price for the asset
            if symbol not in self.asset_best_price:
                self.asset_best_price[symbol] = float(self.Portfolio[symbol].AveragePrice)
                    
            # have we exceded the target?
            if self.Portfolio[symbol].Quantity > 0:
                self.asset_best_price[symbol] = np.maximum(self.asset_best_price[symbol], float(self.Securities[symbol].Price))
                if (float(self.Securities[symbol].Price)-self.asset_best_price[symbol])/self.asset_best_price[symbol] < -self.max_loss_frac:
                    self.Log("RM Exit of Long pos: %s"%symbol)
                    self.Liquidate(symbol, tag="RM")
                    del self.asset_best_price[symbol]
            
            elif self.Portfolio[symbol].Quantity < 0:
                self.asset_best_price[symbol] = np.minimum(self.asset_best_price[symbol], float(self.Securities[symbol].Price))
                if (float(self.Securities[symbol].Price)-self.asset_best_price[symbol])/self.asset_best_price[symbol] > self.max_loss_frac:
                    self.Log("RM Exit of Short pos: %s"%symbol)
                    self.Liquidate(symbol, tag="RM")
                    del self.asset_best_price[symbol]
