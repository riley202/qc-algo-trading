# Derek M Tishler - 2018 - dmtishler@gmail.com
# DEAP Genetic Programming Example for Symbolic Regression Classification on Quant Connect

#DEAP Source: https://github.com/DEAP/deap
#DEAP Docs: https://deap.readthedocs.io/en/master/

from System import *
from QuantConnect import *
from QuantConnect.Algorithm import *
from QuantConnect.Brokerages import BrokerageName
import random
from scipy import stats
import numpy as np
from scipy import stats
#from scipy import stats as sstats
import pandas as pd
import operator
import math
import time

import os

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

from functools import partial

from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

# using random math on random inputs can lead to many warnings(ex try in protected div, undefined math, etc). This cleans the logs for reading evo table. 
# Remove when adjusting/testing pset ops
import warnings
warnings.filterwarnings('ignore')

# how many individuals in our populations
n_pop = 100

seed  = 8675309
random.seed(seed)
np.random.seed(seed)

# simple logic template
def if_then_else(input, output1, output2):
    if input: return output1
    else: return output2

# avoid errors in evaluations
def protectedDiv(left, right):
    if right == 0:
        return 0.
    else:
        return left/right

# we create a fake bool class, this is done to avoid a int-bool confusion in deap operators as they mix up the two easily.
class BOOL:
    pass


class Evolution(object):

    def __init__(self, context):

        # this is here for when you need to log for debugging
        self.context = context
        
        self.n_long_labels  = 0
        self.n_short_labels = 0

        # len of hitory
        self.n_features  = 10#
        
        self.n_samples = 250
        
        self.warmup_count   = self.n_features + self.n_samples + 1

        # persist the evolution, warning though you have to track when its run and on what data
        try_load_saved_pop  = False

        # The primitive set defines what is possible in the program
        self.pset = gp.PrimitiveSetTyped("MAIN", [float]*(self.n_features), float)
        self.pset.addPrimitive(operator.add, [float, float], float)
        self.pset.addPrimitive(operator.sub, [float, float], float)
        self.pset.addPrimitive(operator.mul, [float, float], float)
        self.pset.addPrimitive(protectedDiv, [float, float], float)
        self.pset.addPrimitive(operator.neg, [float], float)
        self.pset.addPrimitive(operator.abs, [float], float)
        self.pset.addPrimitive(np.hypot, [float, float], float)
        self.pset.addPrimitive(np.absolute, [float], float)
        self.pset.addPrimitive(np.fmax, [float, float], float)
        self.pset.addPrimitive(np.fmin, [float, float], float)
        self.pset.addPrimitive(np.sign, [float], float)
        self.pset.addPrimitive(np.square, [float], float)
        self.pset.addPrimitive(math.cos, [float], float)
        self.pset.addPrimitive(math.sin, [float], float)
        
        
        self.pset.addPrimitive(operator.and_, [BOOL, BOOL], BOOL)
        self.pset.addPrimitive(operator.or_, [BOOL, BOOL], BOOL)
        self.pset.addPrimitive(operator.not_, [BOOL], BOOL)

        self.pset.addPrimitive(operator.lt, [float, float], BOOL)
        self.pset.addPrimitive(operator.le, [float, float], BOOL)
        self.pset.addPrimitive(operator.eq, [float, float], BOOL)
        self.pset.addPrimitive(operator.ne, [float, float], BOOL)
        self.pset.addPrimitive(operator.ge, [float, float], BOOL)
        self.pset.addPrimitive(operator.gt, [float, float], BOOL)

        self.pset.addPrimitive(if_then_else, [BOOL, float, float], float, 'ite_float')
        self.pset.addPrimitive(if_then_else, [BOOL, BOOL, BOOL], BOOL, 'ite_bool')

        self.pset.addEphemeralConstant("rand1", lambda: random.random(), float)
        self.pset.addEphemeralConstant("rand-1", lambda: -random.random(), float)

        self.pset.addTerminal(-0.5, float)
        self.pset.addTerminal(-1.0, float)
        self.pset.addTerminal(0.0, float)
        self.pset.addTerminal(0.5, float)
        self.pset.addTerminal(1.0, float)
        self.pset.addTerminal(False, BOOL)
        self.pset.addTerminal(True, BOOL)

        creator.create("FitnessMin", base.Fitness, weights=(-1.0,-1.0, -1.0))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

        self.toolbox = base.Toolbox()
        self.toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=1, max_=3)
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.expr)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("compile", gp.compile, pset=self.pset)

        self.toolbox.register("evaluate", self.evalSymbReg)
        self.toolbox.register("select", tools.selNSGA2)
        self.toolbox.register("mate", gp.cxOnePoint)
        self.toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
        self.toolbox.register("mutate", gp.mutUniform, expr=self.toolbox.expr_mut, pset=self.pset)
        self.toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))   #bloat control
        self.toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17)) #bloat control

        self.stats_fit  = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats_size = tools.Statistics(len)
        self.stats      = tools.MultiStatistics(fitness=self.stats_fit, size=self.stats_size)
        self.stats.register("avg", np.mean, axis=0)
        self.stats.register("std", np.std, axis=0)
        self.stats.register("min", np.min, axis=0)
        self.stats.register("max", np.max, axis=0)

        # persist the evolution, warning though you have to track when its run and on what data
        checkpoint = 'checkpoint.pkl'
        
        self.gen                   = 0
        self.halloffame            = tools.ParetoFront()
        self.logbook               = tools.Logbook()
        self.logbook.header        = ['gen', 'nevals'] + (self.stats.fields if self.stats else [])
        self.population            = self.toolbox.population(n=n_pop)
        self.selected_individuals  = None
        
    def process_batch(self, individual, i):
        # prepare the input features of each sample
        current_step_input = list(100.*self.hist_data.iloc[i-self.n_features-1:i].open.pct_change().dropna().values.flatten().astype(np.float32))
        
        # run sample through program & get probability. clip used to prevent nan/inf issues
        #probability            = np.clip(np.nan_to_num(individual(*current_step_input)), 0.001, 0.999)
        probability            = np.nan_to_num(individual(*current_step_input))
        
        # label for each sample
        dp = 100.*(self.hist_data.close.values[i]-self.hist_data.open.values[i])/self.hist_data.open.values[i]
        if dp >= 0.0:
            label = 1
        else:
            label = 0
        
        return label, probability

    def evalSymbReg(self, individual):
        
        # Transform the tree expression in a callable function
        f = self.toolbox.compile(expr=individual)
        
        # loop over and: create each sample, evaluate it, and compare against the actual result(label)
        idx_steps_to_eval  = np.arange(self.n_features+1, len(self.hist_data.index))
        results            = map(self.process_batch, [f]*len(idx_steps_to_eval), idx_steps_to_eval)
        labels, pred_probs = zip(*results) #unpack
        
        labels = np.array(labels)
        
        # count number of positive/negative class
        self.n_long_labels  = len(np.where(labels == 1)[0])
        self.n_short_labels = len(np.where(labels == 0)[0])
        
        
        # evaluate in batches as way to reduce overfit to older items in rolling history-inputs
        n_eval_groups     = 5
        batch_labels      = np.array_split(labels, n_eval_groups)
        batch_pred_probs  = np.array_split(pred_probs, n_eval_groups)
        batch_losses      = []
        consistency_score = []
        for i in np.arange(len(batch_labels)):
            loss_n = log_loss(batch_labels[i], batch_pred_probs[i], labels=[0,1])
            if not np.isfinite(loss_n):
                loss_n = 25.
            batch_losses.append(loss_n)
            
            if loss_n < 0.68: # let be more strict than -ln(0.5)
                consistency_score.append(1.)
            else:
                consistency_score.append(0.)
                
        # forced negative so every fitness is minimized, easier to read in print logs(commented out below)
        consistency_score = -np.mean(consistency_score)
        # easily influenced by overfit/lucky regions. Have to balance n_samples, batch size, pop size, world peace. Easy job.
        avg_loss          = np.mean(batch_losses) 
        # what is our worst batch? I bet its the recent one...lets improve on that(super difficult metric often flat till endgame)
        max_loss          = np.max(batch_losses) 

        # you HAVE to return a tuple to DEAP when evaluating
        return avg_loss, max_loss, consistency_score

    def evalLive(self, individual):
        
        # most recent sample
        current_step_input = list(100.*self.hist_data.iloc[-self.n_features-1:].open.pct_change().dropna().values.flatten().astype(np.float32))
        
        # Transform the tree expression in a callable function
        compiled_indv = self.toolbox.compile(expr=individual)
        pred_prob = np.clip(np.nan_to_num(compiled_indv(*current_step_input)), 0.001, 0.999)

        if pred_prob >= 0.5:
            signal = 1.
        else:
            signal = -1.

        return signal

    # NOTE, so this looks scary...but it is just a copied eaMuPlusLambda algo from:
    # https://github.com/DEAP/deap/blob/master/deap/algorithms.py
    # explained: http://deap.readthedocs.io/en/master/api/algo.html
    # Since we are using the dead-evolutionary-algo in a weird way, we need to manually set up the evolution which gives us full access.
    # In the DEAP tutorials they just call eaMuPlusLambda or eaSimple and make it looks very clean.
    def OnEvolve(self, cxpb=0.6, mutpb=0.2, lambda_=n_pop*2, verbose=__debug__):

        if self.gen == 0:
            start_time  = time.time()

            invalid_ind = [ind for ind in self.population if not ind.fitness.valid]

            fitnesses   = self.toolbox.map(self.toolbox.evaluate, invalid_ind)

            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            if self.halloffame is not None:
                self.halloffame.update(self.population)

            record = self.stats.compile(self.population) if self.stats else {}
            self.logbook.record(gen=0, nevals=len(invalid_ind), **record)
            if verbose:
                elapsed_time = time.time() - start_time
                #print self.logbook.stream + "\t\t%0.2f sec"%(elapsed_time)
                #self.Log('\n'+self.logbook.stream)
                
            self.context.evo_time = elapsed_time

            self.gen += 1

            self.selected_individuals = self.halloffame[:1]

            # save to file
            #self.Checkpoint()

        else:
            start_time = time.time()

            offspring = algorithms.varOr(self.population, self.toolbox, lambda_, cxpb, mutpb)

            invalid_ind = [ind for ind in offspring]# if not ind.fitness.valid] # force eval of every indv, as history is a moving widnow to eval on

            fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
                
            # Update the hall of fame with the generated individuals
            if self.halloffame is not None:
                self.halloffame.clear() # force eval of every indv, as history is a moving widnow to eval on
                self.halloffame.update(offspring)

            self.population[:] = self.toolbox.select(self.population + offspring, n_pop)

            # Append the current generation statistics to the logbook
            record = self.stats.compile(self.population) if self.stats else {}
            self.logbook.record(gen=self.gen, nevals=len(invalid_ind), **record)
            if verbose:
                elapsed_time = time.time() - start_time
                #print self.logbook.stream + "\t\t%0.2f sec"%(elapsed_time)
                #self.Log('\n'+self.logbook.stream)
                
            self.context.evo_time = elapsed_time

            self.gen += 1

            self.selected_individuals = self.halloffame[:1]

            # save to file
            #self.Checkpoint()

        # using the selected best item
        #signal = self.evalLive(self.halloffame[0])
        
        # but with pareto front we have ANY number of non dominated individuals each gen, just use them all as an ensemble model
        signal = stats.mode([self.evalLive(indv) for indv in self.halloffame]).mode[0]
        
        
        self.context.Log(str(self.gen) + ' : ' + str(self.halloffame[0]))

        return signal
