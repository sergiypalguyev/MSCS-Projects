import os
import time  
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import pandas as pd
import threading
from multiprocessing import Process
import random
import math

from scipy.stats import randint
from functools import partial
import itertools
from numpy import arange
from random import uniform
from scipy.stats import randint as sp_randint

from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.neural_network import MLPClassifier

from sklearn import preprocessing 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_curve, auc

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
import seaborn as sns


import six
import sys
sys.modules['sklearn.externals.six'] = six
import mlrose
import numpy as np
import time

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
from matplotlib import cm

from numba import jit, cuda 
from timeit import default_timer as timer  

import warnings
warnings.filterwarnings("ignore")


def plotThisPlot():
    fig = plt.figure(figsize=(8,12))
    ax1 = fig.add_subplot(2,1,1, projection='3d')
    ax2 = fig.add_subplot(2,1,2, projection='3d') 
    fig.suptitle("Neural Nets RHC")     
    i = {1, 2}
    j = {1, 2}
    k = np.ones((len(i),len(j)))
    max_range = [1, 2, 3]

    X, Y = np.meshgrid(range(len(i)), range(len(j)))
    for i in max_range:
        surf = ax1.plot_surface(X, Y, k, lw=2, label=i, color='b', alpha=0.3)
        ax1.contour(X, Y, k, zdir='z', offset=-1, cmap=cm.coolwarm)
        ax1.contour(X, Y, k, zdir='x', offset=0, cmap=cm.coolwarm)
        ax1.contour(X, Y, k, zdir='y', offset=1, cmap=cm.coolwarm)
        surf._facecolors2d=surf._facecolors3d
        surf._edgecolors2d=surf._edgecolors3d
        surf = ax2.plot_surface(X, Y, k, lw=2, color='g')
        surf._facecolors2d=surf._facecolors3d
        surf._edgecolors2d=surf._edgecolors3d

    fig.text(x=0.05,y=0.95,s="abc\nabc\nabc")
    ax1.set(xlabel="Iterations", ylabel="Restarts", zlabel='Accuracy')
    ax2.set(xlabel="Iterations", ylabel="Restarts", zlabel='Accuracy')
    
    xs = [1,2]
    ys = ['a','b']
    zs = [1,2,'a','b']
    ax1.set(xticks=range(len(xs)), xticklabels=xs,
       yticks=range(len(ys)), yticklabels=ys)
    fig.legend(loc = 'center right', title="stuff")
    #print (title + " max_attempts={a}, # population={b}, mutation={c}, best_fitness={d}".format(a=best[0], b=best[1], c=best[2], d=best[3]))
    plt.tight_layout()
    plt.show()
    plt.clf()


class ProblemBase():
    def __init__(self, verbose=False):
        self.verbose = verbose

    def saveToNewDir(self, fig, directory, filename):
        if not os.path.isdir(directory):
            os.makedirs(directory)
        fig.savefig(directory+filename, bbox_inches='tight')

    def test_mimic(self, title, max_attempts_range=[100], pop_range=[200], keep_pct_range=[0.2]):
        pop_len = range(len(pop_range))
        mut_len = range(len(keep_pct_range))
        i_pop = 0
        i_mut = 0

        color = 0
        colors = ['k', 'b', 'g', 'r', 'c']
        fig = plt.figure(figsize=(8,12))
        ax1 = fig.add_subplot(2,1,1, projection='3d')
        ax2 = fig.add_subplot(2,1,2, projection='3d')     
        fig.suptitle(title + " MIMIC Algorithm")         
        print(title+" MIMIC Algo")
        best = [0, 0, 0, 0]
        for m in max_attempts_range: 
            fitness_arr = np.zeros((len(keep_pct_range),len(pop_range)))
            time_arr = np.zeros((len(keep_pct_range),len(pop_range)))
            i_pop = 0
            for p in pop_range:
                i_mut = 0
                for kp in keep_pct_range:
                    start = time.time()
                    best_state, best_fitness, curve = mlrose.mimic(self.problem_fit,
                                                                pop_size=p,
                                                                keep_pct=kp,
                                                                max_attempts=math.ceil(m),
                                                                max_iters=np.inf,
                                                                curve=True)
                    fitness_arr[i_mut][i_pop]=(best_fitness)
                    time_arr[i_mut][i_pop]=(round(time.time() - start, 2))
                    if best_fitness > best[3]:
                        best[0] = m
                        best[1] = p
                        best[2] = kp
                        best[3] = best_fitness

                    i_mut += 1
                i_pop+=1
                
            X, Y = np.meshgrid(pop_len, mut_len)
            surf = ax1.plot_surface(X, Y, fitness_arr, lw=2, label=m, color=colors[color])
            surf._facecolors2d=surf._facecolors3d
            surf._edgecolors2d=surf._edgecolors3d
            ax2.plot_surface(X, Y, time_arr, lw=2, color=colors[color])
            color+=1

        ax1.set(xlabel="Population", ylabel="Keep %", zlabel='Fitness')
        ax2.set(xlabel="Population", ylabel="Keep %", zlabel='Time (s)')
        ax1.set(xticks=range(len(pop_range)), xticklabels=pop_range, 
            yticks=range(len(keep_pct_range)), yticklabels=keep_pct_range)
        fig.legend(loc='center right', title= 'Attempts')
        plt.tight_layout()
        print (title + " MIMIC max_attempts={a}, # population={b}, keep %={c}, best_fitness={d}".format(a=best[0], b=best[1], c=best[2], d=best[3]))
        #ax1.text(x=0.05,y=0.95, s="max_attempts={a}\n# population={b}\nkeep %={c}\nbest_fitness={d}".format(a=best[0], b=best[1], c=best[2], d=best[3]))
        self.saveToNewDir(fig, "./",title + "_MIMIC.png")
        plt.clf()
    
    def FlipFlop(self, length=8, verbose=False):
        self.problem = 'flipflop{l}'.format(l=length)
        self.verbose = verbose
        fitness_fn = mlrose.FlipFlop()
        self.problem_fit = mlrose.DiscreteOpt(length=length, fitness_fn=fitness_fn, maximize=True)

    def Knapsack(self, length=10, max_weight_pct=0.2, verbose=False):
        def gen_data(length):
            weights = []
            values = []
            max_weight = 50
            max_val = 50
            for i in range(length):
                weights.append(np.random.randint(1, max_weight))
                values.append(np.random.randint(1, max_val))
            return [weights, values]

        self.problem = 'knapsack{l}'.format(l=length)
        self.verbose = verbose
        weights, values = gen_data(length)
        fitness_fn = mlrose.Knapsack(weights, values, max_weight_pct)
        # define optimization problem object
        self.problem_fit = mlrose.DiscreteOpt(length=len(weights), fitness_fn=fitness_fn, maximize=True)

    def FourPeaks(self, length=10, t_pct=0.1, verbose=False):
        self.problem = 'fourpeaks{l}'.format(l=length)
        self.verbose = verbose
        fitness_fn = mlrose.FourPeaks(t_pct=t_pct)
        # define optimization problem object
        self.problem_fit = mlrose.DiscreteOpt(length=length, fitness_fn=fitness_fn, maximize=True)

    def OneMax(self, length=10, verbose=False):
        self.problem = 'onemax{l}'.format(l=length)
        self.verbose = verbose
        np.random.seed(0)
        problem_size = 1000
        fitness = mlrose.OneMax()
        state = np.random.randint(2, size=problem_size)
        self.problem_fit = mlrose.DiscreteOpt(length=problem_size, fitness_fn=fitness, maximize=True)

  
# normal function to run on cpu 
def func(a):      
    start = timer()                            
    for i in range(10000000): 
        a[i]+= 1      
    print("without GPU:", timer()-start) 
  
# function optimized to run on gpu  
@jit(target ="cuda")                          
def func2(): 
    n = 10000000                            
    a = np.ones(n, dtype = np.float64)
    start = timer() 
    for i in range(10000000): 
        a[i]+= 1
    print("with GPU:", timer()-start) 

if __name__ == "__main__":
    #plotThisPlot()

    # # Random Hill Climb Params
    # random_restarts_range = [5, 25, 45, 65, 85]
    # # Simulated Annealing Params
    # decay_range = ['geom', 'exp', 'arith']
    # # Genetic Algorithm Params
    # ga_pop_range = [10, 50, 100, 500, 1000] #[50000, 100000]
    # mutation_range = [0.001, 0.005, 0.01, 0.05, 0.1] #[0.05, 0.1, 0.2]
    # # MIMIC Params
    # mimic_pop_range = [10, 50, 100] #[100000, 200000]
    # keep_pct_range = [0.001, 0.005, 0.01]

    # np.random.seed(100)

    # #attempt_range = [2**x for x in range(1,10,2)]
    # attempt_range = [2**x for x in range(1,6,1)]


    # fp = ProblemBase()
    # ff = ProblemBase()
    # ks = ProblemBase()
    # om = ProblemBase()

    # fp.FourPeaks(length=60, t_pct=0.25, verbose=True)
    # ff.FlipFlop(length=50, verbose=True)
    # ks.Knapsack(length=25, max_weight_pct=0.25, verbose=True)
    # om.OneMax(length=25, verbose=True)

    # print("MIMIC")
    # a =Process(target=fp.test_mimic, args=("Four_Peaks", attempt_range, mimic_pop_range, keep_pct_range))
    # b =Process(target=ff.test_mimic, args=("Flip_Flop", attempt_range, mimic_pop_range, keep_pct_range))
    # c =Process(target=ks.test_mimic, args=("Knapsack", attempt_range, mimic_pop_range, keep_pct_range))
    # d =Process(target=om.test_mimic, args=("OneMax", attempt_range, mimic_pop_range, keep_pct_range))
    # a.start()
    # b.start()
    # c.start()
    # d.start()
    # a.join()
    # b.join()
    # c.join()
    # d.join()
      
    #start = timer() 
    #i = Process(target=func, args=(a)) 
    #print("without GPU:", timer()-start)     
      
    #start = timer() 
    j=Process(target=func2, args=()) 
    #print("with GPU:", timer()-start) 

    j.start()
    j.join()

