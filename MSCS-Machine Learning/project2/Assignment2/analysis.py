import os
import time  
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import pandas as pd
import threading
from multiprocessing import Process
import random

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




import warnings
warnings.filterwarnings("ignore")

class ImportedData:
    Title = ""
    Data = {}

# https://archive.ics.uci.edu/ml/datasets/banknote+authentication
def getBanknoteAuthData(train_size):
    banknote = pd.read_csv('./data_banknote_authentication.data', header=None)
    banknote.columns = ['Wavelet_Variance', 'Wavelet_Skewness', 'Wavelet_Curtosis', 'Image_Entropy', 'Class']
    Y = banknote['Class']
    X = banknote.drop(['Class'], axis=1)
    T = preprocessing.MinMaxScaler().fit_transform(X)
    Xs = pd.DataFrame(T,columns = X.columns)
    Ts = train_test_split(Xs, Y, train_size = train_size)
    return Ts

def saveToNewDir(directory, filename):
    if not os.path.isdir(directory):
        os.makedirs(directory)
    plt.savefig(directory+filename, bbox_inches='tight')
    
def saveFigToNewDir(fig, directory, filename):
    if not os.path.isdir(directory):
        os.makedirs(directory)
    fig.savefig(directory+filename, bbox_inches='tight')

def RunNeuralNetClassifier(ImportedData, Data):
    Title = ImportedData.Title
    print ("---------- Working on NeuralNets for " + Title + " datasets ----------")
    timers = {}

    def CalculateBestHyperParameters(ImportedData, Data):
        params =    {'hidden_layer_sizes': [(sp_randint.rvs(10,200,1),sp_randint.rvs(10,200,1),),(sp_randint.rvs(10,200,1),)],
                    'alpha': np.array([1,0.1,0.01,0.001,0.0001,0])}        
        bestLayers = 0
        bestAlpha = 0
        bestScore = 0
        atKey = 0
        for key, value in Data.items():
            tree_cv = RandomizedSearchCV(MLPClassifier(),params).fit(value["xTrain"], value["yTrain"])
            if (tree_cv.best_score_ > bestScore):
                atKey = key
                bestScore = tree_cv.best_score_
                bestLayers = tree_cv.best_params_['hidden_layer_sizes']
                bestAlpha = tree_cv.best_params_['alpha']

        dtc = MLPClassifier(hidden_layer_sizes=bestLayers, alpha=bestAlpha).fit(Data[atKey]["xTrain"], Data[atKey]["yTrain"])
        yPred = dtc.predict(Data[atKey]["xTest"])
        cvs = cross_val_score(dtc, Data[atKey]["xTrain"], Data[atKey]["yTrain"]).mean()*100
        trs = accuracy_score(Data[atKey]["yTrain"], dtc.predict(Data[atKey]["xTrain"]))*100
        tes = accuracy_score(Data[atKey]["yTest"], yPred)*100
        c_m = confusion_matrix(Data[atKey]["yTest"], yPred)
        if (Title=="BanknoteAuthentication"):
            ps = precision_score(Data[atKey]["yTest"], yPred, average='binary')*100
            rs = recall_score(Data[atKey]["yTest"], yPred, average='binary')*100
            f1 = f1_score(Data[atKey]["yTest"], yPred, average='binary')*100
        else:
            ps = precision_score(Data[atKey]["yTest"], yPred, average='weighted')*100
            rs = recall_score(Data[atKey]["yTest"], yPred, average='weighted')*100
            f1 = f1_score(Data[atKey]["yTest"], yPred, average='weighted')*100

        string = (Title + " NeuralNet calculated BEST parameters are {} Layers, {} Alpha with {} score at {}% Train and {}% Test\nPredicting with {}% Train Accuracy, {}% Test Accuracy, {}% CV Accuracy, {}% Precision Score, {}% Recall Score and {}% \F-1 Score"
            .format(bestLayers, bestAlpha, round(bestScore, 2), atKey, 100-atKey, round(trs,1), round(tes,1), round(cvs,1), round(ps,1), round(rs,1), round(f1,1)))
        
        print (string)

        with open(Title + ".txt", 'a+') as f:
            f.write(string + "\n")

        plt.figure(figsize=(10,10), dpi=150)
        if (Title=="BanknoteAuthentication"):
            group_names = ['True Neg','False Pos','False Neg','True Pos']
            group_counts = ["{0:0.0f}".format(value) for value in c_m.flatten()]
            group_percentages = ["{0:.2%}".format(value) for value in c_m.flatten()/np.sum(c_m)]
            labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
            labels = np.asarray(labels).reshape(2,2)
            sns.heatmap(c_m, annot=labels, fmt='', cmap='coolwarm', square=True, robust=True)
        else:
            categories = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
            group_counts = ["{0:0.0f}".format(value) for value in c_m.flatten()]
            labels = np.asarray(group_counts).reshape(26,26)
            sns.heatmap(c_m, annot=labels, fmt='', cmap='coolwarm', square=True, robust=True, xticklabels = categories, yticklabels = categories)

        plt.xlabel('Predicted Value')
        plt.ylabel('Actual Value')
        plt.title(Title + ' (NeuralNet) Best Parameter Confusion Matrix')
        plt.legend()
        saveToNewDir(Title+"/", Title + "_NeuralNet_BestConfusionMatrix.png")
        plt.gcf().clear()
        plt.clf() 

        return bestLayers, bestAlpha, atKey
    
    # Test different test/train ratios
    def TestTrain(ImportedData, Data, bestKey):
        plt.figure(figsize=(7,7), dpi=70)

        train_scores = []
        test_scores = []
        cv_scores = []
        timer = []
        p_s = []
        r_s = []
        f_1 = []
        value = Data[bestKey]

        for key, value in Data.items():    
            start = time.time()
            mlp = MLPClassifier(hidden_layer_sizes=bestLayers, alpha=bestAlpha).fit(value['xTrain'], value['yTrain'])
            yPred = mlp.predict(value["xTest"])
            train_scores.append(accuracy_score(value['yTrain'], mlp.predict(value['xTrain']))*100)
            cv_scores.append(cross_val_score(mlp, value['xTrain'], value['yTrain'], cv=5).mean()*100)
            test_scores.append(accuracy_score(value['yTest'], yPred)*100)
            if (Title=="BanknoteAuthentication"):
                ps = precision_score(value["yTest"], yPred, average='binary')*100
                rs = recall_score(value["yTest"], yPred, average='binary')*100
                f1 = f1_score(value["yTest"], yPred, average='binary')*100
            else:
                ps = precision_score(value["yTest"], yPred, average='weighted')*100
                rs = recall_score(value["yTest"], yPred, average='weighted')*100
                f1 = f1_score(value["yTest"], yPred, average='weighted')*100            
            p_s.append(ps)
            r_s.append(rs)
            f_1.append(f1)
            timer.append(time.time()-start)
        
        timers["Training"] = timer
        plt.plot(Data.keys(), train_scores, label="Train", lw = 2)
        plt.plot(Data.keys(), test_scores, label="Test", lw = 2)
        plt.plot(Data.keys(), cv_scores, label="CV", lw = 2)
        plt.plot(Data.keys(), p_s, label='Precision', lw = 2)
        plt.plot(Data.keys(), r_s, label='Recall', lw = 2)
        plt.plot(Data.keys(), f_1, label='F1', lw = 2)
        plt.xlabel("Training Set Size")
        plt.ylabel("Accuracy (%)")
        plt.legend(loc='best')
        plt.title(Title + " (NeuralNet) Accuracy with train size")
        saveToNewDir(Title+"/", Title + "_NeuralNet_TrainingSetSize.png")
        plt.clf()
    
    bestLayers, bestAlpha, bestKey = CalculateBestHyperParameters(ImportedData, Data)
    TestTrain(ImportedData, Data, bestKey)

    #Timers Plot
    plt.figure(figsize=(7,7), dpi=70)
    plt.xlabel('Index')
    plt.ylabel('Time')
    plt.xticks(range(1,11,1))
    plt.title(Title + ' (NeuralNet) Training Time Vs. Index')
    #plt.plot([x/5 for x in list(Data.keys())],timers['Training'], label='Training Set Size', lw = 2)
    plt.plot(timers['Training'], label='Training Set Size', lw = 2)
    plt.legend()
    saveToNewDir(Title+"/", Title + "_NeuralNet_Timers.png")
    plt.gcf().clear()
    plt.clf() 
    print ("Done with NeuralNets for " + Title + " datasets")

def NeuralNetOptimizeRHC(ImportedData, Data):
    iterations = [25, 75, 225]
    # RHC parameters
    restarts = [10, 20, 30]

    print ("Optimizing RHC")
    fig = plt.figure(figsize=(12,8))
    ax = fig.gca(projection='3d')
    fig.suptitle("Neural Nets RHC")           
    title = "RHC"
    elapsed_arr = np.zeros((len(restarts),len(iterations)))
    train_arr = np.zeros((len(restarts),len(iterations)))
    test_arr = np.zeros((len(restarts),len(iterations)))
    cv_arr = np.zeros((len(restarts),len(iterations)))
    f1_arr = np.zeros((len(restarts),len(iterations)))

    itr = 0
    for i in iterations:
        rest= 0
        for r in restarts:
            nn = mlrose.NeuralNetwork(hidden_nodes=[20], activation='relu', algorithm='random_hill_climb',
                                restarts = r, max_iters=i, curve=False)
            t0 = time.time()
            nn.fit(Data['xTrain'], Data['yTrain'])
            elapsed_arr[rest][itr] = time.time() - t0
            yPred = nn.predict(Data["xTest"])
            train_arr[rest][itr] = accuracy_score(Data['yTrain'], nn.predict(Data['xTrain']))*100
            test_arr[rest][itr] = accuracy_score(Data['yTest'], yPred)*100
            cv_arr[rest][itr] = cross_val_score(nn, Data['xTrain'], Data['yTrain'], cv=5).mean()*100
            f1_arr[rest][itr] = f1_score(Data["yTest"], yPred, average='binary')*100

            rest += 1
        itr += 1

    X, Y = np.meshgrid(range(len(iterations)), range(len(restarts)))
    ax.contour(X, Y, train_arr, zdir='x', offset=0, color='b')
    ax.contour(X, Y, train_arr, zdir='y', offset=len(restarts)-1, color='b')
    surf = ax.plot_surface(X, Y, train_arr, lw=2, label= 'train', color='b', alpha=0.3)
    surf._facecolors2d=surf._facecolors3d
    surf._edgecolors2d=surf._edgecolors3d
    ax.contour(X, Y, test_arr, zdir='x', offset=0, color='g')
    ax.contour(X, Y, test_arr, zdir='y', offset=len(restarts)-1, color='g')
    surf = ax.plot_surface(X, Y, test_arr, lw=2, label= 'test', color='g', alpha=0.3)
    surf._facecolors2d=surf._facecolors3d
    surf._edgecolors2d=surf._edgecolors3d
    ax.contour(X, Y, cv_arr, zdir='x', offset=0, color='r')
    ax.contour(X, Y, cv_arr, zdir='y', offset=len(restarts)-1, color='r')
    surf = ax.plot_surface(X, Y, cv_arr, lw=2, label= 'CV', color='r', alpha=0.3)
    surf._facecolors2d=surf._facecolors3d
    surf._edgecolors2d=surf._edgecolors3d
    ax.contour(X, Y, f1_arr, zdir='x', offset=0, color='c')
    ax.contour(X, Y, f1_arr, zdir='y', offset=len(restarts)-1, color='c')
    surf = ax.plot_surface(X, Y, f1_arr, lw=2, label= 'F1', color='c', alpha=0.3)
    surf._facecolors2d=surf._facecolors3d
    surf._edgecolors2d=surf._edgecolors3d

    ax.set(xlabel="Iterations", ylabel="Restarts", zlabel='Accuracy')
    ax.set(xticks=range(len(iterations)), xticklabels=iterations, 
        yticks=range(len(restarts)), yticklabels=restarts)
    fig.legend(loc='center right')
    plt.tight_layout()
    #plt.show()
    saveFigToNewDir(fig, "./",title + "_NeuralNet.png")
    plt.clf()

def NeuralNetOptimizeSA(ImportedData, Data):
    iterations = [25, 75, 225]
    
    # SA parameters
    decay = ['geom', 'exp', 'arith']

    print ("Optimizing SA")
    decay_lookup = {
        'geom': mlrose.GeomDecay(),
        'exp': mlrose.ExpDecay(),
        'arith': mlrose.ArithDecay()
        }

    fig = plt.figure(figsize=(12,8))
    ax = fig.gca(projection='3d')
    fig.suptitle("Neural Nets SA")           
    title = "SA"
    elapsed_arr = np.zeros((len(decay),len(iterations)))
    train_arr = np.zeros((len(decay),len(iterations)))
    test_arr = np.zeros((len(decay),len(iterations)))
    cv_arr = np.zeros((len(decay),len(iterations)))
    f1_arr = np.zeros((len(decay),len(iterations)))

    itr = 0
    for i in iterations:
        rest= 0
        for r in decay:
            nn = mlrose.NeuralNetwork(hidden_nodes=[20], algorithm='simulated_annealing',
                                max_iters=i, schedule=decay_lookup[decay[rest]],
                                curve=False)
            t0 = time.time()
            nn.fit(Data['xTrain'], Data['yTrain'])
            elapsed_arr[rest][itr] = time.time() - t0
            yPred = nn.predict(Data["xTest"])
            train_arr[rest][itr] = accuracy_score(Data['yTrain'], nn.predict(Data['xTrain']))*100
            test_arr[rest][itr] = accuracy_score(Data['yTest'], yPred)*100
            cv_arr[rest][itr] = cross_val_score(nn, Data['xTrain'], Data['yTrain'], cv=5).mean()*100
            f1_arr[rest][itr] = f1_score(Data["yTest"], yPred, average='binary')*100

            rest += 1
        itr += 1

    X, Y = np.meshgrid(range(len(iterations)), range(len(decay)))
    ax.contour(X, Y, train_arr, zdir='x', offset=0, color='b')
    ax.contour(X, Y, train_arr, zdir='y', offset=len(decay)-1, color='b')
    surf = ax.plot_surface(X, Y, train_arr, lw=2, label='train', color='b', alpha=0.3)
    surf._facecolors2d=surf._facecolors3d
    surf._edgecolors2d=surf._edgecolors3d
    ax.contour(X, Y, test_arr, zdir='x', offset=0, color='g')
    ax.contour(X, Y, test_arr, zdir='y', offset=len(decay)-1, color='g')
    surf = ax.plot_surface(X, Y, test_arr, lw=2, label='test', color='g', alpha=0.3)
    surf._facecolors2d=surf._facecolors3d
    surf._edgecolors2d=surf._edgecolors3d
    ax.contour(X, Y, cv_arr, zdir='x', offset=0, color='r')
    ax.contour(X, Y, cv_arr, zdir='y', offset=len(decay)-1, color='r')
    surf = ax.plot_surface(X, Y, cv_arr, lw=2, label='CV', color='r', alpha=0.3)
    surf._facecolors2d=surf._facecolors3d
    surf._edgecolors2d=surf._edgecolors3d
    ax.contour(X, Y, f1_arr, zdir='x', offset=0, color='c')
    ax.contour(X, Y, f1_arr, zdir='y', offset=len(decay)-1, color='c')
    surf = ax.plot_surface(X, Y, f1_arr, lw=2, label='F1', color='c', alpha=0.3)
    surf._facecolors2d=surf._facecolors3d
    surf._edgecolors2d=surf._edgecolors3d

    ax.set(xlabel="Iterations", ylabel="Decay", zlabel='Accuracy')
    ax.set(xticks=range(len(iterations)), xticklabels=iterations, 
        yticks=range(len(decay)), yticklabels=decay)
    fig.legend(loc='center right')
    plt.tight_layout()
    #plt.show()
    saveFigToNewDir(fig, "./",title + "_NeuralNet.png")
    plt.clf()

def NeuralNetOptimizeGA(ImportedData, Data):
    iterations = [25, 75, 225]
    populations = [30, 60, 90]
    mutations   = [0.001, 0.01, 0.1]
    print ("Optimizing GA")
    for i in iterations:
        fig = plt.figure(figsize=(12,8))
        ax = fig.gca(projection='3d')
        fig.suptitle("Neural Nets GA {}".format(i))           
        title = "GA"
        elapsed_arr = np.zeros((len(mutations),len(populations)))
        train_arr = np.zeros((len(mutations),len(populations)))
        test_arr = np.zeros((len(mutations),len(populations)))
        cv_arr = np.zeros((len(mutations),len(populations)))
        f1_arr = np.zeros((len(mutations),len(populations)))
        p_itr= 0
        for p in populations:
            m_itr = 0
            for m in mutations:
                nn = mlrose.NeuralNetwork(hidden_nodes=[20], activation='relu', algorithm='genetic_alg',
                                    max_iters=i, pop_size=p, mutation_prob=m,
                                    curve=False)
                t0 = time.time()
                nn.fit(Data['xTrain'], Data['yTrain'])
                elapsed_arr[m_itr][p_itr] = time.time() - t0
                yPred = nn.predict(Data["xTest"])
                train_arr[m_itr][p_itr] = accuracy_score(Data['yTrain'], nn.predict(Data['xTrain']))*100
                test_arr[m_itr][p_itr] = accuracy_score(Data['yTest'], yPred)*100
                cv_arr[m_itr][p_itr] = cross_val_score(nn, Data['xTrain'], Data['yTrain'], cv=5).mean()*100
                f1_arr[m_itr][p_itr] = f1_score(Data["yTest"], yPred, average='binary')*100

                m_itr += 1
            p_itr += 1

        X, Y = np.meshgrid(range(len(populations)), range(len(mutations)))
        ax.contour(X, Y, train_arr, zdir='x', offset=0, color='b')
        ax.contour(X, Y, train_arr, zdir='y', offset=len(mutations)-1, color='b')
        surf = ax.plot_surface(X, Y, train_arr, lw=2, label='train', color='b', alpha=0.3)
        surf._facecolors2d=surf._facecolors3d
        surf._edgecolors2d=surf._edgecolors3d
        ax.contour(X, Y, test_arr, zdir='x', offset=0, color='g')
        ax.contour(X, Y, test_arr, zdir='y', offset=len(mutations)-1, color='g')
        surf = ax.plot_surface(X, Y, test_arr, lw=2, label='test', color='g', alpha=0.3)
        surf._facecolors2d=surf._facecolors3d
        surf._edgecolors2d=surf._edgecolors3d
        ax.contour(X, Y, cv_arr, zdir='x', offset=0, color='r')
        ax.contour(X, Y, cv_arr, zdir='y', offset=len(mutations)-1, color='r')
        surf = ax.plot_surface(X, Y, cv_arr, lw=2, label='CV', color='r', alpha=0.3)
        surf._facecolors2d=surf._facecolors3d
        surf._edgecolors2d=surf._edgecolors3d
        ax.contour(X, Y, f1_arr, zdir='x', offset=0, color='c')
        ax.contour(X, Y, f1_arr, zdir='y', offset=len(mutations)-1, color='c')
        surf = ax.plot_surface(X, Y, f1_arr, lw=2, label='F1', color='c', alpha=0.3)
        surf._facecolors2d=surf._facecolors3d
        surf._edgecolors2d=surf._edgecolors3d

        #ax1.text(0, 0,0,"max_attempts={a}\n# population={b}\nmutation={c}\nbest_fitness={d}".format(a=best[0], b=best[1], c=best[2], d=best[3]), horizontalalignment="left", verticalalignment="top")
        ax.set(xlabel="Populations", ylabel="Mutation", zlabel='Accuracy')
        ax.set(xticks=range(len(populations)), xticklabels=populations, 
            yticks=range(len(mutations)), yticklabels=mutations)
        fig.legend(loc='center right')
        #print (title + " max_attempts={a}, # population={b}, mutation={c}, best_fitness={d}".format(a=best[0], b=best[1], c=best[2], d=best[3]))
        plt.tight_layout()
        #plt.show()
        saveFigToNewDir(fig, "./",title + "_" + str(i) + "itr_NeuralNet.png")
        plt.clf()

class ProblemBase():
    def __init__(self, verbose=False):
        self.verbose = verbose

    def saveToNewDir(self, fig, directory, filename):
        if not os.path.isdir(directory):
            os.makedirs(directory)
        fig.savefig(directory+filename, bbox_inches='tight')

    def test_random_hill(self, title, max_attempts_range=[100], random_restarts_range=[0]):
        print(title+" Random Hill Climbing Algorithm")
        So is there
        fig.suptitle(title+" Random Hill Climb")
        best = [0, 0, 0]
        for m in max_attempts_range:
            fitness_arr = []
            time_arr = []
            for r in random_restarts_range:
                start = time.time()
                best_state, best_fitness, curve = mlrose.random_hill_climb(self.problem_fit,
                                                                        max_attempts=m,
                                                                        max_iters=np.inf,
                                                                        restarts=r,
                                                                        curve=True)
                fitness_arr.append(best_fitness)
                time_arr.append(round(time.time() - start, 2))
                if best_fitness > best[2]:
                    best[0] = m
                    best[1] = r
                    best[2] = best_fitness

            ax1.plot(random_restarts_range, fitness_arr, label=m, lw=2)
            ax2.plot(random_restarts_range, time_arr, lw=2)

        ax1.set(xlabel="# Restarts", ylabel="Fitness")
        ax2.set(xlabel="# Restarts", ylabel="Time(s)")
        fig.legend(loc='center right', title= 'Attempts')
        print (title + " RHC max_attempts={a}, # restarts={b}, best_fitness={c}".format(a=best[0], b=best[1], c=best[2]))
        #ax1.text(x=0.05, y=0.95,s="max_attempts={a}\n# restarts={b}\nbest_fitness={c}".format(a=best[0], b=best[1], c=best[2]))
        plt.tight_layout()
        self.saveToNewDir(fig, "./",title + "_Random_Hill_Climb.png")
        plt.clf()

    def test_simulated_annealing(self, title, max_attempts_range=[100], decay_range=['geom']):
        decay_lookup = {
            'geom': mlrose.GeomDecay(),
            'exp': mlrose.ExpDecay(),
            'arith': mlrose.ArithDecay()
        }
        fig, (ax1, ax2) = plt.subplots(2, figsize=(12,8), dpi=80)
        fig.suptitle(title+" Simmulated Annealing")
        print(title+" Simulated Annealing Algo")
        best = [0, 0, 0]
        for m in max_attempts_range:
            fitness_arr = []
            time_arr = []
            for d in decay_range:
                start = time.time()
                # solve using simulated annealing
                best_state, best_fitness, curve = mlrose.simulated_annealing(self.problem_fit,
                                                                            schedule=decay_lookup[d],
                                                                            max_attempts=m,
                                                                            max_iters=np.inf,
                                                                            curve=True)
                fitness_arr.append(best_fitness)
                time_arr.append(round(time.time() - start, 2))
                if best_fitness > best[2]:
                    best[0] = m
                    best[1] = d
                    best[2] = best_fitness

            ax1.plot(decay_range, fitness_arr, label=m, lw=2)
            ax2.plot(decay_range, time_arr, lw=2)

        ax1.set(xlabel="Decay Range", ylabel="Fitness")
        ax2.set(xlabel="Decay Range", ylabel="Time(s)")
        print (title + " SA max_attempts={a}, # decay={b}, best_fitness={c}".format(a=best[0], b=best[1], c=best[2]))
        fig.legend(loc='center right', title= 'Attempts')
        plt.tight_layout()
        self.saveToNewDir(fig, "./",title + "_Simulated_Annealing.png")
        plt.clf()

    def test_genetic_algorithm(self, title, max_attempts_range=[100], pop_range=[200], mutation_range=[0.1]):
        pop_len = range(len(pop_range))
        mut_len = range(len(mutation_range))
        i_pop = 0
        i_mut = 0

        fig = plt.figure(figsize=(12,8))
        ax1 = fig.add_subplot(2,1,1, projection='3d')
        ax2 = fig.add_subplot(2,1,2, projection='3d') 
        fig.suptitle(title + " Genetic Algorithm")           

        color = 0
        colors = ['b', 'g', 'r']

        best = [0, 0, 0, 0]
        print(title+" Running Genetic Algorithm")
        for m in max_attempts_range:
            fitness_arr = np.zeros((len(mutation_range),len(pop_range)))
            time_arr = np.zeros((len(mutation_range),len(pop_range)))
            i_pop = 0
            for p in pop_range: 
                i_mut = 0
                for mut in mutation_range:
                    start = time.time()
                    # solve using genetic algorithm
                    best_state, best_fitness, curve = mlrose.genetic_alg(self.problem_fit,
                                                                        pop_size=p,
                                                                        mutation_prob=mut,
                                                                        max_attempts=m,
                                                                        max_iters=np.inf,
                                                                        curve=True)
                    fitness_arr[i_mut][i_pop]=(best_fitness)
                    time_arr[i_mut][i_pop]=(round(time.time() - start, 2))
                    if best_fitness > best[3]:
                        best[0] = m
                        best[1] = p
                        best[2] = mut
                        best[3] = best_fitness

                    i_mut += 1
                i_pop+=1

            X, Y = np.meshgrid(pop_len, mut_len)
            ax1.contour(X, Y, fitness_arr, zdir='x', offset=0, color=colors[color])
            ax1.contour(X, Y, fitness_arr, zdir='y', offset=len(mutation_range)-1, color=colors[color])
            surf = ax1.plot_surface(X, Y, fitness_arr, lw=2, label=m, color=colors[color], alpha=0.3)
            surf._facecolors2d=surf._facecolors3d
            surf._edgecolors2d=surf._edgecolors3d
            ax2.plot_surface(X, Y, time_arr, lw=2, color=colors[color], alpha=0.3)
            color+=1

        ax1.set(xlabel="Population", ylabel="Mutation", zlabel='Fitness')
        ax1.set(xticks=range(len(pop_range)), xticklabels=pop_range, 
            yticks=range(len(mutation_range)), yticklabels=mutation_range)
        ax2.set(xlabel="Population", ylabel="Mutation", zlabel='Time (s)')
        ax2.set(xticks=range(len(pop_range)), xticklabels=pop_range, 
            yticks=range(len(mutation_range)), yticklabels=mutation_range)
        fig.legend(loc='center right', title= 'Attempts')
        print (title + " GA max_attempts={a}, # population={b}, mutation={c}, best_fitness={d}".format(a=best[0], b=best[1], c=best[2], d=best[3]))
        plt.tight_layout()
        self.saveToNewDir(fig, "./",title + "_Genetic_Algorithm.png")
        plt.clf()

    def test_mimic(self, title, max_attempts_range=[100], pop_range=[200], keep_pct_range=[0.2]):
        
        pop_len = range(len(pop_range))
        mut_len = range(len(keep_pct_range))
        i_pop = 0
        i_mut = 0

        color = 0
        colors = ['b', 'g', 'r']
        fig = plt.figure(figsize=(12,8))
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
                                                                max_attempts=m,
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
            ax1.contour(X, Y, fitness_arr, zdir='x', offset=0, color=colors[color])
            ax1.contour(X, Y, fitness_arr, zdir='y', offset=len(keep_pct_range)-1, color=colors[color])
            surf = ax1.plot_surface(X, Y, fitness_arr, lw=2, label=m, color=colors[color], alpha=0.3)
            surf._facecolors2d=surf._facecolors3d
            surf._edgecolors2d=surf._edgecolors3d
            ax2.plot_surface(X, Y, time_arr, lw=2, color=colors[color], alpha=0.3)
            color+=1

        ax1.set(xlabel="Population", ylabel="Keep %", zlabel='Fitness')
        ax1.set(xticks=range(len(pop_range)), xticklabels=pop_range, 
            yticks=range(len(keep_pct_range)), yticklabels=keep_pct_range)
        ax2.set(xlabel="Population", ylabel="Keep %", zlabel='Time (s)')
        ax2.set(xticks=range(len(pop_range)), xticklabels=pop_range, 
            yticks=range(len(keep_pct_range)), yticklabels=keep_pct_range)
        fig.legend(loc='center right', title= 'Attempts')
        plt.tight_layout()
        print (title + " MIMIC max_attempts={a}, # population={b}, keep %={c}, best_fitness={d}".format(a=best[0], b=best[1], c=best[2], d=best[3]))
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

if __name__ == "__main__":
    # Backpropagation from Assignment #1
    # scaler = StandardScaler()
    # bankData = ImportedData()
    # bankTemp = {}
    # bankData.Title = "BanknoteAuthentication"
    # for split in range(10, 90, 8):
    #     if(split==100):
    #         split-=1
    #     if(split==0):
    #         split+=1
    #     bankTemp[split] = {}
    #     bankTemp[split]["xTrain"], bankTemp[split]["xTest"], bankTemp[split]["yTrain"], bankTemp[split]["yTest"] = getBanknoteAuthData(split/100.0)
    #     scaler.fit(bankTemp[split]["xTrain"])
    #     bankTemp[split]["xTrain"] = scaler.transform(bankTemp[split]["xTrain"])
    #     bankTemp[split]["xTest"] = scaler.transform(bankTemp[split]["xTest"])
    # bankData.Data = bankTemp
    
    # # Optimization Data
    # bankOpt = ImportedData()
    # bankOpt.Title = "BanknoteAuthentication"
    # tempOpt = {}
    # tempOpt["xTrain"], tempOpt["xTest"], tempOpt["yTrain"], tempOpt["yTest"] = getBanknoteAuthData(0.8)
    # scaler.fit(tempOpt["xTrain"])
    # tempOpt["xTrain"] = scaler.transform(tempOpt["xTrain"])
    # tempOpt["xTest"] = scaler.transform(tempOpt["xTest"])
    # bankOpt.Data = tempOpt

    # w = Process(target=RunNeuralNetClassifier, args=(bankData, bankData.Data))
    # x = Process(target=NeuralNetOptimizeRHC, args=(bankOpt, bankOpt.Data))
    # y = Process(target=NeuralNetOptimizeSA, args=(bankOpt, bankOpt.Data))
    # z = Process(target=NeuralNetOptimizeGA, args=(bankOpt, bankOpt.Data))


    # Random Hill Climb Params
    random_restarts_range = [5, 25, 45, 65, 85]
    # Simulated Annealing Params
    decay_range = ['geom', 'exp', 'arith']
    # Genetic Algorithm Params
    ga_pop_range = [10, 100, 1000] #[50000, 100000]
    mutation_range = [0.001, 0.01, 0.1] #[0.05, 0.1, 0.2]
    # MIMIC Params
    mimic_pop_range = [25, 125, 250]#[100000, 200000]
    keep_pct_range = [0.01, 0.1, 0.3]

    np.random.seed(100)

    attempt_range = [2**x for x in range(1,10,2)]
    attempt_evo_range = [2**x for x in range(2,9,3)]

    fp = ProblemBase()
    ff = ProblemBase()
    ks = ProblemBase()
    om = ProblemBase()

    fp.FourPeaks(length=60, t_pct=0.25, verbose=True)
    ff.FlipFlop(length=50, verbose=True)
    ks.Knapsack(length=25, max_weight_pct=0.25, verbose=True)
    om.OneMax(length=25, verbose=True)

    # print("Random Hill")
    # a = Process(target=fp.test_random_hill, args=("Four_Peaks", attempt_range, random_restarts_range))
    # b = Process(target=ff.test_random_hill, args=("Flip_Flop", attempt_range, random_restarts_range))
    # c = Process(target=ks.test_random_hill, args=("Knapsack", attempt_range, random_restarts_range))
    # d = Process(target=om.test_random_hill, args=("OneMax", attempt_range, random_restarts_range))

    # print("Simulated Annealing")
    # e = Process(target=fp.test_simulated_annealing, args=("Four_Peaks", attempt_range, decay_range))
    # f = Process(target=ff.test_simulated_annealing, args=("Flip_Flop", attempt_range, decay_range))
    # g = Process(target=ks.test_simulated_annealing, args=("Knapsack", attempt_range, decay_range))
    # h = Process(target=om.test_simulated_annealing, args=("OneMax", attempt_range, decay_range))

    # print("Genetic Algorithm")
    # i = Process(target=fp.test_genetic_algorithm, args=("Four_Peaks", attempt_evo_range, ga_pop_range, mutation_range))
    # j = Process(target=ff.test_genetic_algorithm, args=("Flip_Flop", attempt_evo_range, ga_pop_range, mutation_range))
    # k = Process(target=ks.test_genetic_algorithm, args=("Knapsack", attempt_evo_range, ga_pop_range, mutation_range))
    # l = Process(target=om.test_genetic_algorithm, args=("OneMax", attempt_evo_range, ga_pop_range, mutation_range))

    print("MIMIC")
    m = Process(target=fp.test_mimic, args=("Four_Peaks", attempt_evo_range, mimic_pop_range, keep_pct_range))
    n = Process(target=ff.test_mimic, args=("Flip_Flop", attempt_evo_range, mimic_pop_range, keep_pct_range))
    o = Process(target=ks.test_mimic, args=("Knapsack", attempt_evo_range, mimic_pop_range, keep_pct_range))
    p = Process(target=om.test_mimic, args=("OneMax", attempt_evo_range, mimic_pop_range, keep_pct_range))

    # a.start()
    # b.start()
    # c.start()
    # d.start()
    # e.start()
    # f.start()
    # g.start()
    # h.start()
    # i.start()
    # j.start()
    # k.start()
    # l.start()
    m.start()
    n.start()
    o.start()
    p.start()
    # w.start()
    # x.start()
    # y.start()
    # z.start()

    # a.join()
    # b.join()
    # c.join()
    # d.join()
    # e.join()
    # f.join()
    # g.join()
    # h.join()
    # i.join()
    # j.join()
    # k.join()
    # l.join()
    m.join()
    n.join()
    o.join()
    p.join()
    # w.join()
    # x.join()
    # y.join()
    # z.join()