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

# https://archive.ics.uci.edu/ml/datasets/Letter+Recognition
def getLetterRecData(train_size):
    letter = pd.read_csv('./letter-recognition.data', header=None)
    letter.columns = ['Letter', 'xBoxHPos', 'yBoxVPos', 'BoxW', 'BoxH', 'OnPix', 'xBarMean', 'yBarMean', 'x2BarMean', 'y2BarMean', 'xyBarMean', 'x2ybrMean', 'xy2BrMean','xEgeMean', 'xegvyCorr', 'y-egeMean', 'yegvxCorr']
    #Y = letter['Letter']
    Y = [ord(i) for i in letter['Letter']]
    X = letter.drop(['Letter'], axis=1)
    T = preprocessing.MinMaxScaler().fit_transform(X)
    Xs = pd.DataFrame(T,columns = X.columns)
    Ts = train_test_split(Xs, Y, train_size = train_size)
    return Ts

def saveToNewDir(directory, filename):
    if not os.path.isdir(directory):
        os.makedirs(directory)
    plt.savefig(directory+filename, bbox_inches='tight')

def RunDecisionTreeClassifier(ImportedData, Data):
    Title = ImportedData.Title
    print ("---------- Working on DecisionTree for " + Title + " dataset ----------")
    timers = {}

    def CalculateBestHyperParameters(ImportedData, Data):
        params = {"max_depth": range(2,12,1), "min_samples_leaf": range(10,20,1)}
        
        bestLeaf = 0
        bestDepth = 0
        bestScore = 0
        atKey = 0
        timer = []
        for key, value in Data.items():
            start = time.time()
            tree_cv = RandomizedSearchCV(DecisionTreeClassifier(criterion='gini'), params, cv=5).fit(value["xTrain"], value["yTrain"]) 
            timer.append(time.time() - start)    
            if (tree_cv.best_score_ > bestScore):
                atKey = key
                bestScore = tree_cv.best_score_
                bestLeaf = tree_cv.best_params_['min_samples_leaf']
                bestDepth = tree_cv.best_params_['max_depth']
        
        dtc = DecisionTreeClassifier(criterion='gini', max_depth=bestDepth, min_samples_leaf=bestLeaf).fit(Data[atKey]["xTrain"], Data[atKey]["yTrain"])
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

        string = (Title + " Decision Tree calculated BEST parameters are {} Leaves, {} Tree Depth with {} score at {}% Train and {}% Test\nPredicting with {}% Train Accuracy, {}% Test Accuracy, {}% CV Accuracy, {}% Precision Score, {}% Recall Score and {}% \F-1 Score"
            .format(bestLeaf, bestDepth, round(bestScore, 2), atKey, 100-atKey, round(trs,1), round(tes,1), round(cvs,1), round(ps,1), round(rs,1), round(f1,1)))
        
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
            cat_counts = ["{0:0.0f}".format(value) for value in c_m.flatten()]
            lbls = np.asarray(cat_counts).reshape(26,26)
            sns.heatmap(c_m, annot=lbls, fmt='', cmap='coolwarm', square=True, robust=True, xticklabels = categories, yticklabels = categories)

        plt.xlabel('Predicted Value')
        plt.ylabel('Actual Value')
        plt.title(Title + ' (Decision Tree) Best Parameter Confusion Matrix')
        plt.legend()
        saveToNewDir(Title+"/", Title + "_DecisionTree_BestConfusionMatrix.png")
        plt.gcf().clear()
        plt.clf()  

        return bestLeaf, bestDepth, atKey
    
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
        for key, value in Data.items():
            start = time.time()
            dtc = DecisionTreeClassifier(criterion='gini', max_depth=bestDepth, min_samples_leaf=bestLeaf).fit(value["xTrain"], value["yTrain"])
            yPred = dtc.predict(value["xTest"])
            cvs = cross_val_score(dtc, value["xTrain"], value["yTrain"]).mean()*100
            trs = accuracy_score(value["yTrain"], dtc.predict(value["xTrain"]))*100
            tes = accuracy_score(value["yTest"], yPred)*100
            if (Title=="BanknoteAuthentication"):
                ps = precision_score(value["yTest"], yPred, average='binary')*100
                rs = recall_score(value["yTest"], yPred, average='binary')*100
                f1 = f1_score(value["yTest"], yPred, average='binary')*100
            else:
                ps = precision_score(value["yTest"], yPred, average='weighted')*100
                rs = recall_score(value["yTest"], yPred, average='weighted')*100
                f1 = f1_score(value["yTest"], yPred, average='weighted')*100

            cv_scores.append(cvs)
            train_scores.append(trs)
            test_scores.append(tes)
            p_s.append(ps)
            r_s.append(rs)
            f_1.append(f1)

            #print("Accuracy {}% Train, {}% Test, {}% CV, {} F+, {} T+, {} AUC at {}% Train and {}% Test".format(round(trs,0), round(tes,0), round(cvs,0), round(fpr[1],3), round(tpr[1],3), round(auc_score,0), key, 100-key))
            #print("Accuracy {}% Train, {}% Test, {}% CV at {}% Train and {}% Test".format(round(trs,0), round(tes,0), round(cvs,0), key, 100-key))

            timer.append(time.time() - start)

        timers['Training'] = timer
        plt.xticks(list(Data.keys()))
        plt.plot(Data.keys(), train_scores, label='Train', lw = 2)
        plt.plot(Data.keys(), test_scores, label='Test', lw = 2)
        plt.plot(Data.keys(), cv_scores, label='CV', lw = 2)
        plt.plot(Data.keys(), p_s, label='Precision', lw = 2)
        plt.plot(Data.keys(), r_s, label='Recall', lw = 2)
        plt.plot(Data.keys(), f_1, label='F1', lw = 2)

        plt.xlabel('Training Set Size ')
        plt.ylabel('Accuracy')
        plt.title(Title + ' (Decision Tree) Accuracy Vs. Training Set Size')
        plt.legend()
        saveToNewDir(Title + "/", Title + "_DecisionTree_TrainingSetSize.png")
        plt.gcf().clear()
        plt.clf()  

    # Test different tree depth
    def TestDepth(ImportedData, Data, bestKey):
        plt.figure(figsize=(7,7), dpi=70)
        train_scores = []
        test_scores = []
        cv_scores = []
        timer = []
        p_s = []
        r_s = []
        f_1 = []
        value = Data[bestKey]
        tRange = range(2,12,1)
        for depth in tRange:            
            start = time.time()
            dtc = DecisionTreeClassifier(criterion='gini', max_depth = depth, min_samples_leaf=bestLeaf).fit(value["xTrain"], value["yTrain"])
            yPred = dtc.predict(value["xTest"])
            cvs = cross_val_score(dtc, value["xTrain"], value["yTrain"]).mean()*100
            trs = accuracy_score(value["yTrain"], dtc.predict(value["xTrain"]))*100
            tes = accuracy_score(value["yTest"], yPred)*100
            if (Title=="BanknoteAuthentication"):
                ps = precision_score(value["yTest"], yPred, average='binary')*100
                rs = recall_score(value["yTest"], yPred, average='binary')*100
                f1 = f1_score(value["yTest"], yPred, average='binary')*100
            else:
                ps = precision_score(value["yTest"], yPred, average='weighted')*100
                rs = recall_score(value["yTest"], yPred, average='weighted')*100
                f1 = f1_score(value["yTest"], yPred, average='weighted')*100

            cv_scores.append(cvs)
            train_scores.append(trs)
            test_scores.append(tes)
            p_s.append(ps)
            r_s.append(rs)
            f_1.append(f1)

            timer.append(time.time() - start)

        timers['Depth'] = timer
        plt.xticks(tRange)
        plt.plot(tRange, train_scores, label='Train', lw = 2)
        plt.plot(tRange, test_scores, label='Test', lw = 2)
        plt.plot(tRange, cv_scores, label='CV', lw = 2)
        plt.plot(tRange, p_s, label='Precision', lw = 2)
        plt.plot(tRange, r_s, label='Recall', lw = 2)
        plt.plot(tRange, f_1, label='F1', lw = 2)

        plt.xlabel('Tree Depth ')
        plt.ylabel('Accuracy')
        plt.title(Title + ' (DecisionTree) Accuracy Vs. Tree Depth')
        plt.legend()
        saveToNewDir(Title+"/", Title + "_DecisionTree_TreeDepth.png")
        plt.gcf().clear()
        plt.clf() 

    # Test different leaf node sizes
    def TestLeaves(ImportedData, Data, bestKey):
        plt.figure(figsize=(7,7), dpi=70)
        train_scores = []
        test_scores = []
        cv_scores = []
        timer = []
        p_s = []
        r_s = []
        f_1 = []
        value = Data[bestKey]
        tRange = range(10,20,1)
        for leaf_size in tRange:            
            start = time.time()
            dtc = DecisionTreeClassifier(criterion='gini', max_leaf_nodes=leaf_size, max_depth=bestDepth).fit(value["xTrain"], value["yTrain"])
            yPred = dtc.predict(value["xTest"])
            cvs = cross_val_score(dtc, value["xTrain"], value["yTrain"]).mean()*100
            trs = accuracy_score(value["yTrain"], dtc.predict(value["xTrain"]))*100
            tes = accuracy_score(value["yTest"], yPred)*100
            c_m = confusion_matrix(value["yTest"], yPred)
            if (Title=="BanknoteAuthentication"):
                ps = precision_score(value["yTest"], yPred, average='binary')*100
                rs = recall_score(value["yTest"], yPred, average='binary')*100
                f1 = f1_score(value["yTest"], yPred, average='binary')*100
            else:
                ps = precision_score(value["yTest"], yPred, average='weighted')*100
                rs = recall_score(value["yTest"], yPred, average='weighted')*100
                f1 = f1_score(value["yTest"], yPred, average='weighted')*100
            #print ("{}ps {}rs {}f1".format(ps, rs, f1))

            cv_scores.append(cvs)
            train_scores.append(trs)
            test_scores.append(tes)
            p_s.append(ps)
            r_s.append(rs)
            f_1.append(f1)

            #print("Accuracy {}% Train, {}% Test, {}% CV, {} F+, {} T+, {} AUC at {}% Train and {}% Test".format(round(trs,0), round(tes,0), round(cvs,0), round(fpr[1],3), round(tpr[1],3), round(auc_score,0), key, 100-key))
            #print("Accuracy {}% Train, {}% Test, {}% CV at {} Leaves".format(round(trs,0), round(tes,0), round(cvs,0), leaf_size))
            timer.append(time.time() - start)

        timers['Leaf'] = timer
        plt.xticks(tRange)
        plt.plot(tRange, train_scores, label='Train', lw = 2)
        plt.plot(tRange, test_scores, label='Test', lw = 2)
        plt.plot(tRange, cv_scores, label='CV', lw = 2)
        plt.plot(tRange, p_s, label='Precision', lw = 2)
        plt.plot(tRange, r_s, label='Recall', lw = 2)
        plt.plot(tRange, f_1, label='F1', lw = 2)

        plt.xlabel('Tree Leaf Size ')
        plt.ylabel('Accuracy')
        plt.title(Title + ' (DecisionTree) Accuracy Vs. Tree Leaf size')
        plt.legend()
        saveToNewDir(Title+"/", Title + "_DecisionTree_LeafSize.png")
        plt.gcf().clear()
        plt.clf()        
    
    # Test different sizes of Decision Tree forests.
    def TestForest(ImportedData,Data, bestKey):
        plt.figure(figsize=(7,7), dpi=70)
        train_scores = []
        test_scores = []
        cv_scores = []
        timer = []
        p_s = []
        r_s = []
        f_1 = []
        value = Data[bestKey]
        xRange = range(2,12,1)
        for treeNum in xRange:   
            start = time.time()
            rfc = RandomForestClassifier(criterion='gini', n_estimators=int(treeNum), max_depth=bestDepth, min_samples_leaf=bestLeaf).fit(value["xTrain"], value["yTrain"])
            yPred = rfc.predict(value["xTest"])
            cvVector = []
            trainVector = []
            testVector = []
            cmVector = []
            psVector = []
            rsVector = []
            f1Vector = []
            for i in range(10):
                cvVector.append(cross_val_score(rfc, value["xTrain"], value["yTrain"], cv=5).mean()*100)
                trainVector.append(accuracy_score(value["yTrain"], rfc.predict(value["xTrain"]))*100)
                testVector.append(accuracy_score(value["yTest"], yPred)*100)
                if (Title=="BanknoteAuthentication"):
                    psVector.append(precision_score(value["yTest"], yPred, average='binary')*100)
                    rsVector.append(recall_score(value["yTest"], yPred, average='binary')*100)
                    f1Vector.append(f1_score(value["yTest"], yPred, average='binary')*100)
                else:
                    psVector.append(precision_score(value["yTest"], yPred, average='weighted')*100)
                    rsVector.append(recall_score(value["yTest"], yPred, average='weighted')*100)
                    f1Vector.append(f1_score(value["yTest"], yPred, average='weighted')*100)

            cvs = sum(cvVector)/len(cvVector)
            trs = sum(trainVector)/len(trainVector)
            tes = sum(testVector)/len(testVector)
            ps = sum(psVector)/len(psVector)
            rs = sum(rsVector)/len(rsVector)
            f1 = sum(f1Vector)/len(f1Vector)

            cv_scores.append(cvs)
            train_scores.append(trs)
            test_scores.append(tes)
            p_s.append(ps)
            r_s.append(rs)
            f_1.append(f1)

            timer.append(time.time() - start)

        timers['Estimators'] = timer
        plt.xticks(xRange)
        plt.plot(xRange, train_scores, label='Train', lw = 2)
        plt.plot(xRange, test_scores, label='Test', lw = 2)
        plt.plot(xRange, cv_scores, label='CV', lw = 2)
        plt.plot(xRange, p_s, label='Precision', lw = 2)
        plt.plot(xRange, r_s, label='Recall', lw = 2)
        plt.plot(xRange, f_1, label='F1', lw = 2)

        plt.xlabel('Number of Estimators')
        plt.ylabel('Accuracy')
        plt.title(Title + ' (Random Forest) Accuracy Vs. Forest Number of Estimators')
        plt.legend()
        saveToNewDir(Title+"/", Title + "_DecisionTree_RandomForest.png")
        plt.gcf().clear()
        plt.clf()  
    
    bestLeaf, bestDepth, bestKey = CalculateBestHyperParameters(ImportedData, Data)
    TestTrain(ImportedData, Data, bestKey)    
    TestDepth(ImportedData, Data, bestKey)    
    TestLeaves(ImportedData, Data, bestKey)    
    TestForest(ImportedData, Data, bestKey)
    
    #Timers Plot
    plt.figure(figsize=(7,7), dpi=70)
    plt.xlabel('Index')
    plt.ylabel('Time')
    plt.xticks(range(1,11,1))
    plt.title(Title + ' (Random Forest) Training Time Vs. Index')
    #plt.plot([x/8 for x in list(Data.keys())], timers['Training'], label='Training Set Size', lw = 2)
    plt.plot(timers['Training'], label='Training Set Size', lw = 2)
    plt.plot(timers['Depth'], label='Tree Depth Size', lw = 2)
    plt.plot(timers['Leaf'], label='Tree Leaf Size', lw = 2)
    plt.plot(timers['Estimators'], label='Tree Leaf Size', lw = 2)
    plt.legend()
    saveToNewDir(Title+"/", Title + "_DecisionTree_Timers.png")
    plt.gcf().clear()
    plt.clf()  
    print ("Done with DecisionTree for " + Title + " dataset")

def RunAdaBoostClassifier(ImportedData, Data):
    Title = ImportedData.Title
    print ("---------- Working on AdaBoost for " + Title + " dataset ----------")
    timers = {}
    
    def CalculateBestHyperParameters(ImportedData, Data):
        params = {"n_estimators": range(10,110,10), "learning_rate": np.geomspace(0.0001, 0.1, num = 10)}
        
        bestEstimators = 0
        bestRate = 0
        bestScore = 0
        atKey = 0
        for key, value in Data.items():
            tree_cv = RandomizedSearchCV(AdaBoostClassifier(DecisionTreeClassifier(criterion = "gini")), params).fit(value["xTrain"], value["yTrain"])
            if (tree_cv.best_score_ > bestScore):
                atKey = key
                bestScore = tree_cv.best_score_
                bestEstimators = tree_cv.best_params_['n_estimators']
                bestRate = tree_cv.best_params_['learning_rate']

        dtc = AdaBoostClassifier(DecisionTreeClassifier(criterion = "gini"),n_estimators=bestEstimators, learning_rate=bestRate).fit(Data[atKey]["xTrain"], Data[atKey]["yTrain"])
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

        string = (Title + " AdaBoost calculated BEST parameters are {} Estimator, {} Learning Rate with {} score at {}% Train and {}% Test\nPredicting with {}% Train Accuracy, {}% Test Accuracy, {}% CV Accuracy, {}% Precision Score, {}% Recall Score and {}% \F-1 Score"
            .format(bestEstimators, bestRate, round(bestScore, 2), atKey, 100-atKey, round(trs,1), round(tes,1), round(cvs,1), round(ps,1), round(rs,1), round(f1,1)))
        
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
        plt.title(Title + ' (AdaBoost) Best Parameter Confusion Matrix')
        plt.legend()
        saveToNewDir(Title + "/", Title + "_AdaBoost_BestConfusionMatrix.png")
        plt.gcf().clear()
        plt.clf() 

        return bestEstimators, bestRate, atKey
    
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
        for key, value in Data.items():            
            start = time.time()
            abc = AdaBoostClassifier(DecisionTreeClassifier(criterion = "gini"), n_estimators = bestEstimators, learning_rate=bestRate).fit(value["xTrain"], value["yTrain"])
            yPred = abc.predict(value["xTest"])
            cv_scores.append(cross_val_score(abc, value["xTrain"], value["yTrain"]).mean()*100)
            train_scores.append(accuracy_score(value["yTrain"], abc.predict(value["xTrain"]))*100)
            test_scores.append(accuracy_score(value["yTest"], yPred)*100)
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
            timer.append(time.time() - start)
            
        timers['Training'] = timer
        plt.xticks(list(Data.keys()))
        plt.plot(Data.keys(), train_scores, label='Train', lw = 2)
        plt.plot(Data.keys(), test_scores, label='Test', lw = 2)
        plt.plot(Data.keys(), cv_scores, label='CV', lw = 2)
        plt.plot(Data.keys(), p_s, label='Precision', lw = 2)
        plt.plot(Data.keys(), r_s, label='Recall', lw = 2)
        plt.plot(Data.keys(), f_1, label='F1', lw = 2)

        plt.xlabel('Training Set Size ')
        plt.ylabel('Accuracy')
        plt.title(Title + ' (AdaBoost) Accuracy Vs. Training Set Size')
        plt.legend()
        saveToNewDir(Title+"/", Title + "_AdaBoost_TrainingSetSize.png")
        plt.gcf().clear()
        plt.clf()  

    # Test different tree depth
    def TestDepth(ImportedData, Data, bestKey):
        plt.figure(figsize=(7,7), dpi=70)
        train_scores = []
        test_scores = []
        cv_scores = []
        timer = []
        p_s = []
        r_s = []
        f_1 = []        
        value = Data[bestKey]
        tRange = range(2,12,1)
        for depth in tRange:            
            start = time.time()
            abc = AdaBoostClassifier(DecisionTreeClassifier(criterion = "gini", max_depth=depth), n_estimators = bestEstimators, learning_rate=bestRate).fit(value["xTrain"], value["yTrain"])
            yPred = abc.predict(value["xTest"])
            cv_scores.append(cross_val_score(abc, value["xTrain"], value["yTrain"]).mean()*100)
            train_scores.append(accuracy_score(value["yTrain"], abc.predict(value["xTrain"]))*100)
            test_scores.append(accuracy_score(value["yTest"], yPred)*100)
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
            timer.append(time.time() - start)

        timers['Depth'] = timer
        plt.xticks(tRange)
        plt.plot(tRange, train_scores, label='Train', lw = 2)
        plt.plot(tRange, test_scores, label='Test', lw = 2)
        plt.plot(tRange, cv_scores, label='CV', lw = 2)
        plt.plot(tRange, p_s, label='Precision', lw = 2)
        plt.plot(tRange, r_s, label='Recall', lw = 2)
        plt.plot(tRange, f_1, label='F1', lw = 2)

        plt.xlabel('Tree Depth ')
        plt.ylabel('Accuracy')
        plt.title(Title + ' (AdaBoost) Accuracy Vs. Tree Depth')
        plt.legend()
        saveToNewDir(Title+"/", Title + "_AdaBoost_TreeDepth.png")
        plt.gcf().clear()
        plt.clf() 
    
    # Test different leaf node sizes
    def TestLeaves(ImportedData, Data, bestKey):
        plt.figure(figsize=(7,7), dpi=70)
        train_scores = []
        test_scores = []
        cv_scores = []
        timer = []
        p_s = []
        r_s = []
        f_1 = []
        value = Data[bestKey]
        tRange = range(2,12,1)
        for leaf_size in tRange:            
            start = time.time()
            abc = AdaBoostClassifier(DecisionTreeClassifier(criterion = "gini", min_samples_leaf=leaf_size), n_estimators = bestEstimators, learning_rate=bestRate).fit(value["xTrain"], value["yTrain"])
            yPred = abc.predict(value["xTest"])
            cv_scores.append(cross_val_score(abc, value["xTrain"], value["yTrain"]).mean()*100)
            train_scores.append(accuracy_score(value["yTrain"], abc.predict(value["xTrain"]))*100)
            test_scores.append(accuracy_score(value["yTest"], yPred)*100)
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
            timer.append(time.time() - start)

        timers['Leaf'] = timer
        plt.xticks(tRange)
        plt.plot(tRange, train_scores, label='Train', lw = 2)
        plt.plot(tRange, test_scores, label='Test', lw = 2)
        plt.plot(tRange, cv_scores, label='CV', lw = 2)
        plt.plot(tRange, p_s, label='Precision', lw = 2)
        plt.plot(tRange, r_s, label='Recall', lw = 2)
        plt.plot(tRange, f_1, label='F1', lw = 2)

        plt.xlabel('Tree Leaf Size ')
        plt.ylabel('Accuracy')
        plt.title(Title + ' (AdaBoost) Accuracy Vs. Tree Leaf size')
        plt.legend()
        saveToNewDir(Title+"/", Title + "_AdaBoost_LeafSize.png")
        plt.gcf().clear()
        plt.clf()   

    # Test different AdaBoost estimator sizes
    def TestEstimators(ImportedData, Data, bestKey):
        plt.figure(figsize=(7,7), dpi=70)
        train_scores = []
        test_scores = []
        cv_scores = []
        timer = []
        p_s = []
        r_s = []
        f_1 = []
        value = Data[bestKey]
        tRange = range(10,110,10)
        
        for estimator in tRange:            
            start = time.time()
            abc = AdaBoostClassifier(DecisionTreeClassifier(criterion = "gini"), n_estimators = estimator).fit(value["xTrain"], value["yTrain"])
            yPred = abc.predict(value["xTest"])
            cv_scores.append(cross_val_score(abc, value["xTrain"], value["yTrain"]).mean()*100)
            train_scores.append(accuracy_score(value["yTrain"], abc.predict(value["xTrain"]))*100)
            test_scores.append(accuracy_score(value["yTest"], yPred)*100)
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
            timer.append(time.time() - start)

        timers['Estimator'] = timer
        plt.xticks(list(tRange))
        plt.plot(tRange, train_scores, label='Train', lw = 2)
        plt.plot(tRange, test_scores, label='Test', lw = 2)
        plt.plot(tRange, cv_scores, label='CV', lw = 2)
        plt.plot(tRange, p_s, label='Precision', lw = 2)
        plt.plot(tRange, r_s, label='Recall', lw = 2)
        plt.plot(tRange, f_1, label='F1', lw = 2)

        plt.xlabel('Estimator #')
        plt.ylabel('Accuracy')
        plt.title(Title + ' (AdaBoost) Accuracy Vs. Estimator #')
        plt.legend()
        saveToNewDir(Title+"/", Title + "_AdaBoost_Estimator.png")
        plt.gcf().clear()
        plt.clf()  

    # Test different AdaBoost Learning Rates
    def TestLearningRate(ImportedData, Data, bestKey):
        plt.figure(figsize=(7,7), dpi=70)
        train_scores = []
        test_scores = []
        cv_scores = []
        timer = []
        p_s = []
        r_s = []
        f_1 = []
        value = Data[bestKey]
        tRange = np.geomspace(0.0001, 0.1, num = 10) # arange(0.1, 2.0, 0.2)
        
        for rate in tRange:            
            start = time.time()
            abc = AdaBoostClassifier(DecisionTreeClassifier(criterion = "gini"), n_estimators = bestEstimators, learning_rate= rate).fit(value["xTrain"], value["yTrain"])
            yPred = abc.predict(value["xTest"])
            cv_scores.append(cross_val_score(abc, value["xTrain"], value["yTrain"]).mean()*100)
            train_scores.append(accuracy_score(value["yTrain"], abc.predict(value["xTrain"]))*100)
            test_scores.append(accuracy_score(value["yTest"], yPred)*100)
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
            timer.append(time.time() - start)

        timers['LearningRate'] = timer
        plt.xticks(list(tRange))
        plt.plot(tRange, train_scores, label='Train', lw = 2)
        plt.plot(tRange, test_scores, label='Test', lw = 2)
        plt.plot(tRange, cv_scores, label='CV', lw = 2)
        plt.plot(tRange, p_s, label='Precision', lw = 2)
        plt.plot(tRange, r_s, label='Recall', lw = 2)
        plt.plot(tRange, f_1, label='F1', lw = 2)

        plt.xlabel('Learning Rate')
        plt.ylabel('Accuracy')
        plt.title(Title + ' (AdaBoost) Accuracy Vs. LearningRate')
        plt.legend()
        saveToNewDir(Title+"/", Title + "_AdaBoost_LearningRate.png")
        plt.gcf().clear()
        plt.clf()    
    
    bestEstimators, bestRate, bestKey = CalculateBestHyperParameters(ImportedData, Data)
    TestTrain(ImportedData, Data, bestKey)
    TestDepth(ImportedData, Data, bestKey)
    TestLeaves(ImportedData, Data, bestKey)
    TestEstimators(ImportedData, Data, bestKey)
    TestLearningRate(ImportedData, Data, bestKey)

    #Timers Plot
    plt.figure(figsize=(7,7), dpi=70)
    plt.xlabel('Index')
    plt.ylabel('Time')
    plt.xticks(range(1,11,1))
    plt.title(Title + ' (AdaBoost) Training Time Vs. Index')
    #plt.plot([x/5 for x in list(Data.keys())],timers['Training'], label='Training Set Size', lw = 2)
    plt.plot(timers['Training'], label='Training Set Size', lw = 2)
    plt.plot(timers['Depth'], label='Tree Depth Size', lw = 2)
    plt.plot(timers['Leaf'], label='Tree Leaf Size', lw = 2)
    plt.plot(timers['Estimator'], label='Estimator Size', lw = 2)
    plt.plot(timers['LearningRate'], label='Learning Rate', lw = 2)
    plt.legend()
    saveToNewDir(Title + "/", Title + "_AdaBoost_Timers.png")
    plt.gcf().clear()
    plt.clf() 
    print ("Done with AdaBoost for " + Title + " dataset")

def RunKNNClassifier(ImportedData, Data):
    Title = ImportedData.Title
    print ("---------- Working on KNN for " + Title + " dataset ----------")
    timers = {}
        
    def CalculateBestHyperParameters(ImportedData, Data):
        params = {"n_neighbors": range(1, 11, 1), "metric": ["cosine", "manhattan", "euclidean", "minkowski", "hamming", "canberra"]}
        
        bestNeighboars = 0
        bestMetric = 0
        bestScore = 0
        atKey = 0
        for key, value in Data.items():
            tree_cv = RandomizedSearchCV(KNeighborsClassifier(), params).fit(value["xTrain"], value["yTrain"])
            if (tree_cv.best_score_ > bestScore):
                atKey = key
                bestScore = tree_cv.best_score_
                bestNeighboars = tree_cv.best_params_['n_neighbors']
                bestMetric = tree_cv.best_params_['metric']

        dtc = KNeighborsClassifier(n_neighbors=bestNeighboars, metric=bestMetric).fit(Data[atKey]["xTrain"], Data[atKey]["yTrain"])
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

        string = (Title + " KNN calculated BEST parameters are {} Metric, {} Neighbors with {} score at {}% Train and {}% Test\nPredicting with {}% Train Accuracy, {}% Test Accuracy, {}% CV Accuracy, {}% Precision Score, {}% Recall Score and {}% \F-1 Score"
            .format(bestMetric, bestNeighboars, round(bestScore, 2), atKey, 100-atKey, round(trs,1), round(tes,1), round(cvs,1), round(ps,1), round(rs,1), round(f1,1)))
        
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
        plt.title(Title + ' (KNN) Best Parameter Confusion Matrix')
        plt.legend()
        saveToNewDir(Title+"/", Title + "_KNN_BestConfusionMatrix.png")
        plt.gcf().clear()
        plt.clf() 

        return bestNeighboars, bestMetric, atKey
    
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
            knns = KNeighborsClassifier(n_neighbors=bestNeighboars, metric=bestMetric).fit(value['xTrain'], value['yTrain'])
            yPred = knns.predict(value["xTest"])
            train_scores.append(accuracy_score(value['yTrain'], knns.predict(value['xTrain']))*100)
            cv_scores.append(cross_val_score(knns, value['xTrain'], value['yTrain'], cv=5).mean()*100)
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
        plt.title(Title + " (KNN) Accuracy with train size")
        saveToNewDir(Title+"/", Title + "_KNN_TrainingSetSize.png")
        plt.clf()

    # Test different neighbors
    def TestNeighbors(ImportedData, Data, bestKey):
        plt.figure(figsize=(7,7), dpi=70)

        train_scores = []
        test_scores = []
        cv_scores = []
        timer = []
        p_s = []
        r_s = []
        f_1 = []
        value = Data[bestKey]

        tRange = range(1, 11, 1)
        for k in tRange:
            start = time.time()
            knns = KNeighborsClassifier(n_neighbors=k, metric=bestMetric).fit(value['xTrain'], value['yTrain'])
            yPred = knns.predict(value["xTest"])
            test_scores.append(accuracy_score(value['yTest'],yPred)*100)
            cv_scores.append(cross_val_score(knns, value['xTrain'], value['yTrain'], cv=5).mean()*100)
            train_scores.append(accuracy_score(value['yTrain'],knns.predict(value['xTrain']))*100)
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

        timers["Neighbors"] = timer
        plt.plot(tRange, train_scores, label="Train", lw = 2)
        plt.plot(tRange, test_scores, label="Test", lw = 2)
        plt.plot(tRange, cv_scores, label="CV", lw = 2)
        plt.plot(tRange, p_s, label='Precision', lw = 2)
        plt.plot(tRange, r_s, label='Recall', lw = 2)
        plt.plot(tRange, f_1, label='F1', lw = 2)
        plt.xlabel("K neighbours")
        plt.ylabel("Accuracy")
        plt.legend(loc='best')
        plt.title(Title + " (KNN) Accuracy with k")
        saveToNewDir(Title+"/", Title + "_KNN_NeighborSize.png")

        plt.clf()
    
    # Test different metrics
    def TestMetrics(ImportedData, Data, bestKey):
        plt.figure(figsize=(7,7), dpi=70)

        metrics = ["cosine", "manhattan", "euclidean", "minkowski", "hamming", "canberra"]
        train_scores = []
        test_scores = []
        cv_scores = []
        timer = []
        value = Data[bestKey]

        for metric in metrics:        
            start = time.time()
            knns = KNeighborsClassifier(n_neighbors=bestNeighboars, metric=metric).fit(value['xTrain'], value['yTrain'])
            yPred = knns.predict(value["xTest"])
            test_scores.append(accuracy_score(value['yTest'],yPred)*100)
            train_scores.append(accuracy_score(value['yTrain'],knns.predict(value['xTrain']))*100)
            cv_scores.append(cross_val_score(knns, value['xTrain'], value['yTrain'], cv=5).mean()*100)
            timer.append(time.time()-start)

        timers["Metrics"] = timer

        df = pd.DataFrame({'Train': train_scores,
                        'Test': test_scores,
                        'CV': cv_scores}, index=metrics)
        df.plot.bar(rot=0)
        plt.xlabel("K neighbours")
        plt.ylabel("Accuracy")
        plt.legend(loc='best')
        plt.title(Title + " (KNN) Accuracy with metrics")
        saveToNewDir(Title+"/", Title + "_KNN_Metrics.png")

        plt.clf()  
        
        plt.bar(metrics, timer, 1/1.5, label="Training time")
        plt.xlabel("Metrics")
        plt.ylabel("Time (s)")
        plt.title(Title + " (KNN) Training time with kernel")
        saveToNewDir(Title+"/", Title + "_KNN_MetricsTimer.png")

        plt.clf()
        
    bestNeighboars, bestMetric, bestKey = CalculateBestHyperParameters(ImportedData, Data)
    TestTrain(ImportedData, Data, bestKey)
    TestNeighbors(ImportedData, Data, bestKey)
    TestMetrics(ImportedData, Data, bestKey)
    
    #Timers Plot
    plt.figure(figsize=(7,7), dpi=70)
    plt.xlabel('Index')
    plt.ylabel('Time')
    plt.xticks(range(1,11,1))
    plt.title(Title + ' (KNN) Training Time Vs. Index')
    #plt.plot([x/5 for x in list(Data.keys())],timers['Training'], label='Training Set Size', lw = 2)
    plt.plot(timers['Training'], label='Training Set Size', lw = 2)
    plt.plot(timers['Neighbors'], label='Neighbors', lw = 2)
    plt.plot(timers['Metrics'], label='Metrics', lw = 2)
    plt.legend()
    saveToNewDir(Title+"/", Title + "_KNN_Timers.png")
    plt.gcf().clear()
    plt.clf() 
    print ("Done with KNN for " + Title + " dataset")

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
    
    # Test different layers
    def TestLayers(ImportedData, Data, bestKey):
        plt.figure(figsize=(7,7), dpi=70)

        train_scores = []
        test_scores = []
        cv_scores = []
        timer = []
        p_s = []
        r_s = []
        f_1 = []
        value = Data[bestKey]
        tRange = range(2,12,1)
        layer_sizes = ()
        for ix in tRange:
            layer_sizes += (10, )          
            start = time.time()
            mlp = MLPClassifier(hidden_layer_sizes=layer_sizes, alpha=bestAlpha).fit(value['xTrain'], value['yTrain'])
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
        
        timers["Layers"] = timer
        plt.plot(tRange, train_scores, label="Train", lw = 2)
        plt.plot(tRange, test_scores, label="Test", lw = 2)
        plt.plot(tRange, cv_scores, label="CV", lw = 2)
        plt.plot(tRange, p_s, label='Precision', lw = 2)
        plt.plot(tRange, r_s, label='Recall', lw = 2)
        plt.plot(tRange, f_1, label='F1', lw = 2)
        plt.xlabel("Layers #")
        plt.ylabel("Accuracy (%)")
        plt.legend(loc='best')
        plt.title(Title + " (NeuralNet) Accuracy with # of layers")
        saveToNewDir(Title+"/", Title + "_NeuralNet_TrainingLayerNumber.png")
        plt.clf()
    
    # Test different neurons
    def TestNeurons(ImportedData, Data, bestKey):
        plt.figure(figsize=(7,7), dpi=70)

        train_scores = []
        test_scores = []
        cv_scores = []
        timer = []
        p_s = []
        r_s = []
        f_1 = []
        value = Data[bestKey]
        tRange = range(10,110,10)
        for ix in tRange:
            i = int(ix)
            layer_sizes = (i, i, i)   
            start = time.time()
            mlp = MLPClassifier(hidden_layer_sizes=layer_sizes, alpha=bestAlpha).fit(value['xTrain'], value['yTrain'])
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
        
        timers["Neurons"] = timer
        plt.plot(tRange, train_scores, label="Train", lw = 2)
        plt.plot(tRange, test_scores, label="Test", lw = 2)
        plt.plot(tRange, cv_scores, label="CV", lw = 2)
        plt.plot(tRange, p_s, label='Precision', lw = 2)
        plt.plot(tRange, r_s, label='Recall', lw = 2)
        plt.plot(tRange, f_1, label='F1', lw = 2)
        plt.xlabel("Neurons #")
        plt.ylabel("Accuracy (%)")
        plt.legend(loc='best')
        plt.title(Title + " (NeuralNet) Accuracy with # of neurons")
        saveToNewDir(Title+"/", Title + "_NeuralNet_TrainingNeuronsNumber.png")
        plt.clf()
    
    # Test different Alpha Values
    def TestAlpha(ImportedData, Data, bestKey):
        plt.figure(figsize=(7,7), dpi=70)
        train_scores = []
        test_scores = []
        cv_scores = []
        timer = []
        p_s = []
        r_s = []
        f_1 = []
        value = Data[bestKey]
        tRange = np.geomspace(0.0001,0.1,num=10)
        
        for alpha in tRange:            
            start = time.time()
            mlp = MLPClassifier(hidden_layer_sizes=bestLayers, alpha=alpha).fit(value['xTrain'], value['yTrain'])
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

        timers['Alpha'] = timer
        plt.xticks(list(tRange))
        plt.plot(tRange, train_scores, label='Train', lw = 2)
        plt.plot(tRange, test_scores, label='Test', lw = 2)
        plt.plot(tRange, cv_scores, label='CV', lw = 2)
        plt.plot(tRange, p_s, label='Precision', lw = 2)
        plt.plot(tRange, r_s, label='Recall', lw = 2)
        plt.plot(tRange, f_1, label='F1', lw = 2)

        plt.xlabel('Alpha')
        plt.ylabel('Accuracy')
        plt.title(Title + ' (NeuralNet) Accuracy Vs. Alpha')
        plt.legend()
        saveToNewDir(Title+"/", Title + "_NeuralNet_Alpha.png")
        plt.gcf().clear()
        plt.clf()    
    
    bestLayers, bestAlpha, bestKey = CalculateBestHyperParameters(ImportedData, Data)
    TestTrain(ImportedData, Data, bestKey)
    TestLayers(ImportedData, Data, bestKey)
    TestNeurons(ImportedData, Data, bestKey)
    TestAlpha(ImportedData, Data, bestKey)

    #Timers Plot
    plt.figure(figsize=(7,7), dpi=70)
    plt.xlabel('Index')
    plt.ylabel('Time')
    plt.xticks(range(1,11,1))
    plt.title(Title + ' (NeuralNet) Training Time Vs. Index')
    #plt.plot([x/5 for x in list(Data.keys())],timers['Training'], label='Training Set Size', lw = 2)
    plt.plot(timers['Training'], label='Training Set Size', lw = 2)
    plt.plot(timers['Layers'], label='Neural Net Layers #', lw = 2)
    plt.plot(timers['Neurons'], label='Neaural Net Neurons #', lw = 2)
    plt.plot(timers['Alpha'], label='Neaural Net Neurons #', lw = 2)
    plt.legend()
    saveToNewDir(Title+"/", Title + "_NeuralNet_Timers.png")
    plt.gcf().clear()
    plt.clf() 
    print ("Done with NeuralNets for " + Title + " datasets")

def RunSVMClassifier(ImportedData, Data):
    Title = ImportedData.Title
    print ("---------- Working on SVM for " + Title + " dataset ----------")
    timers = {}
    
    def CalculateBestHyperParameters(ImportedData, Data):
        params =    {'kernel': ['linear','poly','rbf','sigmoid'],
                    'C': np.geomspace(0.1,1000,num=10),
                    'gamma': np.geomspace(0.1,1000,num=10)}        
        bestKernel = ''
        bestC = 0
        bestGamma = 0
        bestScore = 0
        atKey = 0
        for key, value in Data.items():
            tree_cv = RandomizedSearchCV(svm.SVC(), params).fit(value["xTrain"], value["yTrain"])
            if (tree_cv.best_score_ > bestScore):
                atKey = key
                bestScore = tree_cv.best_score_
                bestKernel = tree_cv.best_params_['kernel']
                bestC = tree_cv.best_params_['C']
                bestGamma = tree_cv.best_params_['gamma']
        
        dtc = svm.SVC(kernel=bestKernel, C=bestC, gamma=bestGamma).fit(Data[atKey]["xTrain"], Data[atKey]["yTrain"])
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

        string = (Title + " SVM calculated BEST parameters are {} Kernel, {} C value, {} Gamma with {} score at {}% Train and {}% Test\nPredicting with {}% Train Accuracy, {}% Test Accuracy, {}% CV Accuracy, {}% Precision Score, {}% Recall Score and {}% \F-1 Score"
            .format(bestKernel, bestC, bestGamma, round(bestScore, 2), atKey, 100-atKey, round(trs,1), round(tes,1), round(cvs,1), round(ps,1), round(rs,1), round(f1,1)))
        
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
        plt.title(Title + ' (SVM) Best Parameter Confusion Matrix')
        plt.legend()
        saveToNewDir(Title+"/", Title + "_SVM_BestConfusionMatrix.png")
        plt.gcf().clear()
        plt.clf() 

        return bestKernel, bestC, bestGamma, atKey
    
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

        for key, value in Data.items():    
            start = time.time()
            svmsvc = svm.SVC(kernel='linear').fit(value['xTrain'], value['yTrain'])
            yPred = svmsvc.predict(value["xTest"])
            train_scores.append(accuracy_score(value['yTrain'], svmsvc.predict(value['xTrain']))*100)
            cv_scores.append(cross_val_score(svmsvc, value['xTrain'], value['yTrain'], cv=5).mean()*100)
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
        plt.title(Title + " (SVM) Accuracy with train size")
        saveToNewDir(Title+"/", Title + "_SVM_TrainingSetSize.png")
        plt.clf()    

    # Test different Kernels
    def TestKernel(ImportedData, Data, bestKey):
        plt.figure(figsize=(7,7), dpi=70)

        kernels = ['linear','poly','rbf','sigmoid'] 

        graph = []
        train_scores = []
        test_scores = []
        cv_scores = []
        roc_auc = []
        timer = []
        value = Data[bestKey]

        for kernel in kernels:   
            start = time.time()
            svmsvc = svm.SVC(kernel=kernel).fit(value['xTrain'], value['yTrain'])
            yPred = svmsvc.predict(value["xTest"])
            test_scores.append(accuracy_score(value['yTest'],yPred)*100)
            train_scores.append(accuracy_score(value['yTrain'],svmsvc.predict(value['xTrain']))*100)
            cv_scores.append(cross_val_score(svmsvc, value['xTrain'], value['yTrain'], cv=5).mean()*100)

            graph.append(['Training', kernel, accuracy_score(value['yTrain'],svmsvc.predict(value['xTrain']))*100])
            graph.append(['Testing', kernel, accuracy_score(value['yTest'],svmsvc.predict(value['xTest']))*100])
            graph.append(['Validation', kernel, cross_val_score(svmsvc, value['xTrain'], value['yTrain'], cv=5).mean()*100])
            timer.append(time.time()-start)

            timers["Kernel"] = timer

        df = pd.DataFrame(graph, columns=['data', 'kernel', 'Accuracy (%)'])
        df.pivot("kernel", "data", "Accuracy (%)").plot(kind='bar')
        plt.ylabel("Accuracy (%)")
        plt.title(Title + " (SVM) Accuracy with kernel")
        saveToNewDir(Title+"/", Title + "_SVM_Kernel.png")

        plt.clf()

        plt.bar(kernels, timer, 1/1.5, label="Training time")
        plt.xlabel("kernel")
        plt.ylabel("Time (s)")
        plt.title(Title + " (SVM) Training time with kernel")
        saveToNewDir(Title+"/", Title + "_SVM_KernelTimer.png")

        plt.clf()

    # Test different C Values
    def TestCVal(ImportedData, Data, bestKey):
        plt.figure(figsize=(7,7), dpi=70)

        train_scores = []
        test_scores = []
        cv_scores = []
        timer = []
        p_s = []
        r_s = []
        f_1 = []
        value = Data[bestKey]
        
        c_arr = np.geomspace(0.1,1000, num=10)

        for val in c_arr:    
            start = time.time()
            svmsvc = svm.SVC(kernel='linear', C=val).fit(value['xTrain'], value['yTrain'])
            yPred = svmsvc.predict(value["xTest"])
            train_scores.append(accuracy_score(value['yTrain'], svmsvc.predict(value['xTrain']))*100)
            cv_scores.append(cross_val_score(svmsvc, value['xTrain'], value['yTrain'], cv=5).mean()*100)
            test_scores.append(accuracy_score(value['yTest'], svmsvc.predict(value['xTest']))*100)
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
        
        timers["CVal"] = timer
        plt.plot(c_arr, train_scores, label="Train", lw = 2)
        plt.plot(c_arr, test_scores, label="Test", lw = 2)
        plt.plot(c_arr, cv_scores, label="CV", lw = 2)
        plt.plot(c_arr, p_s, label='Precision', lw = 2)
        plt.plot(c_arr, r_s, label='Recall', lw = 2)
        plt.plot(c_arr, f_1, label='F1', lw = 2)
        plt.xlabel("Estimator Size")
        plt.ylabel("Accuracy (%)")
        plt.legend(loc='best')
        plt.title(Title + " (SVM) Accuracy with train size")
        saveToNewDir(Title+"/", Title + "_SVM_CVal.png")
        plt.clf() 

    # Test different Gamma value
    def TestGamma(ImportedData, Data, bestKey):
        plt.figure(figsize=(7,7), dpi=70)

        train_scores = []
        test_scores = []
        cv_scores = []
        timer = []
        p_s = []
        r_s = []
        f_1 = []
        value = Data[bestKey]
        
        gamma = np.geomspace(0.1,1000,num=10)

        for val in gamma:    
            start = time.time()
            svmsvc = svm.SVC(kernel='rbf', gamma = val).fit(value['xTrain'], value['yTrain'])
            yPred = svmsvc.predict(value["xTest"])
            train_scores.append(accuracy_score(value['yTrain'], svmsvc.predict(value['xTrain']))*100)
            cv_scores.append(cross_val_score(svmsvc, value['xTrain'], value['yTrain'], cv=5).mean()*100)
            test_scores.append(accuracy_score(value['yTest'], svmsvc.predict(value['xTest']))*100)
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
        
        timers["Gamma"] = timer
        plt.plot(gamma, train_scores, label="Train", lw = 2)
        plt.plot(gamma, test_scores, label="Test", lw = 2)
        plt.plot(gamma, cv_scores, label="CV", lw = 2)
        plt.plot(gamma, p_s, label='Precision', lw = 2)
        plt.plot(gamma, r_s, label='Recall', lw = 2)
        plt.plot(gamma, f_1, label='F1', lw = 2)
        plt.xlabel("Gamma")
        plt.ylabel("Accuracy (%)")
        plt.legend(loc='best')
        plt.title(Title + " (SVM) Accuracy with Gamma")
        saveToNewDir(Title+"/", Title + "_SVM_Gamma.png")
        plt.clf() 
    
    bestKernel, bestC, bestGamma, bestKey = CalculateBestHyperParameters(ImportedData, Data)
    TestTrain(ImportedData, Data, bestKey)
    TestKernel(ImportedData, Data, bestKey)
    TestCVal(ImportedData, Data, bestKey)
    TestGamma(ImportedData, Data, bestKey)

    #Timers Plot
    plt.figure(figsize=(7,7), dpi=70)
    plt.xlabel('Index')
    plt.ylabel('Time')
    plt.xticks(range(1,11,1))
    plt.title(Title + ' (SVM) Training Time Vs. Index')
    #plt.plot([x/5 for x in list(Data.keys())],timers['Training'], label='Training Set Size', lw = 2)
    plt.plot(timers['Training'], label='Training Set Size', lw = 2)
    plt.plot(timers['Kernel'], label='Training Set Size', lw = 2)
    plt.plot(timers['CVal'], label='Estimator Size', lw = 2)
    plt.plot(timers['Gamma'], label='Estimator Size', lw = 2)
    plt.legend()
    saveToNewDir(Title+"/", Title + "_SVM_Timers.png")
    plt.gcf().clear()
    plt.clf() 
    print ("Done with SVM for " + Title + " dataset")

if __name__ == "__main__":

    scaler = StandardScaler()
    letterData = ImportedData()
    bankData = ImportedData()
    letterTemp = {}
    bankTemp = {}
    letterData.Title = "LetterRecognition"
    bankData.Title = "BanknoteAuthentication"
    for split in range(10, 90, 8):
        if(split==100):
            split-=1
        if(split==0):
            split+=1
        letterTemp[split] = {}
        letterTemp[split]["xTrain"], letterTemp[split]["xTest"], letterTemp[split]["yTrain"], letterTemp[split]["yTest"] = getLetterRecData(split/100.0)
        scaler.fit(letterTemp[split]["xTrain"])
        letterTemp[split]["xTrain"] = scaler.transform(letterTemp[split]["xTrain"])
        letterTemp[split]["xTest"] = scaler.transform(letterTemp[split]["xTest"])
        bankTemp[split] = {}
        bankTemp[split]["xTrain"], bankTemp[split]["xTest"], bankTemp[split]["yTrain"], bankTemp[split]["yTest"] = getBanknoteAuthData(split/100.0)
        scaler.fit(bankTemp[split]["xTrain"])
        bankTemp[split]["xTrain"] = scaler.transform(bankTemp[split]["xTrain"])
        bankTemp[split]["xTest"] = scaler.transform(bankTemp[split]["xTest"])

    letterData.Data = letterTemp
    bankData.Data = bankTemp
    
    RunDecisionTreeClassifier(bankData, bankData.Data)
    RunDecisionTreeClassifier(letterData, letterData.Data)
    RunAdaBoostClassifier(bankData, bankData.Data)
    RunAdaBoostClassifier(letterData, letterData.Data)
    RunKNNClassifier(bankData, bankData.Data)
    RunKNNClassifier(letterData, letterData.Data)
    RunNeuralNetClassifier(bankData, bankData.Data)
    RunNeuralNetClassifier(letterData, letterData.Data)
    RunSVMClassifier(bankData, bankData.Data)
    RunSVMClassifier(letterData, letterData.Data)

    # DecisionTree
    # a = Process(target=RunDecisionTreeClassifier, args = (bankData, bankData.Data)).start()
    # b = Process(target=RunAdaBoostClassifier, args = (bankData, bankData.Data)).start()
    # c = Process(target=RunKNNClassifier, args = (bankData, bankData.Data)).start()   
    # d = Process(target=RunNeuralNetClassifier, args = (bankData, bankData.Data)).start()
    # e = Process(target=RunSVMClassifier, args = (bankData, bankData.Data)).start()

    # a.join()
    # b.join()
    # c.join()
    # d.join()
    # e.join()

    # f = Process(target=RunDecisionTreeClassifier, args = (letterData, letterData.Data)).start()
    # g = Process(target=RunAdaBoostClassifier, args = (letterData, letterData.Data)).start()
    # h = Process(target=RunKNNClassifier, args = (letterData, letterData.Data)).start()
    # i = Process(target=RunNeuralNetClassifier, args = (letterData, letterData.Data)).start()
    # j = Process(target=RunSVMClassifier, args = (letterData, letterData.Data)).start()

    # f.join()
    # g.join()
    # h.join()
    # i.join()
    # j.join()