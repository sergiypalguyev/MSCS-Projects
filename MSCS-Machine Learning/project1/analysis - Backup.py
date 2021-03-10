import os
import time  
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import threading
from multiprocessing import Process

from functools import partial
import itertools

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
from sklearn.metrics import roc_curve, auc

import warnings
warnings.filterwarnings("ignore")

class ImportedData:
    Title = ""
    Data = {}
# https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients
def getDefaultData(train_size):
    banknote = pd.read_csv('./default_of_credit_card_clients.data', header=None)
    banknote.columns = ['LIMIT_BAL','SEX','EDUCATION','MARRIAGE','AGE','PAY_1','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6', 'BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6','PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6','default']
    Y = banknote['default']
    X = banknote.drop(['default'], axis=1)
    T = preprocessing.MinMaxScaler().fit_transform(X)
    Xs = pd.DataFrame(T,columns = X.columns)
    Ts = train_test_split(Xs, Y, train_size = train_size)
    return Ts

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
    banknote = pd.read_csv('./letter-recognition.data', header=None)
    banknote.columns = ['Letter', 'xBoxHPos', 'yBoxVPos', 'BoxW', 'BoxH', 'OnPix', 'xBarMean', 'yBarMean', 'x2BarMean', 'y2BarMean', 'xyBarMean', 'x2ybrMean', 'xy2BrMean','xEgeMean', 'xegvyCorr', 'y-egeMean', 'yegvxCorr']
    
    Y = banknote['Letter']
    X = banknote.drop(['Letter'], axis=1)
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
    print ("Working on DecisionTree for " + Title + " dataset")
    timers = {}
    value = Data[70]

    # Test different test/train ratios
    def TestTrain(ImportedData, Data):
        plt.figure(figsize=(7,7), dpi=70)
        train_scores = []
        test_scores = []
        cv_scores = []
        roc_auc = []
        timer = []
        for key, value in Data.items():  

            start = time.time()
            dtc = DecisionTreeClassifier(criterion='gini', random_state = 100, max_depth=5, min_samples_leaf=5).fit(value["xTrain"], value["yTrain"])
            yPred = dtc.predict(value["xTest"])
            cv_scores.append(cross_val_score(dtc, value["xTrain"], value["yTrain"]).mean()*100)
            train_scores.append(accuracy_score(value["yTrain"], dtc.predict(value["xTrain"]))*100)
            test_scores.append(accuracy_score(value["yTest"], yPred)*100)
            #false_positive_rate, true_positive_rate, thresholds = roc_curve(value["yTest"], yPred)
            #roc_auc.append(auc(false_positive_rate, true_positive_rate)*100)

            timer.append(time.time() - start)

        timers['Training'] = timer
        plt.xticks(list(Data.keys()))
        plt.plot(Data.keys(), train_scores, label='Train', lw = 2)
        plt.plot(Data.keys(), test_scores, label='Test', lw = 2)
        plt.plot(Data.keys(), cv_scores, label='CV', lw = 2)
        #plt.plot(Data.keys(), roc_auc, label='AUC', lw = 2)

        plt.xlabel('Training Set Size ')
        plt.ylabel('Accuracy')
        plt.title(Title + ' (Decision Tree) Accuracy Vs. Training Set Size')
        plt.legend()
        # plt.savefig("./" + Title + "/" + Title + "_DecisionTree_TrainingSetSize.png", bbox_inches='tight')
        saveToNewDir(Title+"/", Title + "_DecisionTree_TrainingSetSize.png")
        plt.gcf().clear()
        plt.clf()  

    # Test different tree depth
    def TestDepth(ImportedData, Data):
        plt.figure(figsize=(7,7), dpi=70)
        train_scores = []
        test_scores = []
        cv_scores = []
        roc_auc = []
        timer = []
        tRange = range(2,22,1)
        for depth in tRange:            
            start = time.time()
            dtc = DecisionTreeClassifier(criterion='gini', random_state = 100, max_depth = depth).fit(value["xTrain"], value["yTrain"])
            yPred = dtc.predict(value["xTest"])
            cv_scores.append(cross_val_score(dtc, value["xTrain"], value["yTrain"]).mean()*100)
            train_scores.append(accuracy_score(value["yTrain"], dtc.predict(value["xTrain"]))*100)
            test_scores.append(accuracy_score(value["yTest"], yPred)*100)
            #false_positive_rate, true_positive_rate, thresholds = roc_curve(value["yTest"], yPred)
            #roc_auc.append(auc(false_positive_rate, true_positive_rate)*100)
            timer.append(time.time() - start)

        timers['Depth'] = timer
        plt.xticks(tRange)
        plt.plot(tRange, train_scores, label='Train', lw = 2)
        plt.plot(tRange, test_scores, label='Test', lw = 2)
        plt.plot(tRange, cv_scores, label='CV', lw = 2)
        #plt.plot(tRange, roc_auc, label='AUC', lw = 2)

        plt.xlabel('Tree Depth ')
        plt.ylabel('Accuracy')
        plt.title(Title + ' (DecisionTree) Accuracy Vs. Tree Depth')
        plt.legend()
        # plt.savefig("./" + Title + "/" + Title + "_DecisionTree_TreeDepth.png",bbox_inches='tight')
        saveToNewDir(Title+"/", Title + "_DecisionTree_TreeDepth.png")
        plt.gcf().clear()
        plt.clf() 

    # Test different leaf node sizes
    def TestLeaves(ImportedData, Data):
        plt.figure(figsize=(7,7), dpi=70)
        train_scores = []
        test_scores = []
        cv_scores = []
        roc_auc = []
        timer = []
        tRange = range(2,22,1)
        for leaf_size in tRange:            
            start = time.time()
            dtc = DecisionTreeClassifier(criterion='gini', random_state = 100, max_leaf_nodes=leaf_size).fit(value["xTrain"], value["yTrain"])
            yPred = dtc.predict(value["xTest"])
            cv_scores.append(cross_val_score(dtc, value["xTrain"], value["yTrain"]).mean()*100)
            train_scores.append(accuracy_score(value["yTrain"], dtc.predict(value["xTrain"]))*100)
            test_scores.append(accuracy_score(value["yTest"], yPred)*100)
            #false_positive_rate, true_positive_rate, thresholds = roc_curve(value["yTest"], yPred)
            #roc_auc.append(auc(false_positive_rate, true_positive_rate)*100)
            timer.append(time.time() - start)

        timers['Leaf'] = timer
        plt.xticks(tRange)
        plt.plot(tRange, train_scores, label='Train', lw = 2)
        plt.plot(tRange, test_scores, label='Test', lw = 2)
        plt.plot(tRange, cv_scores, label='CV', lw = 2)
        #plt.plot(tRange, roc_auc, label='AUC', lw = 2)

        plt.xlabel('Tree Leaf Size ')
        plt.ylabel('Accuracy')
        plt.title(Title + ' (DecisionTree) Accuracy Vs. Tree Leaf size')
        plt.legend()
        #plt.savefig("./" + Title + "/" + Title + "_DecisionTree_LeafSize.png",bbox_inches='tight')
        saveToNewDir(Title+"/", Title + "_DecisionTree_LeafSize.png")
        plt.gcf().clear()
        plt.clf()        
    
    # Test different sizes of Decision Tree forests.
    def TestForest(ImportedData,Data):
        plt.figure(figsize=(7,7), dpi=70)
        train_scores = []
        test_scores = []
        cv_scores = []
        roc_auc = []
        timer = []
        xRange = range(2,12,1)#np.round(np.geomspace(1,50,num=5))
        for treeNum in xRange:   
            start = time.time()
            rfc = RandomForestClassifier(criterion='gini', random_state = 100, n_estimators=int(treeNum)).fit(value["xTrain"], value["yTrain"])
            yPred = rfc.predict(value["xTest"])
            cvVector = []
            trainVector = []
            testVector = []
            aucVector = []
            for i in range(10):
                cvVector.append(cross_val_score(rfc, value["xTrain"], value["yTrain"], cv=5).mean()*100)
                trainVector.append(accuracy_score(value["yTrain"], rfc.predict(value["xTrain"]))*100)
                testVector.append(accuracy_score(value["yTest"], yPred)*100)
                #false_positive_rate, true_positive_rate, thresholds = roc_curve(value["yTest"], yPred)
                #aucVector.append(auc(false_positive_rate, true_positive_rate)*100)

            cv_scores.append(sum(cvVector)/len(cvVector))
            train_scores.append(sum(trainVector)/len(trainVector))
            test_scores.append(sum(testVector)/len(testVector))
            #roc_auc.append(sum(aucVector)/len(aucVector))
            
            timer.append(time.time() - start)

        timers['Estimators'] = timer
        plt.xticks(xRange)
        plt.plot(xRange, train_scores, label='Train', lw = 2)
        plt.plot(xRange, test_scores, label='Test', lw = 2)
        plt.plot(xRange, cv_scores, label='CV', lw = 2)
        #plt.plot(xRange, roc_auc, label='AUC', lw = 2)

        plt.xlabel('Number of Estimators')
        plt.ylabel('Accuracy')
        plt.title(Title + ' (Random Forest) Accuracy Vs. Forest Number of Estimators')
        plt.legend()
        #plt.savefig(Title + "_DecisionTree_RandomForest.png",bbox_inches='tight')
        saveToNewDir(Title+"/", Title + "_DecisionTree_RandomForest.png")
        plt.gcf().clear()
        plt.clf()  
    
    TestTrain(ImportedData, Data)    
    TestDepth(ImportedData, Data)    
    TestLeaves(ImportedData, Data)    
    TestForest(ImportedData, Data)
    
    #Timers Plot
    plt.figure(figsize=(7,7), dpi=70)
    plt.xlabel('Index')
    plt.ylabel('Time')
    plt.xticks(range(1,21,1))
    plt.title(Title + ' (Random Forest) Training Time Vs. Index')
    plt.plot([x/5 for x in list(Data.keys())], timers['Training'], label='Training Set Size', lw = 2)
    plt.plot(timers['Depth'], label='Tree Depth Size', lw = 2)
    plt.plot(timers['Leaf'], label='Tree Leaf Size', lw = 2)
    plt.legend()
    #plt.savefig("./" + Title + "/" + Title + "_DecisionTree_Timers.png",bbox_inches='tight')
    saveToNewDir(Title+"/", Title + "_DecisionTree_Timers.png")
    plt.gcf().clear()
    plt.clf()  
    print ("Done with DecisionTree for " + Title + " dataset")

def RunAdaBoostClassifier(ImportedData, Data):
    Title = ImportedData.Title
    print ("Working on AdaBoost for " + Title + " dataset")
    timers = {}    
    value = Data[70]
    
    # Test different test/train ratios
    def TestTrain(ImportedData, Data):
        plt.figure(figsize=(7,7), dpi=70)
        train_scores = []
        test_scores = []
        cv_scores = []
        roc_auc = []
        timer = []
        for key, value in Data.items():            
            start = time.time()
            abc = AdaBoostClassifier(DecisionTreeClassifier(criterion = "gini", random_state=100), n_estimators = 30, random_state=100).fit(value["xTrain"], value["yTrain"])
            yPred = abc.predict(value["xTest"])
            cv_scores.append(cross_val_score(abc, value["xTrain"], value["yTrain"]).mean()*100)
            train_scores.append(accuracy_score(value["yTrain"], abc.predict(value["xTrain"]))*100)
            test_scores.append(accuracy_score(value["yTest"], yPred)*100)
            #false_positive_rate, true_positive_rate, thresholds = roc_curve(value["yTest"], yPred)
            #roc_auc.append(auc(false_positive_rate, true_positive_rate)*100)
            timer.append(time.time() - start)
            
        timers['Training'] = timer
        plt.xticks(list(Data.keys()))
        plt.plot(Data.keys(), train_scores, label='Train', lw = 2)
        plt.plot(Data.keys(), test_scores, label='Test', lw = 2)
        plt.plot(Data.keys(), cv_scores, label='CV', lw = 2)
        #plt.plot(Data.keys(), roc_auc, label='AUC', lw = 2)

        plt.xlabel('Training Set Size ')
        plt.ylabel('Accuracy')
        plt.title(Title + ' (AdaBoost) Accuracy Vs. Training Set Size')
        plt.legend()
        #plt.savefig("./" + Title + "/" + Title + "_AdaBoost_TrainingSetSize.png", bbox_inches='tight')
        saveToNewDir(Title+"/", Title + "_AdaBoost_TrainingSetSize.png")
        plt.gcf().clear()
        plt.clf()  

    # Test different tree depth
    def TestDepth(ImportedData, Data):
        plt.figure(figsize=(7,7), dpi=70)
        train_scores = []
        test_scores = []
        cv_scores = []
        roc_auc = []
        timer = []
        tRange = range(2,12,1)
        for depth in tRange:            
            start = time.time()
            abc = AdaBoostClassifier(DecisionTreeClassifier(criterion = "gini", random_state=100, max_depth=depth),n_estimators = 30, random_state=100).fit(value["xTrain"], value["yTrain"])
            yPred = abc.predict(value["xTest"])
            cv_scores.append(cross_val_score(abc, value["xTrain"], value["yTrain"]).mean()*100)
            train_scores.append(accuracy_score(value["yTrain"], abc.predict(value["xTrain"]))*100)
            test_scores.append(accuracy_score(value["yTest"], yPred)*100)
            #false_positive_rate, true_positive_rate, thresholds = roc_curve(value["yTest"], yPred)
            #roc_auc.append(auc(false_positive_rate, true_positive_rate)*100)
            timer.append(time.time() - start)

        timers['Depth'] = timer
        plt.xticks(tRange)
        plt.plot(tRange, train_scores, label='Train', lw = 2)
        plt.plot(tRange, test_scores, label='Test', lw = 2)
        plt.plot(tRange, cv_scores, label='CV', lw = 2)
        #plt.plot(tRange, roc_auc, label='AUC', lw = 2)

        plt.xlabel('Tree Depth ')
        plt.ylabel('Accuracy')
        plt.title(Title + ' (AdaBoost) Accuracy Vs. Tree Depth')
        plt.legend()
        #plt.savefig("./" + Title + "/" + Title + "_AdaBoost_TreeDepth.png",bbox_inches='tight')
        saveToNewDir(Title+"/", Title + "_AdaBoost_TreeDepth.png")
        plt.gcf().clear()
        plt.clf() 
    
    # Test different leaf node sizes
    def TestLeaves(ImportedData, Data):
        plt.figure(figsize=(7,7), dpi=70)
        train_scores = []
        test_scores = []
        cv_scores = []
        roc_auc = []
        timer = []
        tRange = range(2,22,1)
        for leaf_size in tRange:            
            start = time.time()
            abc = AdaBoostClassifier(DecisionTreeClassifier(criterion = "gini", random_state=100, min_samples_leaf=leaf_size), n_estimators = 30, random_state=100).fit(value["xTrain"], value["yTrain"])
            yPred = abc.predict(value["xTest"])
            cv_scores.append(cross_val_score(abc, value["xTrain"], value["yTrain"]).mean()*100)
            train_scores.append(accuracy_score(value["yTrain"], abc.predict(value["xTrain"]))*100)
            test_scores.append(accuracy_score(value["yTest"], yPred)*100)
            #false_positive_rate, true_positive_rate, thresholds = roc_curve(value["yTest"], yPred)
            #roc_auc.append(auc(false_positive_rate, true_positive_rate)*100)
            timer.append(time.time() - start)

        timers['Leaf'] = timer
        plt.xticks(tRange)
        plt.plot(tRange, train_scores, label='Train', lw = 2)
        plt.plot(tRange, test_scores, label='Test', lw = 2)
        plt.plot(tRange, cv_scores, label='CV', lw = 2)
        #plt.plot(tRange, roc_auc, label='AUC', lw = 2)

        plt.xlabel('Tree Leaf Size ')
        plt.ylabel('Accuracy')
        plt.title(Title + ' (AdaBoost) Accuracy Vs. Tree Leaf size')
        plt.legend()
        #plt.savefig("./" + Title + "/" + Title + "_AdaBoost_LeafSize.png",bbox_inches='tight')
        saveToNewDir(Title+"/", Title + "_AdaBoost_LeafSize.png")
        plt.gcf().clear()
        plt.clf()   

    # Test different AdaBoost estimator sizes
    def TestEstimators(ImportedData, Data):
        plt.figure(figsize=(7,7), dpi=70)
        train_scores = []
        test_scores = []
        cv_scores = []
        roc_auc = []
        timer = []
        tRange = Data.keys()
        
        for estimator in Data.keys():            
            start = time.time()
            abc = AdaBoostClassifier(DecisionTreeClassifier(criterion = "gini", random_state=100), n_estimators = estimator, random_state=100).fit(value["xTrain"], value["yTrain"])
            yPred = abc.predict(value["xTest"])
            cv_scores.append(cross_val_score(abc, value["xTrain"], value["yTrain"]).mean()*100)
            train_scores.append(accuracy_score(value["yTrain"], abc.predict(value["xTrain"]))*100)
            test_scores.append(accuracy_score(value["yTest"], yPred)*100)
            #false_positive_rate, true_positive_rate, thresholds = roc_curve(value["yTest"], yPred)
            #roc_auc.append(auc(false_positive_rate, true_positive_rate)*100)
            timer.append(time.time() - start)

        timers['Estimator'] = timer
        plt.xticks(list(tRange))
        plt.plot(tRange, train_scores, label='Train', lw = 2)
        plt.plot(tRange, test_scores, label='Test', lw = 2)
        plt.plot(tRange, cv_scores, label='CV', lw = 2)
        #plt.plot(tRange, roc_auc, label='AUC', lw = 2)

        plt.xlabel('Estimator #')
        plt.ylabel('Accuracy')
        plt.title(Title + ' (AdaBoost) Accuracy Vs. Estimator #')
        plt.legend()
        #plt.savefig("./" + Title + "/" + Title + "_AdaBoost_Estimator.png",bbox_inches='tight')
        saveToNewDir(Title+"/", Title + "_AdaBoost_Estimator.png")
        plt.gcf().clear()
        plt.clf()   
    
    TestTrain(ImportedData, Data)
    TestDepth(ImportedData, Data)
    TestLeaves(ImportedData, Data)
    TestEstimators(ImportedData, Data)

    #Timers Plot
    plt.figure(figsize=(7,7), dpi=70)
    plt.xlabel('Index')
    plt.ylabel('Time')
    plt.xticks(range(1,21,1))
    plt.title(Title + ' (AdaBoost) Training Time Vs. Index')
    plt.plot([x/5 for x in list(Data.keys())],timers['Training'], label='Training Set Size', lw = 2)
    plt.plot(timers['Depth'], label='Tree Depth Size', lw = 2)
    plt.plot(timers['Leaf'], label='Tree Leaf Size', lw = 2)
    plt.plot(timers['Estimator'], label='Estimator Size', lw = 2)
    plt.legend()
    #plt.savefig("./" + Title + "/" + Title + "_AdaBoost_Timers.png",bbox_inches='tight')
    saveToNewDir(Title+"/", Title + "_AdaBoost_Timers.png")
    plt.gcf().clear()
    plt.clf() 
    print ("Done with AdaBoost for " + Title + " dataset")

def RunKNNClassifier(ImportedData, Data):
    Title = ImportedData.Title
    print ("Working on KNN for " + Title + " dataset")
    timers = {}
    value = Data[70]
    
    # Test different test/train ratios
    def TestTrain(ImportedData, Data):
        plt.figure(figsize=(7,7), dpi=70)

        train_scores = []
        test_scores = []
        cv_scores = []
        roc_auc = []
        timer = []

        for key, value in Data.items():    
            start = time.time()
            knns = KNeighborsClassifier(n_neighbors=5, n_jobs=-1, weights='distance').fit(value['xTrain'], value['yTrain'])
            yPred = knns.predict(value["xTest"])
            train_scores.append(accuracy_score(value['yTrain'], knns.predict(value['xTrain']))*100)
            cv_scores.append(cross_val_score(knns, value['xTrain'], value['yTrain'], cv=5).mean()*100)
            test_scores.append(accuracy_score(value['yTest'], yPred)*100)
            #false_positive_rate, true_positive_rate, thresholds = roc_curve(value["yTest"], yPred)
            #roc_auc.append(auc(false_positive_rate, true_positive_rate)*100)
            timer.append(time.time()-start)
        
        timers["Training"] = timer
        plt.plot(Data.keys(), train_scores, label="Train")
        plt.plot(Data.keys(), test_scores, label="Test")
        plt.plot(Data.keys(), cv_scores, label="CV")
        #plt.plot(Data.keys(), roc_auc, label='AUC')
        plt.xlabel("Training Set Size")
        plt.ylabel("Accuracy (%)")
        plt.legend(loc='best')
        plt.title(Title + " (KNN) Accuracy with train size")
        #plt.savefig("./" + Title + "/" + Title + "_KNN_TrainingSetSize.png")
        saveToNewDir(Title+"/", Title + "_KNN_TrainingSetSize.png")
        plt.clf()

    # Test different neighbors
    def TestNeighbors(ImportedData, Data):
        plt.figure(figsize=(7,7), dpi=70)

        train_scores = []
        test_scores = []
        cv_scores = []
        roc_auc = []
        timer = []

        tRange = range(1, 21, 1)
        for k in tRange:
            start = time.time()
            knns = KNeighborsClassifier(n_neighbors=k, n_jobs=-1, weights='distance').fit(value['xTrain'], value['yTrain'])
            yPred = knns.predict(value["xTest"])
            test_scores.append(accuracy_score(value['yTest'],yPred)*100)
            cv_scores.append(cross_val_score(knns, value['xTrain'], value['yTrain'], cv=5).mean()*100)
            train_scores.append(accuracy_score(value['yTrain'],knns.predict(value['xTrain']))*100)
            #false_positive_rate, true_positive_rate, thresholds = roc_curve(value["yTest"], yPred)
            #roc_auc.append(auc(false_positive_rate, true_positive_rate)*100)
            timer.append(time.time()-start)

        timers["Neighbors"] = timer
        plt.plot(tRange, train_scores, label="Train")
        plt.plot(tRange, test_scores, label="Test")
        plt.plot(tRange, cv_scores, label="CV")
        #plt.plot(tRange, roc_auc, label='AUC')
        plt.xlabel("K neighbours")
        plt.ylabel("Accuracy")
        plt.legend(loc='best')
        plt.title(Title + " (KNN) Accuracy with k")
        #plt.savefig(Title + "_KNN_NeighborSize.png")
        saveToNewDir(Title+"/", Title + "_KNN_NeighborSize.png")

        plt.clf()
    
    # Test different metrics
    def TestMetrics(ImportedData, Data):
        plt.figure(figsize=(7,7), dpi=70)

        metrics = ["cosine", "manhattan", "euclidean", "minkowski", "hamming", "canberra"]
        train_scores = []
        test_scores = []
        cv_scores = []
        roc_auc = []
        timer = []
        for metric in metrics:        
            start = time.time()
            knns = KNeighborsClassifier(n_neighbors=5, n_jobs=1, metric=metric).fit(value['xTrain'], value['yTrain'])
            yPred = knns.predict(value["xTest"])
            test_scores.append(accuracy_score(value['yTest'],yPred)*100)
            train_scores.append(accuracy_score(value['yTrain'],knns.predict(value['xTrain']))*100)
            cv_scores.append(cross_val_score(knns, value['xTrain'], value['yTrain'], cv=5).mean()*100)
            #false_positive_rate, true_positive_rate, thresholds = roc_curve(value["yTest"], yPred)
            #roc_auc.append(auc(false_positive_rate, true_positive_rate)*100)
            timer.append(time.time()-start)

        timers["Metrics"] = timer

        df = pd.DataFrame({'Train': train_scores,
                        'Test': test_scores,
                        'CV': cv_scores}, index=metrics) #'AUC': roc_auc}, index=metrics)
        df.plot.bar(rot=0)
        plt.xlabel("K neighbours")
        plt.ylabel("Accuracy")
        plt.legend(loc='best')
        plt.title(Title + " (KNN) Accuracy with metrics")
        #plt.savefig(Title + "_KNN_Metrics.png")
        saveToNewDir(Title+"/", Title + "_KNN_Metrics.png")

        plt.clf()  
        
        plt.bar(metrics, timer, 1/1.5, label="Training time")
        plt.xlabel("Metrics")
        plt.ylabel("Time (s)")
        plt.title(Title + " (KNN) Training time with kernel")
        #plt.savefig(Title + "_KNN_MetricsTimer.png")
        saveToNewDir(Title+"/", Title + "_KNN_MetricsTimer.png")

        plt.clf()
    
    TestTrain(ImportedData, Data)
    TestNeighbors(ImportedData, Data)
    TestMetrics(ImportedData, Data)
    
    #Timers Plot
    plt.figure(figsize=(7,7), dpi=70)
    plt.xlabel('Index')
    plt.ylabel('Time')
    plt.xticks(range(1,21,1))
    plt.title(Title + ' (KNN) Training Time Vs. Index')
    plt.plot([x/5 for x in list(Data.keys())],timers['Training'], label='Training Set Size', lw = 2)
    plt.plot(timers['Neighbors'], label='Neighbors', lw = 2)
    plt.legend()
    #plt.savefig("./" + Title + "/" + Title + "_KNN_Timers.png",bbox_inches='tight')
    saveToNewDir(Title+"/", Title + "_KNN_Timers.png")
    plt.gcf().clear()
    plt.clf() 
    print ("Done with KNN for " + Title + " dataset")

def RunNeuralNetClassifier(ImportedData, Data):
    Title = ImportedData.Title
    print ("Working on NeuralNets for " + Title + " datasets")
    timers = {}
    value = Data[70]
    
    # Test different test/train ratios
    def TestTrain(ImportedData, Data):
        plt.figure(figsize=(7,7), dpi=70)

        train_scores = []
        test_scores = []
        cv_scores = []
        roc_auc = []
        timer = []

        for key, value in Data.items():    
            start = time.time()
            mlp = MLPClassifier(hidden_layer_sizes=(10,10,10,10,10), max_iter=10).fit(value['xTrain'], value['yTrain'])
            yPred = mlp.predict(value["xTest"])
            train_scores.append(accuracy_score(value['yTrain'], mlp.predict(value['xTrain']))*100)
            cv_scores.append(cross_val_score(mlp, value['xTrain'], value['yTrain'], cv=5).mean()*100)
            test_scores.append(accuracy_score(value['yTest'], yPred)*100)
            #false_positive_rate, true_positive_rate, thresholds = roc_curve(value["yTest"], yPred)
            #roc_auc.append(auc(false_positive_rate, true_positive_rate)*100)
            timer.append(time.time()-start)
        
        timers["Training"] = timer
        plt.plot(Data.keys(), train_scores, label="Train")
        plt.plot(Data.keys(), test_scores, label="Test")
        plt.plot(Data.keys(), cv_scores, label="CV")
        #plt.plot(Data.keys(), roc_auc, label='AUC')
        plt.xlabel("Training Set Size")
        plt.ylabel("Accuracy (%)")
        plt.legend(loc='best')
        plt.title(Title + " (NeuralNet) Accuracy with train size")
        #plt.savefig("./" + Title + "/" + Title + "_NeuralNet_TrainingSetSize.png")
        saveToNewDir(Title+"/", Title + "_NeuralNet_TrainingSetSize.png")
        plt.clf()
    
    # Test different layers
    def TestLayers(ImportedData, Data):
        plt.figure(figsize=(7,7), dpi=70)

        train_scores = []
        test_scores = []
        cv_scores = []
        roc_auc = []
        timer = []
        value = Data[70]
        tRange = range(2,22,1)
        layer_sizes = ()
        for ix in tRange:
            layer_sizes += (10, )          
            start = time.time()
            mlp = MLPClassifier(hidden_layer_sizes=layer_sizes, max_iter=10).fit(value['xTrain'], value['yTrain'])
            yPred = mlp.predict(value["xTest"])
            train_scores.append(accuracy_score(value['yTrain'], mlp.predict(value['xTrain']))*100)
            cv_scores.append(cross_val_score(mlp, value['xTrain'], value['yTrain'], cv=5).mean()*100)
            test_scores.append(accuracy_score(value['yTest'], yPred)*100)
            #false_positive_rate, true_positive_rate, thresholds = roc_curve(value["yTest"], yPred)
            #roc_auc.append(auc(false_positive_rate, true_positive_rate)*100)
            timer.append(time.time()-start)
        
        timers["Layers"] = timer
        plt.plot(tRange, train_scores, label="Train")
        plt.plot(tRange, test_scores, label="Test")
        plt.plot(tRange, cv_scores, label="CV")
        #plt.plot(tRange, roc_auc, label='AUC')
        plt.xlabel("Layers #")
        plt.ylabel("Accuracy (%)")
        plt.legend(loc='best')
        plt.title(Title + " (NeuralNet) Accuracy with # of layers")
        #plt.savefig("./" + Title + "/" + Title + "_NeuralNet_TrainingLayerNumber.png")
        saveToNewDir(Title+"/", Title + "_NeuralNet_TrainingLayerNumber.png")
        plt.clf()
    
    # Test different neurons
    def TestNeurons(ImportedData, Data):
        plt.figure(figsize=(7,7), dpi=70)

        train_scores = []
        test_scores = []
        cv_scores = []
        roc_auc = []
        timer = []
        value = Data[70]
        tRange = range(2,42,2)#np.round(np.geomspace(10,90,num=20))
        for ix in tRange:
            i = int(ix)
            layer_sizes = (i, i, i, i, i)   
            start = time.time()
            mlp = MLPClassifier(hidden_layer_sizes=layer_sizes, max_iter=10).fit(value['xTrain'], value['yTrain'])
            yPred = mlp.predict(value["xTest"])
            train_scores.append(accuracy_score(value['yTrain'], mlp.predict(value['xTrain']))*100)
            cv_scores.append(cross_val_score(mlp, value['xTrain'], value['yTrain'], cv=5).mean()*100)
            test_scores.append(accuracy_score(value['yTest'], yPred)*100)
            #false_positive_rate, true_positive_rate, thresholds = roc_curve(value["yTest"], yPred)
            #roc_auc.append(auc(false_positive_rate, true_positive_rate)*100)
            timer.append(time.time()-start)
        
        timers["Neurons"] = timer
        plt.plot(tRange, train_scores, label="Train")
        plt.plot(tRange, test_scores, label="Test")
        plt.plot(tRange, cv_scores, label="CV")
        #plt.plot(tRange, roc_auc, label='AUC')
        plt.xlabel("Neurons #")
        plt.ylabel("Accuracy (%)")
        plt.legend(loc='best')
        plt.title(Title + " (NeuralNet) Accuracy with # of neurons")
        #plt.savefig("./" + Title + "/" + Title + "_NeuralNet_TrainingNeuronsNumber.png")
        saveToNewDir(Title+"/", Title + "_NeuralNet_TrainingNeuronsNumber.png")
        plt.clf()
    
    TestTrain(ImportedData, Data)
    TestLayers(ImportedData, Data)
    TestNeurons(ImportedData, Data)

    #Timers Plot
    plt.figure(figsize=(7,7), dpi=70)
    plt.xlabel('Index')
    plt.ylabel('Time')
    plt.xticks(range(1,21,1))
    plt.title(Title + ' (NeuralNet) Training Time Vs. Index')
    plt.plot([x/5 for x in list(Data.keys())],timers['Training'], label='Training Set Size', lw = 2)
    plt.plot(timers['Layers'], label='Neural Net Layers #', lw = 2)
    plt.plot(timers['Neurons'], label='Neaural Net Neurons #', lw = 2)
    plt.legend()
    #plt.savefig("./" + Title + "/" + Title + "_NeuralNet_Timers.png",bbox_inches='tight')
    saveToNewDir(Title+"/", Title + "_NeuralNet_Timers.png")
    plt.gcf().clear()
    plt.clf() 
    print ("Done with NeuralNets for " + Title + " datasets")

def RunSVMClassifier(ImportedData, Data):
    Title = ImportedData.Title
    print ("Working on SVM for " + Title + " dataset")
    timers = {}
    value = Data[70]
    
    # Test different test/train ratios
    def TestTrain(ImportedData, Data):
        plt.figure(figsize=(7,7), dpi=70)

        train_scores = []
        test_scores = []
        cv_scores = []
        roc_auc = []
        timer = []

        for key, value in Data.items():    
            start = time.time()
            svmsvc = svm.SVC(kernel='linear').fit(value['xTrain'], value['yTrain'])
            yPred = svmsvc.predict(value["xTest"])
            train_scores.append(accuracy_score(value['yTrain'], svmsvc.predict(value['xTrain']))*100)
            cv_scores.append(cross_val_score(svmsvc, value['xTrain'], value['yTrain'], cv=5).mean()*100)
            test_scores.append(accuracy_score(value['yTest'], yPred)*100)
            #false_positive_rate, true_positive_rate, thresholds = roc_curve(value["yTest"], yPred)
            #roc_auc.append(auc(false_positive_rate, true_positive_rate)*100)
            timer.append(time.time()-start)
        
        timers["Training"] = timer
        plt.plot(Data.keys(), train_scores, label="Train")
        plt.plot(Data.keys(), test_scores, label="Test")
        plt.plot(Data.keys(), cv_scores, label="CV")
        #plt.plot(Data.keys(), roc_auc, label='AUC')
        plt.xlabel("Training Set Size")
        plt.ylabel("Accuracy (%)")
        plt.legend(loc='best')
        plt.title(Title + " (SVM) Accuracy with train size")
        #plt.savefig("./" + Title + "/" + Title + "_SVM_TrainingSetSize.png")
        saveToNewDir(Title+"/", Title + "_SVM_TrainingSetSize.png")
        plt.clf()    

    # Test different test/train ratios
    def TestKernel(ImportedData, Data):
        plt.figure(figsize=(7,7), dpi=70)

        kernels = ['linear','poly','rbf','sigmoid'] 

        graph = []
        train_scores = []
        test_scores = []
        cv_scores = []
        roc_auc = []
        timer = []

        for kernel in kernels:   
            start = time.time()
            svmsvc = svm.SVC(kernel=kernel).fit(value['xTrain'], value['yTrain'])
            yPred = svmsvc.predict(value["xTest"])
            test_scores.append(accuracy_score(value['yTest'],yPred)*100)
            train_scores.append(accuracy_score(value['yTrain'],svmsvc.predict(value['xTrain']))*100)
            cv_scores.append(cross_val_score(svmsvc, value['xTrain'], value['yTrain'], cv=5).mean()*100)
            #false_positive_rate, true_positive_rate, thresholds = roc_curve(value["yTest"], yPred)
            #roc_auc.append(auc(false_positive_rate, true_positive_rate)*100)

            graph.append(['Training', kernel, accuracy_score(value['yTrain'],svmsvc.predict(value['xTrain']))*100])
            graph.append(['Testing', kernel, accuracy_score(value['yTest'],svmsvc.predict(value['xTest']))*100])
            graph.append(['Validation', kernel, cross_val_score(svmsvc, value['xTrain'], value['yTrain'], cv=5).mean()*100])
            #graph.append(['AUC', kernel, auc(false_positive_rate, true_positive_rate)*100])
            timer.append(time.time()-start)

            timers["Kernel"] = timer

        df = pd.DataFrame(graph, columns=['data', 'kernel', 'Accuracy (%)'])
        df.pivot("kernel", "data", "Accuracy (%)").plot(kind='bar')
        plt.ylabel("Accuracy (%)")
        plt.title(Title + " (SVM) Accuracy with kernel")
        #plt.savefig("./" + Title + "/" + Title + "_SVM_Kernel.png")
        saveToNewDir(Title+"/", Title + "_SVM_Kernel.png")

        plt.clf()

        plt.bar(kernels, timer, 1/1.5, label="Training time")
        plt.xlabel("kernel")
        plt.ylabel("Time (s)")
        plt.title(Title + " (SVM) Training time with kernel")
        #plt.savefig("./" + Title + "/" + Title + "_SVM_KernelTimer.png")
        saveToNewDir(Title+"/", Title + "_SVM_KernelTimer.png")

        plt.clf()

    # Test different test/train ratios
    def TestCVal(ImportedData, Data):
        plt.figure(figsize=(7,7), dpi=70)

        train_scores = []
        test_scores = []
        cv_scores = []
        roc_auc = []
        timer = []
        value = Data[70]
        
        c_arr = np.geomspace(0.2,1,num=4)

        for val in c_arr:    
            start = time.time()
            svmsvc = svm.SVC(kernel='linear', C=val).fit(value['xTrain'], value['yTrain'])
            yPred = svmsvc.predict(value["xTest"])
            train_scores.append(accuracy_score(value['yTrain'], svmsvc.predict(value['xTrain']))*100)
            cv_scores.append(cross_val_score(svmsvc, value['xTrain'], value['yTrain'], cv=5).mean()*100)
            test_scores.append(accuracy_score(value['yTest'], svmsvc.predict(value['xTest']))*100)
            #false_positive_rate, true_positive_rate, thresholds = roc_curve(value["yTest"], yPred)
            #roc_auc.append(auc(false_positive_rate, true_positive_rate)*100)
            timer.append(time.time()-start)
        
        timers["CVal"] = timer
        plt.plot(c_arr, train_scores, label="Train")
        plt.plot(c_arr, test_scores, label="Test")
        plt.plot(c_arr, cv_scores, label="CV")
        #plt.plot(c_arr, roc_auc, label='AUC')
        plt.xlabel("Estimator Size")
        plt.ylabel("Accuracy (%)")
        plt.legend(loc='best')
        plt.title(Title + " (SVM) Accuracy with train size")
        #plt.savefig("./" + Title + "/" + Title + "_SVM_CVal.png")
        saveToNewDir(Title+"/", Title + "_SVM_CVal.png")
        plt.clf() 

    TestTrain(ImportedData, Data)
    TestKernel(ImportedData, Data)
    TestCVal(ImportedData, Data)

    #Timers Plot
    plt.figure(figsize=(7,7), dpi=70)
    plt.xlabel('Index')
    plt.ylabel('Time')
    plt.xticks(range(1,21,1))
    plt.title(Title + ' (SVM) Training Time Vs. Index')
    plt.plot([x/5 for x in list(Data.keys())],timers['Training'], label='Training Set Size', lw = 2)
    plt.plot(timers['CVal'], label='Estimator Size', lw = 2)
    plt.legend()
    #plt.savefig("./" + Title + "/" + Title + "_SVM_Timers.png",bbox_inches='tight')
    saveToNewDir(Title+"/", Title + "_SVM_Timers.png")
    plt.gcf().clear()
    plt.clf() 
    print ("Done with SVM for " + Title + " dataset")

if __name__ == "__main__":

    scaler = StandardScaler()
    defaultData = ImportedData()
    defaultTemp = {}
    defaultData.Title = "Default"
    for split in range(90, 10, -4):
        if(split==100):
            split-=1
        if(split==0):
            split+=1
        defaultTemp[split] = {}
        defaultTemp[split]["xTrain"], defaultTemp[split]["xTest"], defaultTemp[split]["yTrain"], defaultTemp[split]["yTest"] = getDefaultData(split/100.0)
        scaler.fit(defaultTemp[split]["xTrain"])
        defaultTemp[split]["xTrain"] = scaler.transform(defaultTemp[split]["xTrain"])
        defaultTemp[split]["xTest"] = scaler.transform(defaultTemp[split]["xTest"])
    defaultData.Data = defaultTemp
    # RunDecisionTreeClassifier(defaultData, defaultData.Data)

    scaler = StandardScaler()
    letterData = ImportedData()
    letterTemp = {}
    letterData.Title = "LetterRec"
    for split in range(90, 10, -4):
        if(split==100):
            split-=1
        if(split==0):
            split+=1
        letterTemp[split] = {}
        letterTemp[split]["xTrain"], letterTemp[split]["xTest"], letterTemp[split]["yTrain"], letterTemp[split]["yTest"] = getLetterRecData(split/100.0)
        scaler.fit(letterTemp[split]["xTrain"])
        letterTemp[split]["xTrain"] = scaler.transform(letterTemp[split]["xTrain"])
        letterTemp[split]["xTest"] = scaler.transform(letterTemp[split]["xTest"])
    letterData.Data = letterTemp

    scaler = StandardScaler()
    bankData = ImportedData()
    bankTemp = {}
    bankData.Title = "Banknote Authentication"
    for split in range(90, 10, -4):
        if(split==100):
            split-=1
        if(split==0):
            split+=1
        bankTemp[split] = {}
        bankTemp[split]["xTrain"], bankTemp[split]["xTest"], bankTemp[split]["yTrain"], bankTemp[split]["yTest"] = getBanknoteAuthData(split/100.0)
        scaler.fit(bankTemp[split]["xTrain"])
        bankTemp[split]["xTrain"] = scaler.transform(bankTemp[split]["xTrain"])
        bankTemp[split]["xTest"] = scaler.transform(bankTemp[split]["xTest"])
    bankData.Data = bankTemp

    #RunAdaBoostClassifier(djiData, djiData.Data)
    #RunKNNClassifier(djiData, djiData.Data)
    #RunNeuralNetClassifier(djiData, djiData.Data)
    #RunSVMClassifier(djiData, djiData.Data)
    #Process(target=RunDecisionTreeClassifier, args = (djiData, djiData.Data)).start()

    # scaler = StandardScaler()  
    # aData = ImportedData()
    # cData = ImportedData()
    # mData = ImportedData()
    # aTemp = {}
    # cTemp = {}
    # mTemp = {}
    # aData.Title = "Abalone"
    # cData.Title = "Contraceptive"
    # mData.Title = "Mammographic"
    # for split in range(100,0,-5):
    #     if(split==100):
    #         split-=1
    #     if(split==0):
    #         split+=1
    #     aTemp[split] = {}
    #     cTemp[split] = {}
    #     mTemp[split] = {}
    #     aTemp[split]["xTrain"], aTemp[split]["xTest"], aTemp[split]["yTrain"], aTemp[split]["yTest"] = getAbaloneData(train_size=(split / 100.0))
    #     cTemp[split]["xTrain"], cTemp[split]["xTest"], cTemp[split]["yTrain"], cTemp[split]["yTest"]  = getContraceptiveData(train_size=split / 100.0)
    #     mTemp[split]["xTrain"], mTemp[split]["xTest"], mTemp[split]["yTrain"], mTemp[split]["yTest"]  = getMammographicData(train_size=split / 100.0)
	    
    #     scaler.fit(aTemp[split]["xTrain"])
    #     aTemp[split]["xTrain"] = scaler.transform(aTemp[split]["xTrain"])  
    #     aTemp[split]["xTest"] = scaler.transform(aTemp[split]["xTest"])
        
    #     scaler.fit(cTemp[split]["xTrain"])
    #     cTemp[split]["xTrain"] = scaler.transform(cTemp[split]["xTrain"])  
    #     cTemp[split]["xTest"] = scaler.transform(cTemp[split]["xTest"])
        
    #     scaler.fit(mTemp[split]["xTrain"])
    #     mTemp[split]["xTrain"] = scaler.transform(mTemp[split]["xTrain"])  
    #     mTemp[split]["xTest"] = scaler.transform(mTemp[split]["xTest"])
  
    # aData.Data = aTemp
    # cData.Data = cTemp
    # mData.Data = mTemp

    # # DecisionTree
    # #Process(target=RunDecisionTreeClassifier, args = (aData, aData.Data)).start()
    #Process(target=RunDecisionTreeClassifier, args = (letterData, letterData.Data)).start()
    #Process(target=RunDecisionTreeClassifier, args = (defaultData, defaultData.Data)).start()
    #Process(target=RunDecisionTreeClassifier, args = (bankData, bankData.Data)).start()
    # #Process(target=RunDecisionTreeClassifier, args = (cData, cData.Data)).start()
    # #Process(target=RunDecisionTreeClassifier, args = (mData, mData.Data)).start()

    # # # AdaBoost
    # # #Process(target=RunAdaBoostClassifier, args = (aData, aData.Data)).start()
    #Process(target=RunAdaBoostClassifier, args = (letterData, letterData.Data)).start()
    #Process(target=RunAdaBoostClassifier, args = (defaultData, defaultData.Data)).start()
    #Process(target=RunAdaBoostClassifier, args = (bankData, bankData.Data)).start()
    # # Process(target=RunAdaBoostClassifier, args = (cData, cData.Data)).start()
    # # Process(target=RunAdaBoostClassifier, args = (mData, mData.Data)).start()
    
    # # # # KNearestNeighbor
    # # #Process(target=RunKNNClassifier, args = (aData, aData.Data)).start()
    #Process(target=RunKNNClassifier, args = (letterData, letterData.Data)).start()
    #Process(target=RunKNNClassifier, args = (defaultData, defaultData.Data)).start()
    #Process(target=RunKNNClassifier, args = (bankData, bankData.Data)).start()
    # # Process(target=RunKNNClassifier, args = (cData, cData.Data)).start()
    # # Process(target=RunKNNClassifier, args = (mData, mData.Data)).start()

    # # # Neural Nets
    # #Process(target=RunNeuralNetClassifier, args = (aData, aData.Data)).start()
    #Process(target=RunNeuralNetClassifier, args = (letterData, letterData.Data)).start()    
    #Process(target=RunNeuralNetClassifier, args = (defaultData, defaultData.Data)).start()
    #Process(target=RunNeuralNetClassifier, args = (bankData, bankData.Data)).start()
    # # Process(target=RunNeuralNetClassifier, args = (cData, cData.Data)).start()
    # # Process(target=RunNeuralNetClassifier, args = (mData, mData.Data)).start()
    
    # # # # Support Vector Machines
    # # #Process(target=RunSVMClassifier, args = (aData, aData.Data)).start()
    Process(target=RunSVMClassifier, args = (letterData, letterData.Data)).start()
    Process(target=RunSVMClassifier, args = (defaultData, defaultData.Data)).start()
    Process(target=RunSVMClassifier, args = (bankData, bankData.Data)).start()
    # # Process(target=RunSVMClassifier, args = (cData, cData.Data)).start()
    # # Process(target=RunSVMClassifier, args = (mData, mData.Data)).start()