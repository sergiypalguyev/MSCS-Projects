import os
import numpy as np
import pandas as pd
import time
import copy 
from sklearn import metrics
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.decomposition import FactorAnalysis
from sklearn.random_projection import SparseRandomProjection
from sklearn.random_projection import GaussianRandomProjection
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.mixture import GaussianMixture as GMM
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import silhouette_samples, silhouette_score, homogeneity_score
from sklearn.metrics.pairwise import euclidean_distances
import seaborn as sns
from pandas.plotting import parallel_coordinates
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LassoLarsIC
import matplotlib.cm as cm
from sklearn.model_selection import cross_val_score
import threading
from multiprocessing import Process
from sklearn.linear_model  import LogisticRegression  
from sklearn.metrics import precision_score, recall_score,confusion_matrix,classification_report, accuracy_score,f1_score

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

import warnings
warnings.filterwarnings("ignore")

class ImportedData:
    def __init__(self):
        self.Title = ""
        self.Subtitle = ""
        self.Data = {}

# https://archive.ics.uci.edu/ml/datasets/banknote+authentication
def getBanknoteAuthData(train_size):
    banknote = pd.read_csv('./data_banknote_authentication.data', header=None)
    banknote.columns = ['Wavelet_Variance', 'Wavelet_Skewness', 'Wavelet_Curtosis', 'Image_Entropy', 'Class']
    Y = banknote['Class']
    X = banknote.drop(['Class'], axis=1)
    T = preprocessing.MinMaxScaler().fit_transform(X)
    Xs = pd.DataFrame(T,columns = X.columns)
    Ts = train_test_split(Xs, Y, train_size = train_size)

    plt.figure(figsize=(24,12))
    classes = [0,1]
    palette = sns.color_palette('bright', 2)
    colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in classes]
    parallel_coordinates(banknote, class_column='Class', color= colors, alpha=0.1)
    saveToNewDir("BanknoteAuthentication/", "BanknoteAuthentication_Parallel.png")
    plt.gcf().clear()
    plt.clf()

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

    plt.figure(figsize=(24,12))
    classes = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
    palette = sns.color_palette('bright', 26)
    colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in classes]
    parallel_coordinates(letter, class_column='Letter', color= colors, alpha=0.1)
    saveToNewDir("LetterRecognition/", "LetterRecognition_Parallel.png")
    plt.gcf().clear()
    plt.clf()

    return Ts

def saveToNewDir(directory, filename):
    if not os.path.isdir(directory):
        os.makedirs(directory)
    plt.savefig(directory+filename, bbox_inches='tight')

def plotBanknoteData(predictions, n_cluster, range_n_clusters, ImportedData, Data):
    colors = ["#0cc0aa", "#b71c41", "#37b51f", "#6c218e", "#8efd4e", "#ce4fca", "#f1fb1a", "#1f0133", "#d9fdac", "#6108e8", "#809b31", "#fe16f4", "#1b511d", "#e4b8ec", "#051311", "#f8cca6", "#30408d", "#f9bd3a", "#491307", "#82d1f4", "#8f494f", "#588c94", "#ff6b97", "#074d65", "#fa7922", "#7f7bc9", "#ff2a0d", "#a37e1a"]
    classes = predictions[n_cluster]

    # ===== Plot Variance vs Skewness vs Curtosis ===== #
    fig = plt.figure(figsize=(12,12))
    ax = fig.gca(projection='3d')
    for i in range(len(classes)):
        x = Data["xTrain"].T[0][i] + np.random.normal(0, 0.1)
        y = Data["xTrain"].T[1][i] + np.random.normal(0, 0.1)
        z = Data["xTrain"].T[2][i] + np.random.normal(0, 0.1)
        yzs = max(Data["xTrain"].T[1])
        xzs = min(Data["xTrain"].T[0])
        zzs = min(Data["xTrain"].T[2])
        ax.scatter(x, y, z, c=colors[classes[i]], cmap=plt.cm.RdYlGn, marker='.', s=100) # color=colors[classes[i]],
        ax.scatter(x, z, zdir='y', zs=yzs, color='k', marker='.', alpha=0.1)
        ax.scatter(y, z, zdir='x', zs=xzs, color='k', marker='.', alpha=0.1)
        ax.scatter(x, y, zdir='z', zs=zzs, color='k', marker='.', alpha=0.1)
    X_minmax = np.round(np.min(Data["xTrain"].T[0]),2), np.round(np.max(Data["xTrain"].T[0]),2)
    Y_minmax = np.round(np.min(Data["xTrain"].T[1]),2), np.round(np.max(Data["xTrain"].T[1]),2)
    Z_minmax = np.round(np.min(Data["xTrain"].T[2]),2), np.round(np.max(Data["xTrain"].T[2]),2)
    X_range = np.round(np.arange(*X_minmax),2)
    Y_range = np.round(np.arange(*Y_minmax),2)
    Z_range = np.round(np.arange(*Z_minmax),2)
    ax.set_xlim3d(X_minmax)
    ax.set_ylim3d(Y_minmax)
    ax.set_zlim3d(Z_minmax)
    ax.set(xticks=X_range, xticklabels=X_range, yticks=Y_range, yticklabels=Y_range, zticks=Z_range, zticklabels=Z_range)
    saveToNewDir(ImportedData.Title + "/", ImportedData.Title + ImportedData.Subtitle + "_{}-clusters.png".format(range_n_clusters[n_cluster]))
    plt.gcf().clear()
    plt.clf()

def plotLetterData(predictions, n_cluster, range_n_clusters, ImportedData, Data):
    colors = ["#0cc0aa", "#b71c41", "#37b51f", "#6c218e", "#8efd4e", "#ce4fca", "#f1fb1a", "#1f0133", "#d9fdac", "#6108e8", "#809b31", "#fe16f4", "#1b511d", "#e4b8ec", "#051311", "#f8cca6", "#30408d", "#f9bd3a", "#491307", "#82d1f4", "#8f494f", "#588c94", "#ff6b97", "#074d65", "#fa7922", "#7f7bc9", "#ff2a0d", "#a37e1a"]
    classes = predictions[n_cluster]

    # ===== Plot Variance vs Skewness vs Curtosis ===== #
    fig = plt.figure(figsize=(12,12))
    ax = fig.gca(projection='3d')
    for i in range(len(classes)):
        x = Data["xTrain"].T[0][i] + np.random.normal(0, 0.1)
        y = Data["xTrain"].T[1][i] + np.random.normal(0, 0.1)
        z = Data["xTrain"].T[2][i] + np.random.normal(0, 0.1)
        yzs = max(Data["xTrain"].T[1])
        xzs = min(Data["xTrain"].T[0])
        zzs = min(Data["xTrain"].T[2])
        ax.scatter(x, y, z,  cmap = 'prism', marker='.',alpha=0.3, s=100) # color=colors[classes[i]],
        ax.scatter(x, z, zdir='y', zs=yzs, color='k', marker='.', alpha=0.1)
        ax.scatter(y, z, zdir='x', zs=xzs, color='k', marker='.', alpha=0.1)
        ax.scatter(x, y, zdir='z', zs=zzs, color='k', marker='.', alpha=0.1)
        # ax.scatter(Data["xTrain"].T[0][i] + np.random.normal(0, 0.1), Data["xTrain"].T[1][i] + np.random.normal(0, 0.1), Data["xTrain"].T[2][i] + np.random.normal(0, 0.1), color=colors[classes[i]], marker='o',alpha=0.3, s=100)
        # ax.scatter(Data["xTrain"].T[0][i] + np.random.normal(0, 0.1), Data["xTrain"].T[2][i] + np.random.normal(0, 0.1), zdir='y', zs=max(Data["xTrain"].T[1][i]), color='k', marker='.', alpha=0.1)
        # ax.scatter(Data["xTrain"].T[1][i] + np.random.normal(0, 0.1), Data["xTrain"].T[2][i] + np.random.normal(0, 0.1), zdir='x', zs=min(Data["xTrain"].T[0][i]), color='k', marker='.', alpha=0.1)
        # ax.scatter(Data["xTrain"].T[0][i] + np.random.normal(0, 0.1), Data["xTrain"].T[1][i] + np.random.normal(0, 0.1), zdir='z', zs=min(Data["xTrain"].T[2][i]), color='k', marker='.', alpha=0.1)
    X_minmax = np.round(np.min(Data["xTrain"].T[0]),2), np.round(np.max(Data["xTrain"].T[0]),2)
    Y_minmax = np.round(np.min(Data["xTrain"].T[1]),2), np.round(np.max(Data["xTrain"].T[1]),2)
    Z_minmax = np.round(np.min(Data["xTrain"].T[2]),2), np.round(np.max(Data["xTrain"].T[2]),2)
    X_range = np.round(np.arange(*X_minmax),2)
    Y_range = np.round(np.arange(*Y_minmax),2)
    Z_range = np.round(np.arange(*Z_minmax),2)
    ax.set_xlim3d(X_minmax)
    ax.set_ylim3d(Y_minmax)
    ax.set_zlim3d(Z_minmax)
    ax.set(xticks=X_range, xticklabels=X_range, yticks=Y_range, yticklabels=Y_range, zticks=Z_range, zticklabels=Z_range)
    saveToNewDir(ImportedData.Title + "/", ImportedData.Title + ImportedData.Subtitle + "_{}-clusters.png".format(range_n_clusters[n_cluster]))
    plt.gcf().clear()
    plt.clf()

def KMAnalysis(ImportedData, Data):

    range_n_clusters = list(range(2,15))

    pred_onXtrain = []
    predictions = []
    sse = []
    nmi = []
    hgs = []
    cpt = []
    sis = []
    bestSSE = 0
    bestSSEd = 0

    for n_clusters in range_n_clusters:
        k_means = KMeans(n_clusters=n_clusters, random_state=0)
        k_means.fit(Data["xTrain"])
        # pred_onXtrain.append(k_means.predict(Data["xTrain"]))
        # classes = k_means.predict(Data["xTest"])
        xTrain = k_means.labels_
        xTest = k_means.predict(Data["xTest"])
        pred_onXtrain.append(xTrain)
        predictions.append(xTest)
        sse.append(k_means.inertia_)
        nmi.append(normalized_mutual_info_score(Data["yTrain"], k_means.labels_))
        hgs.append(homogeneity_score(Data["yTrain"], k_means.labels_))
        cpt.append(metrics.completeness_score(Data["yTrain"], k_means.labels_))
        sis.append(silhouette_score(Data["xTest"], xTest))

    x1,y1 = [range_n_clusters[-1],sse[-1]]
    x0,y0 = [range_n_clusters[0],sse[0]]
    for x in range(len(sse)):
        x2,y2 = [range_n_clusters[x],sse[x]]
        p1=np.array([x0,y0])
        p2=np.array([x1,y1])
        p3=np.array([x2,y2])
        d = np.linalg.norm(np.cross(p2-p1, p1-p3))/np.linalg.norm(p2-p1)
        if d > bestSSEd:
            bestSSEd = d
            bestSSE = x
    string = (ImportedData.Title + ImportedData.Subtitle + " Best SSE k-Means {} clusters".format(range_n_clusters[bestSSE])\
        + ", SSE: %0.2f" % sse[bestSSE]\
        + ", MutualInfo: %0.2f" % nmi[bestSSE]\
        + ", Homogeneity: %0.2f" % hgs[bestSSE]\
        + ", Completeness: %0.2f" % cpt[bestSSE]\
        + ", Silhouette: %0.2f" % sis[bestSSE])
    print (string)
    with open(ImportedData.Title+"/" + ImportedData.Title + ".txt", 'a+') as f:
        f.write(string + "\n")

    if("BanknoteAuthentication" in ImportedData.Title):
        plotBanknoteData(predictions, bestSSE, range_n_clusters, ImportedData, Data)
    elif("LetterRecognition" in ImportedData.Title):
        plotLetterData(predictions, bestSSE, range_n_clusters, ImportedData, Data)

    fig, ax1 = plt.subplots()
    ax1.set_xlabel('k clusters')
    ax1.set_ylabel('SSE', color='tab:red')
    ax1.plot(range_n_clusters, sse, 'r.-', label="Sum of Squared Errors")
    ax1.axvline(x=range_n_clusters[bestSSE] - 0.04, color='r', linestyle='--', alpha=0.5)
    ax1.tick_params(axis='y', labelcolor='tab:red')
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('Scores', color='tab:blue')  # we already handled the x-label with ax1
    ax2.plot(range_n_clusters, nmi, 'b.-', label="Mutual Info Score")
    ax2.plot(range_n_clusters, hgs, 'g.-', label="Homogeneity Score")
    ax2.plot(range_n_clusters, cpt, 'k.-', label="Completeness Score")
    ax2.plot(range_n_clusters, sis, 'm.-', label="Silhouette Score")
    ax2.set_yticks(ticks=np.arange(0, 1, 0.1))
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    ax2.legend(loc='upper right')
    ax1.legend(loc='upper left')
    plt.grid(linestyle='--', linewidth=0.5, alpha=0.15)
    saveToNewDir(ImportedData.Title+"/", ImportedData.Title + ImportedData.Subtitle + "_Erros_kMeans.png")
    #plt.show()
    plt.gcf().clear()
    plt.clf()

    ImportedData.Data["xTrain"] = pred_onXtrain[bestSSE][:, np.newaxis]
    ImportedData.Data["xTest"] = predictions[bestSSE][:, np.newaxis]

def EMAnalysis(ImportedData, Data):

    pred_onXtrain = []
    predictions = []
    sis = []
    bic=[]
    aic=[]
    bestSIL = 0
    bestSILd = 0
    bestAIC = 0
    bestAICd = 0
    bestBIC = 0
    bestBICd = 0
    range_n_clusters = list(range(2,15))

    for n_clusters in range_n_clusters:
        a = KMeans(n_clusters=n_clusters, random_state=0)
        a.fit(Data["xTrain"])
        # pred_onXtrain.append(k_means.predict(Data["xTrain"]))
        # classes = k_means.predict(Data["xTest"])
        xTrain = a.labels_

        k_means = GaussianMixture(n_components=n_clusters, random_state=0)
        k_means.fit(Data["xTrain"])
        # pred_onXtrain.append(k_means.predict(Data["xTrain"]))
        # classes = k_means.predict(Data["xTest"])
        # xTrain = k_means.predict(Data["xTrain"])
        xTest = k_means.predict(Data["xTest"])
        pred_onXtrain.append(xTrain)
        predictions.append(xTest)
        sis.append(silhouette_score(Data["xTest"], xTest))
        bic.append(k_means.bic(Data["xTrain"]))
        aic.append(k_means.aic(Data["xTrain"]))

    x1,y1 = [range_n_clusters[-1],sis[-1]]
    x0,y0 = [range_n_clusters[0],sis[0]]
    for x in range(len(sis)):
        if x>0:
            if sis[x] > bestSILd:
                bestSILd = sis[x]
                bestSIL = x
            if bic[x] < bestBICd:
                bestBICd = bic[x]
                bestBIC = x
            if aic[x] < bestAICd:
                bestAICd = aic[x]
                bestAIC = x

    string = (ImportedData.Title + ImportedData.Subtitle + " Best Silhouete EM {} clusters".format(range_n_clusters[bestSIL])\
        + ", SIL: %0.2f" % sis[bestSIL]\
        + ", AIC: %0.2f" % aic[bestSIL]\
        + ", BIC: %0.2f\n" % bic[bestSIL])
    string += (ImportedData.Title + ImportedData.Subtitle + " Best BIC EM {} clusters".format(range_n_clusters[bestBIC])\
        + ", SIL: %0.2f" % sis[bestBIC]\
        + ", AIC: %0.2f" % aic[bestBIC]\
        + ", BIC: %0.2f\n" % bic[bestBIC])
    string += (ImportedData.Title + ImportedData.Subtitle + " Best AIC EM {} clusters".format(range_n_clusters[bestAIC])\
        + ", SIL: %0.2f" % sis[bestAIC]\
        + ", AIC: %0.2f" % aic[bestAIC]\
        + ", BIC: %0.2f\n" % bic[bestAIC])
        
    print (string)
    with open(ImportedData.Title+"/" + ImportedData.Title +".txt", 'a+') as f:
        f.write(string + "\n")

    if("BanknoteAuthentication" in ImportedData.Title):
        plotBanknoteData(predictions, bestSIL, range_n_clusters, ImportedData, Data) 
        plotBanknoteData(predictions, bestBIC, range_n_clusters, ImportedData, Data) 
        plotBanknoteData(predictions, bestAIC, range_n_clusters, ImportedData, Data) 
    elif("LetterRecognition" in ImportedData.Title):        
        plotLetterData(predictions, bestSIL, range_n_clusters, ImportedData, Data)
        plotLetterData(predictions, bestBIC, range_n_clusters, ImportedData, Data)
        plotLetterData(predictions, bestAIC, range_n_clusters, ImportedData, Data)

    fig, ax1 = plt.subplots()
    ax1.set_xlabel('k clusters')
    ax1.set_ylabel('AIC & BIC', color='tab:red')
    ax1.plot(range_n_clusters, aic, 'r.-', label="AIC Score")
    ax1.axvline(x=range_n_clusters[bestAIC] - 0.04, color='r', linestyle='--', alpha=0.5)
    ax1.plot(range_n_clusters, bic, 'b.-', label="BIC Score")
    ax1.axvline(x=range_n_clusters[bestBIC] + 0.04, color='b', linestyle='--', alpha=0.5)
    ax1.tick_params(axis='y', labelcolor='tab:red')
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('Silhouette', color='tab:blue')  # we already handled the x-label with ax1
    ax2.plot(range_n_clusters, sis, 'm.-', label="Silhouette Score")
    ax2.axvline(x=range_n_clusters[bestSIL] + 0.00, color='m', linestyle='--', alpha=0.5)
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    ax2.set_yticks(ticks=np.arange(0, 0.5, 0.1))
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    ax1.legend(loc='upper right')
    ax2.legend(loc='upper left')
    plt.grid(linestyle='--', linewidth=0.5, alpha=0.15)
    saveToNewDir(ImportedData.Title+"/", ImportedData.Title + ImportedData.Subtitle + "_Erros_EM.png")
    plt.gcf().clear()
    plt.clf()

    ImportedData.Data["xTrain"] = pred_onXtrain[bestBIC][:, np.newaxis]
    ImportedData.Data["xTest"] = predictions[bestBIC][:, np.newaxis]

def plotCVKurVarGraph(ImportedData):
    def make_patch_spines_invisible(ax):
        ax.set_frame_on(True)
        ax.patch.set_visible(False)
        for sp in ax.spines.values():
            sp.set_visible(False)   

    X_train = copy.deepcopy(ImportedData.Data["xTrain"])
    components = np.shape(X_train)[1]
    pcaX = np.arange(1, components+1)
    cv_scores, mean_arr, var_arr, skew_arr, kurt_arr = [], [], [], [], []
    bestCV = 0
    bestKurt = 0
    for n in pcaX:
        pca = PCA()
        pca.n_components = n
        rescaled = np.float32(X_train)
        reduced = pca.fit_transform(rescaled)
        s = np.shape(reduced)[0]
        mean = np.sum((reduced**1)/s) # Calculate the mean
        mean_arr.append(mean)
        var = np.sum((reduced-mean)**2)/s # Calculate the variance
        var_arr.append(var)
        skew = np.sum((reduced-mean)**3)/s # Calculate the skewness
        skew_arr.append(skew)
        kurt = np.sum((reduced-mean)**4)/s # Calculate the kurtosis
        kurt = kurt/(var**2)-3
        kurt_arr.append(kurt)
        score = np.mean(cross_val_score(pca, rescaled))
        cv_scores.append(score)
        if len(cv_scores)==1 or (score - cv_scores[-1]) > 0.01:
            bestCV = n
        elif score - cv_scores[-2] < 0.01:
            bestCV = n-1
        if len(kurt_arr)==1 or kurt < kurt_arr[-2]:
            bestKurt = n
    
    pca.n_components = components
    rescaled = np.float32(X_train)
    reduced = pca.fit_transform(rescaled)
    var_ratio = np.cumsum(pca.explained_variance_ratio_)

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax3 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax1.plot(pcaX, cv_scores, 'b')
    #plt.plot(pcaX, mean_arr/max(mean_arr), 'm', label='Mean')
    ax3.plot(pcaX, var_arr, 'r')
    #plt.plot(pcaX, skew_arr/max(skew_arr), 'y', label='Skew')
    ax2.plot(pcaX, kurt_arr, 'k')

    ax1.set_xlabel('Components')
    
    ax2.spines["left"].set_position(("axes", -0.2)) # red one
    ax3.spines["left"].set_position(("axes", -0.4)) # red one

    make_patch_spines_invisible(ax2)
    make_patch_spines_invisible(ax3)

    ax2.spines["left"].set_visible(True)
    ax2.yaxis.set_label_position('left')
    ax2.yaxis.set_ticks_position('left')
    ax3.spines["left"].set_visible(True)
    ax3.yaxis.set_label_position('left')
    ax3.yaxis.set_ticks_position('left')

    ax1.set_ylabel('CV Score', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax2.set_ylabel('Kurtosis', color='k')
    ax2.tick_params(axis='y', labelcolor='k')
    ax3.set_ylabel('Variance', color='r')
    ax3.tick_params(axis='y', labelcolor='r')

    fig.legend([ax1, ax2, ax3],     # The line objects
        labels=['CV-Score', 'Kurtosis', 'Variance'],
        loc="center right")   # The labels for each line

    plt.xticks(pcaX, pcaX)
    plt.xlabel('nb of components')
    plt.ylabel('CV scores')
    plt.xlabel('number of components')
    plt.grid(linestyle='--', linewidth=0.5, alpha=0.15)
    saveToNewDir(ImportedData.Title + "/", ImportedData.Title + ImportedData.Subtitle + "-CVKurVarScores.png")
    plt.gcf().clear()
    plt.clf()
    print ("Best CV Score {}@{} components and Kurtosis of {}@{}components".format(cv_scores[bestCV-1], bestCV, kurt_arr[bestKurt-1], bestKurt ))
    return bestCV , bestKurt
def plotFit(ImportedData, algorithm=None):
    scaler=StandardScaler()
    X_train = copy.deepcopy(ImportedData.Data["xTrain"])
    X_test = copy.deepcopy(ImportedData.Data["xTest"])
    y_test = copy.deepcopy(ImportedData.Data["yTest"])
    y_train = copy.deepcopy(ImportedData.Data["yTrain"])
    X_train = pd.DataFrame(preprocessing.StandardScaler().fit_transform(X_train))
    X_train = X_train.astype('float16')
    
    if algorithm != None:
        X_train = algorithm.fit_transform(X_train)
        if isinstance(algorithm, PCA) or isinstance(algorithm, FastICA):
            inverse_X = algorithm.inverse_transform(X_train)
        else:
            inverse_X = None
        X_test = algorithm.transform(X_test)
    else:
        X_train = X_train.to_numpy() 
        inverse_X = None

    LogReg=LogisticRegression(class_weight='balanced')  
    LogReg.fit(X_train,y_train)
    y_pred=LogReg.predict(X_test)

    string = ImportedData.Subtitle + ' Accuracy: {}\n'.format(accuracy_score(y_test,y_pred))
    string += ImportedData.Subtitle + ' F1 score: {}\n'.format(f1_score(y_test,y_pred,average='weighted'))   
    string += ImportedData.Subtitle + ' Recall: {}\n'.format(recall_score(y_test,y_pred,average='weighted'))   
    string += ImportedData.Subtitle + ' Precision: {}\n'.format(precision_score(y_test,y_pred,average='weighted'))  
    string += ImportedData.Subtitle + ' clasification report: \n{}\n'.format(classification_report(y_test,y_pred))  
    string += ImportedData.Subtitle + ' confussion matrix: \n{}\n'.format(confusion_matrix(y_test,y_pred))
    print (string)
    with open(ImportedData.Title + "/" + ImportedData.Title+".txt", 'a+') as f:
        f.write(string + "\n")

    fig = plt.figure(figsize=(12,12))
    ax = fig.gca(projection='3d')
    ax.scatter(X_train[:,0], X_train[:,1], X_train[:,2], c=y_train, marker='.',alpha=0.7, s=100)
    ax.scatter(X_train[:,0], X_train[:,2], zdir='y', zs=max(X_train[:,1]), color='k', marker='.', alpha=0.1)
    ax.scatter(X_train[:,1], X_train[:,2], zdir='x', zs=min(X_train[:,0]), color='k', marker='.', alpha=0.1)
    ax.scatter(X_train[:,0], X_train[:,1], zdir='z', zs=min(X_train[:,2]), color='k', marker='.', alpha=0.1)
    X_minmax = np.round(np.min(X_train[:,0]),2), np.round(np.max(X_train[:,0]),2)
    Y_minmax = np.round(np.min(X_train[:,1]),2), np.round(np.max(X_train[:,1]),2)
    Z_minmax = np.round(np.min(X_train[:,2]),2), np.round(np.max(X_train[:,2]),2)
    ax.set(xlabel="PC1", ylabel="PC2", zlabel='PC3')
    X_range = np.round(np.arange(*X_minmax), 2)
    Y_range = np.round(np.arange(*Y_minmax), 2)
    Z_range = np.round(np.arange(*Z_minmax), 2)
    ax.set_xlim3d(X_minmax)
    ax.set_ylim3d(Y_minmax)
    ax.set_zlim3d(Z_minmax)
    ax.set(xticks=X_range, xticklabels=X_range, yticks=Y_range, yticklabels=Y_range, zticks=Z_range, zticklabels=Z_range)
    saveToNewDir(ImportedData.Title + "/", ImportedData.Title + ImportedData.Subtitle + "-Fit.png")
    plt.gcf().clear()
    plt.clf()

    return X_train, X_test, inverse_X
def plotInverse(ImportedData, algorithm, i_t):
    # pca = algorithm
    # scaler=StandardScaler()
    # X_train = copy.deepcopy(ImportedData.Data["xTrain"])
    # X_test = copy.deepcopy(ImportedData.Data["xTest"])
    # y_test = copy.deepcopy(ImportedData.Data["yTest"])
    y_train = copy.deepcopy(ImportedData.Data["yTrain"])
    # X_train = pd.DataFrame(preprocessing.StandardScaler().fit_transform(X_train))
    # X_train = X_train.astype('float16')
    # X_train = pca.fit_transform(X_train) 
    # i_t = pca.inverse_transform(X_train) 
    fig = plt.figure(figsize=(12,12))
    ax = fig.gca(projection='3d')
    ax.scatter(i_t[:,0], i_t[:,1], i_t[:,2], c=y_train, marker='.',alpha=0.7, s=100)
    ax.scatter(i_t[:,0], i_t[:,2], zdir='y', zs=max(i_t[:,1]), color='k', marker='.', alpha=0.1)
    ax.scatter(i_t[:,1], i_t[:,2], zdir='x', zs=min(i_t[:,0]), color='k', marker='.', alpha=0.1)
    ax.scatter(i_t[:,0], i_t[:,1], zdir='z', zs=min(i_t[:,2]), color='k', marker='.', alpha=0.1)
    X_minmax = np.round(np.min(i_t[:,0]),2), np.round(np.max(i_t[:,0]),2)
    Y_minmax = np.round(np.min(i_t[:,1]),2), np.round(np.max(i_t[:,1]),2)
    Z_minmax = np.round(np.min(i_t[:,2]),2), np.round(np.max(i_t[:,2]),2)
    ax.set(xlabel="PC1", ylabel="PC2", zlabel='PC3')
    X_range = np.round(np.arange(*X_minmax), 2)
    Y_range = np.round(np.arange(*Y_minmax), 2)
    Z_range = np.round(np.arange(*Z_minmax), 2)
    ax.set_xlim3d(X_minmax)
    ax.set_ylim3d(Y_minmax)
    ax.set_zlim3d(Z_minmax)
    ax.set(xticks=X_range, xticklabels=X_range, yticks=Y_range, yticklabels=Y_range, zticks=Z_range, zticklabels=Z_range)
    saveToNewDir(ImportedData.Title + "/", ImportedData.Title + ImportedData.Subtitle + "-Inverse.png")
    plt.gcf().clear()
    plt.clf()
def TestLayers(ImportedData):
    plt.figure(figsize=(7,7), dpi=70)

    train_scores = []
    test_scores = []
    cv_scores = []
    p_s = []
    r_s = []
    f_1 = []
    value = ImportedData.Data
    tRange = range(2,7,1)
    layer_sizes = ()
    for ix in tRange:
        layer_sizes += (5, )     
        mlp = MLPClassifier(hidden_layer_sizes=layer_sizes).fit(value['xTrain'], value['yTrain'])
        yPred = mlp.predict(value["xTest"])
        train_scores.append(accuracy_score(value['yTrain'], mlp.predict(value['xTrain']))*100)
        cv_scores.append(cross_val_score(mlp, value['xTrain'], value['yTrain'], cv=5).mean()*100)
        test_scores.append(accuracy_score(value['yTest'], yPred)*100)
        p_s.append(precision_score(value["yTest"], yPred, average='binary')*100)
        r_s.append(recall_score(value["yTest"], yPred, average='binary')*100)
        f_1.append(f1_score(value["yTest"], yPred, average='binary')*100 )
    
    plt.plot(tRange, train_scores, label="Train", lw = 2)
    plt.plot(tRange, test_scores, label="Test", lw = 2)
    plt.plot(tRange, cv_scores, label="CV", lw = 2)
    plt.plot(tRange, p_s, label='Precision', lw = 2)
    plt.plot(tRange, r_s, label='Recall', lw = 2)
    plt.plot(tRange, f_1, label='F1', lw = 2)
    plt.xlabel("Layers #")
    plt.ylabel("Accuracy (%)")
    plt.legend(loc='best')
    plt.title(ImportedData.Title + " (NeuralNet) Accuracy with # of layers")
    saveToNewDir(ImportedData.Title+"/", ImportedData.Title + ImportedData.Subtitle +"_NeuralNet_TrainingLayerNumber.png")
    plt.clf()
def TestNeurons(ImportedData):
    plt.figure(figsize=(7,7), dpi=70)

    train_scores = []
    test_scores = []
    cv_scores = []
    p_s = []
    r_s = []
    f_1 = []
    value = ImportedData.Data
    tRange = range(5,30,5)
    for ix in tRange:
        i = int(ix)
        layer_sizes = (i, i, i)   
        mlp = MLPClassifier(hidden_layer_sizes=layer_sizes).fit(value['xTrain'], value['yTrain'])
        yPred = mlp.predict(value["xTest"])
        train_scores.append(accuracy_score(value['yTrain'], mlp.predict(value['xTrain']))*100)
        cv_scores.append(cross_val_score(mlp, value['xTrain'], value['yTrain'], cv=5).mean()*100)
        test_scores.append(accuracy_score(value['yTest'], yPred)*100)
        p_s.append(precision_score(value["yTest"], yPred, average='binary')*100)
        r_s.append(recall_score(value["yTest"], yPred, average='binary')*100)
        f_1.append(f1_score(value["yTest"], yPred, average='binary')*100   )
    
    plt.plot(tRange, train_scores, label="Train", lw = 2)
    plt.plot(tRange, test_scores, label="Test", lw = 2)
    plt.plot(tRange, cv_scores, label="CV", lw = 2)
    plt.plot(tRange, p_s, label='Precision', lw = 2)
    plt.plot(tRange, r_s, label='Recall', lw = 2)
    plt.plot(tRange, f_1, label='F1', lw = 2)
    plt.xlabel("Neurons #")
    plt.ylabel("Accuracy (%)")
    plt.legend(loc='best')
    plt.title(ImportedData.Title + " (NeuralNet) Accuracy with # of neurons")
    saveToNewDir(ImportedData.Title+"/", ImportedData.Title + ImportedData.Subtitle +"_NeuralNet_TrainingNeuronsNumber.png")
    plt.clf()
def TestAlpha(ImportedData):
    plt.figure(figsize=(7,7), dpi=70)
    train_scores = []
    test_scores = []
    cv_scores = []
    p_s = []
    r_s = []
    f_1 = []
    value = ImportedData.Data
    tRange = np.geomspace(0.0001,0.001,num=5)
    
    for alpha in tRange:      
        mlp = MLPClassifier(alpha=alpha).fit(value['xTrain'], value['yTrain'])
        yPred = mlp.predict(value["xTest"])
        train_scores.append(accuracy_score(value['yTrain'], mlp.predict(value['xTrain']))*100)
        cv_scores.append(cross_val_score(mlp, value['xTrain'], value['yTrain'], cv=5).mean()*100)
        test_scores.append(accuracy_score(value['yTest'], yPred)*100)
        p_s.append(precision_score(value["yTest"], yPred, average='binary')*100)
        r_s.append(recall_score(value["yTest"], yPred, average='binary')*100)
        f_1.append(f1_score(value["yTest"], yPred, average='binary')*100  )

    plt.xticks(list(tRange))
    plt.plot(tRange, train_scores, label='Train', lw = 2)
    plt.plot(tRange, test_scores, label='Test', lw = 2)
    plt.plot(tRange, cv_scores, label='CV', lw = 2)
    plt.plot(tRange, p_s, label='Precision', lw = 2)
    plt.plot(tRange, r_s, label='Recall', lw = 2)
    plt.plot(tRange, f_1, label='F1', lw = 2)

    plt.xlabel('Alpha')
    plt.ylabel('Accuracy')
    plt.title(ImportedData.Title + ' (NeuralNet) Accuracy Vs. Alpha')
    plt.legend()
    saveToNewDir(ImportedData.Title+"/", ImportedData.Title + ImportedData.Subtitle +"_NeuralNet_Alpha.png")
    plt.gcf().clear()
    plt.clf()    







def PCABankRedux(ImportedData):

    def plotCVKurVarGraph(ImportedData):
        def make_patch_spines_invisible(ax):
            ax.set_frame_on(True)
            ax.patch.set_visible(False)
            for sp in ax.spines.values():
                sp.set_visible(False)   

        X_train = ImportedData.Data["xTrain"]
        components = np.shape(X_train)[1]
        pcaX = np.arange(1, components+1)
        cv_scores, mean_arr, var_arr, skew_arr, kurt_arr = [], [], [], [], []
        best = 0
        for n in pcaX:
            pca = PCA()
            pca.n_components = n
            rescaled = np.float32(X_train)
            reduced = pca.fit_transform(rescaled)
            s = np.shape(reduced)[0]
            mean = np.sum((reduced**1)/s) # Calculate the mean
            mean_arr.append(mean)
            var = np.sum((reduced-mean)**2)/s # Calculate the variance
            var_arr.append(var)
            skew = np.sum((reduced-mean)**3)/s # Calculate the skewness
            skew_arr.append(skew)
            kurt = np.sum((reduced-mean)**4)/s # Calculate the kurtosis
            kurt = kurt/(var**2)-3
            kurt_arr.append(kurt)
            score = np.mean(cross_val_score(pca, rescaled))
            cv_scores.append(score)
            if len(cv_scores)==0 or (score - cv_scores[-1]) > 0.01:
                best = n
            elif score - cv_scores[-1] < 0.01:
                best = n-1
        
        pca.n_components = components
        rescaled = np.float32(X_train)
        reduced = pca.fit_transform(rescaled)
        var_ratio = np.cumsum(pca.explained_variance_ratio_)

        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        ax3 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        ax1.plot(pcaX, cv_scores, 'b')
        #plt.plot(pcaX, mean_arr/max(mean_arr), 'm', label='Mean')
        ax3.plot(pcaX, var_arr, 'r')
        #plt.plot(pcaX, skew_arr/max(skew_arr), 'y', label='Skew')
        ax2.plot(pcaX, kurt_arr, 'k')

        ax1.set_xlabel('Components')
        
        ax2.spines["left"].set_position(("axes", -0.2)) # red one
        ax3.spines["left"].set_position(("axes", -0.4)) # red one

        make_patch_spines_invisible(ax2)
        make_patch_spines_invisible(ax3)

        ax2.spines["left"].set_visible(True)
        ax2.yaxis.set_label_position('left')
        ax2.yaxis.set_ticks_position('left')
        ax3.spines["left"].set_visible(True)
        ax3.yaxis.set_label_position('left')
        ax3.yaxis.set_ticks_position('left')

        ax1.set_ylabel('CV Score', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        ax2.set_ylabel('Kurtosis', color='k')
        ax2.tick_params(axis='y', labelcolor='k')
        ax3.set_ylabel('Variance', color='r')
        ax3.tick_params(axis='y', labelcolor='r')

        fig.legend([ax1, ax2, ax3],     # The line objects
           labels=['CV-Score', 'Kurtosis', 'Variance'],
           loc="center right")   # The labels for each line

        plt.xticks(pcaX, pcaX)
        plt.xlabel('nb of components')
        plt.ylabel('CV scores')
        plt.xlabel('number of components')
        plt.grid(linestyle='--', linewidth=0.5, alpha=0.15)
        saveToNewDir(ImportedData.Title + "/", ImportedData.Title + ImportedData.Subtitle + "-CVKurVarScores.png")
        plt.gcf().clear()
        plt.clf()
        print ("Best CV Score for {} components".format(best))
        return best
    
    def plotFit(ImportedData, algorithm=None):
        scaler=StandardScaler()
        X_train = ImportedData.Data["xTrain"]
        X_test = ImportedData.Data["xTest"]
        y_test = ImportedData.Data["yTest"]
        y_train = ImportedData.Data["yTrain"]
        X_train = pd.DataFrame(preprocessing.StandardScaler().fit_transform(X_train))
        X_train = X_train.astype('float16')
        
        if algorithm != None:
            X_train = algorithm.fit_transform(X_train)
            X_test = algorithm.transform(X_test)
        else:
            X_train = X_train.to_numpy() 

        LogReg=LogisticRegression(class_weight='balanced')  
        LogReg.fit(X_train,y_train)
        y_pred=LogReg.predict(X_test)

        string = ImportedData.Subtitle + ' Accuracy: {}\n'.format(accuracy_score(y_test,y_pred))
        string += ImportedData.Subtitle + ' F1 score: {}\n'.format(f1_score(y_test,y_pred,average='weighted'))   
        string += ImportedData.Subtitle + ' Recall: {}\n'.format(recall_score(y_test,y_pred,average='weighted'))   
        string += ImportedData.Subtitle + ' Precision: {}\n'.format(precision_score(y_test,y_pred,average='weighted'))  
        string += ImportedData.Subtitle + ' clasification report: \n{}\n'.format(classification_report(y_test,y_pred))  
        string += ImportedData.Subtitle + ' confussion matrix: \n{}\n'.format(confusion_matrix(y_test,y_pred))
        print (string)
        with open(ImportedData.Title + "/" + ImportedData.Title+".txt", 'a+') as f:
            f.write(string + "\n")

        fig = plt.figure(figsize=(12,12))
        ax = fig.gca(projection='3d')
        ax.scatter(X_train[:,0], X_train[:,1], X_train[:,2], c=y_train, marker='.',alpha=0.7, s=100)
        ax.scatter(X_train[:,0], X_train[:,2], zdir='y', zs=max(X_train[:,1]), color='k', marker='.', alpha=0.1)
        ax.scatter(X_train[:,1], X_train[:,2], zdir='x', zs=min(X_train[:,0]), color='k', marker='.', alpha=0.1)
        ax.scatter(X_train[:,0], X_train[:,1], zdir='z', zs=min(X_train[:,2]), color='k', marker='.', alpha=0.1)
        X_minmax = np.round(np.min(X_train[:,0]),2), np.round(np.max(X_train[:,0]),2)
        Y_minmax = np.round(np.min(X_train[:,1]),2), np.round(np.max(X_train[:,1]),2)
        Z_minmax = np.round(np.min(X_train[:,2]),2), np.round(np.max(X_train[:,2]),2)
        ax.set(xlabel="PC1", ylabel="PC2", zlabel='PC3')
        X_range = np.round(np.arange(*X_minmax), 2)
        Y_range = np.round(np.arange(*Y_minmax), 2)
        Z_range = np.round(np.arange(*Z_minmax), 2)
        ax.set_xlim3d(X_minmax)
        ax.set_ylim3d(Y_minmax)
        ax.set_zlim3d(Z_minmax)
        ax.set(xticks=X_range, xticklabels=X_range, yticks=Y_range, yticklabels=Y_range, zticks=Z_range, zticklabels=Z_range)
        saveToNewDir(ImportedData.Title + "/", ImportedData.Title + ImportedData.Subtitle + "-Fit.png")
        plt.gcf().clear()
        plt.clf()

        return X_train, X_test

    def plotInverse(ImportedData, algorithm):
        pca = algorithm
        X_train = ImportedData.Data["xTrain"]
        X_test = ImportedData.Data["xTest"]
        y_test = ImportedData.Data["yTest"]
        y_train = ImportedData.Data["yTrain"]
        X_train = np.float32(X_train)
        X_train = pca.fit_transform(X_train) 
        X_test = pca.transform(X_test)
        i_t = pca.inverse_transform(X_train) 
        fig = plt.figure(figsize=(12,12))
        ax = fig.gca(projection='3d')
        ax.scatter(i_t[:,0], i_t[:,1], i_t[:,2], c=y_train, marker='.',alpha=0.7, s=100)
        ax.scatter(i_t[:,0], i_t[:,2], zdir='y', zs=max(i_t[:,1]), color='k', marker='.', alpha=0.1)
        ax.scatter(i_t[:,1], i_t[:,2], zdir='x', zs=min(i_t[:,0]), color='k', marker='.', alpha=0.1)
        ax.scatter(i_t[:,0], i_t[:,1], zdir='z', zs=min(i_t[:,2]), color='k', marker='.', alpha=0.1)
        X_minmax = np.round(np.min(i_t[:,0]),2), np.round(np.max(i_t[:,0]),2)
        Y_minmax = np.round(np.min(i_t[:,1]),2), np.round(np.max(i_t[:,1]),2)
        Z_minmax = np.round(np.min(i_t[:,2]),2), np.round(np.max(i_t[:,2]),2)
        ax.set(xlabel="PC1", ylabel="PC2", zlabel='PC3')
        X_range = np.round(np.arange(*X_minmax), 2)
        Y_range = np.round(np.arange(*Y_minmax), 2)
        Z_range = np.round(np.arange(*Z_minmax), 2)
        ax.set_xlim3d(X_minmax)
        ax.set_ylim3d(Y_minmax)
        ax.set_zlim3d(Z_minmax)
        ax.set(xticks=X_range, xticklabels=X_range, yticks=Y_range, yticklabels=Y_range, zticks=Z_range, zticklabels=Z_range)
        saveToNewDir(ImportedData.Title + "/", ImportedData.Title + ImportedData.Subtitle + "-Inverse.png")
        plt.gcf().clear()
        plt.clf()

    # Test different layers
    def TestLayers(ImportedData):
        plt.figure(figsize=(7,7), dpi=70)

        train_scores = []
        test_scores = []
        cv_scores = []
        p_s = []
        r_s = []
        f_1 = []
        value = ImportedData.Data
        tRange = range(2,12,1)
        layer_sizes = ()
        for ix in tRange:
            layer_sizes += (5, )     
            mlp = MLPClassifier(hidden_layer_sizes=layer_sizes).fit(value['xTrain'], value['yTrain'])
            yPred = mlp.predict(value["xTest"])
            train_scores.append(accuracy_score(value['yTrain'], mlp.predict(value['xTrain']))*100)
            cv_scores.append(cross_val_score(mlp, value['xTrain'], value['yTrain'], cv=5).mean()*100)
            test_scores.append(accuracy_score(value['yTest'], yPred)*100)
            p_s.append(precision_score(value["yTest"], yPred, average='binary')*100)
            r_s.append(recall_score(value["yTest"], yPred, average='binary')*100)
            f_1.append(f1_score(value["yTest"], yPred, average='binary')*100 )
        
        plt.plot(tRange, train_scores, label="Train", lw = 2)
        plt.plot(tRange, test_scores, label="Test", lw = 2)
        plt.plot(tRange, cv_scores, label="CV", lw = 2)
        plt.plot(tRange, p_s, label='Precision', lw = 2)
        plt.plot(tRange, r_s, label='Recall', lw = 2)
        plt.plot(tRange, f_1, label='F1', lw = 2)
        plt.xlabel("Layers #")
        plt.ylabel("Accuracy (%)")
        plt.legend(loc='best')
        plt.title(ImportedData.Title + " (NeuralNet) Accuracy with # of layers")
        saveToNewDir(ImportedData.Title+"/", ImportedData.Title + ImportedData.Subtitle +"_NeuralNet_TrainingLayerNumber.png")
        plt.clf()
    # Test different neurons
    def TestNeurons(ImportedData):
        plt.figure(figsize=(7,7), dpi=70)

        train_scores = []
        test_scores = []
        cv_scores = []
        p_s = []
        r_s = []
        f_1 = []
        value = ImportedData.Data
        tRange = range(5,55,5)
        for ix in tRange:
            i = int(ix)
            layer_sizes = (i, i, i)   
            mlp = MLPClassifier(hidden_layer_sizes=layer_sizes).fit(value['xTrain'], value['yTrain'])
            yPred = mlp.predict(value["xTest"])
            train_scores.append(accuracy_score(value['yTrain'], mlp.predict(value['xTrain']))*100)
            cv_scores.append(cross_val_score(mlp, value['xTrain'], value['yTrain'], cv=5).mean()*100)
            test_scores.append(accuracy_score(value['yTest'], yPred)*100)
            p_s.append(precision_score(value["yTest"], yPred, average='binary')*100)
            r_s.append(recall_score(value["yTest"], yPred, average='binary')*100)
            f_1.append(f1_score(value["yTest"], yPred, average='binary')*100   )
        
        plt.plot(tRange, train_scores, label="Train", lw = 2)
        plt.plot(tRange, test_scores, label="Test", lw = 2)
        plt.plot(tRange, cv_scores, label="CV", lw = 2)
        plt.plot(tRange, p_s, label='Precision', lw = 2)
        plt.plot(tRange, r_s, label='Recall', lw = 2)
        plt.plot(tRange, f_1, label='F1', lw = 2)
        plt.xlabel("Neurons #")
        plt.ylabel("Accuracy (%)")
        plt.legend(loc='best')
        plt.title(ImportedData.Title + " (NeuralNet) Accuracy with # of neurons")
        saveToNewDir(ImportedData.Title+"/", ImportedData.Title + ImportedData.Subtitle +"_NeuralNet_TrainingNeuronsNumber.png")
        plt.clf()
    # Test different Alpha Values
    def TestAlpha(ImportedData):
        plt.figure(figsize=(7,7), dpi=70)
        train_scores = []
        test_scores = []
        cv_scores = []
        p_s = []
        r_s = []
        f_1 = []
        value = ImportedData.Data
        tRange = np.geomspace(0.0001,0.01,num=10)
        
        for alpha in tRange:      
            mlp = MLPClassifier(alpha=alpha).fit(value['xTrain'], value['yTrain'])
            yPred = mlp.predict(value["xTest"])
            train_scores.append(accuracy_score(value['yTrain'], mlp.predict(value['xTrain']))*100)
            cv_scores.append(cross_val_score(mlp, value['xTrain'], value['yTrain'], cv=5).mean()*100)
            test_scores.append(accuracy_score(value['yTest'], yPred)*100)
            p_s.append(precision_score(value["yTest"], yPred, average='binary')*100)
            r_s.append(recall_score(value["yTest"], yPred, average='binary')*100)
            f_1.append(f1_score(value["yTest"], yPred, average='binary')*100  )

        plt.xticks(list(tRange))
        plt.plot(tRange, train_scores, label='Train', lw = 2)
        plt.plot(tRange, test_scores, label='Test', lw = 2)
        plt.plot(tRange, cv_scores, label='CV', lw = 2)
        plt.plot(tRange, p_s, label='Precision', lw = 2)
        plt.plot(tRange, r_s, label='Recall', lw = 2)
        plt.plot(tRange, f_1, label='F1', lw = 2)

        plt.xlabel('Alpha')
        plt.ylabel('Accuracy')
        plt.title(ImportedData.Title + ' (NeuralNet) Accuracy Vs. Alpha')
        plt.legend()
        saveToNewDir(ImportedData.Title+"/", ImportedData.Title + ImportedData.Subtitle +"_NeuralNet_Alpha.png")
        plt.gcf().clear()
        plt.clf()    
    

    RETAINED_VARIANCE = 0.95

    ### Banknote Data
    scaler = StandardScaler()
    ImportedData.Title = "BanknoteAuthentication"
    banknote = pd.read_csv('./data_banknote_authentication.data', header=None)
    banknote.columns = ['Wavelet_Variance', 'Wavelet_Skewness', 'Wavelet_Curtosis', 'Image_Entropy', 'Class']
    Y = banknote['Class']
    X = banknote.drop(['Class'], axis=1)
    df = pd.DataFrame(X, columns=banknote.columns.drop(['Class']))
    Ts = train_test_split(df, Y, train_size = 0.8)
    ImportedData.Data["xTrain"], ImportedData.Data["xTest"], ImportedData.Data["yTrain"], ImportedData.Data["yTest"] = Ts
    ImportedData.Data["xTrain"] = pd.DataFrame(scaler.fit_transform(ImportedData.Data["xTrain"]), columns = df.columns)
    ImportedData.Data["xTest"] = pd.DataFrame(scaler.fit_transform(ImportedData.Data["xTest"]), columns = df.columns)

    pcaData = ImportedData
    icaData = ImportedData
    rpData = ImportedData
    ftData = ImportedData
    
    # print ("-----> plotFit Original Banknote Data <-----")
    # ImportedData.Subtitle = "Original"
    # X_train, X_test = plotFit(ImportedData)
    # ImportedData.Subtitle = "Original"
    # ImportedData.Data["xTrain"] = X_train.astype('double')
    # ImportedData.Data["xTest"] = X_test.astype('double')
    # bestNComponents = plotCVKurVarGraph(ImportedData)
    # ImportedData.Data["xTrain"] = scaler.fit_transform(ImportedData.Data["xTrain"])
    # ImportedData.Data["xTest"] = scaler.fit_transform(ImportedData.Data["xTest"])
    # print ("-----> KMAnalysis Original Banknote Data <-----")
    # KMAnalysis(ImportedData, ImportedData.Data)
    # print ("-----> EMAnalysis Original Banknote Data <-----")
    # EMAnalysis(ImportedData, ImportedData.Data)

    bestNComponents = 3

    pcaData.Subtitle = "PCA"
    print ("-----> plot Variance+CV PCA Banknote Data <-----")
    print ("-----> plotFit PCA Banknote Data {} components<-----".format(bestNComponents))
    pcaX_train, pcaX_test = plotFit(pcaData, PCA(n_components=bestNComponents))
    plotCVKurVarGraph(pcaData)
    print ("-----> plotInverse PCA Banknote Data <-----")
    plotInverse(pcaData, PCA(n_components=bestNComponents))
    pcaData.Data["xTrain"] = pcaX_train.astype('double')
    pcaData.Data["xTest"] = pcaX_test.astype('double')
    pcaData.Data["xTrain"] = scaler.fit_transform(pcaData.Data["xTrain"])
    pcaData.Data["xTest"] = scaler.fit_transform(pcaData.Data["xTest"])
    print ("-----> Training NN <-----")
    TestLayers(pcaData)
    TestNeurons(pcaData)
    TestAlpha(pcaData)
    print ("-----> KMAnalysis PCA Banknote Data <-----")
    pcaDataKM = KMAnalysis(pcaData, pcaData.Data)
    print ("-----> Training NN <-----")
    TestLayers(pcaDataKM)
    TestNeurons(pcaDataKM)
    TestAlpha(pcaDataKM)
    print ("-----> EMAnalysis PCA Banknote Data <-----")
    pcaDataEM = EMAnalysis(pcaData, pcaData.Data)
    print ("-----> Training NN <-----")
    TestLayers(pcaDataEM)
    TestNeurons(pcaDataEM)
    TestAlpha(pcaDataEM)

    rpData.Subtitle = "RP"
    print ("-----> plotFit RP Banknote Data {} components<-----".format(bestNComponents))
    rpX_train, rpX_test = plotFit(rpData, GaussianRandomProjection(n_components=bestNComponents))
    plotCVKurVarGraph(rpData)
    rpData.Data["xTrain"] = rpX_train.astype('double')
    rpData.Data["xTest"] = rpX_test.astype('double')
    rpData.Data["xTrain"] = scaler.fit_transform(rpData.Data["xTrain"])
    rpData.Data["xTest"] = scaler.fit_transform(rpData.Data["xTest"])
    print ("-----> Training NN <-----")
    TestLayers(rpData)
    TestNeurons(rpData)
    TestAlpha(rpData)
    print ("-----> KMAnalysis RP Banknote Data <-----")
    rpDataKM = KMAnalysis(rpData, rpData.Data)
    print ("-----> Training NN <-----")
    TestLayers(rpDataKM)
    TestNeurons(rpDataKM)
    TestAlpha(rpDataKM)
    print ("-----> EMAnalysis RP Banknote Data <-----")
    rpDataEM = EMAnalysis(rpData, rpData.Data)
    print ("-----> Training NN <-----")
    TestLayers(rpDataEM)
    TestNeurons(rpDataEM)
    TestAlpha(rpDataEM)

    ftData.Subtitle = "FT"
    print ("-----> plotFit FA Banknote Data {} components <-----".format(bestNComponents))
    ftX_train, ftX_test = plotFit(ftData, FactorAnalysis(n_components=bestNComponents))
    plotCVKurVarGraph(ftData)
    ftData.Data["xTrain"] = ftX_train.astype('double')
    ftData.Data["xTest"] = ftX_test.astype('double')
    ftData.Data["xTrain"] = scaler.fit_transform(ftData.Data["xTrain"])
    ftData.Data["xTest"] = scaler.fit_transform(ftData.Data["xTest"])
    print ("-----> Training NN <-----")
    TestLayers(ftData)
    TestNeurons(ftData)
    TestAlpha(ftData)
    print ("-----> KMAnalysis FT Banknote Data <-----")
    ftDataKM = KMAnalysis(ftData, ftData.Data)
    print ("-----> Training NN <-----")
    TestLayers(ftDataKM)
    TestNeurons(ftDataKM)
    TestAlpha(ftDataKM)
    print ("-----> EMAnalysis FT Banknote Data <-----")
    ftDataEM = EMAnalysis(ftData, ftData.Data)
    print ("-----> Training NN <-----")
    TestLayers(ftDataEM)
    TestNeurons(ftDataEM)
    TestAlpha(ftDataEM)

    icaData.Subtitle = "ICA"
    print ("-----> plotFit ICA Banknote Data {} components <-----".format(bestNComponents))
    icaX_train, icaX_test = plotFit(icaData, FastICA(n_components=bestNComponents))
    plotCVKurVarGraph(icaData)
    print ("-----> plotInverse ICA Banknote Data <-----")
    plotInverse(ImportedData, FastICA(n_components=bestNComponents))
    icaData.Data["xTrain"] = icaX_train.astype('double')
    icaData.Data["xTest"] = icaX_test.astype('double')
    icaData.Data["xTrain"] = scaler.fit_transform(icaData.Data["xTrain"])
    icaData.Data["xTest"] = scaler.fit_transform(icaData.Data["xTest"])
    print ("-----> Training NN <-----")
    TestLayers(icaData)
    TestNeurons(icaData)
    TestAlpha(icaData)
    print ("-----> KMAnalysis ICA Banknote Data <-----")
    icaDataKM = KMAnalysis(icaData, icaData.Data)
    print ("-----> Training NN <-----")
    TestLayers(icaDataKM)
    TestNeurons(icaDataKM)
    TestAlpha(icaDataKM)
    print ("-----> EMAnalysis ICA Banknote Data <-----")
    icaDataEM = EMAnalysis(icaData, icaData.Data)
    print ("-----> Training NN <-----")
    TestLayers(fticaDataEMData)
    TestNeurons(icaDataEM)
    TestAlpha(icaDataEM)

    ### Letter Data

    # scaler = StandardScaler()
    # ImportedData.Title = "LetterRecognition"
    # letter = pd.read_csv('./letter-recognition.data', header=None)
    # letter.columns = ['Letter', 'xBoxHPos', 'yBoxVPos', 'BoxW', 'BoxH', 'OnPix', 'xBarMean', 'yBarMean', 'x2BarMean', 'y2BarMean', 'xyBarMean', 'x2ybrMean', 'xy2BrMean','xEgeMean', 'xegvyCorr', 'y-egeMean', 'yegvxCorr']
    # Y = [ord(i) for i in letter['Letter']]
    # X = letter.drop(['Letter'], axis=1)
    # df = pd.DataFrame(X, columns=letter.columns.drop(['Letter']))
    # Ts = train_test_split(df, Y, train_size = 0.8)
    # ImportedData.Data["xTrain"], ImportedData.Data["xTest"], ImportedData.Data["yTrain"], ImportedData.Data["yTest"] = Ts
    # ImportedData.Data["xTrain"] = pd.DataFrame(scaler.fit_transform(ImportedData.Data["xTrain"]), columns = df.columns)
    # ImportedData.Data["xTest"] = pd.DataFrame(scaler.fit_transform(ImportedData.Data["xTest"]), columns = df.columns)

    # pcaData = ImportedData
    # icaData = ImportedData
    # rpData = ImportedData
    # ftData = ImportedData

    # print ("-----> plotFit Original Letter Data <-----")
    # ImportedData.Subtitle = "Original"
    # X_train, X_test = plotFit(ImportedData)
    # ImportedData.Data["xTrain"] = X_train.astype('double')
    # ImportedData.Data["xTest"] = X_test.astype('double')
    # bestNComponents = plotCVKurVarGraph(ImportedData)
    # ImportedData.Data["xTrain"] = scaler.fit_transform(ImportedData.Data["xTrain"])
    # ImportedData.Data["xTest"] = scaler.fit_transform(ImportedData.Data["xTest"])
    # print ("-----> KMAnalysis Original Letter Data <-----")
    # KMAnalysis(ImportedData, ImportedData.Data)
    # print ("-----> EMAnalysis Original Letter Data <-----")
    # EMAnalysis(ImportedData, ImportedData.Data)

    # bestNComponents = 11

    # pcaData.Subtitle = "PCA"
    # print ("-----> plotFit PCA Letter Data {} components<-----".format(bestNComponents))
    # pcaX_train, pcaX_test = plotFit(pcaData, PCA(n_components=bestNComponents))
    # plotCVKurVarGraph(pcaData)
    # print ("-----> plotInverse PCA Letter Data <-----")
    # plotInverse(pcaData, PCA(n_components=bestNComponents))
    # pcaData.Data["xTrain"] = pcaX_train.astype('double')
    # pcaData.Data["xTest"] = pcaX_test.astype('double')
    # pcaData.Data["xTrain"] = scaler.fit_transform(pcaData.Data["xTrain"])
    # pcaData.Data["xTest"] = scaler.fit_transform(pcaData.Data["xTest"])
    # print ("-----> KMAnalysis PCA Letter Data <-----")
    # KMAnalysis(pcaData, pcaData.Data)
    # print ("-----> EMAnalysis PCA Letter Data <-----")
    # EMAnalysis(pcaData, pcaData.Data)

    # rpData.Subtitle = "RP"
    # print ("-----> plotFit RP Letter Data {} components<-----".format(bestNComponents))
    # rpX_train, rpX_test = plotFit(rpData, GaussianRandomProjection(n_components=bestNComponents))
    # plotCVKurVarGraph(rpData)
    # rpData.Data["xTrain"] = rpX_train.astype('double')
    # rpData.Data["xTest"] = rpX_test.astype('double')
    # rpData.Data["xTrain"] = scaler.fit_transform(rpData.Data["xTrain"])
    # rpData.Data["xTest"] = scaler.fit_transform(rpData.Data["xTest"])
    # print ("-----> KMAnalysis RP Letter Data <-----")
    # KMAnalysis(rpData, rpData.Data)
    # print ("-----> EMAnalysis RP Letter Data <-----")
    # EMAnalysis(rpData, rpData.Data)

    # ftData.Subtitle = "FT"
    # print ("-----> plotFit FA Letter Data {} components <-----".format(bestNComponents))
    # ftX_train, ftX_test = plotFit(ftData, FactorAnalysis(n_components=bestNComponents))
    # plotCVKurVarGraph(ftData)
    # ftData.Data["xTrain"] = ftX_train.astype('double')
    # ftData.Data["xTest"] = ftX_test.astype('double')
    # ftData.Data["xTrain"] = scaler.fit_transform(ftData.Data["xTrain"])
    # ftData.Data["xTest"] = scaler.fit_transform(ftData.Data["xTest"])
    # print ("-----> KMAnalysis FT Letter Data <-----")
    # KMAnalysis(ftData, ftData.Data)
    # print ("-----> EMAnalysis FT Letter Data <-----")
    # EMAnalysis(ftData, ftData.Data)

    # icaData.Subtitle = "ICA"
    # print ("-----> plotFit ICA Letter Data {} components <-----".format(bestNComponents))
    # icaX_train, icaX_test = plotFit(icaData, FastICA(n_components=bestNComponents))
    # plotCVKurVarGraph(icaData)
    # print ("-----> plotInverse ICA Letter Data <-----")
    # plotInverse(ImportedData, FastICA(n_components=bestNComponents))
    # icaData.Data["xTrain"] = icaX_train.astype('double')
    # icaData.Data["xTest"] = icaX_test.astype('double')
    # icaData.Data["xTrain"] = scaler.fit_transform(icaData.Data["xTrain"])
    # icaData.Data["xTest"] = scaler.fit_transform(icaData.Data["xTest"])
    # print ("-----> KMAnalysis ICA Letter Data <-----")
    # KMAnalysis(icaData, icaData.Data)
    # print ("-----> EMAnalysis ICA Letter Data <-----")
    # EMAnalysis(icaData, icaData.Data)

def PCALetterRedux(ImportedData):
    scaler = StandardScaler()
    ImportedData.Title = "LetterRecognition"
    letter = pd.read_csv('./letter-recognition.data', header=None)
    letter.columns = ['Letter', 'xBoxHPos', 'yBoxVPos', 'BoxW', 'BoxH', 'OnPix', 'xBarMean', 'yBarMean', 'x2BarMean', 'y2BarMean', 'xyBarMean', 'x2ybrMean', 'xy2BrMean','xEgeMean', 'xegvyCorr', 'y-egeMean', 'yegvxCorr']
    Y = [ord(i) for i in letter['Letter']]
    X = letter.drop(['Letter'], axis=1)
    df = pd.DataFrame(X, columns=letter.columns.drop(['Letter']))
    Ts = train_test_split(df, Y, train_size = 0.8)
    ImportedData.Data["xTrain"], ImportedData.Data["xTest"], ImportedData.Data["yTrain"], ImportedData.Data["yTest"] = Ts
    
    pcaBank = PCA(n_components=0.999)
    rescaled = np.float32(ImportedData.Data["xTrain"])
    reduced = pcaBank.fit_transform(rescaled)
    pcaY = np.cumsum(pcaBank.explained_variance_ratio_)
    varLen = len(pcaBank.explained_variance_ratio_)+1
    pcaX = range(1, varLen)
    plt.plot(pcaX, pcaY)
    plt.xticks(np.round(np.arange(1, varLen),0), np.round(np.arange(1, varLen),0))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.grid(linestyle='--', linewidth=0.5, alpha=0.15)
    saveToNewDir(ImportedData.Title + "/", ImportedData.Title + "-PCAVariance.png")
    plt.gcf().clear()
    plt.clf()

    pca = PCA(n_components=0.95)
    X_train = ImportedData.Data["xTrain"]
    X_test = ImportedData.Data["xTest"]
    y_test = ImportedData.Data["yTest"]
    y_train = ImportedData.Data["yTrain"]
    X_train = np.float32(X_train)
    X_train = pca.fit_transform(X_train) 
    X_test = pca.transform(X_test)
    LogReg=LogisticRegression(class_weight='balanced')  
    LogReg.fit(X_train,y_train)
    y_pred=LogReg.predict(X_test)
    print('Accuracy:',accuracy_score(y_test,y_pred))  
    print('F1 score:',f1_score(y_test,y_pred,average='weighted'))   
    print('Recall:',recall_score(y_test,y_pred,average='weighted'))   
    print('Precision:',precision_score(y_test,y_pred,average='weighted'))  
    print('\n clasification report:\n',classification_report(y_test,y_pred))  
    print('\n confussion matrix:\n',confusion_matrix(y_test,y_pred))
    fig = plt.figure(figsize=(12,12))
    ax = fig.gca(projection='3d')
    ax.scatter(X_train[:,0], X_train[:,1], X_train[:,2], c=y_train, marker='.',alpha=0.7, s=100)
    ax.scatter(X_train[:,0], X_train[:,2], zdir='y', zs=max(X_train[:,1]), color='k', marker='.', alpha=0.1)
    ax.scatter(X_train[:,1], X_train[:,2], zdir='x', zs=min(X_train[:,0]), color='k', marker='.', alpha=0.1)
    ax.scatter(X_train[:,0], X_train[:,1], zdir='z', zs=min(X_train[:,2]), color='k', marker='.', alpha=0.1)
    X_minmax = np.round(np.min(X_train[:,0]),2), np.round(np.max(X_train[:,0]),2)
    Y_minmax = np.round(np.min(X_train[:,1]),2), np.round(np.max(X_train[:,1]),2)
    Z_minmax = np.round(np.min(X_train[:,2]),2), np.round(np.max(X_train[:,2]),2)
    ax.set(xlabel="PC1", ylabel="PC2", zlabel='PC3')
    X_range = np.round(np.arange(*X_minmax), 2)
    Y_range = np.round(np.arange(*Y_minmax), 2)
    Z_range = np.round(np.arange(*Z_minmax), 2)
    ax.set_xlim3d(X_minmax)
    ax.set_ylim3d(Y_minmax)
    ax.set_zlim3d(Z_minmax)
    ax.set(xticks=X_range, xticklabels=X_range, yticks=Y_range, yticklabels=Y_range, zticks=Z_range, zticklabels=Z_range)
    saveToNewDir(ImportedData.Title + "/", ImportedData.Title + "-PCAreducedFit.png")
    plt.gcf().clear()
    plt.clf()
    i_t = pca.inverse_transform(X_train) 
    fig = plt.figure(figsize=(12,12))
    ax = fig.gca(projection='3d')
    ax.scatter(i_t[:,0], i_t[:,1], i_t[:,2], c=y_train, marker='.',alpha=0.7, s=100)
    ax.scatter(i_t[:,0], i_t[:,2], zdir='y', zs=max(i_t[:,1]), color='k', marker='.', alpha=0.1)
    ax.scatter(i_t[:,1], i_t[:,2], zdir='x', zs=min(i_t[:,0]), color='k', marker='.', alpha=0.1)
    ax.scatter(i_t[:,0], i_t[:,1], zdir='z', zs=min(i_t[:,2]), color='k', marker='.', alpha=0.1)
    X_minmax = np.round(np.min(i_t[:,0]),2), np.round(np.max(i_t[:,0]),2)
    Y_minmax = np.round(np.min(i_t[:,1]),2), np.round(np.max(i_t[:,1]),2)
    Z_minmax = np.round(np.min(i_t[:,2]),2), np.round(np.max(i_t[:,2]),2)
    ax.set(xlabel="PC1", ylabel="PC2", zlabel='PC3')
    X_range = np.round(np.arange(*X_minmax), 2)
    Y_range = np.round(np.arange(*Y_minmax), 2)
    Z_range = np.round(np.arange(*Z_minmax), 2)
    ax.set_xlim3d(X_minmax)
    ax.set_ylim3d(Y_minmax)
    ax.set_zlim3d(Z_minmax)
    ax.set(xticks=X_range, xticklabels=X_range, yticks=Y_range, yticklabels=Y_range, zticks=Z_range, zticklabels=Z_range)
    saveToNewDir(ImportedData.Title + "/", ImportedData.Title + "-PCAreducedInverse.png")
    plt.gcf().clear()
    plt.clf()

    ImportedData.Title = "LetterRecognition"
    ImportedData.Subtitle = "PCA"
    
    ImportedData.Data["xTrain"] = X_train.astype('double')
    ImportedData.Data["xTest"] = X_test.astype('double')

    KMAnalysis(ImportedData, ImportedData.Data)
    EMAnalysis(ImportedData, ImportedData.Data)
    # a = Process(target=KMAnalysis, args=(ImportedData, ImportedData.Data,))
    # b = Process(target=EMAnalysis, args=(ImportedData, ImportedData.Data,))
    # a.start()
    # b.start()
    # a.join()
    # b.join()
    # return ImportedData

def ICABankRedux(ImportedData):
    scaler = StandardScaler()
    ImportedData.Title = "BanknoteAuthentication"
    banknote = pd.read_csv('./data_banknote_authentication.data', header=None)
    banknote.columns = ['Wavelet_Variance', 'Wavelet_Skewness', 'Wavelet_Curtosis', 'Image_Entropy', 'Class']
    Y = banknote['Class']
    X = banknote.drop(['Class'], axis=1)
    df = pd.DataFrame(X, columns=banknote.columns.drop(['Class']))
    rescaled = pd.DataFrame(preprocessing.StandardScaler().fit_transform(df), columns = df.columns)
    
    icaBank = FastICA(n_components=3)
    rescaled = np.float32(rescaled)
    reduced = icaBank.fit_transform(rescaled)

    fig = plt.figure(figsize=(12,12))
    ax = fig.gca(projection='3d')
    ax.scatter(reduced[:,0], reduced[:,1], reduced[:,2], c=Y, marker='.',alpha=0.7, s=100)
    ax.scatter(reduced[:,0], reduced[:,2], zdir='y', zs=max(reduced[:,1]), color='k', marker='.', alpha=0.1)
    ax.scatter(reduced[:,1], reduced[:,2], zdir='x', zs=min(reduced[:,0]), color='k', marker='.', alpha=0.1)
    ax.scatter(reduced[:,0], reduced[:,1], zdir='z', zs=min(reduced[:,2]), color='k', marker='.', alpha=0.1)
    X_minmax = np.round(np.min(reduced[:,0]),2), np.round(np.max(reduced[:,0]),2)
    Y_minmax = np.round(np.min(reduced[:,1]),2), np.round(np.max(reduced[:,1]),2)
    Z_minmax = np.round(np.min(reduced[:,2]),2), np.round(np.max(reduced[:,2]),2)
    ax.set(xlabel="PC1", ylabel="PC2", zlabel='PC3')
    X_range = np.round(np.arange(*X_minmax), 2)
    Y_range = np.round(np.arange(*Y_minmax), 2)
    Z_range = np.round(np.arange(*Z_minmax), 2)
    ax.set_xlim3d(X_minmax)
    ax.set_ylim3d(Y_minmax)
    ax.set_zlim3d(Z_minmax)
    ax.set(xticks=X_range, xticklabels=X_range, yticks=Y_range, yticklabels=Y_range, zticks=Z_range, zticklabels=Z_range)
    #ax.view_init(elev=45., azim=-45)
    saveToNewDir(ImportedData.Title + "/", ImportedData.Title + "-ICAreducedFT.png")
    plt.gcf().clear()
    plt.clf()

    i_t = icaBank.inverse_transform(reduced)
    fig = plt.figure(figsize=(12,12))
    ax = fig.gca(projection='3d')
    ax.scatter(i_t[:,0], i_t[:,1], i_t[:,2], c=Y, marker='.',alpha=0.7, s=100)
    ax.scatter(i_t[:,0], i_t[:,2], zdir='y', zs=max(i_t[:,1]), color='k', marker='.', alpha=0.1)
    ax.scatter(i_t[:,1], i_t[:,2], zdir='x', zs=min(i_t[:,0]), color='k', marker='.', alpha=0.1)
    ax.scatter(i_t[:,0], i_t[:,1], zdir='z', zs=min(i_t[:,2]), color='k', marker='.', alpha=0.1)
    X_minmax = np.round(np.min(i_t[:,0]),2), np.round(np.max(i_t[:,0]),2)
    Y_minmax = np.round(np.min(i_t[:,1]),2), np.round(np.max(i_t[:,1]),2)
    Z_minmax = np.round(np.min(i_t[:,2]),2), np.round(np.max(i_t[:,2]),2)
    ax.set(xlabel="PC1", ylabel="PC2", zlabel='PC3')
    X_range = np.round(np.arange(*X_minmax), 2)
    Y_range = np.round(np.arange(*Y_minmax), 2)
    Z_range = np.round(np.arange(*Z_minmax), 2)
    ax.set_xlim3d(X_minmax)
    ax.set_ylim3d(Y_minmax)
    ax.set_zlim3d(Z_minmax)
    ax.set(xticks=X_range, xticklabels=X_range, yticks=Y_range, yticklabels=Y_range, zticks=Z_range, zticklabels=Z_range)
    #ax.view_init(elev=45., azim=-45)
    saveToNewDir(ImportedData.Title + "/", ImportedData.Title + "-ICAreducedIT.png")
    plt.gcf().clear()
    plt.clf()

    ImportedData.Title = "BanknoteAuthentication"
    ImportedData.Subtitle = "ICA"

    df_reduced = preprocessing.MinMaxScaler().fit_transform(reduced)
    df_reduced = pd.DataFrame(reduced)
    Ts = train_test_split(df_reduced, Y, train_size = 0.8)
    ImportedData.Data["xTrain"], ImportedData.Data["xTest"], ImportedData.Data["yTrain"], ImportedData.Data["yTest"] = Ts
    ImportedData.Data["xTrain"] = scaler.fit_transform(ImportedData.Data["xTrain"])
    ImportedData.Data["xTest"] = scaler.fit_transform(ImportedData.Data["xTest"])

    KMAnalysis(ImportedData, ImportedData.Data)
    EMAnalysis(ImportedData, ImportedData.Data)
    # a = Process(target=KMAnalysis, args=(ImportedData, ImportedData.Data,))
    # b = Process(target=EMAnalysis, args=(ImportedData, ImportedData.Data,))
    # a.start()
    # b.start()
    # a.join()
    # b.join()
    # return ImportedData
    
def ICALetterRedux(ImportedData):
    scaler = StandardScaler()
    ImportedData.Title = "LetterRecognition"
    letter = pd.read_csv('./letter-recognition.data', header=None)
    letter.columns = ['Letter', 'xBoxHPos', 'yBoxVPos', 'BoxW', 'BoxH', 'OnPix', 'xBarMean', 'yBarMean', 'x2BarMean', 'y2BarMean', 'xyBarMean', 'x2ybrMean', 'xy2BrMean','xEgeMean', 'xegvyCorr', 'y-egeMean', 'yegvxCorr']
    Y = [ord(i) for i in letter['Letter']]
    X = letter.drop(['Letter'], axis=1)
    df = pd.DataFrame(X, columns=letter.columns.drop(['Letter']))
    rescaled = pd.DataFrame(preprocessing.StandardScaler().fit_transform(df), columns = df.columns)

    icaLetter = FastICA(n_components=10)
    rescaled = np.float32(rescaled)
    reduced = icaLetter.fit_transform(rescaled)

    fig = plt.figure(figsize=(12,12))
    ax = fig.gca(projection='3d')
    ax.scatter(reduced[:,0], reduced[:,1], reduced[:,2], c=Y, marker='.',alpha=0.7, s=100)
    ax.scatter(reduced[:,0], reduced[:,2], zdir='y', zs=max(reduced[:,1]), color='k', marker='.', alpha=0.1)
    ax.scatter(reduced[:,1], reduced[:,2], zdir='x', zs=min(reduced[:,0]), color='k', marker='.', alpha=0.1)
    ax.scatter(reduced[:,0], reduced[:,1], zdir='z', zs=min(reduced[:,2]), color='k', marker='.', alpha=0.1)
    X_minmax = np.round(np.min(reduced[:,0]),2), np.round(np.max(reduced[:,0]),2)
    Y_minmax = np.round(np.min(reduced[:,1]),2), np.round(np.max(reduced[:,1]),2)
    Z_minmax = np.round(np.min(reduced[:,2]),2), np.round(np.max(reduced[:,2]),2)
    ax.set(xlabel="PC1", ylabel="PC2", zlabel='PC3')
    X_range = np.round(np.arange(*X_minmax), 2)
    Y_range = np.round(np.arange(*Y_minmax), 2)
    Z_range = np.round(np.arange(*Z_minmax), 2)
    ax.set_xlim3d(X_minmax)
    ax.set_ylim3d(Y_minmax)
    ax.set_zlim3d(Z_minmax)
    ax.set(xticks=X_range, xticklabels=X_range, yticks=Y_range, yticklabels=Y_range, zticks=Z_range, zticklabels=Z_range)
    #ax.view_init(elev=45., azim=-45)
    saveToNewDir(ImportedData.Title + "/", ImportedData.Title + "-ICAreducedFT.png")
    plt.gcf().clear()
    plt.clf()
  
    i_t = icaLetter.inverse_transform(reduced)
    fig = plt.figure(figsize=(12,12))
    ax = fig.gca(projection='3d')
    ax.scatter(i_t[:,0], i_t[:,1], i_t[:,2], c=Y, marker='.',alpha=0.7, s=100)
    ax.scatter(i_t[:,0], i_t[:,2], zdir='y', zs=max(i_t[:,1]), color='k', marker='.', alpha=0.1)
    ax.scatter(i_t[:,1], i_t[:,2], zdir='x', zs=min(i_t[:,0]), color='k', marker='.', alpha=0.1)
    ax.scatter(i_t[:,0], i_t[:,1], zdir='z', zs=min(i_t[:,2]), color='k', marker='.', alpha=0.1)
    X_minmax = np.round(np.min(i_t[:,0]),2), np.round(np.max(i_t[:,0]),2)
    Y_minmax = np.round(np.min(i_t[:,1]),2), np.round(np.max(i_t[:,1]),2)
    Z_minmax = np.round(np.min(i_t[:,2]),2), np.round(np.max(i_t[:,2]),2)
    ax.set(xlabel="PC1", ylabel="PC2", zlabel='PC3')
    X_range = np.round(np.arange(*X_minmax), 2)
    Y_range = np.round(np.arange(*Y_minmax), 2)
    Z_range = np.round(np.arange(*Z_minmax), 2)
    ax.set_xlim3d(X_minmax)
    ax.set_ylim3d(Y_minmax)
    ax.set_zlim3d(Z_minmax)
    ax.set(xticks=X_range, xticklabels=X_range, yticks=Y_range, yticklabels=Y_range, zticks=Z_range, zticklabels=Z_range)
    #ax.view_init(elev=45., azim=-45)
    saveToNewDir(ImportedData.Title + "/", ImportedData.Title + "-ICAreducedIT.png")
    plt.gcf().clear()
    plt.clf()

    ImportedData.Title = "LetterRecognition"
    ImportedData.Subtitle = "ICA"

    df_reduced = preprocessing.MinMaxScaler().fit_transform(reduced)
    df_reduced = pd.DataFrame(reduced)
    Ts = train_test_split(df_reduced, Y, train_size = 0.8)
    ImportedData.Data["xTrain"], ImportedData.Data["xTest"], ImportedData.Data["yTrain"], ImportedData.Data["yTest"] = Ts
    ImportedData.Data["xTrain"] = scaler.fit_transform(ImportedData.Data["xTrain"])
    ImportedData.Data["xTest"] = scaler.fit_transform(ImportedData.Data["xTest"])

    KMAnalysis(ImportedData, ImportedData.Data)
    EMAnalysis(ImportedData, ImportedData.Data)
    # a = Process(target=KMAnalysis, args=(ImportedData, ImportedData.Data,))
    # b = Process(target=EMAnalysis, args=(ImportedData, ImportedData.Data,))
    # a.start()
    # b.start()
    # a.join()
    # b.join()
    # return ImportedData

def RPBankRedux(ImportedData):
    scaler = StandardScaler()
    ImportedData.Title = "BanknoteAuthentication"
    banknote = pd.read_csv('./data_banknote_authentication.data', header=None)
    banknote.columns = ['Wavelet_Variance', 'Wavelet_Skewness', 'Wavelet_Curtosis', 'Image_Entropy', 'Class']
    Y = banknote['Class']
    X = banknote.drop(['Class'], axis=1)
    df = pd.DataFrame(X, columns=banknote.columns.drop(['Class']))
    rescaled = pd.DataFrame(preprocessing.StandardScaler().fit_transform(df), columns = df.columns)

    rpBank = GaussianRandomProjection(n_components=3)
    rescaled = np.float32(rescaled)
    reduced = rpBank.fit_transform(rescaled)
    fig = plt.figure(figsize=(12,12))
    ax = fig.gca(projection='3d')
    ax.scatter(reduced[:,0], reduced[:,1], reduced[:,2], c=Y, marker='.',alpha=0.7, s=100)
    ax.scatter(reduced[:,0], reduced[:,2], zdir='y', zs=max(reduced[:,1]), color='k', marker='.', alpha=0.1)
    ax.scatter(reduced[:,1], reduced[:,2], zdir='x', zs=min(reduced[:,0]), color='k', marker='.', alpha=0.1)
    ax.scatter(reduced[:,0], reduced[:,1], zdir='z', zs=min(reduced[:,2]), color='k', marker='.', alpha=0.1)
    X_minmax = np.round(np.min(reduced[:,0]),2), np.round(np.max(reduced[:,0]),2)
    Y_minmax = np.round(np.min(reduced[:,1]),2), np.round(np.max(reduced[:,1]),2)
    Z_minmax = np.round(np.min(reduced[:,2]),2), np.round(np.max(reduced[:,2]),2)
    ax.set(xlabel="PC1", ylabel="PC2", zlabel='PC3')
    X_range = np.round(np.arange(*X_minmax), 2)
    Y_range = np.round(np.arange(*Y_minmax), 2)
    Z_range = np.round(np.arange(*Z_minmax), 2)
    ax.set_xlim3d(X_minmax)
    ax.set_ylim3d(Y_minmax)
    ax.set_zlim3d(Z_minmax)
    ax.set(xticks=X_range, xticklabels=X_range, yticks=Y_range, yticklabels=Y_range, zticks=Z_range, zticklabels=Z_range)
    #ax.view_init(elev=45., azim=-45)
    saveToNewDir(ImportedData.Title + "/", ImportedData.Title + "-RPreduced.png")
    plt.gcf().clear()
    plt.clf()

    ImportedData.Title = "BanknoteAuthentication"
    ImportedData.Subtitle = "RP"

    df_reduced = preprocessing.MinMaxScaler().fit_transform(reduced)
    df_reduced = pd.DataFrame(reduced)
    Ts = train_test_split(df_reduced, Y, train_size = 0.8)
    ImportedData.Data["xTrain"], ImportedData.Data["xTest"], ImportedData.Data["yTrain"], ImportedData.Data["yTest"] = Ts
    ImportedData.Data["xTrain"] = scaler.fit_transform(ImportedData.Data["xTrain"])
    ImportedData.Data["xTest"] = scaler.fit_transform(ImportedData.Data["xTest"])

    KMAnalysis(ImportedData, ImportedData.Data)
    EMAnalysis(ImportedData, ImportedData.Data)
    # a = Process(target=KMAnalysis, args=(ImportedData, ImportedData.Data,))
    # b = Process(target=EMAnalysis, args=(ImportedData, ImportedData.Data,))
    # a.start()
    # b.start()
    # a.join()
    # b.join()
    # return ImportedData

def RPLetterRedux(ImportedData):
    scaler = StandardScaler()
    ImportedData.Title = "LetterRecognition"
    letter = pd.read_csv('./letter-recognition.data', header=None)
    letter.columns = ['Letter', 'xBoxHPos', 'yBoxVPos', 'BoxW', 'BoxH', 'OnPix', 'xBarMean', 'yBarMean', 'x2BarMean', 'y2BarMean', 'xyBarMean', 'x2ybrMean', 'xy2BrMean','xEgeMean', 'xegvyCorr', 'y-egeMean', 'yegvxCorr']
    Y = [ord(i) for i in letter['Letter']]
    X = letter.drop(['Letter'], axis=1)
    df = pd.DataFrame(X, columns=letter.columns.drop(['Letter']))
    rescaled = pd.DataFrame(preprocessing.StandardScaler().fit_transform(df), columns = df.columns)

    rcLetter = GaussianRandomProjection(n_components=10)
    rescaled = np.float32(rescaled)
    reduced = rcLetter.fit_transform(rescaled)

    fig = plt.figure(figsize=(12,12))
    ax = fig.gca(projection='3d')
    ax.scatter(reduced[:,0], reduced[:,1], reduced[:,2], c=Y,  marker='.', alpha=0.7, s=100)
    ax.scatter(reduced[:,0], reduced[:,2], zdir='y', zs=max(reduced[:,1]), color='k', marker='.', alpha=0.1)
    ax.scatter(reduced[:,1], reduced[:,2], zdir='x', zs=min(reduced[:,0]), color='k', marker='.', alpha=0.1)
    ax.scatter(reduced[:,0], reduced[:,1], zdir='z', zs=min(reduced[:,2]), color='k', marker='.', alpha=0.1)
    X_minmax = np.round(np.min(reduced[:,0]),2), np.round(np.max(reduced[:,0]),2)
    Y_minmax = np.round(np.min(reduced[:,1]),2), np.round(np.max(reduced[:,1]),2)
    Z_minmax = np.round(np.min(reduced[:,4]),2), np.round(np.max(reduced[:,4]),2)
    X_range = np.round(np.arange(*X_minmax), 2)
    Y_range = np.round(np.arange(*Y_minmax), 2)
    Z_range = np.round(np.arange(*Z_minmax), 2)
    ax.set(xlabel="PC1", ylabel="PC2", zlabel='PC3')
    ax.set_xlim3d(X_minmax)
    ax.set_ylim3d(Y_minmax)
    ax.set_zlim3d(Z_minmax)
    ax.set(xticks=X_range, xticklabels=X_range, yticks=Y_range, yticklabels=Y_range, zticks=Z_range, zticklabels=Z_range)
    saveToNewDir(ImportedData.Title + "/", ImportedData.Title + "-RPreduced.png")
    plt.gcf().clear()
    plt.clf()

    ImportedData.Title = "LetterRecognition"
    ImportedData.Subtitle = "RP"

    df_reduced = preprocessing.MinMaxScaler().fit_transform(reduced)
    df_reduced = pd.DataFrame(reduced)
    Ts = train_test_split(df_reduced, Y, train_size = 0.8)
    ImportedData.Data["xTrain"], ImportedData.Data["xTest"], ImportedData.Data["yTrain"], ImportedData.Data["yTest"] = Ts
    ImportedData.Data["xTrain"] = scaler.fit_transform(ImportedData.Data["xTrain"])
    ImportedData.Data["xTest"] = scaler.fit_transform(ImportedData.Data["xTest"])

    KMAnalysis(ImportedData, ImportedData.Data)
    EMAnalysis(ImportedData, ImportedData.Data)
    # a = Process(target=KMAnalysis, args=(ImportedData, ImportedData.Data,))
    # b = Process(target=EMAnalysis, args=(ImportedData, ImportedData.Data,))
    # a.start()
    # b.start()
    # a.join()
    # b.join()
    # return ImportedData

def FABankRedux(ImportedData):
    scaler = StandardScaler()
    ImportedData.Title = "BanknoteAuthentication"
    banknote = pd.read_csv('./data_banknote_authentication.data', header=None)
    banknote.columns = ['Wavelet_Variance', 'Wavelet_Skewness', 'Wavelet_Curtosis', 'Image_Entropy', 'Class']
    Y = banknote['Class']
    X = banknote.drop(['Class'], axis=1)
    df = pd.DataFrame(X, columns=banknote.columns.drop(['Class']))
    rescaled = pd.DataFrame(preprocessing.StandardScaler().fit_transform(df), columns = df.columns)

    faBank = FactorAnalysis(n_components=3)
    #rescaled = np.float16(rescaled)
    reduced = faBank.fit_transform(rescaled)

    fig = plt.figure(figsize=(12,12))
    ax = fig.gca(projection='3d')
    ax.scatter(reduced[:,0], reduced[:,1], reduced[:,2], c=Y, marker='.',alpha=0.7, s=100)
    ax.scatter(reduced[:,0], reduced[:,2], zdir='y', zs=max(reduced[:,1]), color='k', marker='.', alpha=0.1)
    ax.scatter(reduced[:,1], reduced[:,2], zdir='x', zs=min(reduced[:,0]), color='k', marker='.', alpha=0.1)
    ax.scatter(reduced[:,0], reduced[:,1], zdir='z', zs=min(reduced[:,2]), color='k', marker='.', alpha=0.1)
    X_minmax = np.round(np.min(reduced[:,0]),2), np.round(np.max(reduced[:,0]),2)
    Y_minmax = np.round(np.min(reduced[:,1]),2), np.round(np.max(reduced[:,1]),2)
    Z_minmax = np.round(np.min(reduced[:,2]),2), np.round(np.max(reduced[:,2]),2)
    ax.set(xlabel="PC1", ylabel="PC2", zlabel='PC3')
    X_range = np.round(np.arange(*X_minmax), 2)
    Y_range = np.round(np.arange(*Y_minmax), 2)
    Z_range = np.round(np.arange(*Z_minmax), 2)
    ax.set_xlim3d(X_minmax)
    ax.set_ylim3d(Y_minmax)
    ax.set_zlim3d(Z_minmax)
    ax.set(xticks=X_range, xticklabels=X_range, yticks=Y_range, yticklabels=Y_range, zticks=Z_range, zticklabels=Z_range)
    saveToNewDir(ImportedData.Title + "/", ImportedData.Title + "-FAreduced.png")
    plt.gcf().clear()
    plt.clf()

    ImportedData.Title = "BanknoteAuthentication"
    ImportedData.Subtitle = "FA"

    df_reduced = preprocessing.MinMaxScaler().fit_transform(reduced)
    df_reduced = pd.DataFrame(reduced)
    Ts = train_test_split(df_reduced, Y, train_size = 0.8)
    ImportedData.Data["xTrain"], ImportedData.Data["xTest"], ImportedData.Data["yTrain"], ImportedData.Data["yTest"] = Ts
    ImportedData.Data["xTrain"] = scaler.fit_transform(ImportedData.Data["xTrain"])
    ImportedData.Data["xTest"] = scaler.fit_transform(ImportedData.Data["xTest"])

    KMAnalysis(ImportedData, ImportedData.Data)
    EMAnalysis(ImportedData, ImportedData.Data)
    # a = Process(target=KMAnalysis, args=(ImportedData, ImportedData.Data,))
    # b = Process(target=EMAnalysis, args=(ImportedData, ImportedData.Data,))
    # a.start()
    # b.start()
    # a.join()
    # b.join()
    # return ImportedData

def FALetterRedux(ImportedData):
    scaler = StandardScaler()
    ImportedData.Title = "LetterRecognition"
    letter = pd.read_csv('./letter-recognition.data', header=None)
    letter.columns = ['Letter', 'xBoxHPos', 'yBoxVPos', 'BoxW', 'BoxH', 'OnPix', 'xBarMean', 'yBarMean', 'x2BarMean', 'y2BarMean', 'xyBarMean', 'x2ybrMean', 'xy2BrMean','xEgeMean', 'xegvyCorr', 'y-egeMean', 'yegvxCorr']
    Y = [ord(i) for i in letter['Letter']]
    X = letter.drop(['Letter'], axis=1)
    df = pd.DataFrame(X, columns=letter.columns.drop(['Letter']))
    rescaled = pd.DataFrame(preprocessing.StandardScaler().fit_transform(df), columns = df.columns)

    faLetter = FactorAnalysis(n_components=10)
    #rescaled = np.float16(rescaled)
    reduced = faLetter.fit_transform(rescaled)

    fig = plt.figure(figsize=(12,12))
    ax = fig.gca(projection='3d')
    ax.scatter(reduced[:,0], reduced[:,1], reduced[:,2], c=Y,  marker='.', alpha=0.7, s=100)
    ax.scatter(reduced[:,0], reduced[:,2], zdir='y', zs=max(reduced[:,1]), color='k', marker='.', alpha=0.1)
    ax.scatter(reduced[:,1], reduced[:,2], zdir='x', zs=min(reduced[:,0]), color='k', marker='.', alpha=0.1)
    ax.scatter(reduced[:,0], reduced[:,1], zdir='z', zs=min(reduced[:,2]), color='k', marker='.', alpha=0.1)
    X_minmax = np.round(np.min(reduced[:,0]),2), np.round(np.max(reduced[:,0]),2)
    Y_minmax = np.round(np.min(reduced[:,1]),2), np.round(np.max(reduced[:,1]),2)
    Z_minmax = np.round(np.min(reduced[:,4]),2), np.round(np.max(reduced[:,4]),2)
    X_range = np.round(np.arange(*X_minmax), 2)
    Y_range = np.round(np.arange(*Y_minmax), 2)
    Z_range = np.round(np.arange(*Z_minmax), 2)
    ax.set(xlabel="PC1", ylabel="PC2", zlabel='PC3')
    ax.set_xlim3d(X_minmax)
    ax.set_ylim3d(Y_minmax)
    ax.set_zlim3d(Z_minmax)
    ax.set(xticks=X_range, xticklabels=X_range, yticks=Y_range, yticklabels=Y_range, zticks=Z_range, zticklabels=Z_range)
    saveToNewDir(ImportedData.Title + "/", ImportedData.Title + "-FAreduced.png")
    plt.gcf().clear()
    plt.clf()

    ImportedData.Title = "LetterRecognition"
    ImportedData.Subtitle = "FA"

    df_reduced = preprocessing.MinMaxScaler().fit_transform(reduced)
    df_reduced = pd.DataFrame(reduced)
    Ts = train_test_split(df_reduced, Y, train_size = 0.8)
    ImportedData.Data["xTrain"], ImportedData.Data["xTest"], ImportedData.Data["yTrain"], ImportedData.Data["yTest"] = Ts
    ImportedData.Data["xTrain"] = scaler.fit_transform(ImportedData.Data["xTrain"])
    ImportedData.Data["xTest"] = scaler.fit_transform(ImportedData.Data["xTest"])

    KMAnalysis(ImportedData, ImportedData.Data)
    EMAnalysis(ImportedData, ImportedData.Data)
    # a = Process(target=KMAnalysis, args=(ImportedData, ImportedData.Data,))
    # b = Process(target=EMAnalysis, args=(ImportedData, ImportedData.Data,))
    # a.start()
    # b.start()
    # a.join()
    # b.join()
    # return ImportedData

def PlotBankData(ImportedData):
    ImportedData.Title = "BanknoteAuthentication"
    banknote = pd.read_csv('./data_banknote_authentication.data', header=None)
    banknote.columns = ['Wavelet_Variance', 'Wavelet_Skewness', 'Wavelet_Curtosis', 'Image_Entropy', 'Class']
    Y = banknote['Class']
    X = banknote.drop(['Class'], axis=1)
    df = pd.DataFrame(X, columns=banknote.columns.drop(['Class']))
    rescaled = pd.DataFrame(preprocessing.StandardScaler().fit_transform(df), columns = df.columns)
    nRescaled = rescaled.to_numpy()

    fig = plt.figure(figsize=(12,12))
    ax = fig.gca(projection='3d')
    ax.scatter(nRescaled[:,0], nRescaled[:,1], nRescaled[:,2], c=Y,  marker='.', alpha=0.7, s=100)
    ax.scatter(nRescaled[:,0], nRescaled[:,2], zdir='y', zs=max(nRescaled[:,1]), color='k', marker='.', alpha=0.1)
    ax.scatter(nRescaled[:,1], nRescaled[:,2], zdir='x', zs=min(nRescaled[:,0]), color='k', marker='.', alpha=0.1)
    ax.scatter(nRescaled[:,0], nRescaled[:,1], zdir='z', zs=min(nRescaled[:,2]), color='k', marker='.', alpha=0.1)
    X_minmax = np.round(np.min(nRescaled[:,0]),2), np.round(np.max(nRescaled[:,0]),2)
    Y_minmax = np.round(np.min(nRescaled[:,1]),2), np.round(np.max(nRescaled[:,1]),2)
    Z_minmax = np.round(np.min(nRescaled[:,2]),2), np.round(np.max(nRescaled[:,2]),2)
    ax.set(xlabel="Variance", ylabel="Skewness", zlabel='Curtosis')
    X_range = np.round(np.arange(*X_minmax), 2)
    Y_range = np.round(np.arange(*Y_minmax), 2)
    Z_range = np.round(np.arange(*Z_minmax), 2)
    ax.set_xlim3d(X_minmax)
    ax.set_ylim3d(Y_minmax)
    ax.set_zlim3d(Z_minmax)
    ax.set(xticks=X_range, xticklabels=X_range, yticks=Y_range, yticklabels=Y_range, zticks=Z_range, zticklabels=Z_range)
    saveToNewDir(ImportedData.Title + "/", ImportedData.Title + "-original.png")
    plt.gcf().clear()
    plt.clf()

    Ts = train_test_split(rescaled, Y, train_size = 0.8)
    ImportedData.Data["xTrain"], ImportedData.Data["xTest"], ImportedData.Data["yTrain"], ImportedData.Data["yTest"] = Ts
    ImportedData.Data["xTrain"] = scaler.fit_transform(ImportedData.Data["xTrain"])
    ImportedData.Data["xTest"] = scaler.fit_transform(ImportedData.Data["xTest"])

    ImportedData.Title = "BanknoteAuthentication"
    ImportedData.Subtitle = "Original"
    KMAnalysis(ImportedData, ImportedData.Data)
    EMAnalysis(ImportedData, ImportedData.Data)
    # a = Process(target=KMAnalysis, args=(ImportedData, ImportedData.Data,))
    # b = Process(target=EMAnalysis, args=(ImportedData, ImportedData.Data,))
    # a.start()
    # b.start()
    # a.join()
    # b.join()

def PlotLetterData(ImportedData):
    ImportedData.Title = "LetterRecognition"
    letter = pd.read_csv('./letter-recognition.data', header=None)
    letter.columns = ['Letter', 'xBoxHPos', 'yBoxVPos', 'BoxW', 'BoxH', 'OnPix', 'xBarMean', 'yBarMean', 'x2BarMean', 'y2BarMean', 'xyBarMean', 'x2ybrMean', 'xy2BrMean','xEgeMean', 'xegvyCorr', 'y-egeMean', 'yegvxCorr']
    Y = [ord(i) for i in letter['Letter']]
    X = letter.drop(['Letter'], axis=1)
    df = pd.DataFrame(X, columns=letter.columns.drop(['Letter']))
    rescaled = pd.DataFrame(preprocessing.StandardScaler().fit_transform(df), columns = df.columns)
    nRescaled = rescaled.to_numpy()
    
    fig = plt.figure(figsize=(12,12))
    ax = fig.gca(projection='3d')
    ax.scatter(nRescaled[:,0], nRescaled[:,1], nRescaled[:,2], c=Y,  marker='.', alpha=0.7, s=100)
    ax.scatter(nRescaled[:,0], nRescaled[:,2], zdir='y', zs=max(nRescaled[:,1]), color='k', marker='.', alpha=0.1) #c=Yi
    ax.scatter(nRescaled[:,1], nRescaled[:,2], zdir='x', zs=min(nRescaled[:,0]), color='k', marker='.', alpha=0.1)
    ax.scatter(nRescaled[:,0], nRescaled[:,1], zdir='z', zs=min(nRescaled[:,2]), color='k', marker='.', alpha=0.1)
    X_minmax = np.round(np.min(nRescaled[:,0]),2), np.round(np.max(nRescaled[:,0]),2)
    Y_minmax = np.round(np.min(nRescaled[:,1]),2), np.round(np.max(nRescaled[:,1]),2)
    Z_minmax = np.round(np.min(nRescaled[:,2]),2), np.round(np.max(nRescaled[:,2]),2)
    ax.set(xlabel="XBox", ylabel="YBox", zlabel='ONPix')
    X_range = np.round(np.arange(*X_minmax), 2)
    Y_range = np.round(np.arange(*Y_minmax), 2)
    Z_range = np.round(np.arange(*Z_minmax), 2)
    ax.set_xlim3d(X_minmax)
    ax.set_ylim3d(Y_minmax)
    ax.set_zlim3d(Z_minmax)
    ax.set(xticks=X_range, xticklabels=X_range, yticks=Y_range, yticklabels=Y_range, zticks=Z_range, zticklabels=Z_range)
    #ax.view_init(elev=45., azim=-45)
    saveToNewDir(ImportedData.Title + "/", ImportedData.Title + "-original.png")
    plt.gcf().clear()
    plt.clf()
    
    Ts = train_test_split(rescaled, Y, train_size = 0.8)
    ImportedData.Data["xTrain"], ImportedData.Data["xTest"], ImportedData.Data["yTrain"], ImportedData.Data["yTest"] = Ts
    ImportedData.Data["xTrain"] = scaler.fit_transform(ImportedData.Data["xTrain"])
    ImportedData.Data["xTest"] = scaler.fit_transform(ImportedData.Data["xTest"])
    ImportedData.Title = "LetterRecognition"
    ImportedData.Subtitle = "Original"
    KMAnalysis(ImportedData, ImportedData.Data)
    EMAnalysis(ImportedData, ImportedData.Data)
    # a = Process(target=KMAnalysis, args=(ImportedData, ImportedData.Data,))
    # b = Process(target=EMAnalysis, args=(ImportedData, ImportedData.Data,))
    # a.start()
    # b.start()
    # a.join()
    # b.join()

def getBankScore():
    ImportedData.Title = "BanknoteAuthentication"
    banknote = pd.read_csv('./data_banknote_authentication.data', header=None)
    banknote.columns = ['Wavelet_Variance', 'Wavelet_Skewness', 'Wavelet_Curtosis', 'Image_Entropy', 'Class']
    Y = banknote['Class']
    X = banknote.drop(['Class'], axis=1)
    df = pd.DataFrame(X, columns=banknote.columns.drop(['Class']))
    Ts = train_test_split(df, Y, train_size = 0.8)
    ImportedData.Data["xTrain"], ImportedData.Data["xTest"], ImportedData.Data["yTrain"], ImportedData.Data["yTest"] = Ts
    
    n_components = np.arange(1, 5)

    plt.figure()
    pca_scores, fa_scores = [], []
    pca = PCA()
    fa = FactorAnalysis()
    for n in n_components:
        pca.n_components = n
        fa.n_components = n
        pca_scores.append(np.mean(cross_val_score(pca, ImportedData.Data["xTrain"])))
        fa_scores.append(np.mean(cross_val_score(fa, ImportedData.Data["xTrain"])))
    plt.plot(n_components, pca_scores, 'b', label='PCA scores')
    plt.plot(n_components, fa_scores, 'r', label='FA scores')

    plt.xticks(n_components, n_components)
    plt.xlabel('nb of components')
    plt.ylabel('CV scores')
    plt.legend(loc='lower right')
    saveToNewDir(ImportedData.Title + "/", ImportedData.Title + "-PCAvsFA.png")
    plt.gcf().clear()
    plt.clf()

    return pca_scores, fa_scores

def getLetterScore():
    ImportedData.Title = "LetterRecognition"
    letter = pd.read_csv('./letter-recognition.data', header=None)
    letter.columns = ['Letter', 'xBoxHPos', 'yBoxVPos', 'BoxW', 'BoxH', 'OnPix', 'xBarMean', 'yBarMean', 'x2BarMean', 'y2BarMean', 'xyBarMean', 'x2ybrMean', 'xy2BrMean','xEgeMean', 'xegvyCorr', 'y-egeMean', 'yegvxCorr']
    Y = [ord(i) for i in letter['Letter']]
    X = letter.drop(['Letter'], axis=1)
    df = pd.DataFrame(X, columns=letter.columns.drop(['Letter']))
    Ts = train_test_split(df, Y, train_size = 0.8)
    ImportedData.Data["xTrain"], ImportedData.Data["xTest"], ImportedData.Data["yTrain"], ImportedData.Data["yTest"] = Ts
    
    n_components = np.arange(1, 17)
    pca = PCA()
    fa = FactorAnalysis()

    plt.figure()
    pca_scores, fa_scores = [], []
    pca = PCA()
    fa = FactorAnalysis()
    for n in n_components:
        pca.n_components = n
        fa.n_components = n
        pca_scores.append(np.mean(cross_val_score(pca, ImportedData.Data["xTrain"])))
        fa_scores.append(np.mean(cross_val_score(fa, ImportedData.Data["xTrain"])))
    plt.plot(n_components, pca_scores, 'b', label='PCA scores')
    plt.plot(n_components, fa_scores, 'r', label='FA scores')

    plt.xticks(n_components, n_components)
    plt.xlabel('nb of components')
    plt.ylabel('CV scores')
    plt.legend(loc='lower right')
    saveToNewDir(ImportedData.Title + "/", ImportedData.Title + "-PCAvsFA.png")
    plt.gcf().clear()
    plt.clf()

    return pca_scores, fa_scores

if __name__ == "__main__":

    TRAIN_SIZE = 0.8
    getBanknoteAuthData(TRAIN_SIZE)
    getLetterRecData(TRAIN_SIZE)

    ### Banknote Data
    scaler = StandardScaler()
    bankData = ImportedData()
    bankData.Title = "BanknoteAuthentication"
    banknote = pd.read_csv('./data_banknote_authentication.data', header=None)
    banknote.columns = ['Wavelet_Variance', 'Wavelet_Skewness', 'Wavelet_Curtosis', 'Image_Entropy', 'Class']
    Y = banknote['Class']
    X = banknote.drop(['Class'], axis=1)
    df = pd.DataFrame(X, columns=banknote.columns.drop(['Class']))
    Ts = train_test_split(df, Y, train_size = 0.8)
    bankData.Data["xTrain"], bankData.Data["xTest"], bankData.Data["yTrain"], bankData.Data["yTest"] = Ts
    bankData.Data["xTrain"] = pd.DataFrame(scaler.fit_transform(bankData.Data["xTrain"]), columns = df.columns)
    bankData.Data["xTest"] = pd.DataFrame(scaler.fit_transform(bankData.Data["xTest"]), columns = df.columns)

    orgData = copy.deepcopy(bankData)
    pcaData = copy.deepcopy(bankData)
    icaData = copy.deepcopy(bankData)
    rpData = copy.deepcopy(bankData)
    ftData = copy.deepcopy(bankData)
    
    bestNComponents = 3

    print ("-----> plotFit Original Banknote Data <-----")
    orgData.Subtitle = "Original"
    X_train, X_test, inverse_X = plotFit(orgData)
    print ("-----> plot Variance+CV Original Banknote Data <-----")
    bestCV, bestKurt = plotCVKurVarGraph(orgData)
    orgData.Data["xTrain"] = copy.deepcopy(X_train.astype('double'))
    orgData.Data["xTest"] = copy.deepcopy(X_test.astype('double'))
    orgData.Data["xTrain"] = scaler.fit_transform(orgData.Data["xTrain"])
    orgData.Data["xTest"] = scaler.fit_transform(orgData.Data["xTest"])
    print ("-----> Training Original NN <-----")
    a = copy.deepcopy(orgData)
    b = copy.deepcopy(orgData)
    c = copy.deepcopy(orgData)
    TestLayers(a)
    TestNeurons(a)
    TestAlpha(a)
    print ("-----> KMAnalysis Original_KM Banknote Data <-----")
    KMAnalysis(b, b.Data)
    print ("-----> Training Original_KM NN <-----")
    b.Subtitle = "Original_KM"
    TestLayers(b)
    TestNeurons(b)
    TestAlpha(b)
    print ("-----> EMAnalysis Original_EM Banknote Data <-----")
    EMAnalysis(c, c.Data)
    print ("-----> Training Original_EM NN <-----")
    c.Subtitle = "Original_EM"
    TestLayers(c)
    TestNeurons(c)
    TestAlpha(c)

    del a
    del b
    del c
    del orgData
    del bestCV
    del bestKurt
    

    pcaData.Subtitle = "PCA"
    print ("-----> plotFit PCA Banknote Data {} components<-----".format(bestNComponents))
    z = copy.deepcopy(pcaData)
    x = copy.deepcopy(pcaData)
    pcaX_train, pcaX_test, inverse_X = plotFit(pcaData, PCA(n_components=bestNComponents))
    print ("-----> plotInverse PCA Letter Data <-----")
    plotInverse(x, PCA(n_components=bestNComponents), inverse_X)
    print ("-----> plot Variance+CV PCA Banknote Data <-----")
    bestCV, bestKurt = plotCVKurVarGraph(pcaData)
    pcaData.Data["xTrain"] = copy.deepcopy(pcaX_train.astype('double'))
    pcaData.Data["xTest"] = copy.deepcopy(pcaX_test.astype('double'))
    pcaData.Data["xTrain"] = scaler.fit_transform(pcaData.Data["xTrain"])
    pcaData.Data["xTest"] = scaler.fit_transform(pcaData.Data["xTest"])
    print ("-----> Training PCA NN <-----")
    a = copy.deepcopy(pcaData)
    b = copy.deepcopy(pcaData)
    c = copy.deepcopy(pcaData)
    TestLayers(a)
    TestNeurons(a)
    TestAlpha(a)
    print ("-----> KMAnalysis PCA Banknote Data <-----")
    KMAnalysis(b, b.Data)
    print ("-----> Training PCA_KM NN <-----")
    b.Subtitle = "PCA_KM"
    TestLayers(b)
    TestNeurons(b)
    TestAlpha(b)
    print ("-----> EMAnalysis PCA Banknote Data <-----")
    EMAnalysis(c, c.Data)
    print ("-----> Training PCA_EM NN <-----")
    c.Subtitle = "PCA_EM"
    TestLayers(c)
    TestNeurons(c)
    TestAlpha(c)

    del a
    del b
    del c
    del x
    del z
    del pcaData
    del bestCV
    del bestKurt

    rpData.Subtitle = "RP"
    print ("-----> plotFit RP Banknote Data {} components<-----".format(bestNComponents))
    rpX_train, rpX_test, inverse_X = plotFit(rpData, GaussianRandomProjection(n_components=bestNComponents))
    print ("-----> plot Variance+CV RP Banknote Data <-----")
    bestCV, bestKurt = plotCVKurVarGraph(rpData)
    rpData.Data["xTrain"] = copy.deepcopy(rpX_train.astype('double'))
    rpData.Data["xTest"] = copy.deepcopy(rpX_test.astype('double'))
    rpData.Data["xTrain"] = scaler.fit_transform(rpData.Data["xTrain"])
    rpData.Data["xTest"] = scaler.fit_transform(rpData.Data["xTest"])
    print ("-----> Training RP NN <-----")
    a = copy.deepcopy(rpData)
    b = copy.deepcopy(rpData)
    c = copy.deepcopy(rpData)
    TestLayers(a)
    TestNeurons(a)
    TestAlpha(a)
    print ("-----> KM-Analysis RP Banknote Data <-----")
    KMAnalysis(b, b.Data)
    print ("-----> Training RP_KM NN <-----")
    b.Subtitle = "RP_KM"
    TestLayers(b)
    TestNeurons(b)
    TestAlpha(b)
    print ("-----> EM-Analysis RP Banknote Data <-----")
    EMAnalysis(c, c.Data)
    print ("-----> Training RP_EM NN <-----")
    c.Subtitle = "RP_EM"
    TestLayers(c)
    TestNeurons(c)
    TestAlpha(c)

    del a
    del b
    del c
    del rpData
    del bestCV
    del bestKurt

    ftData.Subtitle = "FT"
    print ("-----> plotFit FA Banknote Data {} components <-----".format(bestNComponents))
    ftX_train, ftX_test, inverse_X = plotFit(ftData, FactorAnalysis(n_components=bestNComponents))
    print ("-----> plot Variance+CV FT Banknote Data <-----")
    bestCV, bestKurt = plotCVKurVarGraph(ftData)
    ftData.Data["xTrain"] = copy.deepcopy(ftX_train.astype('double'))
    ftData.Data["xTest"] = copy.deepcopy(ftX_test.astype('double'))
    ftData.Data["xTrain"] = scaler.fit_transform(ftData.Data["xTrain"])
    ftData.Data["xTest"] = scaler.fit_transform(ftData.Data["xTest"])
    print ("-----> Training FT NN <-----")
    a = copy.deepcopy(ftData)
    b = copy.deepcopy(ftData)
    c = copy.deepcopy(ftData)
    TestLayers(a)
    TestNeurons(a)
    TestAlpha(a)
    print ("-----> KM-Analysis FT Banknote Data <-----")
    KMAnalysis(b, b.Data)
    print ("-----> Training FT_KM NN <-----")
    b.Subtitle = "FT_KM"
    TestLayers(b)
    TestNeurons(b)
    TestAlpha(b)
    print ("-----> EM-Analysis FT Banknote Data <-----")
    EMAnalysis(c, c.Data)
    print ("-----> Training FT_EM NN <-----")
    c.Subtitle = "FT_EM"
    TestLayers(c)
    TestNeurons(c)
    TestAlpha(c)

    del a
    del b
    del c
    del ftData
    del bestCV
    del bestKurt

    icaData.Subtitle = "ICA"
    print ("-----> plotFit ICA Banknote Data {} components <-----".format(bestNComponents))
    z = copy.deepcopy(icaData)
    x = copy.deepcopy(icaData)
    icaX_train, icaX_test, inverse_X = plotFit(icaData, FastICA(n_components=bestNComponents))
    print ("-----> plotInverse ICA Letter Data <-----")
    plotInverse(x, FastICA(n_components=bestNComponents), inverse_X)
    print ("-----> plot Variance+CV ICA Banknote Data <-----")
    bestCV, bestKurt = plotCVKurVarGraph(icaData)
    icaData.Data["xTrain"] = copy.deepcopy(icaX_train.astype('double'))
    icaData.Data["xTest"] = copy.deepcopy(icaX_test.astype('double'))
    icaData.Data["xTrain"] = scaler.fit_transform(icaData.Data["xTrain"])
    icaData.Data["xTest"] = scaler.fit_transform(icaData.Data["xTest"])
    print ("-----> Training ICA NN <-----")
    a = copy.deepcopy(icaData)
    b = copy.deepcopy(icaData)
    c = copy.deepcopy(icaData)
    TestLayers(a)
    TestNeurons(a)
    TestAlpha(a)
    print ("-----> KM-Analysis ICA Banknote Data <-----")
    KMAnalysis(b, b.Data)
    print ("-----> Training ICA_KM NN <-----")
    b.Subtitle = "ICA_KM"
    TestLayers(b)
    TestNeurons(b)
    TestAlpha(b)
    print ("-----> EM-Analysis ICA Banknote Data <-----")
    EMAnalysis(c, c.Data)
    print ("-----> Training ICA_EM NN <-----")
    c.Subtitle = "ICA_EM"
    TestLayers(c)
    TestNeurons(c)
    TestAlpha(c)

    del a
    del b
    del c
    del x
    del z
    del icaData
    del bestCV
    del bestKurt

    Letter Data

    scaler = StandardScaler()
    letterData = ImportedData()
    letterData.Title = "LetterRecognition"
    letter = pd.read_csv('./letter-recognition.data', header=None)
    letter.columns = ['Letter', 'xBoxHPos', 'yBoxVPos', 'BoxW', 'BoxH', 'OnPix', 'xBarMean', 'yBarMean', 'x2BarMean', 'y2BarMean', 'xyBarMean', 'x2ybrMean', 'xy2BrMean','xEgeMean', 'xegvyCorr', 'y-egeMean', 'yegvxCorr']
    Y = [ord(i) for i in letter['Letter']]
    X = letter.drop(['Letter'], axis=1)
    df = pd.DataFrame(X, columns=letter.columns.drop(['Letter']))
    Ts = train_test_split(df, Y, train_size = 0.8)
    letterData.Data["xTrain"], letterData.Data["xTest"], letterData.Data["yTrain"], letterData.Data["yTest"] = Ts
    letterData.Data["xTrain"] = pd.DataFrame(scaler.fit_transform(letterData.Data["xTrain"]), columns = df.columns)
    letterData.Data["xTest"] = pd.DataFrame(scaler.fit_transform(letterData.Data["xTest"]), columns = df.columns)

    orgData = copy.deepcopy(letterData)
    pcaData = copy.deepcopy(letterData)
    icaData = copy.deepcopy(letterData)
    rpData = copy.deepcopy(letterData)
    ftData = copy.deepcopy(letterData)
    
    bestNComponents = 11

    print ("-----> plotFit Original Letter Data <-----")
    orgData.Subtitle = "Original"
    X_train, X_test, inverse_X = plotFit(orgData)
    print ("-----> plot Variance+CV Original Letter Data <-----")
    bestCV, bestKurt = plotCVKurVarGraph(orgData)
    orgData.Data["xTrain"] = copy.deepcopy(X_train.astype('double'))
    orgData.Data["xTest"] = copy.deepcopy(X_test.astype('double'))
    orgData.Data["xTrain"] = scaler.fit_transform(orgData.Data["xTrain"])
    orgData.Data["xTest"] = scaler.fit_transform(orgData.Data["xTest"])
    b = copy.deepcopy(orgData)
    c = copy.deepcopy(orgData)
    print ("-----> KMAnalysis Original_KM Letter Data <-----")
    KMAnalysis(b, b.Data)
    print ("-----> EMAnalysis Original_EM Letter Data <-----")
    EMAnalysis(c, c.Data)

    del b
    del c
    del orgData
    del bestCV
    del bestKurt

    pcaData.Subtitle = "PCA"
    print ("-----> plotFit PCA Letter Data {} components<-----".format(bestNComponents))
    a = copy.deepcopy(pcaData)
    b = copy.deepcopy(pcaData)
    pcaX_train, pcaX_test, inverse_X = plotFit(a, PCA(n_components=bestNComponents))
    print ("-----> plotInverse PCA Letter Data <-----")
    plotInverse(a, PCA(n_components=bestNComponents), inverse_X)
    print ("-----> plot Variance+CV PCA Letter Data <-----")
    bestCV, bestKurt = plotCVKurVarGraph(pcaData)
    pcaData.Data["xTrain"] = copy.deepcopy(pcaX_train.astype('double'))
    pcaData.Data["xTest"] = copy.deepcopy(pcaX_test.astype('double'))
    pcaData.Data["xTrain"] = scaler.fit_transform(pcaData.Data["xTrain"])
    pcaData.Data["xTest"] = scaler.fit_transform(pcaData.Data["xTest"])
    print ("-----> Training PCA NN <-----")
    c = copy.deepcopy(pcaData)
    d = copy.deepcopy(pcaData)
    print ("-----> KMAnalysis PCA Letter Data <-----")
    KMAnalysis(c, c.Data)
    print ("-----> EMAnalysis PCA Letter Data <-----")
    EMAnalysis(d, d.Data)

    del a
    del b
    del c
    del d
    del pcaData
    del bestCV
    del bestKurt

    rpData.Subtitle = "RP"
    print ("-----> plotFit RP Letter Data {} components<-----".format(bestNComponents))
    a = copy.deepcopy(icaData)
    rpX_train, rpX_test, inverse_X = plotFit(a, GaussianRandomProjection(n_components=bestNComponents))
    print ("-----> plot Variance+CV RP Letter Data <-----")
    bestCV, bestKurt = plotCVKurVarGraph(rpData)
    rpData.Data["xTrain"] = copy.deepcopy(rpX_train.astype('double'))
    rpData.Data["xTest"] = copy.deepcopy(rpX_test.astype('double'))
    rpData.Data["xTrain"] = scaler.fit_transform(rpData.Data["xTrain"])
    rpData.Data["xTest"] = scaler.fit_transform(rpData.Data["xTest"])
    b = copy.deepcopy(rpData)
    c = copy.deepcopy(rpData)
    print ("-----> KM-Analysis RP Letter Data <-----")
    KMAnalysis(b, b.Data)
    print ("-----> EM-Analysis RP Letter Data <-----")
    EMAnalysis(c, c.Data)

    del a
    del b
    del c
    del rpData
    del bestCV
    del bestKurt

    ftData.Subtitle = "FT"
    print ("-----> plotFit FA Letter Data {} components <-----".format(bestNComponents))
    a = copy.deepcopy(icaData)
    ftX_train, ftX_test, inverse_X = plotFit(a, FactorAnalysis(n_components=bestNComponents))
    print ("-----> plot Variance+CV FT Letter Data <-----")
    bestCV, bestKurt = plotCVKurVarGraph(ftData)
    ftData.Data["xTrain"] = copy.deepcopy(ftX_train.astype('double'))
    ftData.Data["xTest"] = copy.deepcopy(ftX_test.astype('double'))
    ftData.Data["xTrain"] = scaler.fit_transform(ftData.Data["xTrain"])
    ftData.Data["xTest"] = scaler.fit_transform(ftData.Data["xTest"])
    b = copy.deepcopy(ftData)
    c = copy.deepcopy(ftData)
    print ("-----> KM-Analysis FT Letter Data <-----")
    KMAnalysis(b, b.Data)
    print ("-----> EM-Analysis FT Letter Data <-----")
    EMAnalysis(c, c.Data)

    del a
    del b
    del c
    del ftData
    del bestCV
    del bestKurt
        
    icaData11.Subtitle = "ICA"
    print ("-----> plotFit ICA Letter Data {} components <-----".format(11))
    a = copy.deepcopy(icaData11)
    b = copy.deepcopy(icaData11)
    icaX_train, icaX_test, inverse_X = plotFit(a, FastICA(n_components=11))
    print ("-----> plotInverse ICA Letter Data <-----")
    plotInverse(b, FastICA(n_components=11), inverse_X)
    print ("-----> plot Variance+CV ICA Letter Data <-----")
    bestCV, bestKurt = plotCVKurVarGraph(icaData11)
    icaData11.Data["xTrain"] = copy.deepcopy(icaX_train.astype('double'))
    icaData11.Data["xTest"] = copy.deepcopy(icaX_test.astype('double'))
    icaData11.Data["xTrain"] = scaler.fit_transform(icaData11.Data["xTrain"])
    icaData11.Data["xTest"] = scaler.fit_transform(icaData11.Data["xTest"])
    c = copy.deepcopy(icaData11)
    d = copy.deepcopy(icaData11)
    print ("-----> KM-Analysis ICA Letter Data <-----")
    KMAnalysis(c, c.Data)
    print ("-----> EM-Analysis ICA Letter Data <-----")
    EMAnalysis(d, d.Data)

    del a
    del b
    del c
    del d
    del icaData11
    del bestCV
    del bestKurt