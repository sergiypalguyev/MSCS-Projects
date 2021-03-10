# Your Agent for solving Raven's Progressive Matrices. You MUST modify this file.
#
# You may also create and submit new files in addition to modifying this file.
#
# Make sure your file retains methods with the signatures:
# def __init__(self)
# def Solve(self,problem)
#
# These methods will be necessary for the project's main method to run.

# Install Pillow and uncomment this line to access image processing.
from PIL import Image, ImageFilter, ImageEnhance, ImageOps, ImageChops
import numpy as np
import os
import copy
import random
from random import randint

def toNumpyMatrix(image):
    return np.array(image).clip(max=1)

def diffStatistics(pixelsA, pixelsB, thresh=0.98):
    matrixComp = np.equal(pixelsA, pixelsB).astype(np.uint8)
    
    mean = np.mean(matrixComp)
    rms = np.sqrt(np.mean(np.square(pixelsA-pixelsB)))
    #BoundingBox = ImageChops.difference(Image.fromarray(pixelsA), Image.fromarray(pixelsB)).getbbox()
    BoundingBox = 1

    return mean > thresh, mean, rms, BoundingBox

def equate2x2(figuresDictionary, solutionsDictionary, pixA, pixCompare, pixSolution, compareThreshold=0.97, SolutionThreshold = 0.97):
    retVal = -1
    bestMeanDiff = 1
    similarA, meanA, rmsA, bboxA = diffStatistics(pixA, pixCompare, compareThreshold)
    if similarA:
        for key, value in solutionsDictionary.items():
            similar, mean, rms, bbox = diffStatistics(pixSolution, value, SolutionThreshold)
            if similar and abs(mean-meanA) < bestMeanDiff:
                bestMeanDiff = abs(mean-meanA)
                retVal = int(key)
    return retVal

def getNumPyImages(figuresDictionary):
    return {"A" : toNumpyMatrix(Image.open(figuresDictionary['A'].visualFilename).convert('L')),
            "B" : toNumpyMatrix(Image.open(figuresDictionary['B'].visualFilename).convert('L')),
            "C" : toNumpyMatrix(Image.open(figuresDictionary['C'].visualFilename).convert('L')),
            "D" : toNumpyMatrix(Image.open(figuresDictionary['D'].visualFilename).convert('L')), 
            "E" : toNumpyMatrix(Image.open(figuresDictionary['E'].visualFilename).convert('L')), 
            "F" : toNumpyMatrix(Image.open(figuresDictionary['F'].visualFilename).convert('L')), 
            "G" : toNumpyMatrix(Image.open(figuresDictionary['G'].visualFilename).convert('L')), 
            "H" : toNumpyMatrix(Image.open(figuresDictionary['H'].visualFilename).convert('L'))}

def getNumPyAnswers(figuresDictionary):
    return {"1" : toNumpyMatrix(Image.open(figuresDictionary['1'].visualFilename).convert('L')),
            "2" : toNumpyMatrix(Image.open(figuresDictionary['2'].visualFilename).convert('L')),
            "3" : toNumpyMatrix(Image.open(figuresDictionary['3'].visualFilename).convert('L')),
            "4" : toNumpyMatrix(Image.open(figuresDictionary['4'].visualFilename).convert('L')), 
            "5" : toNumpyMatrix(Image.open(figuresDictionary['5'].visualFilename).convert('L')), 
            "6" : toNumpyMatrix(Image.open(figuresDictionary['6'].visualFilename).convert('L')), 
            "7" : toNumpyMatrix(Image.open(figuresDictionary['7'].visualFilename).convert('L')), 
            "8" : toNumpyMatrix(Image.open(figuresDictionary['8'].visualFilename).convert('L'))}

def getImagesBWRatio(figuresDictionary):
    npImgA=toNumpyMatrix(Image.open(figuresDictionary['A'].visualFilename).convert('L'))
    npImgB=toNumpyMatrix(Image.open(figuresDictionary['B'].visualFilename).convert('L'))
    npImgC=toNumpyMatrix(Image.open(figuresDictionary['C'].visualFilename).convert('L'))
    npImgD=toNumpyMatrix(Image.open(figuresDictionary['D'].visualFilename).convert('L'))
    npImgE=toNumpyMatrix(Image.open(figuresDictionary['E'].visualFilename).convert('L'))
    npImgF=toNumpyMatrix(Image.open(figuresDictionary['F'].visualFilename).convert('L'))
    npImgG=toNumpyMatrix(Image.open(figuresDictionary['G'].visualFilename).convert('L'))
    npImgH=toNumpyMatrix(Image.open(figuresDictionary['H'].visualFilename).convert('L'))
    npImgARatio = np.count_nonzero(npImgA==0)/np.size(npImgA)
    npImgBRatio = np.count_nonzero(npImgB==0)/np.size(npImgB)
    npImgCRatio = np.count_nonzero(npImgC==0)/np.size(npImgC)
    npImgDRatio = np.count_nonzero(npImgD==0)/np.size(npImgD)
    npImgERatio = np.count_nonzero(npImgE==0)/np.size(npImgE)
    npImgFRatio = np.count_nonzero(npImgF==0)/np.size(npImgF)
    npImgGRatio = np.count_nonzero(npImgG==0)/np.size(npImgG)
    npImgHRatio = np.count_nonzero(npImgH==0)/np.size(npImgH)
    return {"A" : npImgARatio,
            "B" : npImgBRatio,
            "C" : npImgCRatio,
            "D" : npImgDRatio, 
            "E" : npImgERatio, 
            "F" : npImgFRatio, 
            "G" : npImgGRatio, 
            "H" : npImgHRatio}

def getAnswersBWRatio(figuresDictionary):
    npAns1=toNumpyMatrix(Image.open(figuresDictionary['1'].visualFilename).convert('L'))
    npAns2=toNumpyMatrix(Image.open(figuresDictionary['2'].visualFilename).convert('L'))
    npAns3=toNumpyMatrix(Image.open(figuresDictionary['3'].visualFilename).convert('L'))
    npAns4=toNumpyMatrix(Image.open(figuresDictionary['4'].visualFilename).convert('L'))
    npAns5=toNumpyMatrix(Image.open(figuresDictionary['5'].visualFilename).convert('L'))
    npAns6=toNumpyMatrix(Image.open(figuresDictionary['6'].visualFilename).convert('L'))
    npAns7=toNumpyMatrix(Image.open(figuresDictionary['7'].visualFilename).convert('L'))
    npAns8=toNumpyMatrix(Image.open(figuresDictionary['8'].visualFilename).convert('L'))
    npAns1Ratio = np.count_nonzero(npAns1==0)/np.size(npAns1)
    npAns2Ratio = np.count_nonzero(npAns2==0)/np.size(npAns2)
    npAns3Ratio = np.count_nonzero(npAns3==0)/np.size(npAns3)
    npAns4Ratio = np.count_nonzero(npAns4==0)/np.size(npAns4)
    npAns5Ratio = np.count_nonzero(npAns5==0)/np.size(npAns5)
    npAns6Ratio = np.count_nonzero(npAns6==0)/np.size(npAns6)
    npAns7Ratio = np.count_nonzero(npAns7==0)/np.size(npAns7)
    npAns8Ratio = np.count_nonzero(npAns8==0)/np.size(npAns8)
    return {"1" : npAns1Ratio,
            "2" : npAns2Ratio,
            "3" : npAns3Ratio,
            "4" : npAns4Ratio,
            "5" : npAns5Ratio,
            "6" : npAns6Ratio,
            "7" : npAns7Ratio,
            "8" : npAns8Ratio}

def getImageDiffs(images):
    horizontal =   {"AB" : np.abs(images["A"] - images["B"]),
                    "BC" : np.abs(images["B"] - images["C"]),
                    "AC" : np.abs(images["A"] - images["C"]),
                    "DE" : np.abs(images["D"] - images["E"]),
                    "EF" : np.abs(images["E"] - images["F"]),
                    "DF" : np.abs(images["D"] - images["F"]),
                    "GH" : np.abs(images["G"] - images["H"])}
    
    vertical =     {"AD" : np.abs(images["A"] - images["D"]),
                    "DG" : np.abs(images["D"] - images["G"]),
                    "AG" : np.abs(images["A"] - images["G"]),
                    "BE" : np.abs(images["B"] - images["E"]),
                    "EH" : np.abs(images["E"] - images["H"]),
                    "BH" : np.abs(images["B"] - images["H"]),
                    "CF" : np.abs(images["C"] - images["F"])}
    
    fwdDiagonal =  {"BD" : np.abs(images["B"] - images["D"]),
                    "CE" : np.abs(images["C"] - images["E"]),
                    "EG" : np.abs(images["E"] - images["G"]),
                    "CG" : np.abs(images["C"] - images["G"]),
                    "FH" : np.abs(images["F"] - images["H"])}
    
    bckDiagonal =  {"AE" : np.abs(images["A"] - images["E"]),
                    "BF" : np.abs(images["B"] - images["F"]),
                    "DH" : np.abs(images["D"] - images["H"])}

    return  {"horizontal"   : horizontal,
             "vertical"     : vertical,
             "fwdDiagonal"  : fwdDiagonal,
             "bckDiagonal"  : bckDiagonal}

def imgDiffAnalysis(imgDiffs, images, answers):

    picks = {}
    solidFill = False
    threshold = 0.003
    
    # Evaluate across horizontal image comparison 
    if (imgDiffs["horizontal"]["AB"] < threshold and
        imgDiffs["horizontal"]["BC"] < threshold and
        imgDiffs["horizontal"]["AC"] < threshold and
        imgDiffs["horizontal"]["DE"] < threshold and
        imgDiffs["horizontal"]["EF"] < threshold and
        imgDiffs["horizontal"]["DF"] < threshold and
        imgDiffs["horizontal"]["GH"] < threshold) :
        val1 = images["H"] - threshold
        val2 = images["H"] + threshold
        for key, value in answers.items():
            if val1 < value < val2:
                picks[key] = value
        if len(picks)==1:
            #print ("A return", picks)
            return picks
        #print ("A", picks)
    
    # Evaluate across vertical image comparison 
    if (imgDiffs["vertical"]["AD"] < threshold and
        imgDiffs["vertical"]["DG"] < threshold and
        imgDiffs["vertical"]["AG"] < threshold and
        imgDiffs["vertical"]["BE"] < threshold and
        imgDiffs["vertical"]["EH"] < threshold and
        imgDiffs["vertical"]["BH"] < threshold and
        imgDiffs["vertical"]["CF"] < threshold) :
        val1 = images["F"] - threshold
        val2 = images["F"] + threshold
        for key, value in answers.items():
            if val1 < value < val2:
                picks[key] = value
        if len(picks)==1:
            #print ("B return", picks)
            return picks
        #print ("B",picks)

    # Evaluate across corner comparison
    if (imgDiffs["horizontal"]["AC"] < threshold and
        imgDiffs["horizontal"]["DF"] < threshold and
        imgDiffs["vertical"]["AG"] < threshold and
        imgDiffs["vertical"]["BH"] < threshold) :
        val1 = images["G"] - threshold
        val2 = images["G"] + threshold
        val3 = images["C"] - threshold
        val4 = images["C"] + threshold
        for key, value in answers.items():
            if val1 < value < val2 or val3 < value < val4:
                picks[key] = value
        if len(picks)==1:
            #print ("C return", picks)
            return picks
        #print ("C",picks)

    # Evaluate across increasing image comparison 
    if (imgDiffs["fwdDiagonal"]["BD"] < threshold and
        imgDiffs["fwdDiagonal"]["CG"] < threshold and
        images["F"] - images["C"] > threshold and
        images["H"] - images["G"] > threshold) :
        for key, value in answers.items():
            if value >= images["H"] + threshold:
                picks[key] = value
        if len(picks)==1:
            #print ("D return", picks)
            return picks
        #print ("D", picks)

    # Evaluate across decreasing image comparison 
    if (imgDiffs["fwdDiagonal"]["BD"] < threshold and
        imgDiffs["fwdDiagonal"]["CG"] < threshold and
        images["C"] - images["F"] > threshold and
        images["G"] - images["H"] > threshold) :
        for key, value in answers.items():
            if value <= images["H"] - threshold:
                picks[key] = value
        if len(picks)==1:
            #print ("E return", picks)
            return picks
        #print ("E", picks)
    
    # Evaluate across back diagonal
    if(imgDiffs["bckDiagonal"]["AE"] < threshold and
       imgDiffs["bckDiagonal"]["BF"] < threshold and
       imgDiffs["bckDiagonal"]["DH"] < threshold):
        val1 = images["E"] - threshold
        for key, value in answers.items():
            if val1<= value <= images["E"] + threshold:
                picks[key] = value
        if len(picks)==1:
            #print ("F return", picks)
            return picks
        #print ("F", picks)
    
    return picks

def imgOverlayAnalysis(npImages, npAnswers):
    img1 = npImages["A"]
    img2 = npImages["B"]
    img3 = npImages["C"]
    tempImg = np.array(img1, copy=True)
    tempImg.fill(1)

    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            tempImg[i,j] = img1[i,j] ^ img2[i,j]

    a = np.count_nonzero((img3 - tempImg)==0)

    img1 = npImages["D"]
    img2 = npImages["E"]
    img3 = npImages["F"]
    tempImg = np.array(img1, copy=True)
    tempImg.fill(1)

    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            tempImg[i,j] = img1[i,j] ^ img2[i,j]

    b = np.count_nonzero((img3 - tempImg)==0)

    img1 = npImages["A"]
    img2 = npImages["D"]
    img3 = npImages["G"]
    tempImg = np.array(img1, copy=True)
    tempImg.fill(1)

    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            tempImg[i,j] = img1[i,j] ^ img2[i,j]

    c = np.count_nonzero((img3 - tempImg)==0)

    img1 = npImages["B"]
    img2 = npImages["E"]
    img3 = npImages["H"]
    tempImg = np.array(img1, copy=True)
    tempImg.fill(1)

    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            tempImg[i,j] = img1[i,j] ^ img2[i,j]

    d = np.count_nonzero((img3 - tempImg)==0)
    #print ("ABC XOR 0s=",a, " DEF XOR 0s=",b, " ADG XOR 0s=",c, " BEH XOR 0s=",d)

    threshold = 2000
    img1 = npImages["G"]
    img2 = npImages["H"]
    img3 = npImages["C"]
    img4 = npImages["F"]
    #XOR by rows and columns
    if(a<threshold and b<threshold and c<threshold and d<threshold):
        rowImg = np.array(img1, copy=True)
        colImg = np.array(img1, copy=True)
        rowImg.fill(1)
        colImg.fill(1)
        
        for i in range(img1.shape[0]):
            for j in range(img1.shape[1]):
                rowImg[i,j] = img1[i,j] ^ img2[i,j]
                colImg[i,j] = img3[i,j] ^ img4[i,j]
        
        answer = ""
        rowAns = colAns = 10000
        for key, ansImg in npAnswers.items():
            rowTemp = np.count_nonzero((rowImg-ansImg)==0)
            colTemp = np.count_nonzero((colImg-ansImg)==0)
            if (colTemp < colAns and rowTemp < rowAns):
                rowAns = rowTemp
                colAns = colTemp
                answer = key    

        return answer
    return -1
    
def imgCompositeAnalysis(npImages, npAnswers):
    img1 = npImages["A"]
    img2 = npImages["B"]
    img3 = npImages["C"]
    tempImg = np.array(img1, copy=True)
    tempImg.fill(1)

    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            tempImg[i,j] = 0
            if(img1[i,j]== 0 or img2[i,j]==0):
                tempImg[i,j] = 1

    a = np.count_nonzero((img3 - tempImg)==0)

    img1 = npImages["D"]
    img2 = npImages["E"]
    img3 = npImages["F"]
    tempImg = np.array(img1, copy=True)
    tempImg.fill(1)

    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            tempImg[i,j] = 0
            if(img1[i,j]== 0 or img2[i,j]==0):
                tempImg[i,j] = 1

    b = np.count_nonzero((img3 - tempImg)==0)

    
    img1 = npImages["A"]
    img2 = npImages["D"]
    img3 = npImages["G"]
    tempImg = np.array(img1, copy=True)
    tempImg.fill(1)

    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            tempImg[i,j] = 0
            if(img1[i,j]== 0 or img2[i,j]==0):
                tempImg[i,j] = 1

    c = np.count_nonzero((img3 - tempImg)==0)

    
    img1 = npImages["B"]
    img2 = npImages["E"]
    img3 = npImages["H"]
    tempImg = np.array(img1, copy=True)
    tempImg.fill(1)

    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            tempImg[i,j] = 0
            if(img1[i,j]== 0 or img2[i,j]==0):
                tempImg[i,j] = 1

    d = np.count_nonzero((img3 - tempImg)==0)

    threshold = 1000
    img1 = npImages["G"]
    img2 = npImages["H"]
    img3 = npImages["C"]
    img4 = npImages["F"]
    #XOR by rows and columns
    if(a<threshold and b<threshold and c<threshold and d<threshold):
        rowImg = np.array(img1, copy=True)
        colImg = np.array(img1, copy=True)
        rowImg.fill(1)
        colImg.fill(1)
        
        for i in range(img1.shape[0]):
            for j in range(img1.shape[1]):
                rowImg[i,j] = 0
                if(img1[i,j]== 0 or img2[i,j]==0):
                    rowImg[i,j] = 1
                colImg[i,j] = 0
                if(img3[i,j]== 0 or img4[i,j]==0):
                    colImg[i,j] = 1
        
        answer = ""
        rowAns = colAns = 10000
        for key, ansImg in npAnswers.items():
            rowTemp = np.count_nonzero((rowImg-ansImg)==0)
            colTemp = np.count_nonzero((colImg-ansImg)==0)
            if (colTemp < colAns and rowTemp < rowAns):
                rowAns = rowTemp
                colAns = colTemp
                answer = key    
        return answer
    return -1
    
def imgCommonAnalysis(npImages, npAnswers):
    img1 = npImages["A"]
    img2 = npImages["B"]
    img3 = npImages["C"]
    tempImg = np.array(img1, copy=True)
    tempImg.fill(1)

    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            tempImg[i,j] = 0
            if(img1[i,j]== 0 and img2[i,j]==0):
                tempImg[i,j] = 1

    a = np.count_nonzero((img3 - tempImg)==0)

    img1 = npImages["D"]
    img2 = npImages["E"]
    img3 = npImages["F"]
    tempImg = np.array(img1, copy=True)
    tempImg.fill(1)

    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            tempImg[i,j] = 0
            if(img1[i,j]== 0 and img2[i,j]==0):
                tempImg[i,j] = 1

    b = np.count_nonzero((img3 - tempImg)==0)

    
    img1 = npImages["A"]
    img2 = npImages["D"]
    img3 = npImages["G"]
    tempImg = np.array(img1, copy=True)
    tempImg.fill(1)

    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            tempImg[i,j] = 0
            if(img1[i,j]== 0 and img2[i,j]==0):
                tempImg[i,j] = 1

    c = np.count_nonzero((img3 - tempImg)==0)

    
    img1 = npImages["B"]
    img2 = npImages["E"]
    img3 = npImages["H"]
    tempImg = np.array(img1, copy=True)
    tempImg.fill(1)

    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            tempImg[i,j] = 0
            if(img1[i,j]== 0 and img2[i,j]==0):
                tempImg[i,j] = 1

    d = np.count_nonzero((img3 - tempImg)==0)

    threshold = 1000
    img1 = npImages["G"]
    img2 = npImages["H"]
    img3 = npImages["C"]
    img4 = npImages["F"]
    #AND by rows and columns
    if(a<threshold and b<threshold and c<threshold and d<threshold):
        rowImg = np.array(img1, copy=True)
        colImg = np.array(img1, copy=True)
        rowImg.fill(1)
        colImg.fill(1)
        
        for i in range(img1.shape[0]):
            for j in range(img1.shape[1]):
                rowImg[i,j] = 0
                if(img1[i,j]== 0 and img2[i,j]==0):
                    rowImg[i,j] = 1
                colImg[i,j] = 0
                if(img3[i,j]== 0 and img4[i,j]==0):
                    colImg[i,j] = 1
        
        answer = ""
        rowAns = colAns = 10000
        for key, ansImg in npAnswers.items():
            rowTemp = np.count_nonzero((rowImg-ansImg)==0)
            colTemp = np.count_nonzero((colImg-ansImg)==0)
            if (colTemp < colAns and rowTemp < rowAns):
                rowAns = rowTemp
                colAns = colTemp
                answer = key    
        return answer
    return -1

def imgSumAnalysis(images, answers, npImages, npAnswers):

    findThreshold = 0.4
    answerThreshold = 0.2
    rowMatch, colMatch = False, False

    if abs(images["A"] - abs(images["B"] + images["C"])) < findThreshold and abs(images["D"] - abs(images["E"] + images["F"])) < findThreshold:
        rowMatch = True
    if abs(images["A"] - abs(images["D"] + images["G"])) < findThreshold and abs(images["B"] - abs(images["E"] + images["H"])) < findThreshold:
        colMatch = True

    if rowMatch and colMatch:
        rowRateRange = [images["G"] - images["H"] - answerThreshold, images["G"] - images["H"] + answerThreshold]
        colRateRange = [images["C"] - images["F"] - answerThreshold, images["C"] - images["F"] + answerThreshold]
        answerRange = [min(rowRateRange[0],colRateRange[0]), max(rowRateRange[1],colRateRange[1])]

        results = []

        for ansKey, answer in answers.items():
            if (0.225 <= answer <= 0.235):
                for imgKey, image in images.items():
                    a = np.count_nonzero((npAnswers[ansKey] - npImages[imgKey])==1)
                    if (0 < a < 1000 and ansKey not in results):
                        results.append(ansKey)

        if len(results) == 1:
            return results[0]

    return -1

def imgEdgeAnalysis(imgDiffs, images, answers, thresholdAdd, thresholdDel):
    picks = {}
    copyPicks = {}
    if(abs(imgDiffs["horizontal"]["AB"]-imgDiffs["horizontal"]["BC"]) < thresholdAdd and abs(imgDiffs["horizontal"]["DE"]-imgDiffs["horizontal"]["EF"]) < thresholdAdd):
        target = imgDiffs["horizontal"]["GH"]
        for key,value in answers.items():
            common = images["H"] - value
            #print (key, "common, target, subtracted", common, target, abs(common - target))
            if abs(common - target) < thresholdAdd:
                picks[key] = value
        t = list(picks.values())
        if(len(picks)==2 and t[0] == t[1]):
            return picks
        copyPicks = copy.deepcopy(picks)
        if len(picks) > 1:
            for keyAns, ans in picks.items():
                for keyImg, img in images.items():
                    #print("ans",keyAns, ans,"img", keyImg, img,"diff", abs(ans-img))
                    if(abs(ans-img) < thresholdDel):
                        del copyPicks[keyAns]
                        break
        #print ("G", picks, copyPicks)
        if(len(copyPicks)==0):
            return picks
        return copyPicks
    
    picks = {}  
    copyPicks = {}
    if(abs(imgDiffs["vertical"]["AD"]-imgDiffs["vertical"]["DG"]) < thresholdAdd and abs(imgDiffs["vertical"]["BE"]-imgDiffs["vertical"]["EH"]) < thresholdAdd):
        target = imgDiffs["vertical"]["CF"]
        results = []
        for key,value in answers.items():
            common = images["F"] - value
            #print (key, "common, target, subtracted", common, target, abs(common - target))
            if abs(common - target) < thresholdAdd:
                picks[key] = value
        t = list(picks.values())
        if(len(picks)==2 and t[0] == t[1]):
            return picks
        copyPicks = copy.deepcopy(picks)
        if len(picks) > 1:
            for keyAns, ans in picks.items():
                for keyImg, img in images.items():
                    #print("ans",keyAns, ans,"img", keyImg, img,"diff", abs(ans-img))
                    if(abs(ans-img) < thresholdDel):
                        del copyPicks[keyAns]
                        break
        #print ("H", picks, copyPicks)
        if(len(copyPicks)==0):
            return picks
        return copyPicks
    
    return copyPicks

class Agent:

    # The default constructor for your Agent. Make sure to execute any
    # processing necessary before your Agent starts solving problems here.
    #
    # Do not add any variables to this signature; they will not be used by
    # main().
    def __init__(self):
        pass

    def solve3x3(self, problem):
        retVal = -1

        imageDictionary = {}
        figuresDictionary = {}
        for name in ['A','B','C','D','E','F','G','H','1','2','3','4','5','6','7','8']:
            imageDictionary[name] = Image.open(problem.figures[name].visualFilename)
            figuresDictionary[name] = toNumpyMatrix(imageDictionary[name])
            
        solutionsDictionary = {k: figuresDictionary[k] for k in figuresDictionary.keys() & {'1','2','3','4','5','6','7','8'}}

        hMirrorA = toNumpyMatrix(ImageOps.mirror(imageDictionary['A']))
        hMirrorB = toNumpyMatrix(ImageOps.mirror(imageDictionary['B']))
        hMirrorC = toNumpyMatrix(ImageOps.mirror(imageDictionary['C']))
        hMirrorD = toNumpyMatrix(ImageOps.mirror(imageDictionary['D']))
        hMirrorE = toNumpyMatrix(ImageOps.mirror(imageDictionary['E']))
        hMirrorF = toNumpyMatrix(ImageOps.mirror(imageDictionary['F']))
        hMirrorG = toNumpyMatrix(ImageOps.mirror(imageDictionary['G']))
        hMirrorH = toNumpyMatrix(ImageOps.mirror(imageDictionary['H']))
        vMirrorA = toNumpyMatrix(ImageOps.flip(imageDictionary['A']))
        vMirrorB = toNumpyMatrix(ImageOps.flip(imageDictionary['B']))
        vMirrorC = toNumpyMatrix(ImageOps.flip(imageDictionary['C']))
        vMirrorD = toNumpyMatrix(ImageOps.flip(imageDictionary['D']))
        vMirrorE = toNumpyMatrix(ImageOps.flip(imageDictionary['E']))
        vMirrorF = toNumpyMatrix(ImageOps.flip(imageDictionary['F']))
        vMirrorG = toNumpyMatrix(ImageOps.flip(imageDictionary['G']))
        vMirrorH = toNumpyMatrix(ImageOps.flip(imageDictionary['H']))
    
        if retVal == -1: # A == B == C
            similar1, mean1, rms1, bbox1 = diffStatistics(figuresDictionary['A'], figuresDictionary['B'])
            similar2, mean2, rms2, bbox2 = diffStatistics(figuresDictionary['B'], figuresDictionary['C'])
            bestMeanDiff = 1
            if similar1 and similar2:
                for key, value in solutionsDictionary.items():
                    similar, mean, rms, bbox = diffStatistics(figuresDictionary['G'], value)
                    if similar and abs(mean-mean1) < bestMeanDiff:
                        bestMeanDiff = abs(mean-mean1)
                        retVal = int(key)
        if retVal == -1: # A == D == G
            similar1, mean1, rms1, bbox1 = diffStatistics(figuresDictionary['A'], figuresDictionary['D'])
            similar2, mean2, rms2, bbox2 = diffStatistics(figuresDictionary['D'], figuresDictionary['G'])
            bestMeanDiff = 1
            if similar1 and similar2:
                for key, value in solutionsDictionary.items():
                    similar, mean, rms, bbox = diffStatistics(figuresDictionary['C'], value)
                    if similar and abs(mean-mean1) < bestMeanDiff:
                        bestMeanDiff = abs(mean-mean1)
                        retVal = int(key)
        if retVal == -1: # percentage
            similar1, mean1, rms1, bbox1 = diffStatistics(figuresDictionary['A'], figuresDictionary['B'])
            similar2, mean2, rms2, bbox2 = diffStatistics(figuresDictionary['B'], figuresDictionary['C'])
            similar12, mean12, rms12, bbox12 = diffStatistics(figuresDictionary['A'], figuresDictionary['C'])
            similar3, mean3, rms3, bbox3 = diffStatistics(figuresDictionary['D'], figuresDictionary['E'])
            similar4, mean4, rms4, bbox4 = diffStatistics(figuresDictionary['E'], figuresDictionary['F'])
            similar34, mean34, rms34, bbox34 = diffStatistics(figuresDictionary['D'], figuresDictionary['F'])
            similar5, mean5, rms5, bbox5 = diffStatistics(figuresDictionary['G'], figuresDictionary['H'])

            similar1, mean6, rms6, bbox1 = diffStatistics(figuresDictionary['A'], figuresDictionary['D'])
            similar2, mean7, rms7, bbox2 = diffStatistics(figuresDictionary['D'], figuresDictionary['G'])
            similar3, mean8, rms8, bbox3 = diffStatistics(figuresDictionary['B'], figuresDictionary['E'])
            similar4, mean9, rms9, bbox4 = diffStatistics(figuresDictionary['E'], figuresDictionary['H'])
            similar5, mean0, rms0, bbox5 = diffStatistics(figuresDictionary['C'], figuresDictionary['F'])

            rms1 = round(rms1,3)
            rms2 = round(rms2,3)
            rms12 = round(rms12,3)
            rms3 = round(rms3,3)
            rms4 = round(rms4,3)
            rms34 = round(rms34,3)
            rms5 = round(rms5,3)
            rms6 = round(rms6,3)
            rms7 = round(rms7,3)
            rms8 = round(rms8,3)
            rms9 = round(rms9,3)
            rms0 = round(rms0,3)

            k = round(rms2-rms0,3)
            j = round(rms7-rms5,3)
            
            for key, value in solutionsDictionary.items():
                similar, meana, rmsa, bbox = diffStatistics(figuresDictionary['F'], value)
                similar, meanb, rmsb, bbox = diffStatistics(figuresDictionary['H'], value)
                rmsa = round(rmsa,3)
                rmsb = round(rmsb,3)
                if(meana==round(mean0-k,2) and meanb==round(mean5-j,2)):
                    retVal = int(key)
        return retVal
    
    def solve2x2(self, problem):

        retVal = -1

        imageDictionary = {}
        figuresDictionary = {}
        for name in ['A','B','C','1','2','3','4','5','6']:
            imageDictionary[name] = Image.open(problem.figures[name].visualFilename)
            figuresDictionary[name] = toNumpyMatrix(imageDictionary[name])
            
        solutionsDictionary = {k: figuresDictionary[k] for k in figuresDictionary.keys() & {'1','2','3','4','5','6'}}

        hMirrorA = toNumpyMatrix(ImageOps.mirror(imageDictionary['A']))
        hMirrorB = toNumpyMatrix(ImageOps.mirror(imageDictionary['B']))
        hMirrorC = toNumpyMatrix(ImageOps.mirror(imageDictionary['C']))
        vMirrorA = toNumpyMatrix(ImageOps.flip(imageDictionary['A']))
        vMirrorB = toNumpyMatrix(ImageOps.flip(imageDictionary['B']))
        vMirrorC = toNumpyMatrix(ImageOps.flip(imageDictionary['C']))
    
        if retVal == -1: # A == B
            retVal = equate2x2(figuresDictionary, solutionsDictionary, figuresDictionary['A'], figuresDictionary['B'], figuresDictionary['C'])
        if retVal == -1: # A == C
            retVal = equate2x2(figuresDictionary, solutionsDictionary, figuresDictionary['A'], figuresDictionary['C'], figuresDictionary['B'])
        if retVal == -1: # A horizontal mirror B
            retVal = equate2x2(figuresDictionary, solutionsDictionary, hMirrorA, figuresDictionary['B'], hMirrorC)
        if retVal == -1: # A horizontal mirror C
            retVal = equate2x2(figuresDictionary, solutionsDictionary, hMirrorA, figuresDictionary['C'], hMirrorB)
        if retVal == -1: # A vertical mirror B
            retVal = equate2x2(figuresDictionary, solutionsDictionary, vMirrorA, figuresDictionary['B'], vMirrorC)
        if retVal == -1: # A vertical mirror C
            retVal = equate2x2(figuresDictionary, solutionsDictionary, vMirrorA, figuresDictionary['C'], vMirrorB)
        if retVal == -1: # A is outline of B solid
            retVal = equate2x2(figuresDictionary, solutionsDictionary, figuresDictionary['A'], figuresDictionary['B'],figuresDictionary['C'], 0.96, 0.90)
        if retVal == -1: # A is outline of C solid
            retVal = equate2x2(figuresDictionary, solutionsDictionary, figuresDictionary['A'], figuresDictionary['C'],figuresDictionary['B'], 0.96, 0.90)
        if retVal == -1: # Rotate image
            degrees = [-45,-90,-135,-180,-225,-270,-315]
            background = Image.new('RGBA', imageDictionary['A'].size, 'white')

            for angle in degrees:
                #print ("Rotate Image ", angle, " degrees")

                RotateA = imageDictionary['A'].rotate(angle)
                RotateB = imageDictionary['B'].rotate(angle)
                RotateC = imageDictionary['C'].rotate(angle)
                
                rotateA = toNumpyMatrix(Image.composite(RotateA, background, RotateA))
                rotateB = toNumpyMatrix(Image.composite(RotateB, background, RotateB))
                rotateC = toNumpyMatrix(Image.composite(RotateC, background, RotateC))

                if retVal == -1: # A horizontal mirror B
                    similarA, meanA, rmsA, bboxA = diffStatistics(rotateA, figuresDictionary['B'])
                    #print ("A rotated ",angle," degrees to B mean=",meanA," rms=",rmsA," diff=",abs(meanA-rmsA))
                    retVal = equate2x2(figuresDictionary, solutionsDictionary, rotateA, figuresDictionary['B'], rotateC)
                if retVal == -1: # A horizontal mirror C
                    similarA, meanA, rmsA, bboxA = diffStatistics(rotateA, figuresDictionary['C'])
                    #print ("A rotated ",angle," degrees to C mean=",meanA," rms=",rmsA," diff=",abs(meanA-rmsA))
                    retVal = equate2x2(figuresDictionary, solutionsDictionary, rotateA, figuresDictionary['C'], rotateB)
                if retVal > 0:
                    break
        if retVal == -1:
            retVal = randint(1, 6)
        
        return retVal
    
    # The primary method for solving incoming Raven's Progressive Matrices.
    # For each problem, your Agent's Solve() method will be called. At the
    # conclusion of Solve(), your Agent should return an int representing its
    # answer to the question: 1, 2, 3, 4, 5, or 6. Strings of these ints
    # are also the Names of the individual RavensFigures, obtained through
    # RavensFigure.getName(). Return a negative number to skip a problem.
    #
    # Make sure to return your answer *as an integer* at the end of Solve().
    # Returning your answer as a string may cause your program to crash.
    def Solve(self, problem):
        retVal = -1

        if (problem.problemType == '2x2'):
            retVal = self.solve2x2(problem)
            return retVal
        if (problem.problemType == '3x3'):
            retVal = self.solve3x3(problem)

            npImages = getNumPyImages(problem.figures)
            npAnswers = getNumPyAnswers(problem.figures)

            images = getImagesBWRatio(problem.figures)
            answers = getAnswersBWRatio(problem.figures)
            imgDiffs = getImageDiffs(images)
            picks = imgDiffAnalysis(imgDiffs, images, answers)
            k=-1
            if len(picks)==1:
                k = int(list(picks.keys())[0])
                print (problem.name, "final 3x3 Diffs\t", k) 
            elif (1 < len(picks) <= 3):
                lowVal = 1
                lowKey = 0
                for i in picks.keys():
                    val = picks[i]
                    if val == lowVal:
                        eq = False
                        for j in images:
                            same, mean, rms, bbox = diffStatistics(answers[i], images[j], 0.99)
                            sub=toNumpyMatrix(ImageChops.subtract(Image.open(problem.figures[i].visualFilename).convert('L'),Image.open(problem.figures[j].visualFilename).convert('L') ))
                            if np.count_nonzero(sub==0)/np.size(sub) == 1.0:
                                eq = True
                                break
                        if (eq==False):
                            lowVal = val
                            lowKey = int(i)
                    elif val < lowVal:
                        lowVal = val
                        lowKey = int(i)
                k = lowKey
                print (problem.name, "final 3x3 Low\t", k)

            if (k==-1):
                k=int(imgCompositeAnalysis(npImages, npAnswers))
                if (k>-1):
                    print (problem.name, "final 3x3 Composite\t", k)
            if (k==-1):
                k = int(imgOverlayAnalysis(npImages, npAnswers))
                if (k>-1):
                    print (problem.name, "final 3x3 Overlay\t", k)
            if (k==-1):
                k = int(imgCommonAnalysis(npImages, npAnswers))
                if (k>-1):
                    print (problem.name, "final 3x3 Common\t", k)
            if (k==-1):
                k = int(imgSumAnalysis(images, answers, npImages, npAnswers))
                if (k>-1):
                    print (problem.name, "final 3x3 Sum\t", k)
            if (k==-1):
                picks = imgEdgeAnalysis(imgDiffs, images, answers, 0.05, 0.02)
                if len(picks)==1:
                    k = int(list(picks.keys())[0])
                elif len(picks)==2:
                    a = int(list(picks.keys())[0])
                    b = int(list(picks.keys())[1])
                    if(b >= a):
                        k = a
                    else:
                        k = b
                if (k>-1):
                    print (problem.name, "final 3x3 Edge\t", k) 
            if (k==-1):
                kRange = ['1','2','3','4','5','6','7','8']
                for keyAns, ans in answers.items():
                    for keyImg, img in images.items():
                        #print (keyAns, keyImg, abs(ans-img))
                        if(abs(ans-img) < 0.01 and keyAns in kRange):
                            kRange.remove(keyAns)
                            break
                if(len(kRange)>0):
                    #print (kRange)
                    trKey = -1
                    trVal = 1
                    for key, val in answers.items():
                        if key in kRange and val<trVal:
                            trKey = key
                            trVal = val
                    k = int(random.choice(trKey))
                else:
                    k = randint(1,8)
                print (problem.name, "final 3x3 Guess\t", k)

            return k