# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 16:37:00 2022

@author: matti
"""

import random,string,math,csv
import numpy as np
import matplotlib.pyplot as plt

with open('C:/Users/matti/TF/training.csv', 'r') as f:
    data = list(csv.reader(f,delimiter = ','))


data = np.array(data)
xs = np.copy(data)
num_rows, num_col = xs.shape

xs = np.delete(xs,0,0)
xs = np.delete(xs,num_col-1,1)
xs = np.delete(xs,num_col-2,1)

xs = np.delete(xs,0,1)
xs = xs.astype(np.float64)

(numPoints,numFeatures) = xs.shape
xs = np.add(xs, np.random.normal(0.0, 0.0001, xs.shape))

sSelector = np.array([row[-1] == 's' for row in data[1:]])
bSelector = np.array([row[-1] == 'b' for row in data[1:]])

weights = np.array([float(row[-2]) for row in data[1:]])

sumWeights = np.sum(weights)
sumSWeights = np.sum(weights[sSelector])
sumBWeights = np.sum(weights[bSelector])

randomPermutation = random.sample(range(len(xs)), len(xs))
numPointsTrain = int(numPoints*0.9)
numPointsValidation = numPoints - numPointsTrain

xsTrain = xs[randomPermutation[:numPointsTrain]]
xsValidation = xs[randomPermutation[numPointsTrain:]]

sSelectorTrain = sSelector[randomPermutation[:numPointsTrain]]
bSelectorTrain = bSelector[randomPermutation[:numPointsTrain]]
sSelectorValidation = sSelector[randomPermutation[numPointsTrain:]]
bSelectorValidation = bSelector[randomPermutation[numPointsTrain:]]

weightsTrain = weights[randomPermutation[:numPointsTrain]]
weightsValidation = weights[randomPermutation[numPointsTrain:]]

sumWeightsTrain = np.sum(weightsTrain)
sumSWeightsTrain = np.sum(weightsTrain[sSelectorTrain])
sumBWeightsTrain = np.sum(weightsTrain[bSelectorTrain])


xsTrainTranspose = xsTrain.transpose()
weightsBalancedTrain = np.array([0.5 * weightsTrain[i]/sumSWeightsTrain
                                 if sSelectorTrain[i]
                                 else 0.5 * weightsTrain[i]/sumBWeightsTrain\
                                 for i in range(numPointsTrain)])
    
numBins = 10
logPs = np.empty([numFeatures, numBins])
binMaxs = np.empty([numFeatures, numBins])
binIndexes = np.array(range(0, numPointsTrain+1, int(numPointsTrain/numBins)))

for fI in range(numFeatures):
    # index permutation of sorted feature column
    indexes = xsTrainTranspose[fI].argsort()

    for bI in range(numBins):
        # upper bin limits
        binMaxs[fI, bI] = xsTrainTranspose[fI, indexes[binIndexes[bI+1]-1]]
        # training indices of points in a bin
        indexesInBin = indexes[binIndexes[bI]:binIndexes[bI+1]]
        # sum of signal weights in bin
        wS = np.sum(weightsBalancedTrain[indexesInBin]
                    [sSelectorTrain[indexesInBin]])
        # sum of background weights in bin
        wB = np.sum(weightsBalancedTrain[indexesInBin]
                    [bSelectorTrain[indexesInBin]])
        # log probability of being a signal in the bin
        logPs[fI, bI] = math.log(wS/(wS+wB))
        
def score(x):
    logP = 0
    for fI in range(numFeatures):
        bI = 0
        # linear search for the bin index of the fIth feature
        # of the signal
        while bI < len(binMaxs[fI]) - 1 and x[fI] > binMaxs[fI, bI]:
            bI += 1
        logP += logPs[fI, bI] - math.log(0.5)
    return logP




def AMS(s,b):
    assert s >= 0
    assert b >= 0
    bReg = 10.
    return math.sqrt(2 * ((s + b + bReg) * 
                          math.log(1 + s / (b + bReg)) - s))

validationScores = np.array([score(x) for x in xsValidation])
tIIs = validationScores.argsort()
wFactor = 1.* numPoints / numPointsValidation

s = np.sum(weightsValidation[sSelectorValidation])
b = np.sum(weightsValidation[bSelectorValidation])

amss = np.empty([len(tIIs)])
amsMax = 0
threshold = 0.0

for tI in range(len(tIIs)):
    # don't forget to renormalize the weights to the same sum 
    # as in the complete training set
    amss[tI] = AMS(max(0,s * wFactor),max(0,b * wFactor))
    if amss[tI] > amsMax:
        amsMax = amss[tI]
        threshold = validationScores[tIIs[tI]]
        #print tI,threshold
    if sSelectorValidation[tIIs[tI]]:
        s -= weightsValidation[tIIs[tI]]
    else:
        b -= weightsValidation[tIIs[tI]]
        
with open('C:/Users/matti/TF/test.csv', 'r') as f:
    test = list(csv.reader(f,delimiter = ','))

test = np.array(test)
xsTest = np.copy(test)
num_rows, num_col = xsTest.shape


xsTest = np.delete(xsTest,0,0)

#xsTest = np.delete(xs,num_col-1,1)
xsTest = np.delete(xsTest,0,1)
xsTest = xsTest.astype(np.float64)

testIds = np.array([int(row[0]) for row in test[1:]])
testScores = np.array([score(x) for x in xsTest])
testInversePermutation = testScores.argsort()
testPermutation = list(testInversePermutation)
for tI,tII in zip(range(len(testInversePermutation)),
                  testInversePermutation):
    testPermutation[tII] = tI

submission = np.array([[str(testIds[tI]),str(testPermutation[tI]+1),
                       's' if testScores[tI] >= threshold else 'b'] 
            for tI in range(len(testIds))])


submission = np.append([['EventId','RankOrder','Class']],
                        submission, axis=0)

np.savetxt("submission.csv",submission,fmt='%s',delimiter=',')
