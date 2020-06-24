#importing libraries
import numpy as np
import pandas as pd
import math
import operator
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

#loading dataset
df=pd.read_csv("C:/Users/havish/Desktop/iris.csv")

#splitting the data into train and test
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.25)

#Calculating the euclidean distance 
def euclideanDistance(instance1, instance2, length):
	distance = 0
	for x in range(length):
		distance += pow((instance1[x] - instance2[x]), 2)
	return math.sqrt(distance)

#Applying KNN Algorithm
def KNearestNeighbour(trainingSet, testInstance, k):
    distances = {}
    length = testInstance.shape[1]
    for x in range(len(trainingSet)):
        dist = E_Distance(testInstance, trainingSet.iloc[x], length)
        distances[x] = dist[0]
    sortdist = sorted(distances.items(), key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(sortdist[x][0])
    Count = {}  # to get most frequent class of rows
    for x in range(len(neighbors)):
        response = trainingSet.iloc[neighbors[x]][-1]
        if response in Count:
            Count[response] += 1
        else:
            Count[response] = 1
    sortcount = sorted(Count.items(), key=operator.itemgetter(1), reverse=True)
    return (sortcount[0][0], neighbors)

#calculating accuracy
def getAccuracy(testSet, predictions):
correct = 0
for x in range(len(testSet)):
if testSet[x][-1] is predictions[x]:
correct += 1

return (correct/float(len(testSet))) * 100.0
accuracy = getAccuracy(testSet, predictions)

#checking the result for different k values 
k = 1
k1 = 3
k2 = 11
result, neigh = KNearestNeighbour(dataset, test, k)
result1, neigh1 = KNearestNeighbour(dataset, test, k1)
result2, neigh2 = KNearestNeighbour(dataset, test, k2)
print(result)
print(neigh)
print(result1)
print(neigh1)
print(result2)
print(neigh2)



