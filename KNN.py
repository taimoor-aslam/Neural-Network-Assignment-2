import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import csv
import math
import operator
from sklearn import metrics, preprocessing
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

# =============================================================================
# Manual Implementataion 
# =============================================================================

data=pd.read_csv("F:/Semester 8/NN/Datasets/Simple/data.csv")
data=data.dropna()
x=data.drop(['label'],axis=1)
y=data['label']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
normalized_x=preprocessing.normalize(x)

# =============================================================================
# manhattenDistance Formula 
# =============================================================================

def manhattenDistance(data1, data2,length):
    distance = 0
    for x in range(length):
        distance += abs(data1[x] - data2[x])
    return (distance) 

# =============================================================================
# InfinityNorm Formula
# =============================================================================

def InfinityNorm(data1, data2,length):
    distance = []
    for x in range(length):
        distance.append(abs(data1[x] - data2[x]))
    return max(distance) 

# =============================================================================
# euclideanDistenace Formula
# =============================================================================

def euclideanDistenace(data1,data2,length):
    distance=0
    for x in range(length):
        distance+= pow((data1[x] - data2[x]),2)
    return math.sqrt(distance)

# =============================================================================
# Finding Neighbors
# =============================================================================

def getNeighbour(trainingset, testInstance, k):
    distance = []
    
    length = len(testInstance) - 1
    
    for x in range(len(trainingset)):
        dist = euclideanDistenace(testInstance, trainingset[x],length)
        distance.append((trainingset[x],dist))
    distance.sort(key= operator.itemgetter(1)) 
    neighbors = []
    
    for x in range(k):
        neighbors.append(distance[x][0])
    return neighbors

# =============================================================================
# Getting lebal which occurs more in neighbors
# =============================================================================

def getResponse(neighbors):
    classVotes = {}
    
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        
        if response in classVotes:
            classVotes[response] += 1
            
        else:
            classVotes[response] = 1
            
        sortedVotes = sorted(classVotes.items(),key=operator.itemgetter(1), reverse=True) 
        return sortedVotes[0][0]
    
# =============================================================================
#     Calculating the acuuracy
# =============================================================================
    
def getAccuracy(testset,perdiction):
    correct = 0
    
    for y in range(len(testset)):
        if testset[y][-1] == perdiction[y]:
            correct += 1
            #print(correct)
   # print(correct)
    return (correct/float(len(testset)))*100.0 

# =============================================================================
# Calculation
# =============================================================================

new_y_train=y_train.to_numpy().tolist()
x_train['label']=new_y_train
new_x_train=x_train.to_numpy().tolist()
trainset=new_x_train
new_x_test=x_test.to_numpy().tolist()
k=3
responce=[]
for testInstance in range(len(new_x_test)):  
                        neighbors = getNeighbour(trainset,new_x_test[testInstance],k)
                        res = getResponse(neighbors)
                        responce.append(res)
new_y_test=y_test.to_numpy().tolist()
x_test['label']=new_y_test
new_x_test=x_test.to_numpy().tolist()       
accuracy = getAccuracy(new_x_test, responce)
print(accuracy)

