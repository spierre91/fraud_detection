#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 14:49:13 2018

@author: spierre91
"""
import json
import pandas as pd 
from sklearn.neighbors import KNeighborsClassifier
import numpy as np 
from sklearn import model_selection

#read in model tuning parameters from json file 
with open('../Configuration/model_tuning.json') as f:
   data = json.load(f)
model = pd.DataFrame(data)

KNN_MODEL_DATA_CSV_PATH = model['training_data'][0]#the data used in the KNN model with labels for pump and dump and normal behavior 
NUMBEROF_NEIGHBORS = model['numberof_neighbors'][0]#the number of neighbors to be used in the KNN algo
NUMBER_OF_KNN_RUNS = model['number of accuracy runs'][0] #the number of times the KNN algorithm is run for train test split accuracy measure 
TARGET_THRESHOLD = model['target_threshold'][0]#parameter used to label training data as pump and dump

#function to calculate average accuracy across 1000 runs 
def knn_accuracy(NUMBER_OF_KNN_RUNS):
    accuracy = 0.0
    for i in range(NUMBER_OF_KNN_RUNS):
        #define the training data
        df = pd.read_csv(KNN_MODEL_DATA_CSV_PATH, index_col = 0)

        #labeling the training data with a rule: trades*returns> 60000 gets label=1, else: 0
        df['label'] = np.where(df['return'] * df['number_of_trades'] >= TARGET_THRESHOLD, 1, 0)

        #select the first four columns as features for training
        X_train = df.iloc[:, :4].values
        #the targets or labels are "1" for pump and "0" for no pump
        y_train = df.iloc[:, 4].values
        
        #Call the KNN method
        classifier = KNeighborsClassifier(n_neighbors = NUMBEROF_NEIGHBORS)
        #fit the training data to the KNN model 
        classifier.fit(X_train, y_train)
        
        #split training data into training set and set set
        #80% of the data is for training, 20% of the data is for testing 
        X_train_val, X_test_val, y_train_val, y_test_val \
        = model_selection.train_test_split(X_train,y_train,test_size=0.2)
        
        #take a sum of each accuracy result and calculate the average 
        accuracy += classifier.score(X_test_val, y_test_val)
    accuracy = (accuracy / (NUMBER_OF_KNN_RUNS)*100.0)
    return accuracy
print(knn_accuracy(NUMBER_OF_KNN_RUNS))
