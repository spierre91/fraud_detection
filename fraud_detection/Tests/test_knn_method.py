#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 11:47:55 2018

@author: spierre91
"""
import unittest 
import numpy as np
import pandas as pd 
import json
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

#read in model tuning parameters from json file 
with open('../Configuration/model_tuning.json') as f:
   data = json.load(f)
model = pd.DataFrame(data)
KNN_MODEL_DATA_CSV_PATH = model['training_data'][0]#the data used in the KNN model with labels for pump and dump and normal behavior 
TARGET_THRESHOLD = model['target_threshold'][0]#parameter used to label training data as pump and dump
NUMBEROF_NEIGHBORS = model['numberof_neighbors'][0]#the number of neighbors to be used in the KNN algo
DUMP_THRESHOLD = model['dump_threshold'][0]#the threshold for the decrease in returns followed by a local max used in the KNN+rule engine


def knn_engine(test_data, returns_threshold):
    #define the training data
    df = pd.read_csv(KNN_MODEL_DATA_CSV_PATH, index_col = 0)
    
    dump_magnitude_list = [100]#list of dips in magnitude followed by max returns

    #assumption: labeling the training data with a rule: trades*returns> TARGET_THRESHOLD gets label=1, else: 0
    #the choice for this rule is based on exploratory data analysis of historical trade data on the binance exchange
    #on the dates of June - September. Cases of normal market behavior statistical have a 
    #value of returns*trades several orders of magnitude less than the Target_threshold
    #The labeling doesn't account for crypto whales (large volume, small number of trades), 
    #though the KNN algo should detect whales based on the distances of additional features: returns, volatility and volume
    df['label'] = np.where(df['return'] * df['number_of_trades'] >= TARGET_THRESHOLD, 1, 0)
    
    #select the first four columns as features for training
    X_train = df.iloc[:, :4].values
    #the targets or labels are "1" for pump and "0" for no pump
    y_train = df.iloc[:, 4].values
    
    #initializing lists
    X_test_features = []
    X_transform_features = []
    y_pred = []
    rule_knn_result = []
    knn_output =[]
    returns = test_data[0]
    volatility = test_data[1]
    volume = test_data[2]
    trades = test_data[3]
    #appending features: returns, volatility, volume and trades 
    for i in range(len(returns)): 
        X_test_features.append([[returns[i], volatility[i],  volume[i], trades[i]]])
        
        #standardize data (the method subtracts the mean of feature values and divides by std)
        scaler = StandardScaler()
        scaler.fit(X_train)
        
        #transform applys the standardaztion method to train and test features
        X_train = scaler.transform(X_train)
        X_transform_features.append(scaler.transform(X_test_features[i]))
        
        #this part of the code trains the KNN algorithm with K=5 nearest neighbors
        classifier = KNeighborsClassifier(n_neighbors=NUMBEROF_NEIGHBORS)
        classifier.fit(X_train, y_train)
        
        #returns the list of target/label given test features
        y_pred.append(classifier.predict(X_transform_features[i]))
    #appending target values pump and dump: 1, normal market behavior: 0
    for i in range(len(returns)): 
        #targets are label as pump and dump if they are classified by knn and satisfy the dump rule
        if y_pred[i] == 1 and dump_magnitude_list[i] > DUMP_THRESHOLD:
            rule_knn_result.append(y_pred[i])
        else:
            rule_knn_result.append([0])
    #return a dictionary with label, values of maximum returns, and timestamps 
    for i in range(len(returns)):    
        knn_output.append({'target{}'.format(i): rule_knn_result[i],
                          'return max':returns[i], 'dump': dump_magnitude_list[i]})
    return knn_output 

class FeaturesTestCase(unittest.TestCase):
    #[returns, volatility, volume, trades]
    #test with all features large
    def test_largefeatures(self):
        test_all_features_large = [[100.0], [20.0], [10000.0], [10000.0]]
        returns_threshold = 10.0
        self.assertEqual(knn_engine(test_all_features_large, returns_threshold)[0]['target0'][0], 1)
    #[returns, volatility, volume, trades]
    #test with large volume, large returns (model whale behavior) 
    def test_largevolume(self):
        test_largevolume = [[50.0], [2.0], [1000000.0], [10.0]]
        returns_threshold = 10.0
        self.assertEqual(knn_engine(test_largevolume, returns_threshold)[0]['target0'][0], 1)
    #[returns, volatility, volume, trades]
    #test with small features     
    def test_smallfeatures(self):
        #[returns, volatility, volume, trades]
        #test with all features small
        test_all_features_small = [[1.0], [2.0], [1.0], [1.0]]
        returns_threshold = 10.0
        self.assertEqual(knn_engine(test_all_features_small, returns_threshold)[0]['target0'][0], 0)
        
if __name__ == '__main__':
    unittest.main()
