#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 15:45:21 2018

@author: spierre91
"""
import json
import pandas as pd 
import sys
import numpy as np
from influxdb import InfluxDBClient
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
sys.path.append('..') 
from Tools import utility_library

#read in model parameters, windowing for resampling, and database configurations 
with open('../Configuration/window_assumptions.json') as f:
   data = json.load(f)
window = pd.DataFrame(data)
with open('../Configuration/model_tuning.json') as f:
   data = json.load(f)
model = pd.DataFrame(data)
with open('../Configuration/database_config.json') as f:
   data = json.load(f)
database = pd.DataFrame(data)


#defining parameters for window sizes and KNN model
STEP_FORWARD = np.asscalar(window['step_forward'][0])#size of steps forward in time after the max return
WINDOW_SIZE =  window['window_size'][0] #size of resampling window 
TARGET_THRESHOLD = model['target_threshold'][0]#parameter used to label training data as pump and dump
NUMBEROF_NEIGHBORS = model['numberof_neighbors'][0]#the number of neighbors to be used in the KNN algo
DUMP_THRESHOLD = model['dump_threshold'][0]#the threshold for the decrease in returns followed by a local max used in the KNN+rule engine
KNN_MODEL_DATA_CSV_PATH = model['training_data'][0]#the data used in the KNN model with labels for pump and dump and normal behavior 
SENSITIVITY_FACTOR = model['sensitivity_factor'][0]#factor tunable by user input, used with user input to increase or decrease sensitivity of the KNN algo

#database configurations 
URL = database['url'][0]#the url string to connect to AWS servers
PORT = database['port'][0]#connection port
USERNAME = database['username'][0]#username for AWS server login
PASSWORD = database['password'][0]#password for AWS server login
DATABASE = database['database'][0]#name of database
CHUNK_SIZE = database['chunk_size'][0]#data queries are broken into chunks 

client = InfluxDBClient(URL, PORT, USERNAME, PASSWORD, DATABASE)#influx client 

#main driver function that runn knn engine 
#takes no input
#calls utility functions, and knn_engine and returns the target label for pump 
#and dump classification 
def main():
    retention_policy = sys.argv[1] 
    exchange = sys.argv[2] 
    pair = sys.argv[3] 
    fromTime = sys.argv[4]
    toTime = sys.argv[5] 
    threshold = sys.argv[6] 
    returns_threshold = SENSITIVITY_FACTOR*threshold
    query = create_query(retention_policy, exchange, pair, fromTime, toTime)
    query_result = client.query(query, chunked = True, chunk_size = CHUNK_SIZE)
    df = pd.DataFrame(query_result.get_points())    
    knn_result = knn_engine(df, returns_threshold)
    print(knn_result)
    return knn_result

#query influx for historical trade data 
#takes retention policy, name of exchange, coin pair, from and to times as arguments
#returns a string for the query of influx 
def create_query(retention_policy, exchange, pair, fromTime, toTime):
    query = "select * from {}.trade_data where exchange='{}' and pair='{}' \
    and time>{}ms - 1h and time<{}ms + 1h;".format(retention_policy, exchange, 
    pair, fromTime, toTime)
    return query

#calculates the size of the dump given the local minimum values in returns
#takes historical trade data as input
#returns a list of the maginitudes of the decrease in returns 

def get_dumpsize_list(df, minreturns_dataframe, maxreturns_dataframe, returns_threshold):    
    dump_magnitude_list =[] 
    #if the larges return value in max returns dataframe is less than the threshold, set dump =0
    if max(maxreturns_dataframe['largest returns']) < returns_threshold:
            for i in range(len(maxreturns_dataframe['largest returns'])):  
                dump_magnitude_list.append(0)
    else:
    #iterate over dictionary and store minimum and maximum return values in a list 
        for key_min, value_min in minreturns_dataframe.iteritems():
            value_min = list(value_min)
        for key_max, value_max in maxreturns_dataframe.iteritems():
            value_max = list(value_max)      
        for i in range(len(value_min)):
        #if the minimum value is negative, the dump is the maximum return plus the absolute value of the minium return 
            if value_min[i] <= 0:
                dump_magnitude_list.append(value_max[i] + np.abs(value_min[i]))
        #if the minimum value is negative, the dump is the maximum return plus the absolute value of the minium return 
            elif value_min[i] > 0:
                dump_magnitude_list.append(value_max[i] - value_min[i])
    return dump_magnitude_list 

#the k-neiarest neighbors algorithm used to classify pump and dump 
#take the test data, the list of largest dips in returns, and the returns threshold as input 
#returns a list of dictionaries with target label, maximum value of return, timestamp of maximum, dump size, 
#and timestamp of minimum
def knn_engine(test_data, returns_threshold):
    #define the training data
    df = pd.read_csv(KNN_MODEL_DATA_CSV_PATH, index_col = 0)
    
    minreturns_dataframe = utility_library.get_minreturns_dataframe(test_data) #dataframe of minimum returns and timestamps
    maxreturns_dataframe = utility_library.get_maxreturns_dataframe(test_data) #dataframe of maximum returns and timestamps
    dump_magnitude_list = get_dumpsize_list(test_data, minreturns_dataframe, maxreturns_dataframe, returns_threshold)#list of dips in magnitude followed by max returns

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
    
    #list of times of max returns
    timeof_maxreturn = maxreturns_dataframe
    timeof_maxreturn = list(timeof_maxreturn.index)
    
    #list of times of min returns
    timeof_minreturn = maxreturns_dataframe 
    timeof_minreturn = list(timeof_minreturn.index)
     
    #initializing lists
    X_test_features = []
    X_transform_features = []
    y_pred = []
    rule_knn_result = []
    knn_output =[]
    #storing values of maximum volatilities, returns, volumes and trades in lists
    maxvolatility_list = utility_library.get_maxvolatility_dataframe(test_data)
    maxvolatility_list = list(maxvolatility_list['largest volatility'])
    maxtrades_list = utility_library.get_maxnumberoftrades_dataframe(test_data)
    maxtrades_list = list(maxtrades_list['largest number of trades'])
    maxvolume_list = utility_library.get_maxvolume_dataframe(test_data)
    maxvolume_list = list(maxvolume_list['largest volume'])
    maxreturns_list = utility_library.get_maxreturns_dataframe(test_data) 
    maxreturns_list = list(maxreturns_list['largest returns'])
    #appending features: returns, volatility, volume and trades 
    for i in range(len(maxreturns_list)): 
        X_test_features.append([[maxreturns_list[i], maxvolatility_list[i], maxvolume_list[i], maxtrades_list[i]]])
        
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
    for i in range(len(maxreturns_list)): 
        #targets are label as pump and dump if they are classified by knn and satisfy the dump rule
        if y_pred[i] == 1 and dump_magnitude_list[i] > DUMP_THRESHOLD:
            rule_knn_result.append(y_pred[i])
        else:
            rule_knn_result.append([0])
    #return a dictionary with label, values of maximum returns, and timestamps 
    for i in range(len(maxreturns_list)):    
        knn_output.append({'target{}'.format(i): rule_knn_result[i], 'time of max return': timeof_maxreturn[i],
                          'return max':maxreturns_list[i], 'time of min return': timeof_minreturn[i], 'dump': dump_magnitude_list[i]})
    return knn_output 

if __name__ == "__main__":
    main()
