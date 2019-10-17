#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 23:19:31 2018

@author: spierre91
"""
import json
import pandas as pd 
import matplotlib.dates as mdates
from datetime import timedelta
import numpy as np 
import sys
sys.path.append("..") 
from influxdb import InfluxDBClient

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

STEP_FORWARD = np.asscalar(window['step_forward'][0])#size of steps forward in time after the max return
WINDOW_SIZE =  window['window_size'][0] #size of resampling window 
TARGET_THRESHOLD = model['target_threshold'][0]#parameter used to label training data as pump and dump
NUMBEROF_NEIGHBORS = model['numberof_neighbors'][0]#the number of neighbors to be used in the KNN algo
DUMP_THRESHOLD = model['dump_threshold'][0]#the threshold for the decrease in returns followed by a local max used in the KNN+rule engine
KNN_MODEL_DATA_CSV_PATH = model['training_data'][0]#the data used in the KNN model with labels for pump and dump and normal behavior 
SENSITIVITY_FACTOR = model['sensitivity_factor'][0]#factor tunable by user input, used with user input to increase or decrease sensitivity of the KNN algo

URL = database['url'][0]#the url string to connect to AWS servers
PORT = database['port'][0]#connection port
USERNAME = database['username'][0]#username for AWS server login
PASSWORD = database['password'][0]#password for AWS server login
DATABASE = database['database'][0]#name of database
CHUNK_SIZE = database['chunk_size'][0]#data queries are broken into chunks 

client = InfluxDBClient(URL, PORT, USERNAME, PASSWORD, DATABASE)#influx client 

#get the returns threshold 
def get_threshold():
    retention_policy = sys.argv[1] 
    exchange = sys.argv[2] 
    pair = sys.argv[3] 
    fromTime = sys.argv[4]
    toTime = sys.argv[5] 
    threshold = sys.argv[6] 
    returns_threshold = SENSITIVITY_FACTOR*threshold 
    return returns_threshold 

#gets the resampled volume, 
#takes the historical trade data as an argument, 
#returns a dataframe with open/high/low/close
def volume_dataframe(df):
    length = len(df)
    if length == 0:
        return "Empty DataFrame"
    df.index = pd.to_datetime(df['time'], unit = 'ns')
    df.index.map(mdates.date2num)
    df_volume = df['size'].resample(WINDOW_SIZE).mean()
    df_volume = pd.DataFrame(df_volume)    
    return df_volume

#gets the resampled open/high/low/close in prices, 
#takes the historical trade data as an argument, 
#returns a dataframe with open/high/low/close
def ohlc_dataframe(df):
    length = len(df)
    if length == 0:
        return "Empty DataFrame"    
    df.index = pd.to_datetime(df['time'], unit = 'ns')
    df.index.map(mdates.date2num)
    df_ohlc = df['price'].resample(WINDOW_SIZE).ohlc()
    df_ohlc.reset_index(inplace = True)    
    return df_ohlc

#gets the price returns from resample open and close prices, 
#takes the df_ohlc as an argument, 
#returns a dataframe with price returns           
def return_dataframe(df):
    df_ohlc = ohlc_dataframe(df)
    series_return = ((df_ohlc['close'] - df_ohlc['open']) / 
                     df_ohlc['open']) * 100.0    
    df_return = pd.DataFrame(series_return)
    df_return.columns = ['returns']
    df_return.set_index(df_ohlc['time'], inplace = True)   
    return df_return

#gets the number of trades that occurec within a time period, 
#takes the historical trades as an argument, 
#returns a data frame of resampled number of executed trades
def executions_dataframe(df):
    length = len(df)
    if length == 0:
        return "Empty DataFrame"
    df.index = pd.to_datetime(df['time'], unit = 'ns')
    df.index.map(mdates.date2num)
    df_execute = df['time'].resample(rule = WINDOW_SIZE).count()
    df_execute = pd.DataFrame(df_execute)
    df_execute.columns = ['execute']
    return df_execute

#gets the volatility in returns (standard deviation in returns) within a time period
#takes the historical trade data as an argument
#returns volatility in returns in a dataframe 
def get_volatility_dataframe(df):
    length = len(df)
    if length == 0:
        return "Empty DataFrame"
    df.index = pd.to_datetime(df['time'], unit = 'ns')
    df.index.map(mdates.date2num)
    df_ohlc = df['price'].resample('15min').ohlc()
    df_ohlc.reset_index(inplace = True)
    series_return = ((df_ohlc['close'] - df_ohlc['open']) / 
                     df_ohlc['open']) * 100.0    
    df_return = pd.DataFrame(series_return)
    df_return.columns = ['returns']
    df_return.set_index(df_ohlc['time'], inplace = True)
    volatility = df_return.resample('30min').std()
    volatility.dropna(inplace = True)     
    volatility.columns = ['volatility']
    return volatility

#combined the summary statistics (number of trades, returns, volume and volatility) into one dataframe
#takes the historical trade data as an argument 
#returns a data frame with summary statistics 
def summary_statistics_dataframe(df):
    executed_trades = executions_dataframe(df)
    returns = return_dataframe(df)
    volume = volume_dataframe(df)
    volatility = get_volatility_dataframe(df)
    frames = [returns, executed_trades, volume, volatility]
    summary_statistics = pd.concat(frames, axis = 1)
    summary_statistics.fillna(0, inplace = True)
    return summary_statistics 

#return the largest returns above a defined threshold along with timestamps
#takes the historical trade data as input 
#returns a dataframe with local maxima in returns with timestamps 
def get_maxreturns_dataframe(df):
    returns_threshold = get_threshold()
    length = len(df)
    if length == 0:
        return "Empty DataFrame"
    df_return = return_dataframe(df)
    returns_count = df_return['returns'][df_return['returns'] > 
                                 returns_threshold].count()
    if returns_count == 0:
       maxreturns_dataframe = df_return[df_return['returns'] > 
                                      0.0].nlargest(1, 'returns')   
                                     
    else:
       maxreturns_dataframe = df_return[df_return['returns'] > 
                                    returns_threshold].nlargest(returns_count, 'returns')
    maxreturns_dataframe.columns = ['largest returns']
    return maxreturns_dataframe

#return the volumes at the timestamps of the maximum returns 
#takes the historical trade data as input 
#returns a dataframe with volumes at the time of max returns with timestamps 
def get_maxvolume_dataframe(df):
    summary_statistics = summary_statistics_dataframe(df)
    maxreturns_dataframe = get_maxreturns_dataframe(df) 
    length = len(df)
    if length == 0:
        return "Empty DataFrame"
    maxvolume_dataframe = summary_statistics.loc[maxreturns_dataframe.index]['size']
    maxvolume_dataframe= pd.DataFrame(maxvolume_dataframe)
    maxvolume_dataframe.columns = ['largest volume']
    return maxvolume_dataframe    

#return the number of trades at the timestamps of the maximum returns 
#takes the historical trade data as input 
#returns a dataframe with number of trades at the time of max returns with timestamps 
def get_maxnumberoftrades_dataframe(df):
    summary_statistics = summary_statistics_dataframe(df)
    maxreturns_dataframe = get_maxreturns_dataframe(df)
    length = len(df)
    if length == 0:
        return "Empty DataFrame"
    maxtrades_dataframe = summary_statistics.loc[maxreturns_dataframe.index]['execute']
    maxtrades_dataframe = pd.DataFrame(maxtrades_dataframe)
    maxtrades_dataframe.columns = ['largest number of trades']
    return maxtrades_dataframe  

#return the volatility at the timestamps of the maximum returns 
#takes the historical trade data as input 
#returns a dataframe with volatility at the time of max returns with timestamps 
def get_maxvolatility_dataframe(df):
    summary_statistics = summary_statistics_dataframe(df)
    maxreturns_dataframe = get_maxreturns_dataframe(df) 
    length = len(df)
    if length == 0:
        return "Empty DataFrame"
    maxvolatility_dataframe = summary_statistics.loc[maxreturns_dataframe.index]['volatility']
    maxvolatility_dataframe = pd.DataFrame(maxvolatility_dataframe)
    maxvolatility_dataframe.columns = ['largest volatility']
    #print(maxvolatility_dataframe)
    return maxvolatility_dataframe  

#takes steps forward in times after the time of maximum returns 
#takes historical trade data as an argument
#returns a list of times after the time of maximum returns
def timestep_forward_list(df, size):
        maxreturns_dataframe = get_maxreturns_dataframe(df)
        time_aftermax_return_list = []
        for i in maxreturns_dataframe:
                time_aftermax_return_list.append(maxreturns_dataframe.index 
                +  timedelta(minutes = size*STEP_FORWARD))                  
        return time_aftermax_return_list
    
##given the time steps forward, this returns the local  minimum after the local max along with its timestamp
#takes historical trade data as an argument
#returns a dataframe of local minimum returns and timestamps 
def get_minreturns_dataframe(df):
    #get dataframe of returns and timestamps
    df_returns = return_dataframe(df)
    #get dataframe of local maximum returns and timestamps
    maxreturns_dataframe = get_maxreturns_dataframe(df)
    #define three steps forward in time after local maximum returns
    time1 = timestep_forward_list(df, 1)     
    time2 = timestep_forward_list(df, 2)   
    time3 = timestep_forward_list(df, 3) 
    #define a list of local minimum returns at each timestamp
    minreturns_value1 = df_returns.loc[time1[0]]
    minreturns_value2 = df_returns.loc[time2[0]]
    minreturns_value3 = df_returns.loc[time3[0]]   
    alerts_list = []
    minimumum_returns_times = []
    minimumum_returns_values = []
    #define a dictionary with key/values corresponding to timestamps/minimum returns
    for i in range(len(maxreturns_dataframe)):
        alerts_list.append({'alert{}'.format(i):
            {minreturns_value1.index[i] : minreturns_value1.iloc[i]['returns'], 
             minreturns_value2.index[i] : minreturns_value2.iloc[i]['returns'],
             minreturns_value3.index[i] : minreturns_value3.iloc[i]['returns'] },})
        #store minimum returns in list
        minimumum_returns_times.append(min(alerts_list[i]['alert{}'.format(i)], 
                                           key = lambda x : alerts_list[i]['alert{}'.format(i)].get(x)))   
        #store time stamps of minimum returns in list
        minimumum_returns_values.append(df_returns.loc[min(alerts_list[i]['alert{}'.format(i)], 
                                 key = lambda x : alerts_list[i]['alert{}'.format(i)].get(x))]['returns'])  
    #create a dictionary with minimum returns and their corresponding timestamps 
    minimumum_returns_dataframe = {'time':minimumum_returns_times, 'smallest returns':minimumum_returns_values }
    #convert dictionary into a dataframe
    minimumum_returns_dataframe = pd.DataFrame(minimumum_returns_dataframe)
    #set the time column as the index
    minimumum_returns_dataframe.set_index('time', inplace = True)
    return minimumum_returns_dataframe 

