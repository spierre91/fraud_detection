#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 11:29:19 2018

@author: spierre91
"""

import numpy as np
import pandas as pd
import matplotlib.dates as mdates
from matplotlib import style
import datetime as dt
from datetime import timedelta


#this function returns the maximum return within a 30 min time period
def getMaxReturn(filename):
    df1 = pd.read_json(filename)
    df1['tradeID']=df1['a']
    df1['Price'] = df1['p']
    df1['Quantity'] = df1['q']
    df1['First tradeID'] = df1['f']
    df1['Last tradeID'] = df1['l']
    df1['Timestamp']= df1['T']
    df1['Maker?'] = df1['m']
    df1['Match?'] = df1['M']
    df1 = df1[['Price', 'Quantity', 'Timestamp',]]

    df1.index = pd.to_datetime(df1['Timestamp'], unit= 'ms')
    df1.index.map(mdates.date2num)

    df_ohlc = df1['Price'].resample('30min').ohlc()
    df_volume = df1['Quantity'].resample('30min').sum()

    df_ohlc.reset_index(inplace=True)
    df_ohlc['Timestamp'] = df_ohlc['Timestamp'].map(mdates.date2num)


    df1_PCT_change = ((df_ohlc['close']-df_ohlc['open'])/df_ohlc['open'])*100.0

    return df1_PCT_change.max()
    
#this function returns the maximum price volatility within a 30 min time period
def Volatility(filename):
    df = pd.read_json(filename)

    df['tradeID']=df['a']
    df['Price'] = df['p']
    df['Quantity'] = df['q']
    df['First tradeID'] = df['f']
    df['Last tradeID'] = df['l']
    df['Timestamp']= df['T']
    df['Maker?'] = df['m']
    df['Match?'] = df['M']
    df = df[['Price', 'Quantity', 'Timestamp',]]
    
    df.index = pd.to_datetime(df['Timestamp'], unit= 'ms')
    df.index.map(mdates.date2num)

    df_ohlc = df['Price'].resample('30min').ohlc()
    df_volume = df['Quantity'].resample('30min').sum()

    df_ohlc.reset_index(inplace=True)

    df_ohlc['Timestamp'] = df_ohlc['Timestamp'].map(mdates.date2num)
    df_return = ((df_ohlc['close']-df_ohlc['open'])/df_ohlc['open'])*100.0
    return df_return.std()

#this function returns the maximum percent change in volume within a 30 min time period
def getVolume(filename):
    df = pd.read_json(filename)
    
    df['tradeID']=df['a']
    df['Price'] = df['p']
    df['Quantity'] = df['q']
    df['First tradeID'] = df['f']
    df['Last tradeID'] = df['l']
    df['Timestamp']= df['T']
    df['Maker?'] = df['m']
    df['Match?'] = df['M']
    df = df[['Price', 'Quantity', 'Timestamp',]]
    
    df.index = pd.to_datetime(df['Timestamp'], unit= 'ms')
    df.index.map(mdates.date2num)

    df_ohlc = df['Price'].resample('30min').ohlc()
    df_volume = df['Quantity'].resample('30min').mean()

    df_ohlc.reset_index(inplace=True)

    df_ohlc['Timestamp'] = df_ohlc['Timestamp'].map(mdates.date2num)
    df_return = np.log( df_ohlc['open']/ df_ohlc['close'])
    return df_volume.pct_change().max()

#this function returns the number of executed trades within a 30 minute period 
def getNumberOfExecutions(filename):
    df = pd.read_json(filename)
    
    df['tradeID']=df['a']
    df['Price'] = df['p']
    df['Quantity'] = df['q']
    df['First tradeID'] = df['f']
    df['Last tradeID'] = df['l']
    df['Timestamp']= df['T']
    df['Maker?'] = df['m']
    df['Match?'] = df['M']
    df = df[['Price', 'Quantity', 'Timestamp',]]
    
    df.index = pd.to_datetime(df['Timestamp'], unit= 'ms')
    df.index.map(mdates.date2num)

    df_execute = df['Timestamp'].resample(rule='30min').count()
    return df_execute.max()
#this function calculate the maximum decrease in return out of windows= 1,2,3 hours after the maximum pump 
def dump(filename):
        df = pd.read_json(filename)
        df['Price'] = df['p']
        df['Timestamp']= df['T']
        df = df[['Price',  'Timestamp',]]
        df.index = pd.to_datetime(df['Timestamp'], unit= 'ms')
        df.index.map(mdates.date2num)

        df=df[(df.index > '2018-6-1 01:00:00') & 
                (df.index <= '2018-9-1 01:00:00')]
        df_ohlc = df['Price'].resample('30min').ohlc()
        

        df_return = ((df_ohlc['close']-df_ohlc['open'])/df_ohlc['open'])*100.0
        maxT= df_return.idxmax()
        hour1 = maxT + timedelta(hours =1)
        hour2 = maxT + timedelta(hours =2)
        hour3 = maxT + timedelta(hours =3)
        
        df_minreturn= {'Min Return': [df_return[hour1],df_return[hour2],df_return[hour3]]}
        df_minreturn = pd.DataFrame(df_minreturn)
        minreturn= df_minreturn['Min Return'].min()
        PDump = ((df_return[maxT]-minreturn)/(df_return[maxT]))*100.0
        print(df_minreturn.head())
        if PDump > 50.0:
            dump = df_return[maxT]+np.abs(minreturn)
        return dump

#print out feautures for ML input. This example prints of the max 0.5 hourly return, volume 
#and price volatility. This example is from aggregate trade data for SYS/BTC (Sept 18-20, 2018)
print(getVolume('Binanceagg_SYSBTC-1537195772962.json'))
print(Volatility('Binanceagg_SYSBTC-1537195772962.json'))
print(getMaxReturn('Binanceagg_SYSBTC-1537195772962.json'))
