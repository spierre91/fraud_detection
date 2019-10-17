#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 20:29:09 2018

@author: spierre91
"""



# importing the requests library 
import requests 
import ast
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.10; rv:62.0) Gecko/20100101 Firefox/62.0'}
r_buy = requests.get('http://ec2-13-59-155-247.us-east-2.compute.amazonaws.com:8086/query?db=marketdata&precision=n&q=SELECT%20*%20FROM%20one_month.quote_data%20%20WHERE%20exchange=%27gemini%27AND%20pair=%27BTC/USD%27%20AND%20side=%27Buy%27%20ORDER%20BY%20DESC%20LIMIT%20100',  auth=('influx', 'solidus123'), headers=headers)

r_sell = requests.get('http://ec2-13-59-155-247.us-east-2.compute.amazonaws.com:8086/query?db=marketdata&precision=n&q=SELECT%20*%20FROM%20one_month.quote_data%20%20WHERE%20exchange=%27gemini%27AND%20pair=%27BTC/USD%27%20AND%20side=%27Sell%27%20ORDER%20BY%20DESC%20LIMIT%20100',  auth=('influx', 'solidus123'), headers=headers)

# api-endpoint 
dicts_buy = r_buy.text
data_b = ast.literal_eval(dicts_buy)
data_buy = data_b['results']
df_buy = pd.DataFrame(data_buy)
list_buy = df_buy['series'][0][0]
#print(list_buy)

dicts_sell = r_sell.text
data_s = ast.literal_eval(dicts_buy)
data_sell = data_s['results']
df_sell = pd.DataFrame(data_sell)
list_sell = df_sell['series'][0][0]


depth=100

i=0   
time_list = [] 
for i in range(0,depth):
    df_buy['timestamp'] = list_buy['values'][i][0]
    time_list.append(df_buy['timestamp'][0])
    
print(time_list)
#
i=0   
buy_price_list = [] 
for i in range(0,depth):
    df_buy['price'] = list_buy['values'][i][3]
    buy_price_list.append(df_buy['price'][0])
    
print(buy_price_list)
#
i=0   
buy_quantity_list = [] 
for i in range(0,depth):
    df_buy['quantity'] = list_buy['values'][i][5]
    buy_quantity_list.append(df_buy['quantity'][0])
    
print(buy_quantity_list)

i=0   
sell_price_list = [] 
for i in range(0,depth):
    df_sell['price'] = list_sell['values'][i][3]
    sell_price_list.append(df_sell['price'][0])
    
print(sell_price_list)
#
i=0   
sell_quantity_list = [] 
for i in range(0,depth):
    df_sell['quantity'] = list_sell['values'][i][5]
    sell_quantity_list.append(df_sell['quantity'][0])
    
print(sell_quantity_list)

dataframe = {'Buy_Price':buy_price_list,'Buy_quantity':buy_quantity_list, 
             'Sell_Price':sell_price_list,'Sell_quantity':sell_quantity_list,
              'Time': time_list}
df_new = pd.DataFrame(dataframe)
df_new.set_index('Time', inplace=True)
print(df_new)

sns.set()
plt.scatter(df_new['Buy_Price'], df_new['Buy_quantity'])
plt.scatter(df_new['Sell_Price'], df_new['Sell_quantity'])
plt.title('BTC/USD Order Book')
plt.xlabel('Price')
plt.ylabel('quantity')
