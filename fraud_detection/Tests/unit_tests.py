#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 16:23:16 2018

@author: spierre91
"""

import pandas as pd 
import json
import unittest 
import sys
sys.path.append("..") 
from Tools import utility_library
from KNN_Algorithm import main_knn_script
from KNN_Algorithm import knnaccuracy

#read in model tuning parameters from json file 
with open('../Configuration/model_tuning.json') as f:
   data = json.load(f)
model = pd.DataFrame(data)
NUMBER_OF_KNN_RUNS = model['number of accuracy runs'][0] #the number of times the KNN algorithm is run for train test split accuracy measure 

class FeaturesTestCase(unittest.TestCase):
    def integration_test(self, pair, fromTime, toTime, threshold):
        sys.argv = ['pump_and_dump.py', 'five_year', 'binance', 
                    pair, fromTime, toTime, threshold]
        return sys.argv
    #each test asserts that an empty dataframe was received 
    def test_maxreturns(self):
        result = self.integration_test('LTC/USDT', 1513136227600, 1513146227600, 1)#dummy system input
        df_emptydict = {}
        empty_test = utility_library.get_maxreturns_dataframe(df_emptydict)
        self.assertEqual(empty_test, "Empty DataFrame")    
    def test_volume(self):
        result = self.integration_test('LTC/USDT', 1513136227600, 1513146227600, 1)#dummy system input
        df_emptydict = {}
        empty_test = utility_library.volume_dataframe(df_emptydict)
        self.assertEqual(empty_test, "Empty DataFrame")               
    def test_ohlc(self):
        result = self.integration_test('LTC/USDT', 1513136227600, 1513146227600, 1)#dummy system input
        df_emptydict = {}
        empty_test = utility_library.ohlc_dataframe(df_emptydict)
        self.assertEqual(empty_test, "Empty DataFrame") 
    def test_trades(self):
        result = self.integration_test('LTC/USDT', 1513136227600, 1513146227600, 1)#dummy system input
        df_emptydict = {}
        empty_test = utility_library.executions_dataframe(df_emptydict)
        self.assertEqual(empty_test, "Empty DataFrame") 
    def test_volatility(self):
        result = self.integration_test('LTC/USDT', 1513136227600, 1513146227600, 1)#dummy system input
        df_emptydict = {}
        empty_test = utility_library.get_volatility_dataframe(df_emptydict)
        self.assertEqual(empty_test, "Empty DataFrame") 
    #the accuracy test asserts that the accuracy is greater than 90%
    def test_accuracy(self):   
        accuracy = knnaccuracy.knn_accuracy(NUMBER_OF_KNN_RUNS)
        self.assertTrue(accuracy > 90.0)

if __name__ == '__main__':
    unittest.main()

