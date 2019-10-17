#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 14:28:48 2018

@author: spierre91
"""
import sys
sys.path.append("..") 
from KNN_Algorithm import main_knn_script
import unittest 
import numpy as np

#integration test suite tests KNN algo with examples of normal market behavior and pump and dump
class KNNTestCase(unittest.TestCase):
    
    #runs the main driver code in knn_script, 
    #takes pair, from/to time,and threshold as argument
    #returns the target label as out put
    def integration_test(self, pair, fromTime, toTime, threshold):
        sys.argv = ['pump_and_dump.py', 'five_year', 'binance', 
                    pair, fromTime, toTime, threshold]
        result = main_knn_script.main() 
        return result  
      
    #test checks if known normal market case returns a target label of zero
    #historical trade data LTC/USDT is used
    def test_LTCUSDT(self):
       result = self.integration_test('LTC/USDT', 1513136227600, 1513146227600, 1)[0]['target0']
       test_result = result[0]
       self.assertEqual(test_result, 0)
       
    #test asserts if known normal market case returns a target label of zero
    #historical trade data IOTA/USDT is used
    def test_IOTAUSDT(self):
       result = self.integration_test('IOTA/USDT', 1527759030847, 1527769030847, 1)[0]['target0']
       test_result = result[0]
       self.assertEqual(test_result, 0)   

    #test asserts that four pump and dumps occur with SYS/BTC
    def test_SYSBTC_multiplealerts(self):
       test_result = []
       expected = [1, 1, 1, 1]
       result = self.integration_test('SYS/BTC', 1530642019645, 1530655914001, 1)
       for i in range(len(result)):
           test_result.append(np.asscalar(result[i]['target{}'.format(i)][0]))
       self.assertEqual(test_result, expected)  
       
    #test asserts that one pump and dump occurs with SYS/BTC
    def test_SYSBTC_onealert(self):
       result = self.integration_test('SYS/BTC', 1530655914001, 1530665914001, 1)[0]['target0']
       test_result = np.asscalar(result[0])
       self.assertEqual(test_result, 1)  
       
    #test asserts that one pump and dump occurs with OST/BTC
    def test_OSTBTC_onealert(self):
       result = self.integration_test('OST/BTC', 1530444145781, 1530544145781, 1)[0]['target0']
       test_result = np.asscalar(result[0])
       self.assertEqual(test_result, 1)  
       
    #test asserts that three pump and dumps occurs with OST/BTC
    def test_OSTBTC_multiplealerts(self):
       test_result = []
       expected = [1, 1, 1]
       result = self.integration_test('OST/BTC', 1533624693431, 1533824693431, 1)
       for i in range(len(result)):
           test_result.append(np.asscalar(result[i]['target{}'.format(i)][0]))
       self.assertEqual(test_result, expected)    

if __name__ == '__main__':
    unittest.main()

