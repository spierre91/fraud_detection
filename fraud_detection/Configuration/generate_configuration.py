#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 11:15:52 2018

@author: spierre91
"""

import json
import pandas as pd 
import numpy as np



model_tuning = {'numberof_neighbors':[5], 'target_threshold': [60000.0], 
                'training_data': ["trainingdata.csv"], 'sensitivity_factor': [10],
                'dump_threshold': [0.0], 'number of accuracy runs': [1000]}

window_assumptions = {'step_forward': [30], 'window_size': ["30min"]}

database_config = {}
with open('window_assumptions.json', 'w',  ) as outfile:
    json.dump(window_assumptions, outfile)
#    
with open('model_tuning.json', 'w') as outfile:
    json.dump(model_tuning, outfile)
    
with open('database_config.json', 'w') as outfile:
    json.dump(database_config, outfile)

window = pd.read_json('window_assumptions.json')
model = pd.read_json('model_tuning.json')
database = pd.read_json('database_config.json')
