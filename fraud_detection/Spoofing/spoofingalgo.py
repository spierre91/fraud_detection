#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 13:21:50 2019

@author: spierre91
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.stats import multivariate_normal
from sklearn.metrics import f1_score

def feature_normalize(dataset):
    mu = np.mean(dataset,axis=0)
    sigma = np.std(dataset.T,axis=0)
    return (dataset - mu)/sigma

def estimateGaussian(dataset):
    mu = np.mean(dataset, axis=0)
    #sigma = np.cov(dataset.T)
    sigma = np.cov(dataset.T)
    return mu, sigma
    
def multivariateGaussian(dataset, mu,sigma):
    p = multivariate_normal(mean=mu, cov=sigma)    
    return p.pdf(dataset)
'''
Define a threshold called epsilon that can be used to differentiate between 
normal and anomalous matched and cancelled volumes. The threshold is based on the f1 score
calculated with labelled data
'''
def selectThresholdByCV(probs,gt):
    best_epsilon = 0
    best_f1 = 0
    f = 0
    stepsize = (max(probs) - min(probs)) / 1000;
    epsilons = np.arange(min(probs),max(probs),stepsize)
    for epsilon in np.nditer(epsilons):
        predictions = (probs < epsilon) 
        f = f1_score(gt, predictions,average='binary')
        if f > best_f1:
            best_f1 = f
            best_epsilon = epsilon  
    return best_f1, best_epsilon

tr_data = pd.read_csv('tr_data.csv') 
cv_data = pd.read_csv('cv_data.csv') 
gt_data = pd.read_csv('gt_data.csv')
print(tr_data)
tr_data = tr_data[['matched volume', 'cancelled volume']]
print(tr_data)
mu, sigma = estimateGaussian(tr_data)
mu = [mu[0], mu[1]]

n_training_samples = tr_data.shape[0]
n_dim = tr_data.shape[1]

print('Number of datapoints in training set: %d' % n_training_samples)
print('Number of dimensions/features: %d' % n_dim)

plt.ylabel('Matched Volume')
plt.xlabel('Cancelled Volume')
plt.plot(tr_data['cancelled volume'], tr_data['matched volume'],'bx')
plt.show()

mu, sigma = estimateGaussian(tr_data)
p = multivariateGaussian(tr_data,mu,sigma)

##selecting optimal value of epsilon using cross validation
p_cv = multivariateGaussian(cv_data,mu,sigma)
fscore, ep = selectThresholdByCV(p_cv,gt_data)
print(fscore, ep)

#selecting outlier datapoints 
outliers = list(np.asarray(np.where(p < ep))[0])
print("Outliers: ", tr_data['cancelled volume'].iloc[outliers],tr_data['matched volume'].iloc[outliers])
plt.figure()
plt.ylabel('Matched Volume')
plt.xlabel('Cancelled Volume')
plt.plot(tr_data['cancelled volume'], tr_data['matched volume'], 'bx')
plt.plot(tr_data['cancelled volume'].iloc[outliers],tr_data['matched volume'].iloc[outliers],'ro')
plt.show()
