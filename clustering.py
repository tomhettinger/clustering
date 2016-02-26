# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 15:17:18 2016

@author: Tom
"""

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np

import plotting

READ_PATH = u'C:\\Users\\Tom\\Desktop\\clustering\\data.csv'


def standardize(df):
    X = df.drop(['id', 'group'], axis=1)
    X.ix[X.gender == 'f', 'gender'] = 1.0
    X.ix[X.gender == 'm', 'gender'] = -1.0
    X['gender'] = pd.to_numeric(X.gender)
    X = StandardScaler().fit_transform(X.as_matrix())
    return X


def build_cluster_model(X, k):
    """Build a KMeans cluster model with 'k' clusters."""
    km = KMeans(n_clusters=k, n_init=10, max_iter=300, n_jobs=1)
    y_pred = km.fit_predict(X)
    return y_pred, km

def k2_confusion_matrix(df):
    print df.groupby(['k2_class', 'group']).size()
    temp = df.copy(deep=True)
    # Rename true values    
    temp.ix[temp.group != 'Docile-Male', 'group'] = 'other'
    # Rename predicted values    
    temp.ix[temp.k2_class == 0, 'k2_class'] = 'other'
    temp.ix[temp.k2_class == 1, 'k2_class'] = 'Docile-Male'
    labs = ['Docile-Male', 'other']
    print labs
    print confusion_matrix(temp.group.tolist(), temp.k2_class.tolist(), labels=labs)


def k3_confusion_matrix(df):
    print df.groupby(['k3_class', 'group']).size()
    temp = df.copy(deep=True)
    # Rename true values
    temp.ix[temp.group == 'Tall-Active', 'group'] = 'Tall-Active or Tall-Slender'
    temp.ix[temp.group == 'Tall-Slender', 'group'] = 'Tall-Active or Tall-Slender'
    temp.ix[temp.group == 'Heavy-Active-Female', 'group'] = 'Heavy-Active-Female or Short-Active'
    temp.ix[temp.group == 'Short-Active', 'group'] = 'Heavy-Active-Female or Short-Active'
    # Rename predicted values
    temp.ix[temp.k3_class == 0, 'k3_class'] = 'Tall-Active or Tall-Slender'
    temp.ix[temp.k3_class == 1, 'k3_class'] = 'Docile-Male'
    temp.ix[temp.k3_class == 2, 'k3_class'] = 'Heavy-Active-Female or Short-Active'
    labs = ['Tall-Active or Tall-Slender', 'Docile-Male', 'Heavy-Active-Female or Short-Active']
    print labs
    print confusion_matrix(temp.group.tolist(), temp.k3_class.tolist(), labels=labs)


def k5_confusion_matrix(df):
    print df.groupby(['k5_class', 'group']).size()
    temp = df.copy(deep=True)
    # Rename predicted values
    temp.ix[temp.k5_class == 0, 'k5_class'] = 'Tall-Slender'
    temp.ix[temp.k5_class == 1, 'k5_class'] = 'Tall-Active'
    temp.ix[temp.k5_class == 2, 'k5_class'] = 'Docile-Male'
    temp.ix[temp.k5_class == 3, 'k5_class'] = 'Short-Active'
    temp.ix[temp.k5_class == 4, 'k5_class'] = 'Heavy-Active-Female'
    labs = ['Tall-Slender', 'Tall-Active', 'Docile-Male', 'Short-Active', 'Heavy-Active-Female']
    print labs
    print confusion_matrix(temp.group.tolist(), temp.k5_class.tolist(), labels=labs)


def main():
    # Read in the data
    sample = pd.read_csv(READ_PATH)
    print sample.info()
    print sample.describe()
    print sample.describe(include=['O'])
   
    # Create summary plots
    plotting.violin(sample)
    plotting.pairplot(sample)
    plotting.pairplot_kde(sample)
    plotting.heatmap(sample)
    plotting.swarmplot(sample)
   
    # Standardize variables
    X = standardize(sample)
    
    # Build models with k=2 through k=10
    models = []
    for k in range(2, 11, 1):
        y_pred, km = build_cluster_model(X, k)
        models.append(('k%d_class' % k, km))
        sample['k%d_class' % k] = y_pred        
        
    # Inertia Analysis
    inertia = np.array([m[1].inertia_ for m in models])
    k_value = np.arange(2, 11, 1)
    plotting.inertia(k_value, inertia)
    plotting.d_inertia(k_value, inertia)
     
    # Pair plots with new color coding
    for k in range(2, 11, 1):
        plotting.pairplot(sample, group='k%d_class' % k)
   
    # Comparison of k2 model with original groupings
    k2_confusion_matrix(sample)
    # Comparison of k3 model with original groupings
    k3_confusion_matrix(sample)
    # Comparison of k5 model with original groupings
    k5_confusion_matrix(sample)

    # Feature-Feature plots comparing pred and truth
    plotting.compare_model(df=sample, model='k5_class', x='heartrate', y='height')
    plotting.compare_model(df=sample, model='k5_class', x='weight', y='height')

if __name__ == "__main__":
    main()
    
    
    
    
    