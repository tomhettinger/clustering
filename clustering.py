# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 15:17:18 2016

@author: Tom
"""

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
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
   
   
if __name__ == "__main__":
    main()