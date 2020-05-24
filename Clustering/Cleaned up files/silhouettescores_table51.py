# -*- coding: utf-8 -*-
"""
This file generates the values of silhouettescores.txt as shown in Table 5.1 in the report. 
Input: numpy file with all data points.
Output: print statements containing the average silhouette scores for a range of number of clusters.
This file was run a total of 10 times in order to consider every combination of model and experiment. 
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

#load data
Data = np.load('../Datares/tensor_daily_mean_5D.npy')

#fix NaN values, set them to average
NanINDX = np.argwhere(np.isnan(Data))
for i in range(len(NanINDX)):
    Data[NanINDX[i][0],NanINDX[i][1],NanINDX[i][2],NanINDX[i][3],NanINDX[i][4]] = 200
        
mdl = 0     #choose model, need to rerun again for mdl=0,1,2,3,4
exp = 0     #choose experiment, need to rerun again for exp=0,1
SubData = Data[:,:,:,mdl,exp]       #pick the right data

#standardize data
std = 1
for i in range(SubData.shape[1]):
    SubData[:,i,:] = SubData[:,i,:]-SubData[:,i,:].mean()
    SubData[:,i,:] = SubData[:,i,:]/(SubData[:,i,:].std(ddof = std))

st = 0                           #consider station Marsdiep
SubDataStation = SubData[:,:,st] #pick the right data

#calculate average silhouette scores for various numbers of clusters
range_n_clusters = [2,3,4,5,6,7,8,9,10,11,12]
for n_clusters in range_n_clusters:
    clusterer = KMeans(n_clusters=n_clusters, random_state=10).fit(SubDataStation)
    cluster_labels = clusterer.labels_
    silhouette_avg = silhouette_score(SubDataStation, cluster_labels)    
    print("For n_clusters =", n_clusters, 
          "The average silhouette_score is :", silhouette_avg)