# -*- coding: utf-8 -*-
"""
This file generates the figure as shown in Figure 5.7 in the report.
In this file, we thus cluster the whole dataset, then look at the labels of the first and last 10 years.
Input: numpy array will all data points
Output: figure 5.7, with on the left cluster labels of the entire dataset, on the right cluster labels of first and last 10 years. 
"""
import datetime
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans

#define function that transforms counts how many cluster labels there are in one specific month. 
def MonthCounter(Labels,n_clusters,StartTime):
    Dat = [[] for i in range(n_clusters)]
    Dag = StartTime
    for i in range(len(Labels)):
        Dat[Labels[i]].append(Dag.month)
        Dag += datetime.timedelta(days = 1)
    return(Dat) 
     
#load data               
Data = np.load('../Datares/tensor_daily_mean_5D.npy')

#fix NaN values, set them to average
NanINDX = np.argwhere(np.isnan(Data))
for i in range(len(NanINDX)):
    Data[NanINDX[i][0],NanINDX[i][1],NanINDX[i][2],NanINDX[i][3],NanINDX[i][4]] = 200

#select model, experiment and corresponding data
mdl = 0
exp = 1
SubData = Data[:,:,:,mdl,exp]

#standardize data
std = 1
for i in range(SubData.shape[1]):
    SubData[:,i,:] = SubData[:,i,:]-SubData[:,i,:].mean()
    SubData[:,i,:] = SubData[:,i,:]/(SubData[:,i,:].std(ddof = std))

st = 0                           #consider station Marsdiep
SubDataStation = SubData[:,:,st] #pick the right data

range_n_clusters = [2]          #choose number of clusters

#perform clustering on entire dataset
for n_clusters in range_n_clusters:
    data1 = SubDataStation
    clusterer = KMeans(n_clusters=n_clusters, random_state=10).fit(data1)
    cluster_labels = clusterer.labels_

cluster_labels1 = cluster_labels[:3650]     #consider first 10 years
cluster_labels2 = cluster_labels[-3650:]    #consider last 10 years

Start1 = datetime.date(2006,1,1)
Start2 = datetime.date(2086,1,1)

Month_Counter = MonthCounter(cluster_labels,n_clusters,Start1)  #counter for left figure
Month_Counter1 = MonthCounter(cluster_labels1,n_clusters,Start1) #counter for first 10 years in right figure
Month_Counter2 = MonthCounter(cluster_labels2,n_clusters,Start2) #counter for last 10 years in right figure

#combine Month_Counter1 and Month_Counter to plot them in one histogram
MC = []
for i in range(len(Month_Counter)):
    MC.append(Month_Counter1[i])
    MC.append(Month_Counter2[i])

#plot the cluster labels per month in two plots next to each other    
LAB = len(Month_Counter1)*['First 10 Years','Last 10 Years']
fig, axes = plt.subplots(nrows=1, ncols=2,figsize = (15,8))
ax0, ax1= axes.flatten()
ax0.hist(Month_Counter, 12, density=True, histtype='bar')
ax1.hist(MC, 12, density=True, histtype='bar',label=LAB, color=['lightsteelblue','blue','lightsalmon','red'])
ax1.legend(prop={'size': 10})