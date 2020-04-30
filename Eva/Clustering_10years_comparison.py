# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 09:56:56 2020

@author: simon
"""
import datetime
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

def MonthCounter(Labels,n_clusters):
    Dat = [[] for i in range(n_clusters)]
    Dag = datetime.date(2006,1,1)
    for i in range(len(Labels)):
        Dat[Labels[i]].append(Dag.month)
        Dag += datetime.timedelta(days = 1)
    return(Dat)
    
def YearCounter(Labels,n_clusters):
    Dat = [[] for i in range(n_clusters)]
    Dag = datetime.date(2006,1,1)
    for i in range(len(Labels)):
        Dat[Labels[i]].append(Dag.year)
        Dag += datetime.timedelta(days = 1)
    return(Dat)

Data = np.load('../Datares/tensor_daily_mean_5D.npy')
NanINDX = np.argwhere(np.isnan(Data))
for i in range(len(NanINDX)):
    Data[NanINDX[i]] = 200
st = 0
STATIONS = ['Marsdiep Noord','Doove Balg West',
                'Vliestroom','Doove Balg Oost',
                'Blauwe Slenk Oost','Harlingen Voorhaven','Dantziggat',
                'Zoutkamperlaag Zeegat','Zoutkamperlaag',
                'Harlingen Havenmond West']
"""
0 'Marsdiep Noord'
1 'Doove Balg West',
2 'Vliestroom'
3'Doove Balg Oost'
4 'Blauwe Slenk Oost',
5 'Harlingen Voorhaven',
6 'Dantziggat',
7 'Zoutkamperlaag Zeegat'
8 'Zoutkamperlaag',
9 'Harlingen Havenmond West'
"""

mdl = 0
MODELS = ['CNRM-CERFACS-CNRM-CM5','ICHEC-EC-EARTH', 
          'IPSL-IPSL-CM5A-MR','MOHC-HadGEM2-ES','MPI-M-MPI-ESM-LR']
"""
0 'CNRM-CERFACS-CNRM-CM5'
1 'ICHEC-EC-EARTH'
2 'IPSL-IPSL-CM5A-MR'
3 'MOHC-HadGEM2-ES'
4 'MPI-M-MPI-ESM-LR'
"""
exp = 0
EXPERIMENTS = ['rcp45','rcp85']

SubData = Data[:,:,:,mdl,exp]

std = 1

for i in range(SubData.shape[1]):
    SubData[:,i,:] = SubData[:,i,:]-SubData[:,i,:].mean()
    SubData[:,i,:] = SubData[:,i,:]/(SubData[:,i,:].std(ddof = std))

SubDataStation = SubData[:,:,st] #Select the station that we are going to analyse

range_n_clusters = [3]

for n_clusters in range_n_clusters:
    #2006-2016
    data1 = SubDataStation[:][3650:]
    clusterer = KMeans(n_clusters=n_clusters, random_state=10).fit(data1)
    cluster_labels = clusterer.labels_
    silhouette_avg = silhouette_score(data1, cluster_labels)   
    print("For n_clusters =", n_clusters, 
          "The average silhouette_score is :", silhouette_avg)
    sample_silhouette_values = silhouette_samples(data1, cluster_labels)
    #2086-2096
    data2 = SubDataStation[:][-3650:]
    clusterer2 = KMeans(n_clusters=n_clusters, random_state=10).fit(data2)
    cluster_labels2 = clusterer2.labels_
    silhouette_avg = silhouette_score(data2, cluster_labels2)   
    print("For n_clusters =", n_clusters, 
          "The average silhouette_score is :", silhouette_avg)
    sample_silhouette_values = silhouette_samples(data1, cluster_labels)

Month_Counter = MonthCounter(cluster_labels,n_clusters)
Year_Counter = YearCounter(cluster_labels,n_clusters)
Year_Counter2 = YearCounter(cluster_labels2,n_clusters)
Month_Counter2 = MonthCounter(cluster_labels2,n_clusters)

fig, axes = plt.subplots(nrows=2, ncols=2)
ax0, ax1, ax2, ax3 = axes.flatten()


ax0.hist(Month_Counter, 12, density=True, histtype='bar')
ax0.legend(prop={'size': 10})
ax0.set_title('Divide in months')

ax2.hist(Month_Counter2, 12, density=True, histtype='bar')
ax2.legend(prop={'size': 10})
ax2.set_title('Divide in months')

ax1.hist(Year_Counter, 10, density=True, histtype='bar')
ax1.legend(prop={'size': 10})
ax1.set_title('Divide in years')

ax3.hist(Year_Counter2, 10, density=True, histtype='bar')
ax3.legend(prop={'size': 10})
ax3.set_title('Divide in years')
