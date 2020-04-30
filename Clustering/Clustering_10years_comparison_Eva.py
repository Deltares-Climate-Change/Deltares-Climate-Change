# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 09:56:56 2020

@author: Eva
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

mdl = 4
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

n_clusters = 3
std = 1
for i in range(SubData.shape[1]):
    SubData[:,i,:] = SubData[:,i,:]-SubData[:,i,:].mean()
    SubData[:,i,:] = SubData[:,i,:]/(SubData[:,i,:].std(ddof = std))
    
def clustering(SubData,n_clusters,st):    
    SubDataStation = SubData[:,:,st] #Select the station that we are going to analyse
    #2006-2016
    data1 = SubDataStation[:][3650:]
    clusterer = KMeans(n_clusters=n_clusters, random_state=10).fit(data1)
    cluster_labels = clusterer.labels_
    #2086-2096
    data2 = SubDataStation[:][-3650:]
    clusterer2 = KMeans(n_clusters=n_clusters, random_state=10).fit(data2)
    cluster_labels2 = clusterer2.labels_
    Month_Counter_first = MonthCounter(cluster_labels,n_clusters)
    Month_Counter_last = MonthCounter(cluster_labels2,n_clusters)
    return(Month_Counter_first,Month_Counter_last)


fig, axes = plt.subplots(nrows=2, ncols=5)
ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9  = axes.flatten()
#ax10, ax11, ax12, ax13, ax14, ax15, ax17, ax18, ax19

#station 1
Month_Counter_first1,Month_Counter_last1 = clustering(SubData,n_clusters, 1)
ax0.hist(Month_Counter_first1, 12, density=True, histtype='bar')
ax0.set_title('Divide in months')
ax5.hist(Month_Counter_last1, 12, density=True, histtype='bar')
ax5.set_title('Divide in months')

#station 2
Month_Counter_first2,Month_Counter_last2 = clustering(SubData,n_clusters, 2)
ax1.hist(Month_Counter_first2, 12, density=True, histtype='bar')
ax1.set_title('Divide in months')
ax6.hist(Month_Counter_last2, 12, density=True, histtype='bar')
ax6.set_title('Divide in months')

#station 3
Month_Counter_first3,Month_Counter_last3 = clustering(SubData,n_clusters, 3)
ax2.hist(Month_Counter_first3, 12, density=True, histtype='bar')
ax2.set_title('Divide in months')
ax7.hist(Month_Counter_last3, 12, density=True, histtype='bar')
ax7.set_title('Divide in months')

#station 4
Month_Counter_first4,Month_Counter_last4 = clustering(SubData,n_clusters, 4)
ax3.hist(Month_Counter_first4, 12, density=True, histtype='bar')
ax3.set_title('Divide in months')
ax8.hist(Month_Counter_last4, 12, density=True, histtype='bar')
ax8.set_title('Divide in months')

#station 5
Month_Counter_first5,Month_Counter_last5 = clustering(SubData,n_clusters, 5)
ax4.hist(Month_Counter_first5, 12, density=True, histtype='bar')
ax4.set_title('Divide in months')
ax9.hist(Month_Counter_last5, 12, density=True, histtype='bar')
ax9.set_title('Divide in months')