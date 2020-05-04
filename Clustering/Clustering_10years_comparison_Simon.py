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

def MonthCounter(Labels,n_clusters,StartTime):
    Dat = [[] for i in range(n_clusters)]
    Dag = StartTime
    for i in range(len(Labels)):
        Dat[Labels[i]].append(Dag.month)
        Dag += datetime.timedelta(days = 1)
    return(Dat)
    
def YearCounter(Labels,n_clusters,StartTime):
    Dat = [[] for i in range(n_clusters)]
    Dag = StartTime
    for i in range(len(Labels)):
        Dat[Labels[i]].append(Dag.year)
        Dag += datetime.timedelta(days = 1)
    return(Dat)

Data = np.load('../Datares/tensor_daily_mean_5D.npy')
NanINDX = np.argwhere(np.isnan(Data))
for i in range(len(NanINDX)):
    Data[NanINDX[i][0],NanINDX[i][1],NanINDX[i][2],NanINDX[i][3],NanINDX[i][4]] = 200
<<<<<<< HEAD


=======
>>>>>>> 38c2fc8fcfc40b7206e31ca816f93d44eec34424
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

range_n_clusters = [4]

for n_clusters in range_n_clusters:
    data1 = SubDataStation
    clusterer = KMeans(n_clusters=n_clusters, random_state=10).fit(data1)
    cluster_labels = clusterer.labels_
    silhouette_avg = silhouette_score(data1, cluster_labels)   
    print("For n_clusters =", n_clusters, 
          "The average silhouette_score is :", silhouette_avg)
    sample_silhouette_values = silhouette_samples(data1, cluster_labels)

cluster_labels1 = cluster_labels[:3650]
cluster_labels2 = cluster_labels[-3650:]

Start1 = datetime.date(2006,1,1)
Start2 = datetime.date(2086,1,1)

Month_Counter = MonthCounter(cluster_labels,n_clusters,Start1)
Year_Counter = YearCounter(cluster_labels,n_clusters,Start1)

Month_Counter1 = MonthCounter(cluster_labels1,n_clusters,Start1)
Year_Counter1 = YearCounter(cluster_labels1,n_clusters,Start1)

Year_Counter2 = YearCounter(cluster_labels2,n_clusters,Start2)
Month_Counter2 = MonthCounter(cluster_labels2,n_clusters,Start2)


MC = []
YC = []
for i in range(len(Month_Counter)):
    MC.append(Month_Counter1[i])
    MC.append(Month_Counter2[i])
    YC.append(Year_Counter1[i])
    YC.append(Year_Counter2[i])
    
LAB = len(Month_Counter1)*['First 10 Years','Last 10 Years']


fig, axes = plt.subplots(nrows=2, ncols=2)
ax0, ax1, ax2, ax3 = axes.flatten()
Col = [(0.1, 0.2, 0.5),(0.1, 0.5, 0.8),(0.3, 0.2, 0.5),(0.3, 0.5, 0.8),(0.5, 0.2, 0.5),(0.5, 0.5, 0.8),(0.7, 0.2, 0.5),(0.7, 0.5, 0.8)]
ax0.hist(Month_Counter, 12, density=True, histtype='bar')
ax0.set_title('Divide in months')

ax2.hist(MC, 12, density=True, histtype='bar',label=LAB)
ax2.set_title('Divide in months')
ax2.legend(prop={'size': 10})

ax1.hist(Year_Counter, 10, density=True, histtype='bar',label=LAB)
ax1.set_title('Divide in years')

ax3.hist(YC, 10, density=True, histtype='bar')
ax3.set_title('Divide in years')
ax3.legend(prop={'size': 10})


